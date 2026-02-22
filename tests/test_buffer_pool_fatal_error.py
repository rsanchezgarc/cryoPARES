"""
Tests for BufferPool / AsyncMRCWriter fatal-error propagation.

These tests do not require real particles or volumes — they inject failures
directly into the writer thread and verify that the producer thread receives
the original exception rather than the misleading "swap to non-empty buffer"
RuntimeError.
"""
import threading
import time

import numpy as np
import pytest
import torch

from cryoPARES.simulation.simulateParticlesHelper import AsyncMRCWriter, BufferPool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pool(images_per_file: int = 4, image_shape=(8, 8)):
    device = torch.device("cpu")
    return BufferPool(images_per_file=images_per_file, image_shape=image_shape, device=device)


def _small_batch(n: int = 2, image_shape=(8, 8)) -> torch.Tensor:
    return torch.zeros(n, *image_shape)


# ---------------------------------------------------------------------------
# 1. fatal_error set before submit_batch → raises immediately
# ---------------------------------------------------------------------------

def test_submit_batch_raises_when_fatal_error_already_set():
    pool = _make_pool()
    pool.fatal_error = ValueError("disk full")

    with pytest.raises(RuntimeError, match="AsyncMRCWriter failed") as exc_info:
        pool.submit_batch(_small_batch())

    # The original exception must be chained
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert "disk full" in str(exc_info.value.__cause__)


# ---------------------------------------------------------------------------
# 2. fatal_error set while producer is waiting for buffer_available
#    (simulates writer crashing while both buffers are full)
# ---------------------------------------------------------------------------

def test_submit_batch_unblocks_when_fatal_error_set_during_wait():
    """
    Fill the active buffer, then set fatal_error from another thread
    while the producer is blocked in the wait loop.
    """
    pool = _make_pool(images_per_file=2)  # tiny buffer so it fills fast

    # Fill the active buffer completely
    pool.submit_batch(_small_batch(2))
    # Now ready='a', active='b', count_b=0
    # Fill 'b' as well by faking it so the pool looks full
    with pool.lock:
        pool.count_b = 2  # pretend 'b' is also full
        pool.ready = 'a'  # keep 'a' as ready (not consumed)

    errors_seen = []

    def producer():
        try:
            # This should block because ready is not None AND count_b==images_per_file
            pool.submit_batch(_small_batch(1))
        except RuntimeError as e:
            errors_seen.append(e)

    t = threading.Thread(target=producer)
    t.start()

    # Give the producer time to enter the wait
    time.sleep(0.05)

    # Inject fatal error from "writer thread"
    with pool.lock:
        pool.fatal_error = OSError("permission denied")
        pool.buffer_available.notify_all()

    t.join(timeout=2.0)
    assert not t.is_alive(), "Producer thread did not unblock"
    assert len(errors_seen) == 1
    assert isinstance(errors_seen[0].__cause__, OSError)


# ---------------------------------------------------------------------------
# 3. _swap_buffers raises fatal_error instead of "non-empty buffer" error
# ---------------------------------------------------------------------------

def test_swap_buffers_raises_fatal_error_not_sanity_check():
    pool = _make_pool(images_per_file=2)

    # Simulate the inconsistent state the old bug would create:
    # ready=None but count_b > 0  (the "other" buffer is non-empty)
    with pool.lock:
        pool.active = 'a'
        pool.count_a = 2   # active buffer is full
        pool.count_b = 2   # other buffer is also full (inconsistent!)
        pool.ready = None
        # Set fatal_error as the new code does
        pool.fatal_error = MemoryError("OOM in numpy conversion")

    with pytest.raises(RuntimeError, match="AsyncMRCWriter failed") as exc_info:
        with pool.lock:
            pool._swap_buffers()

    assert isinstance(exc_info.value.__cause__, MemoryError)


# ---------------------------------------------------------------------------
# 4. AsyncMRCWriter: writer failure propagates through check_error()
# ---------------------------------------------------------------------------

def test_async_writer_check_error_exposes_writer_exception(tmp_path):
    pool = _make_pool(images_per_file=4)
    writer = AsyncMRCWriter(
        buffer_pool=pool,
        out_dir=str(tmp_path),
        basename="test",
        px_A=1.0,
    )
    writer.start()

    # Inject an error directly (bypassing the real writer loop)
    sentinel = RuntimeError("forced writer failure")
    with pool.lock:
        pool.fatal_error = sentinel
        pool.buffer_available.notify_all()
    writer.error = sentinel

    with pytest.raises(RuntimeError, match="AsyncMRCWriter failed"):
        writer.check_error()


# ---------------------------------------------------------------------------
# 5. End-to-end: writer crashes on first write → producer gets the error,
#    NOT the "swap to non-empty buffer" sanity check error
# ---------------------------------------------------------------------------

def test_writer_crash_propagates_to_producer(tmp_path, monkeypatch):
    """
    Patch _write_mrc to raise on the first call, then drive a full simulation
    loop via submit_batch calls and verify the producer sees the real error.
    """
    pool = _make_pool(images_per_file=2, image_shape=(4, 4))
    writer = AsyncMRCWriter(
        buffer_pool=pool,
        out_dir=str(tmp_path),
        basename="test",
        px_A=1.0,
    )

    call_count = [0]
    orig_write = writer._write_mrc

    def failing_write(data, path):
        call_count[0] += 1
        raise IOError("simulated disk failure")

    monkeypatch.setattr(writer, "_write_mrc", failing_write)
    writer.start()

    # Submit enough batches to trigger at least one file write (images_per_file=2,
    # batch_size=2 → first submit fills buffer and signals writer immediately).
    caught = []
    for _ in range(20):
        time.sleep(0.01)  # give writer thread a chance to fail
        try:
            pool.submit_batch(_small_batch(2, image_shape=(4, 4)))
            writer.check_error()
        except RuntimeError as e:
            caught.append(e)
            break

    writer.stop_event.set()

    assert caught, "Producer should have received an error"
    err = caught[0]
    # Must be chained to the original IOError, not the sanity-check RuntimeError
    assert isinstance(err.__cause__, IOError), (
        f"Expected IOError cause, got: {err.__cause__!r}"
    )
    assert "simulated disk failure" in str(err.__cause__)