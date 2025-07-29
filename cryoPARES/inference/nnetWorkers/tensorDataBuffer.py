# streaming_buffer.py
#
# Optimised two‑stage inference buffer + extensive test‑suite.
# Run:  python streaming_buffer.py


import itertools
from typing import Any, Callable, Dict, List, Union


import torch
from torch import nn


TensorOrList = Union[torch.Tensor, List[Any]]




class StreamingBuffer(nn.Module):
    """
    Accumulates items selected by a first stage until `buffer_size` items are
    ready, then calls `processing_fn` (the second stage) with a *contiguous*
    batch.  Uses ring buffers to avoid reallocations.
    """


    def __init__(
        self,
        buffer_size: int,
        processing_fn: Callable[[Dict[str, TensorOrList]], Any],
        max_buffer_capacity: int | None = None,
    ) -> None:
        super().__init__()


        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive")


        self.buffer_size = int(buffer_size)
        self.processing_fn = processing_fn
        self.max_capacity = int(max_buffer_capacity or buffer_size * 2)
        if self.max_capacity < self.buffer_size:
            raise ValueError("max_buffer_capacity must be ≥ buffer_size")


        self._tensor_buffers: dict[str, torch.Tensor] = {}
        self._list_buffers: dict[str, list[Any]] = {}
        self._tensor_workspaces: dict[str, torch.Tensor] = {}
        self._list_workspaces: dict[str, list[Any]] = {}


        self._head: int = 0      # first valid element
        self._length: int = 0    # number of valid elements


    # ------------------------------------------------------------------ #
    #                 internal helpers (allocate / copy)                 #
    # ------------------------------------------------------------------ #
    def _ensure_storage(self, key: str, value: TensorOrList) -> None:
        if isinstance(value, torch.Tensor):
            if key not in self._tensor_buffers:
                buf_shape = (self.max_capacity, *value.shape[1:])
                self._tensor_buffers[key] = torch.empty(
                    buf_shape, dtype=value.dtype, device=value.device
                )
                ws_shape = (self.buffer_size, *value.shape[1:])
                self._tensor_workspaces[key] = torch.empty(
                    ws_shape, dtype=value.dtype, device=value.device
                )
        elif isinstance(value, list):
            if key not in self._list_buffers:
                self._list_buffers[key] = [None] * self.max_capacity
                self._list_workspaces[key] = [None] * self.buffer_size
        else:
            raise TypeError(
                f"Unsupported type for key '{key}': {type(value).__name__}"
            )


    def _copy_from_ring(self, key: str, length: int, dst_start: int) -> None:
        if length == 0:
            return
        first = min(length, self.max_capacity - self._head)
        second = length - first


        if key in self._tensor_buffers:
            src, dst = self._tensor_buffers[key], self._tensor_workspaces[key]
            dst[dst_start : dst_start + first].copy_(src[self._head : self._head + first])
            if second:
                dst[dst_start + first : dst_start + length].copy_(src[0:second])
        else:
            src, dst = self._list_buffers[key], self._list_workspaces[key]
            dst[dst_start : dst_start + first] = src[self._head : self._head + first]
            if second:
                dst[dst_start + first : dst_start + length] = src[0:second]


    def _copy_into_ring(
        self, key: str, src: TensorOrList, src_start: int, count: int
    ) -> None:
        if count == 0:
            return
        tail = (self._head + self._length) % self.max_capacity
        first = min(count, self.max_capacity - tail)
        second = count - first


        if key in self._tensor_buffers:
            dst = self._tensor_buffers[key]
            dst[tail : tail + first].copy_(src[src_start : src_start + first])
            if second:
                dst[0:second].copy_(src[src_start + first : src_start + count])
        else:
            dst = self._list_buffers[key]
            dst[tail : tail + first] = src[src_start : src_start + first]
            if second:
                dst[0:second] = src[src_start + first : src_start + count]


    # ------------------------------------------------------------------ #
    #                                API                                 #
    # ------------------------------------------------------------------ #
    def add_batch(self, batch: Dict[str, TensorOrList]) -> Any | None:
        if not batch:
            return None
        batch_size = len(next(iter(batch.values())))
        if batch_size == 0:
            return None


        for k, v in batch.items():
            self._ensure_storage(k, v)


        results: list[Any] = []
        new_pos = 0


        while self._length + (batch_size - new_pos) >= self.buffer_size:
            take_ring = min(self._length, self.buffer_size)
            take_new = self.buffer_size - take_ring


            for k, v in batch.items():
                self._copy_from_ring(k, take_ring, 0)
                if take_new:
                    if isinstance(v, torch.Tensor):
                        self._tensor_workspaces[k][take_ring : self.buffer_size].copy_(
                            v[new_pos : new_pos + take_new]
                        )
                    else:
                        ws = self._list_workspaces[k]
                        ws[take_ring : self.buffer_size] = v[
                            new_pos : new_pos + take_new
                        ]


            proc_input = {
                k: (self._tensor_workspaces[k][: self.buffer_size]
                    if k in self._tensor_workspaces
                    else self._list_workspaces[k][: self.buffer_size])
                for k in batch
            }
            results.append(self.processing_fn(**proc_input))


            self._head = (self._head + take_ring) % self.max_capacity
            self._length -= take_ring
            new_pos += take_new


        remain = batch_size - new_pos
        if remain:
            if self._length + remain > self.max_capacity:
                raise RuntimeError("StreamingBuffer overflow")
            for k, v in batch.items():
                self._copy_into_ring(k, v, new_pos, remain)
            self._length += remain


        if not results:
            return None
        if len(results) == 1:
            return results[0]
        return self._combine_results(results)


    def flush(self) -> Any | None:
        if self._length == 0:
            return None


        proc_input: dict[str, TensorOrList] = {}
        for k in itertools.chain(self._tensor_buffers, self._list_buffers):
            n = self._length
            first = min(n, self.max_capacity - self._head)
            second = n - first


            if k in self._tensor_buffers:
                src = self._tensor_buffers[k]
                tmp = torch.empty(
                    (n, *src.shape[1:]), dtype=src.dtype, device=src.device
                )
                tmp[:first].copy_(src[self._head : self._head + first])
                if second:
                    tmp[first:].copy_(src[0:second])
                proc_input[k] = tmp
            else:
                src = self._list_buffers[k]
                out = [None] * n
                out[:first] = src[self._head : self._head + first]
                if second:
                    out[first:] = src[0:second]
                proc_input[k] = out


        self._head = 0
        self._length = 0


        return self.processing_fn(**proc_input)


    # ------------------------------------------------------------------ #
    @staticmethod
    def _combine_results(all_results: List[Any]) -> Any:
        all_results = [r for r in all_results if r is not None]
        if not all_results:
            return None


        first = all_results[0]
        if isinstance(first, torch.Tensor):
            return torch.cat(all_results, dim=0)
        if isinstance(first, (list, tuple)):
            merged = []
            for i in range(len(first)):
                parts = [r[i] for r in all_results]
                if isinstance(parts[0], torch.Tensor):
                    merged.append(torch.cat(parts, dim=0))
                elif isinstance(parts[0], list):
                    merged.append(list(itertools.chain.from_iterable(parts)))
                else:
                    merged.append(parts)
            return merged
        return all_results  # scalars → list




# ======================================================================
#                              TESTS                                    #
# ======================================================================


def _tensor(start: int, count: int, col_dim: int = 1) -> torch.Tensor:
    return (
        torch.arange(start, start + count)
        .unsqueeze(1)
        .expand(count, col_dim)
        .float()
    )




def test_order_preservation():
    seen: List[int] = []


    def proc_fn(data):
        seen.extend(data[:, 0].int().tolist())
        return data.clone()


    buf = StreamingBuffer(4, proc_fn, 8)
    buf.add_batch({"data": _tensor(0, 3)})
    assert seen == [] and buf._length == 3


    buf.add_batch({"data": _tensor(3, 5)})
    assert seen == list(range(8)) and buf._length == 0


    assert buf.flush() is None
    print("test_order_preservation ✓")




def test_various_chunking():
    scenarios = [
        ("Small, no processing",      5, 4,  [],           9),
        ("Small, triggers",           7, 8,  [10],         5),
        ("Large, multiples exact",    5, 15, [10, 10],     0),
        ("Large, multiples left",     5, 18, [10, 10],     3),
        ("Empty, large left",         0, 23, [10, 10],     3),
        ("Empty, small batch",        0, 7,  [],           7),
        ("Exact fit in two calls",    0, 6,  [],           6),
        ("Buffer » Batch space left", 3, 4,  [],           7),
    ]


    for name, init_n, new_n, exp_chunks, exp_left in scenarios:
        processed: List[int] = []


        def proc_fn(data):
            processed.append(data.shape[0])
            return data.shape[0]


        buf = StreamingBuffer(10, proc_fn, 20)
        if init_n:
            buf.add_batch({"data": _tensor(0, init_n)})


        processed.clear()
        buf.add_batch({"data": _tensor(100, new_n)})


        assert processed == exp_chunks, f"{name}: chunks {processed} ≠ {exp_chunks}"
        assert buf._length == exp_left, f"{name}: leftovers {buf._length} ≠ {exp_left}"


        if buf._length:
            flush_sz = buf.flush()
            assert flush_sz == exp_left and buf._length == 0


    print("test_various_chunking ✓")




def test_mixed_types():
    def proc_fn(names, scores):
        return names, scores


    buf = StreamingBuffer(3, proc_fn, 6)
    buf.add_batch({"names": ["A", "B"], "scores": torch.ones(2, 1)})
    assert buf._length == 2


    names_scores = buf.add_batch(
        {"names": ["C", "D", "E"], "scores": torch.ones(3, 1)}
    )
    names, scores = names_scores
    assert names == ["A", "B", "C"] and scores.shape[0] == 3 and buf._length == 2


    names_left, scores_left = buf.flush()
    assert names_left == ["D", "E"] and scores_left.shape[0] == 2
    print("test_mixed_types ✓")




# ------------------------------------------------------------------ #
#                        NEW ADDITIONAL TESTS                        #
# ------------------------------------------------------------------ #
def test_repeated_small_batches():
    processed: List[int] = []


    def proc_fn(data):
        processed.append(data.shape[0])
        return data.shape[0]


    buf = StreamingBuffer(3, proc_fn, 6)
    buf.add_batch({"data": _tensor(0, 1)})
    buf.add_batch({"data": _tensor(1, 1)})
    assert buf._length == 2 and processed == []


    buf.add_batch({"data": _tensor(2, 2)})
    assert processed == [3] and buf._length == 1


    flushed = buf.flush()
    assert flushed == 1 and buf._length == 0
    print("test_repeated_small_batches ✓")




def test_flush_empty():
    buf = StreamingBuffer(4, lambda **kw: None)
    assert buf.flush() is None
    print("test_flush_empty ✓")




def test_capacity_boundary():
    processed: List[int] = []


    def proc_fn(data):
        processed.append(data.shape[0])
        return data.shape[0]


    buf = StreamingBuffer(4, proc_fn, 4)  # capacity == buffer_size
    buf.add_batch({"data": _tensor(0, 3)})
    buf.add_batch({"data": _tensor(3, 2)})


    assert processed == [4] and buf._length == 1
    assert buf.flush() == 1 and buf._length == 0
    print("test_capacity_boundary ✓")




def test_tuple_result_combination():
    def proc_fn(names, vecs):
        return names, vecs


    buf = StreamingBuffer(2, proc_fn, 5)
    names = [f"n{i}" for i in range(5)]
    combined = buf.add_batch({"names": names, "vecs": _tensor(0, 5)})


    # two chunks processed → combined list returned
    proc_names, proc_vecs = combined
    assert proc_names == names[:4] and proc_vecs.shape[0] == 4 and buf._length == 1


    left_names, left_vecs = buf.flush()
    assert left_names == [names[4]] and left_vecs.shape[0] == 1
    print("test_tuple_result_combination ✓")




# ======================================================================
if __name__ == "__main__":
    torch.manual_seed(0)


    test_order_preservation()
    test_various_chunking()
    test_mixed_types()
    test_repeated_small_batches()
    test_flush_empty()
    test_capacity_boundary()
    test_tuple_result_combination()


    print("\nAll tests passed ✨")
