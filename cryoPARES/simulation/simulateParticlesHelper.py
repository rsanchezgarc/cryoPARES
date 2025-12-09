"""
Cryo-EM particle stack simulator with projection subtraction.

Key idea: at low SNR, fit scalar 'a' in a frequency band where
particle ≈ a*(CTF*projection) + noise, then subtract a*(CTF*projection).
"""

import math
import os
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

import mrcfile
import numpy as np
import pandas as pd
import starfile
import torch
import torch.fft as fft
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Import existing library functions
from cryoPARES.geometry.convert_angles import euler_angles_to_matrix
from cryoPARES.projmatching.projmatchingUtils.fourierOperations import (
    compute_dft_3d, _fourier_proj_to_real_2d,
)
from torch_fourier_slice.slice_extraction import extract_central_slices_rfft_3d
from torch_fourier_shift import fourier_shift_image_2d
from cryoPARES.datamanager.ctf.rfft_ctf import corrupt_with_ctf, compute_ctf_rfft
from cryoPARES.constants import RELION_EULER_CONVENTION
from starstack.particlesStar import ParticlesStarSet

# ---------------------
# Utilities
# ---------------------

def _hann_1d(n: int, device: torch.device) -> torch.Tensor:
    x = torch.linspace(0, math.pi, steps=n, device=device)
    return 0.5 * (1 - torch.cos(2 * x))

def _hann_2d(h: int, w: int, device: torch.device) -> torch.Tensor:
    wx = _hann_1d(w, device=device)
    wy = _hann_1d(h, device=device)
    return wy[:, None] * wx[None, :]

def rfft2c(x: torch.Tensor) -> torch.Tensor:
    return fft.rfft2(x, norm="ortho")

def irfft2c(X: torch.Tensor, s: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    return fft.irfft2(X, s=s, norm="ortho")

def complex_abs2(x: torch.Tensor) -> torch.Tensor:
    return (x.real ** 2 + x.imag ** 2)

def fftfreq_2d(h: int, w: int, px_A: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute 2D frequency grid for RFFT layout.

    Returns:
        FY: Y-frequencies (H, W//2+1)
        FX: X-frequencies (H, W//2+1)
        R: Radial frequencies (H, W//2+1)
    """
    fy = torch.fft.fftfreq(h, d=px_A, device=device)  # (H,)
    fx = torch.fft.rfftfreq(w, d=px_A, device=device)  # (W//2+1,)
    FY, FX = torch.meshgrid(fy, fx, indexing='ij')  # (H, W//2+1) each
    R = torch.sqrt(FY**2 + FX**2)
    return FY, FX, R

# ---------------------
# STAR handling and dataset
# ---------------------

class ParticlesDataset(Dataset):
    """
    Lazy-loading dataset for simulation that extracts metadata from STAR files.
    Follows particlesDataset.py pattern with lazy ParticlesStarSet initialization.
    Pre-computes CTF in __getitem__ to enable fully vectorized simulation loop.
    """
    def __init__(self, star_path: str, px_A: float, image_size: int, device: str, apply_ctf: bool, n_first_particles: Optional[int] = None):
        """
        Args:
            star_path: Path to STAR file
            px_A: Pixel size in Angstroms
            image_size: Image size in pixels (for CTF computation)
            device: Device for CTF computation
            apply_ctf: Whether to compute CTF (if False, returns None for ctf)
            n_first_particles: Optional: Limit to the first N particles
        """
        self.star_path = star_path
        self.px_A = px_A
        self.image_size = image_size
        self.device = device
        self.apply_ctf = apply_ctf
        self.n_first_particles = n_first_particles
        self._particles = None  # Lazy-loaded

    @property
    def particles(self) -> ParticlesStarSet:
        """Lazy-load ParticlesStarSet to avoid data locks in multiprocessing."""
        if self._particles is None:
            self._particles = ParticlesStarSet(self.star_path)
            if self.n_first_particles is not None:
                self._particles.particles_md = self._particles.particles_md.head(self.n_first_particles)
        return self._particles

    def __len__(self):
        return len(self.particles)

    def __getitem__(self, idx: int):
        """
        Extract metadata for a single particle and pre-compute CTF.
        Returns dict with angles, shifts, and pre-computed CTF tensor.
        """
        r = self.particles.particles_md.iloc[idx]

        # Extract optics group parameters (following particlesDataset.py line 259-260)
        optics_group_num = int(r.get('rlnOpticsGroup', 1))
        optics_data = self.particles.optics_md.query(f'rlnOpticsGroup == {optics_group_num}')

        # Angles (degrees)
        rot  = float(r.get("rlnAngleRot", 0.0))
        tilt = float(r.get("rlnAngleTilt", 0.0))
        psi  = float(r.get("rlnAnglePsi", 0.0))

        # Shifts: Angstroms -> pixels (dy, dx)
        sx_A = float(r.get("rlnOriginXAngst", 0.0))
        sy_A = float(r.get("rlnOriginYAngst", 0.0))
        dy_px = sy_A / self.px_A
        dx_px = sx_A / self.px_A

        # CTF parameters
        dfu = float(r.get("rlnDefocusU", optics_data.get("rlnDefocusU", pd.Series([15000.0])).iloc[0]))
        dfv = float(r.get("rlnDefocusV", optics_data.get("rlnDefocusV", pd.Series([15000.0])).iloc[0]))
        dfa = float(r.get("rlnDefocusAngle", 0.0))
        volt = float(optics_data["rlnVoltage"].iloc[0])
        cs = float(optics_data["rlnSphericalAberration"].iloc[0])
        w = float(optics_data["rlnAmplitudeContrast"].iloc[0])
        phase = float(r.get("rlnPhaseShift", 0.0))
        bfac = float(r.get("rlnBfactor", 0.0))

        # Pre-compute CTF if needed (defers computation to dataset, avoiding loop in simulation)
        # Note: CTF computed on CPU to avoid CUDA fork issues with DataLoader workers
        if self.apply_ctf:
            ctf = compute_ctf_rfft(
                image_size=self.image_size, sampling_rate=self.px_A,
                dfu=dfu, dfv=dfv, dfang=dfa,
                volt=volt, cs=cs, w=w,
                phase_shift=phase, bfactor=bfac,
                fftshift=True, device="cpu"  # Compute on CPU, will be moved to GPU in main process
            )  # (H, W//2+1)
        else:
            ctf = None

        return {
            "angles_deg": np.array([rot, tilt, psi], dtype=np.float32),
            "shift_px": np.array([dy_px, dx_px], dtype=np.float32),
            "ctf": ctf,  # Pre-computed CTF tensor or None
        }

# ---------------------
# Subtraction scaling (T0 band-limited LS fit)
# ---------------------

def subtract_projection_T0(particle: torch.Tensor,
                           projection: torch.Tensor,
                           ctf: torch.Tensor,
                           px_A: float,
                           band_lo_A: float,
                           band_hi_A: float,
                           power_quantile: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fit scalar a for a * (CTF * projection) to match particle in a band; return noise-only image and a."""
    H, W = particle.shape
    P = rfft2c(particle)
    S = rfft2c(projection)
    FY, FX, R = fftfreq_2d(H, W, px_A, device=particle.device)
    band = (R >= 1.0 / band_hi_A) & (R <= 1.0 / band_lo_A)
    Ppow = complex_abs2(P)
    thresh = torch.quantile(Ppow[band], power_quantile)
    mask = band & (Ppow <= thresh)
    CS = ctf * S
    denom = torch.sum(complex_abs2(CS) * mask) + 1e-12
    A = torch.sum(torch.conj(CS) * P * mask) / denom
    a = A.real
    sub = particle - irfft2c(CS * a, s=(H, W))
    return sub, a

# ---------------------
# Double-buffered async I/O with CUDA streams
# ---------------------

class BufferPool:
    """
    Manages double-buffered CPU-pinned memory for async GPU→CPU transfers.
    Uses CUDA streams and events for non-blocking transfers with explicit synchronization.
    """

    def __init__(self, images_per_file: int, image_shape: Tuple[int, int], device: torch.device):
        self.device = device
        self.images_per_file = images_per_file
        self.image_shape = image_shape

        # Allocate CPU-side buffers (pinned if CUDA)
        if device.type == 'cuda':
            # Pinned memory enables fast DMA transfers
            self.buffer_a = torch.empty(
                (images_per_file, *image_shape),
                dtype=torch.float32,
                pin_memory=True
            )
            self.buffer_b = torch.empty(
                (images_per_file, *image_shape),
                dtype=torch.float32,
                pin_memory=True
            )
            # CUDA events for synchronization
            self.event_a = torch.cuda.Event()
            self.event_b = torch.cuda.Event()
            # Stream for async copies
            self.stream = torch.cuda.Stream(device=device)
        else:
            # CPU-only: regular numpy arrays
            self.buffer_a = np.empty((images_per_file, *image_shape), dtype=np.float32)
            self.buffer_b = np.empty((images_per_file, *image_shape), dtype=np.float32)
            self.event_a = None
            self.event_b = None
            self.stream = None

        # Counters for valid data in each buffer
        self.count_a = 0
        self.count_b = 0

        # Which buffer is currently being filled by GPU ('a' or 'b')
        self.active = 'a'

        # Which buffer is ready for writer (None, 'a', or 'b')
        self.ready = None

        # Thread synchronization
        self.lock = threading.Lock()
        self.buffer_ready = threading.Condition(self.lock)  # Signal to writer thread
        self.buffer_available = threading.Condition(self.lock)  # Signal to GPU thread

    def submit_batch(self, batch_gpu: torch.Tensor) -> None:
        """
        Submit a batch from GPU to CPU buffer (non-blocking).

        Args:
            batch_gpu: Tensor on GPU (B, H, W)
        """
        batch_size = batch_gpu.shape[0]

        with self.lock:
            # Wait until we have space in a buffer that we can safely use
            # This ensures we never try to swap to a non-empty buffer
            while True:
                # Check current buffer space
                if self.active == 'a':
                    space = self.images_per_file - self.count_a
                else:
                    space = self.images_per_file - self.count_b

                # If we have space in the active buffer, we're good to go
                if space > 0:
                    break

                # Active buffer is full. We need to swap. But first check if a buffer is being written
                if self.ready is not None:
                    # A buffer is currently being written - we must wait for it to be consumed
                    # (Can't swap if both buffers are full)
                    self.buffer_available.wait(timeout=0.1)
                    continue  # Re-check after wait

                # Active buffer is full AND no buffer is being written - we can proceed and swap
                break

            # Get active buffer and its metadata
            if self.active == 'a':
                buf = self.buffer_a
                count = self.count_a
                event = self.event_a
            else:
                buf = self.buffer_b
                count = self.count_b
                event = self.event_b

            # Determine how many images we can fit in this buffer
            to_copy = min(batch_size, space)

        # Non-blocking GPU→CPU copy (outside lock for performance)
        if self.device.type == 'cuda':
            with torch.cuda.stream(self.stream):
                # Async copy to pinned CPU memory
                buf[count:count+to_copy].copy_(batch_gpu[:to_copy], non_blocking=True)
            # Record completion event in the stream
            event.record(self.stream)
        else:
            # CPU-only: direct copy
            if isinstance(buf, np.ndarray):
                buf[count:count+to_copy] = batch_gpu[:to_copy].numpy()
            else:
                buf[count:count+to_copy] = batch_gpu[:to_copy]

        # Update counter and check for buffer swap
        with self.lock:
            if self.active == 'a':
                self.count_a += to_copy
                if self.count_a >= self.images_per_file:
                    # Before swapping, wait if the other buffer isn't empty yet
                    while self.ready is not None:
                        self.buffer_available.wait(timeout=0.1)
                    self._swap_buffers()
            else:
                self.count_b += to_copy
                if self.count_b >= self.images_per_file:
                    # Before swapping, wait if the other buffer isn't empty yet
                    while self.ready is not None:
                        self.buffer_available.wait(timeout=0.1)
                    self._swap_buffers()

        # If we couldn't fit the entire batch, recursively submit the rest
        if to_copy < batch_size:
            self.submit_batch(batch_gpu[to_copy:])

    def _swap_buffers(self) -> None:
        """Swap active and ready buffers (must be called under lock)."""
        # Switch to the other buffer
        next_buffer = 'b' if self.active == 'a' else 'a'

        # Sanity check: the other buffer should be empty
        next_count = self.count_b if next_buffer == 'b' else self.count_a
        if next_count > 0:
            raise RuntimeError(f"Internal error: trying to swap to non-empty buffer {next_buffer} (count={next_count})")

        # Mark current active buffer as ready for writing
        self.ready = self.active
        # Switch active pointer
        self.active = next_buffer
        # Signal writer thread
        self.buffer_ready.notify()

    def finish(self) -> None:
        """Flush any remaining partial buffer."""
        with self.lock:
            # First, wait for any existing ready buffer to be consumed
            while self.ready is not None:
                self.buffer_available.wait(timeout=0.1)

            # Now mark the active buffer as ready if it has data
            if self.active == 'a' and self.count_a > 0:
                self.ready = 'a'
                self.buffer_ready.notify()
            elif self.active == 'b' and self.count_b > 0:
                self.ready = 'b'
                self.buffer_ready.notify()

    def get_ready_buffer(self) -> Optional[Tuple[torch.Tensor, int, Optional[torch.cuda.Event]]]:
        """
        Get the ready buffer for writing (called by writer thread under lock).
        Returns (buffer, count, event) or None if no buffer is ready.
        """
        if self.ready is None:
            return None

        if self.ready == 'a':
            return self.buffer_a, self.count_a, self.event_a
        else:
            return self.buffer_b, self.count_b, self.event_b

    def mark_buffer_consumed(self) -> None:
        """Mark the ready buffer as consumed (called by writer thread under lock)."""
        if self.ready == 'a':
            self.count_a = 0
        elif self.ready == 'b':
            self.count_b = 0
        self.ready = None
        # Signal GPU thread that a buffer is now available
        self.buffer_available.notify()


class AsyncMRCWriter:
    """Background thread for writing MRC files using double-buffered async I/O."""

    def __init__(self, buffer_pool: BufferPool, out_dir: str, basename: str, px_A: float):
        self.buffer_pool = buffer_pool
        self.out_dir = out_dir
        self.basename = basename
        self.px_A = px_A
        self.stack_paths = []
        self.worker_thread = None
        self.stop_event = threading.Event()
        self.error = None

    def start(self):
        """Start the background writer thread."""
        self.worker_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.worker_thread.start()

    def _writer_loop(self):
        """Worker thread: wait for full buffer, synchronize, then write."""
        shard_idx = 0

        try:
            while not self.stop_event.is_set():
                # Wait for a ready buffer
                with self.buffer_pool.lock:
                    while self.buffer_pool.ready is None and not self.stop_event.is_set():
                        self.buffer_pool.buffer_ready.wait(timeout=0.1)

                    if self.stop_event.is_set():
                        # Check one more time for any final buffer before exiting
                        result = self.buffer_pool.get_ready_buffer()
                        if result is None:
                            return
                    else:
                        # Get buffer metadata
                        result = self.buffer_pool.get_ready_buffer()
                        if result is None:
                            continue

                    buf, count, event = result

                # *** EXPLICIT SYNCHRONIZATION ***
                # Wait for all GPU→CPU transfers to this buffer to complete
                if event is not None:
                    event.synchronize()  # Blocks writer thread until GPU transfer done

                # Convert to numpy for writing (must copy to avoid race condition)
                if self.buffer_pool.device.type == 'cuda':
                    buf_np = buf[:count].numpy().copy()  # Create copy to avoid overwrite
                else:
                    if isinstance(buf, np.ndarray):
                        buf_np = buf[:count].copy()
                    else:
                        buf_np = buf[:count].numpy().copy()

                # Write to disk
                out_path = os.path.join(self.out_dir, f"{self.basename}_{shard_idx:04d}.mrcs")
                self._write_mrc(buf_np, out_path)
                shard_idx += 1

                # Mark buffer as consumed
                with self.buffer_pool.lock:
                    self.buffer_pool.mark_buffer_consumed()

        except Exception as e:
            self.error = e
            # CRITICAL: Clean up buffer state to unblock GPU thread
            # If we don't do this, the GPU thread will wait forever
            with self.buffer_pool.lock:
                self.buffer_pool.ready = None
                self.buffer_pool.buffer_available.notify_all()
            # Stop the writer loop
            self.stop_event.set()

    def _write_mrc(self, data: np.ndarray, path: str):
        """Write numpy array to MRC file."""
        with mrcfile.new(path, overwrite=True) as m:
            m.set_data(data.astype(np.float32))
            m.voxel_size = (self.px_A, self.px_A, self.px_A)
        self.stack_paths.append(path)

    def finish(self):
        """Signal writer to flush remaining data and wait for completion."""
        # Flush any partial buffer
        self.buffer_pool.finish()

        # Wait for writer thread to consume the final buffer
        # (poll until ready flag is cleared, meaning writer consumed it)
        import time
        max_wait = 60  # 60 seconds max
        start = time.time()
        while time.time() - start < max_wait:
            with self.buffer_pool.lock:
                if self.buffer_pool.ready is None:
                    # Writer has consumed the final buffer
                    break
            time.sleep(0.01)  # 10ms poll interval
        else:
            # Timeout - writer is stuck
            print("Warning: Writer thread did not consume final buffer within 60s")

        # Signal writer thread to stop
        self.stop_event.set()

        # Wake up writer thread if it's waiting
        with self.buffer_pool.lock:
            self.buffer_pool.buffer_ready.notify()

        # Wait for writer thread to finish
        if self.worker_thread:
            self.worker_thread.join(timeout=300)  # 5 minute timeout

        if self.error:
            raise self.error

    def get_stack_paths(self) -> List[str]:
        """Get list of written stack paths."""
        return self.stack_paths

    def check_error(self) -> None:
        """Check if writer thread has encountered an error and raise it if so."""
        if self.error:
            raise RuntimeError(f"Writer thread failed: {self.error}") from self.error

# ---------------------
# Main simulator
# ---------------------

class CryoEMSimulator:
    """
    Modes:
      - 'central_slice': projection + optional CTF + noise
      - 'noise_additive': subtract GT, then add jittered projection (scaled)
    """

    def __init__(self, volume_mrc: str, device: str = "cpu", normalize_volume: bool = False):
        self.device = torch.device(device)
        with mrcfile.open(volume_mrc, permissive=True) as mrc:
            vol_np = np.asarray(mrc.data.copy(), dtype=np.float32)
            if vol_np.ndim == 4:
                vol_np = vol_np.squeeze()
            assert vol_np.ndim == 3, "Volume must be 3D"
        vol = torch.from_numpy(vol_np).to(self.device)
        self.vol_size = vol.shape[-1]
        if normalize_volume:
            vol = (vol - vol.mean()) / (vol.std() + 1e-8)
        # Use library function to compute volume FFT
        vol_rfft, vol_shape, _ = compute_dft_3d(vol, pad_length=0)
        self.vol_rfft = vol_rfft.to(torch.complex64)
        self.base_image_shape = (int(vol_shape[-1]), int(vol_shape[-1]))

    def _project_central_slice(self, R: torch.Tensor) -> torch.Tensor:
        # Use library functions with correct parameters
        projs_rfft = extract_central_slices_rfft_3d(
            self.vol_rfft,
            image_shape=self.base_image_shape,
            rotation_matrices=R,
            fftfreq_max=None,
            zyx_matrices=False
        )
        return _fourier_proj_to_real_2d(projs_rfft, pad_length=None)

    def run(self,
            in_star: str,
            out_dir: str,
            basename: str,
            images_per_file: int,
            batch_size: int,
            simulation_mode: str,
            apply_ctf: bool,
            snr: Optional[float],
            num_workers: int,
            angle_jitter_deg: float = 0.0,
            angle_jitter_frac: float = 0.0,
            shift_jitter_A: float = 0.0,
            shift_jitter_frac: float = 0.0,
            bandpass_lo_A: float = 6.0,
            bandpass_hi_A: float = 25.0,
            sub_bp_lo_A: float = 8.0,
            sub_bp_hi_A: float = 20.0,
            sub_power_q: float = 0.85,
            px_A: float = 1.0,
            disable_tqdm: bool = False,
            n_first_particles: Optional[int] = None) -> List[str]:

        assert simulation_mode in ("central_slice", "noise_additive")
        os.makedirs(out_dir, exist_ok=True)

        dataset = ParticlesDataset(
            star_path=in_star,
            px_A=px_A,
            image_size=self.vol_size,
            device=self.device,
            apply_ctf=apply_ctf,
            n_first_particles=n_first_particles
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=False,
                            persistent_workers=True if num_workers > 0 else False)

        H_part = W_part = self.vol_size

        # Initialize double-buffered system
        buffer_pool = BufferPool(
            images_per_file=images_per_file,
            image_shape=(H_part, W_part),
            device=self.device
        )
        writer = AsyncMRCWriter(
            buffer_pool=buffer_pool,
            out_dir=out_dir,
            basename=basename,
            px_A=px_A
        )
        writer.start()

        total_particles = len(dataset)
        pbar = tqdm(total=total_particles, desc="Simulating particles", unit="particle", disable=disable_tqdm)

        for batch in loader:
            euls_gt = torch.as_tensor(batch["angles_deg"], device=self.device)        # (B, 3)
            shifts_gt_px = torch.as_tensor(batch["shift_px"], device=self.device)     # (B, 2) (dy, dx)
            # Pre-computed CTFs from dataset (if apply_ctf=True, else None)
            # CTFs computed on CPU in workers, moved to GPU here
            if apply_ctf:
                ctfs_list = batch["ctf"]
                if isinstance(ctfs_list, list):
                    ctfs = torch.stack(ctfs_list, dim=0).to(self.device)  # (B, H, W//2+1)
                else:
                    ctfs = ctfs_list.to(self.device)
            else:
                ctfs = None
            B = euls_gt.shape[0]

            angle_jit = torch.zeros_like(euls_gt)
            if angle_jitter_deg > 0.0:
                angle_jit += angle_jitter_deg * (torch.rand_like(angle_jit) - 0.5) * 2.0
            if angle_jitter_frac > 0.0:
                angle_jit += angle_jitter_frac * (torch.rand_like(angle_jit) - 0.5) * torch.abs(euls_gt)

            shift_jit_px = torch.zeros_like(shifts_gt_px)
            if shift_jitter_A > 0.0:
                shift_jit_px += (shift_jitter_A / px_A) * (torch.rand_like(shift_jit_px) - 0.5) * 2.0
            if shift_jitter_frac > 0.0:
                shift_jit_px += shift_jitter_frac * (torch.rand_like(shift_jit_px) - 0.5) * torch.abs(shifts_gt_px)

            euls_jit = euls_gt + angle_jit
            # Use library function for rotation matrices
            R_gt = euler_angles_to_matrix(torch.deg2rad(euls_gt), convention=RELION_EULER_CONVENTION)
            R_jit = euler_angles_to_matrix(torch.deg2rad(euls_jit), convention=RELION_EULER_CONVENTION)

            # Vectorized projection and CTF application
            if simulation_mode == "central_slice":
                projs = self._project_central_slice(R_gt)  # (B, H, W)

                if apply_ctf:
                    # Fully vectorized CTF application using pre-computed CTFs from dataset
                    projs_fft = torch.fft.rfft2(projs)  # (B, H, W//2+1)
                    projs_fft = torch.fft.fftshift(projs_fft, dim=-2)  # Shift first dim only

                    # Apply pre-computed CTFs (batched multiplication, no loops!)
                    projs_fft = projs_fft * ctfs  # (B, H, W//2+1) * (B, H, W//2+1)

                    projs_fft = torch.fft.ifftshift(projs_fft, dim=-2)  # Undo shift
                    out_imgs = torch.fft.irfft2(projs_fft, s=(H_part, W_part))  # (B, H, W)
                else:
                    out_imgs = projs

                # Apply shifts (vectorized using batch-capable fourier_shift_image_2d)
                # Negate shifts for RELION convention: shifts are corrections
                shifts_vec = torch.stack([-shifts_gt_px[:, 1], -shifts_gt_px[:, 0]], dim=1)  # (B, 2): [-dy, -dx]
                out_imgs = fourier_shift_image_2d(out_imgs, shifts_vec)

            else:
                # noise_additive mode - partially vectorized (subtraction_T0 requires per-particle loop)
                projs_gt = self._project_central_slice(R_gt)
                projs_jit = self._project_central_slice(R_jit)

                out_imgs = []
                for i in range(B):
                    # Use pre-computed CTF (needs fftshift=False for this mode, so recompute if needed)
                    # TODO: Consider pre-computing both fftshift=True and False variants in dataset
                    if apply_ctf:
                        # For now, recompute CTF without fftshift for noise_additive mode
                        # This mode is rarely used, so acceptable
                        raise NotImplementedError(
                            "noise_additive mode with apply_ctf=True requires CTF with fftshift=False. "
                            "Use central_slice mode instead, or implement dual CTF computation in dataset."
                        )

                    proj_gt = fourier_shift_image_2d(
                        projs_gt[i],
                        torch.tensor([-shifts_gt_px[i, 1], -shifts_gt_px[i, 0]], device=self.device)
                    )

                    noise_only, _ = subtract_projection_T0(
                        particle=None,  # TODO: need to load actual particle for this mode
                        projection=proj_gt, ctf=None, px_A=px_A,
                        band_lo_A=sub_bp_lo_A, band_hi_A=sub_bp_hi_A, power_quantile=sub_power_q
                    )

                    total_shift = shifts_gt_px[i] + shift_jit_px[i]
                    proj_jit = fourier_shift_image_2d(
                        projs_jit[i],
                        torch.tensor([-total_shift[1], -total_shift[0]], device=self.device)
                    )

                    out_imgs.append(noise_only + proj_jit)

                out_imgs = torch.stack(out_imgs, dim=0)

            # Vectorized windowing (apply to entire batch)
            win = _hann_2d(H_part, W_part, device=self.device)
            out_imgs = out_imgs * win[None, :, :]  # Broadcast over batch dimension

            # Vectorized noise addition with robust edge case handling
            if snr is not None and snr > 0:
                # Compute variance per image (B, 1, 1)
                sig_var = out_imgs.var(dim=(-2, -1), keepdim=True)

                # Vectorized fallback: use 1.0 where variance is <= 0
                # This replaces: noise_std = math.sqrt(sig_var / snr) if sig_var > 0 else 1.0
                noise_std = torch.where(
                    sig_var > 0,
                    torch.sqrt(sig_var / snr),
                    torch.tensor(1.0, device=sig_var.device, dtype=sig_var.dtype)
                )

                # Add noise (fully vectorized)
                out_imgs = out_imgs + torch.randn_like(out_imgs) * noise_std

            # Submit to buffer pool (non-blocking GPU→CPU transfer)
            buffer_pool.submit_batch(out_imgs)

            # Check for writer errors (fail fast instead of hanging)
            writer.check_error()

            # Clean up GPU memory immediately
            if simulation_mode == "central_slice":
                del out_imgs, projs
                if ctfs is not None:
                    del ctfs
            else:
                del out_imgs, projs_gt, projs_jit
            del euls_gt, angle_jit, shift_jit_px, shifts_gt_px

            if torch.cuda.is_available():
                torch.cuda.synchronize(device=self.device)
            pbar.update(B)

        pbar.close()

        # Signal completion to buffer pool
        buffer_pool.finish()

        # Wait for writer to finish and get results
        writer.finish()

        return writer.get_stack_paths()

# ---------------------
# Public API
# ---------------------

def run_simulation(
        volume: str,
        in_star: str,
        output_dir: str,
        basename: str,
        images_per_file: int,
        batch_size: int,
        simulation_mode: str = "central_slice",
        apply_ctf: bool = True,
        snr: Optional[float] = None,
        num_workers: int = 0,
        angle_jitter_deg: float = 0.0,
        angle_jitter_frac: float = 0.0,
        shift_jitter_A: float = 0.0,
        shift_jitter_frac: float = 0.0,
        bandpass_lo_A: float = 6.0,
        bandpass_hi_A: float = 25.0,
        sub_bp_lo_A: float = 8.0,
        sub_bp_hi_A: float = 20.0,
        sub_power_q: float = 0.85,
        px_A: float = 1.0,
        device: str = "cpu",
        normalize_volume: bool = False,
        disable_tqdm: bool = False,
        n_first_particles: Optional[int] = None,
) -> List[str]:
    sim = CryoEMSimulator(volume_mrc=volume, device=device, normalize_volume=normalize_volume)
    return sim.run(
        in_star=in_star,
        out_dir=output_dir,
        basename=basename,
        images_per_file=images_per_file,
        batch_size=batch_size,
        simulation_mode=simulation_mode,
        apply_ctf=apply_ctf,
        snr=snr,
        num_workers=num_workers,
        angle_jitter_deg=angle_jitter_deg,
        angle_jitter_frac=angle_jitter_frac,
        shift_jitter_A=shift_jitter_A,
        shift_jitter_frac=shift_jitter_frac,
        bandpass_lo_A=bandpass_lo_A,
        bandpass_hi_A=bandpass_hi_A,
        sub_bp_lo_A=sub_bp_lo_A,
        sub_bp_hi_A=sub_bp_hi_A,
        sub_power_q=sub_power_q,
        px_A=px_A,
        disable_tqdm=disable_tqdm,
        n_first_particles=n_first_particles,
    )

# ---------------------
# Multi-GPU sharding helper
# ---------------------

def run_simulation_sharded(
        volume: str,
        in_star: str,
        output_dir: str,
        basename: str,
        images_per_file: int,
        batch_size: int,
        simulation_mode: str,
        apply_ctf: bool,
        snr: Optional[float],
        angle_jitter_deg: float,
        angle_jitter_frac: float,
        shift_jitter_A: float,
        shift_jitter_frac: float,
        bandpass_lo_A: float,
        bandpass_hi_A: float,
        sub_bp_lo_A: float,
        sub_bp_hi_A: float,
        sub_power_q: float,
        px_A: float,
        gpus: List[int],
        normalize_volume: bool = False,
        disable_tqdm: bool = False,
        n_first_particles: Optional[int] = None,
) -> List[str]:
    pset_full = ParticlesStarSet(in_star)
    particles_to_shard = pset_full.particles_md
    if n_first_particles is not None:
        particles_to_shard = particles_to_shard.head(n_first_particles)

    n_total = len(particles_to_shard)
    n_gpus = len(gpus)
    shard_sizes = [n_total // n_gpus + (1 if i < (n_total % n_gpus) else 0) for i in range(n_gpus)]

    os.makedirs(output_dir, exist_ok=True)

    stack_paths_all: List[str] = []
    start_idx = 0
    for gi, gpu_id in enumerate(gpus):
        end_idx = start_idx + shard_sizes[gi]
        if start_idx >= end_idx:
            continue

        shard_star = os.path.join(output_dir, f".shard_{gpu_id}.star")
        particles_df = particles_to_shard.iloc[start_idx:end_idx].copy()

        star_dict = {"particles": particles_df}
        if hasattr(pset_full, "optics_md") and pset_full.optics_md is not None:
            star_dict["optics"] = pset_full.optics_md
        starfile.write(star_dict, shard_star, overwrite=True)

        shard_basename = f"{basename}_gpu{gpu_id}"
        out_paths = run_simulation(
            volume=volume,
            in_star=shard_star,
            output_dir=output_dir,
            basename=shard_basename,
            images_per_file=images_per_file,
            batch_size=batch_size,
            simulation_mode=simulation_mode,
            apply_ctf=apply_ctf,
            snr=snr,
            num_workers=0,
            angle_jitter_deg=angle_jitter_deg,
            angle_jitter_frac=angle_jitter_frac,
            shift_jitter_A=shift_jitter_A,
            shift_jitter_frac=shift_jitter_frac,
            bandpass_lo_A=bandpass_lo_A,
            bandpass_hi_A=bandpass_hi_A,
            sub_bp_lo_A=sub_bp_lo_A,
            sub_bp_hi_A=sub_bp_hi_A,
            sub_power_q=sub_power_q,
            px_A=px_A,
            device=f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu",
            normalize_volume=normalize_volume,
            disable_tqdm=disable_tqdm,
        )

        stack_paths_all.extend(out_paths)
        try:
            os.remove(shard_star)
        except OSError:
            pass

        start_idx = end_idx

    return stack_paths_all
