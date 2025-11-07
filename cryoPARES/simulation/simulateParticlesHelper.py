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

# ---------------------
# STAR handling and dataset
# ---------------------

@dataclass
class ParticlesStarSet:
    particles_md: pd.DataFrame
    optics_md: Optional[pd.DataFrame]

    @classmethod
    def load(cls, star_path: str) -> "ParticlesStarSet":
        data = starfile.read(star_path)
        if isinstance(data, dict):
            particles = data.get("particles", None)
            optics = data.get("optics", None)
            if particles is None:
                particles = data
                optics = None
        else:
            particles = data
            optics = None
        return cls(particles_md=particles, optics_md=optics)

    def __len__(self):
        return len(self.particles_md)

class ParticlesDataset(Dataset):
    """Precomputes numeric fields so the default collate works (no pandas, no None)."""
    def __init__(self, pset: ParticlesStarSet, px_A: float):
        self.pset = pset
        self.px_A = px_A

    def __len__(self):
        return len(self.pset.particles_md)

    def __getitem__(self, idx: int):
        r = self.pset.particles_md.iloc[idx]
        # Note: starfile library reads "_rlnXxx" columns as "rlnXxx" (without underscore)
        # angles (deg)
        rot  = float(r.get("rlnAngleRot", 0.0))
        tilt = float(r.get("rlnAngleTilt", 0.0))
        psi  = float(r.get("rlnAnglePsi", 0.0))
        # shifts -> pixels (dy, dx)
        sx_A = float(r.get("rlnOriginXAngst", 0.0))
        sy_A = float(r.get("rlnOriginYAngst", 0.0))
        dy_px = sy_A / self.px_A
        dx_px = sx_A / self.px_A
        # CTF params
        return {
            "angles_deg": np.array([rot, tilt, psi], dtype=np.float32),   # (3,)
            "shift_px":  np.array([dy_px, dx_px], dtype=np.float32),      # (2,)
            "dfu":  float(r.get("rlnDefocusU", 15000.0)),
            "dfv":  float(r.get("rlnDefocusV", 15000.0)),
            "dfa":  float(r.get("rlnDefocusAngle", 0.0)),
            "volt": float(r.get("rlnVoltage", 300.0)),
            "cs":   float(r.get("rlnSphericalAberration", 2.7)),
            "w":    float(r.get("rlnAmplitudeContrast", 0.07)),
            "phase":float(r.get("rlnPhaseShift", 0.0)),
            "bfac": float(r.get("rlnBfactor", 0.0)),
            # No 'img' key (avoids None in batches)
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
            pset: ParticlesStarSet,
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
            disable_tqdm: bool = False) -> List[str]:

        assert simulation_mode in ("central_slice", "noise_additive")
        os.makedirs(out_dir, exist_ok=True)

        dataset = ParticlesDataset(pset, px_A=px_A)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=False)

        H_part = W_part = self.vol_size

        # DEBUG: Simple in-memory accumulation (no buffer)
        all_images = []
        stack_paths = []

        total_particles = len(pset)
        pbar = tqdm(total=total_particles, desc="Simulating particles", unit="particle", disable=disable_tqdm)

        for batch in loader:
            euls_gt = torch.as_tensor(batch["angles_deg"], device=self.device)        # (B, 3)
            shifts_gt_px = torch.as_tensor(batch["shift_px"], device=self.device)     # (B, 2) (dy, dx)
            dfu_batch  = torch.as_tensor(batch["dfu"],  device=self.device)
            dfv_batch  = torch.as_tensor(batch["dfv"],  device=self.device)
            dfang_batch= torch.as_tensor(batch["dfa"],  device=self.device)
            volt_kV    = torch.as_tensor(batch["volt"], device=self.device)
            cs_mm      = torch.as_tensor(batch["cs"],   device=self.device)
            amp_contrast = torch.as_tensor(batch["w"],    device=self.device)
            phase_shift  = torch.as_tensor(batch["phase"],device=self.device)
            bfactors     = torch.as_tensor(batch["bfac"], device=self.device)
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

            if simulation_mode == "central_slice":
                projs = self._project_central_slice(R_gt)  # (B, H, W)
            else:
                projs_gt = self._project_central_slice(R_gt)
                projs_jit = self._project_central_slice(R_jit)

            out_imgs = []
            for i in range(B):
                H, W = H_part, W_part
                dfu = dfu_batch[i].item()
                dfv = dfv_batch[i].item()
                dfa = dfang_batch[i].item()
                volt = volt_kV[i].item()
                cs = cs_mm[i].item()
                w = amp_contrast[i].item()
                phase = phase_shift[i].item()
                bfac = bfactors[i].item()

                if simulation_mode == "central_slice":
                    proj = projs[i]
                    if apply_ctf:
                        # Use library function for CTF application
                        _, img = corrupt_with_ctf(
                            image=proj, sampling_rate=px_A,
                            dfu=dfu, dfv=dfv, dfang=dfa,
                            volt=volt, cs=cs, w=w,
                            phase_shift=phase, bfactor=bfac, fftshift=True
                        )
                    else:
                        img = proj
                    dy, dx = shifts_gt_px[i]
                    if dy != 0 or dx != 0:
                        img = fourier_shift_image_2d(img, torch.tensor([-dy, -dx], device=self.device))
                    out_imgs.append(img)
                else:
                    # Get CTF for subtraction (rfft layout, fftshift=False)
                    ctf = compute_ctf_rfft(
                        image_size=H, sampling_rate=px_A,
                        dfu=dfu, dfv=dfv, dfang=dfa,
                        volt=volt, cs=cs, w=w,
                        phase_shift=phase, bfactor=bfac,
                        fftshift=False, device=self.device
                    )

                    proj_gt = projs_gt[i].clone()
                    if shifts_gt_px[i, 0] != 0 or shifts_gt_px[i, 1] != 0:
                        proj_gt = fourier_shift_image_2d(
                            proj_gt,
                            shifts=torch.tensor([-shifts_gt_px[i, 1], -shifts_gt_px[i, 0]], device=self.device),
                        )
                    noise_only, _ = subtract_projection_T0(
                        particle=particle,
                        projection=proj_gt,
                        ctf=ctf,
                        px_A=px_A,
                        band_lo_A=sub_bp_lo_A,
                        band_hi_A=sub_bp_hi_A,
                        power_quantile=sub_power_q,
                    )

                    total_shift_jit_px = shifts_gt_px[i] + shift_jit_px[i]
                    proj_jit = projs_jit[i].clone()
                    if total_shift_jit_px[0] != 0 or total_shift_jit_px[1] != 0:
                        proj_jit = fourier_shift_image_2d(
                            proj_jit,
                            shifts=torch.tensor([-total_shift_jit_px[1], -total_shift_jit_px[0]], device=self.device),
                        )
                    if apply_ctf:
                        # Apply CTF using irfft2 (matching old code style for consistency with ctf from compute_ctf_rfft)
                        proj_jit_rfft = torch.fft.rfft2(proj_jit)
                        proj_jit = torch.fft.irfft2(proj_jit_rfft * ctf, s=(H, W))

                    out_imgs.append(noise_only + proj_jit)

            out_imgs = torch.stack(out_imgs, dim=0)  # (B, H, W)

            if snr is not None and snr > 0:
                sig_pow = out_imgs.pow(2).mean(dim=(1, 2), keepdim=True)
                noise_var = (sig_pow / snr).clamp(min=1e-12)
                noise = torch.randn_like(out_imgs) * torch.sqrt(noise_var)
                out_imgs = out_imgs + noise

            win = _hann_2d(H_part, W_part, device=self.device)
            out_imgs = out_imgs * win

            # DEBUG: Accumulate in memory (move to CPU immediately)
            out_imgs_cpu = out_imgs.cpu().numpy()
            for i in range(out_imgs_cpu.shape[0]):
                all_images.append(out_imgs_cpu[i])

            # Write to disk when we reach images_per_file
            while len(all_images) >= images_per_file:
                shard_idx = len(stack_paths)
                out_path = os.path.join(out_dir, f"{basename}_{shard_idx:04d}.mrcs")
                stack_data = np.stack(all_images[:images_per_file], axis=0)
                with mrcfile.new(out_path, overwrite=True) as mrc:
                    mrc.set_data(stack_data.astype(np.float32))
                    mrc.voxel_size = (px_A, px_A, px_A)
                stack_paths.append(out_path)
                all_images = all_images[images_per_file:]

            # Clean up GPU memory immediately
            if simulation_mode == "central_slice":
                del out_imgs, projs
            else:
                del out_imgs, projs_gt, projs_jit
            del euls_gt, angle_jit, shift_jit_px, shifts_gt_px, out_imgs_cpu
            del dfu_batch, dfv_batch, dfang_batch, volt_kV, cs_mm, amp_contrast, phase_shift, bfactors

            if torch.cuda.is_available():
                torch.cuda.synchronize(device=self.device)
            pbar.update(B)

        pbar.close()

        # DEBUG: Write remaining images (partial last shard)
        if len(all_images) > 0:
            shard_idx = len(stack_paths)
            out_path = os.path.join(out_dir, f"{basename}_{shard_idx:04d}.mrcs")
            stack_data = np.stack(all_images, axis=0)
            with mrcfile.new(out_path, overwrite=True) as mrc:
                mrc.set_data(stack_data.astype(np.float32))
                mrc.voxel_size = (px_A, px_A, px_A)
            stack_paths.append(out_path)

        return stack_paths

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
) -> List[str]:
    pset = ParticlesStarSet.load(in_star)
    sim = CryoEMSimulator(volume_mrc=volume, device=device, normalize_volume=normalize_volume)
    return sim.run(
        pset=pset,
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
) -> List[str]:
    pset_full = ParticlesStarSet.load(in_star)
    n_total = len(pset_full)
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
        particles_df = pset_full.particles_md.iloc[start_idx:end_idx].copy()

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
