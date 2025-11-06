"""
Cryo-EM particle stack simulator with projection subtraction.

Implements the method from:
"A new algorithm for particle weighted subtraction to decrease signals from
unwanted components in single particle analysis"
Fernández-Giménez et al., JSB 2023

Key principle: At low SNR (<0.05), fit scalar 'a' in frequency band where
particle ≈ a*(CTF*projection) + noise, then subtract a*(CTF*projection).
"""

import math
import os
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import mrcfile
import starfile

from starstack.particlesStar import ParticlesStarSet
from cryoPARES.constants import (
    RELION_ANGLES_NAMES, RELION_EULER_CONVENTION, RELION_SHIFTS_NAMES,
)
from cryoPARES.geometry.convert_angles import euler_angles_to_matrix
from cryoPARES.projmatching.projmatchingUtils.fourierOperations import (
    compute_dft_3d, _fourier_proj_to_real_2d,
)
from torch_fourier_slice.slice_extraction import extract_central_slices_rfft_3d
from torch_fourier_shift import fourier_shift_image_2d
from cryoPARES.datamanager.ctf.rfft_ctf import corrupt_with_ctf, compute_ctf_rfft


# ======================== Utility Functions ======================== #

def _deg_eulers_to_R(eulers_deg: torch.Tensor) -> torch.Tensor:
    """Convert Euler angles in degrees to rotation matrices."""
    return euler_angles_to_matrix(torch.deg2rad(eulers_deg), convention=RELION_EULER_CONVENTION)


def _isnan_or_none(x) -> bool:
    """Check if value is None or NaN."""
    try:
        return x is None or (isinstance(x, float) and math.isnan(x))
    except Exception:
        return False


def _get_px_A_from_optics_row(row) -> float:
    """Extract pixel size in Angstroms from optics/particles row."""
    for k in ("rlnImagePixelSize", "rlnPixelSize", "rlnDetectorPixelSize"):
        if k in row and not (isinstance(row[k], float) and math.isnan(row[k])):
            try:
                return float(row[k])
            except Exception:
                return float(pd.to_numeric(row[k], errors="coerce"))
    raise ValueError("No pixel size found in optics/particles")


def _sample_angle_jitter(B: int, max_deg: float, frac: float, device, gen=None) -> torch.Tensor:
    """Sample random angle jitter for a fraction of particles."""
    if max_deg <= 0.0 or frac <= 0.0:
        return torch.zeros((B, 3), device=device, dtype=torch.float32)
    mask = (torch.rand((B,), device=device, generator=gen) < frac).float().unsqueeze(1)
    jitter = (torch.rand((B, 3), device=device, generator=gen) * 2.0 - 1.0) * max_deg
    return jitter * mask


def _sample_shift_jitter_px(B: int, max_A: float, frac: float, px_A: float, device, gen=None) -> torch.Tensor:
    """Sample random shift jitter in pixels for a fraction of particles."""
    if max_A <= 0.0 or frac <= 0.0 or px_A <= 0:
        return torch.zeros((B, 2), device=device, dtype=torch.float32)
    mask = (torch.rand((B,), device=device, generator=gen) < frac).float().unsqueeze(1)
    jitter_A = (torch.rand((B, 2), device=device, generator=gen) * 2.0 - 1.0) * max_A
    return (jitter_A / float(px_A)) * mask


def _batch_ctf_params(md_rows: List[pd.Series], device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List]:
    """Extract CTF parameters from STAR rows as batched tensors.

    Returns:
        dfu: (B,) defocus U in Angstroms
        dfv: (B,) defocus V in Angstroms
        dfang: (B,) defocus angle in degrees
        phase_shift: (B,) phase shift in degrees
        bfactors: List of bfactor values (None or float) per particle
    """
    dfu_list = [float(row.get("rlnDefocusU", 15000.0)) for row in md_rows]
    dfv_list = [float(row.get("rlnDefocusV", 15000.0)) for row in md_rows]
    dfang_list = [float(row.get("rlnDefocusAngle", 0.0)) for row in md_rows]
    phase_shift_list = [float(row.get("rlnPhaseShift", 0.0)) if "rlnPhaseShift" in row else 0.0 for row in md_rows]
    bfactors = [None if _isnan_or_none(row.get("rlnCtfBfactor", None)) else float(row["rlnCtfBfactor"]) for row in md_rows]

    return (
        torch.tensor(dfu_list, device=device, dtype=torch.float32),
        torch.tensor(dfv_list, device=device, dtype=torch.float32),
        torch.tensor(dfang_list, device=device, dtype=torch.float32),
        torch.tensor(phase_shift_list, device=device, dtype=torch.float32),
        bfactors
    )


def _batch_shifts_px(md_rows: List[pd.Series], px_A: float, device) -> torch.Tensor:
    """Extract shifts in pixels from STAR rows as (B, 2) tensor."""
    shifts = []
    for row in md_rows:
        sx_A = float(row.get(RELION_SHIFTS_NAMES[0], 0.0))
        sy_A = float(row.get(RELION_SHIFTS_NAMES[1], 0.0))
        shifts.append([sx_A / px_A, sy_A / px_A])
    return torch.tensor(shifts, device=device, dtype=torch.float32)


# ===================== Projection Subtraction ===================== #

@torch.no_grad()
def subtract_projection_T0(
        particle: torch.Tensor,
        projection: torch.Tensor,
        ctf: torch.Tensor,
        px_A: float,
        band_lo_A: float,
        band_hi_A: float,
        power_quantile: float = 0.30,
) -> Tuple[torch.Tensor, float]:
    """
    Subtract projection from particle using T0 scaling (Fernández-Giménez et al., 2023).

    At low SNR, particle = a*signal + noise, where signal = CTF*projection.
    Fit scalar 'a' in frequency band to estimate signal amplitude, then subtract.

    Args:
        particle: (H,W) experimental particle (normalized: background mean=0, std=1)
        projection: (H,W) clean projection from volume (NOT normalized)
        ctf: (H, W//2+1) CTF in rfft layout (real, can be negative for phase flips)
        px_A: pixel size in Angstroms
        band_lo_A: low-resolution cutoff (e.g., 40 Å)
        band_hi_A: high-resolution cutoff (e.g., 8 Å)
        power_quantile: discard |CTF*P|^2 below this quantile in band

    Returns:
        noise_only: particle with signal subtracted
        a_fit: fitted scale factor
    """
    H, W = particle.shape
    device = particle.device

    # Fourier transforms
    Y = torch.fft.rfft2(particle)  # (H, W//2+1) complex
    P = torch.fft.rfft2(projection)  # (H, W//2+1) complex
    X = ctf * P  # CTF-affected projection

    # Frequency grid in Å^-1
    fy = torch.fft.fftfreq(H, d=px_A, device=device)
    fx = torch.fft.rfftfreq(W, d=px_A, device=device)
    FY, FX = torch.meshgrid(fy, fx, indexing="ij")
    freq_radius = torch.sqrt(FX ** 2 + FY ** 2)

    # Define frequency band
    f_lo = 1.0 / band_lo_A if band_lo_A > 0 else 0.0
    f_hi = 1.0 / band_hi_A if band_hi_A > 0 else float('inf')
    band_mask = (freq_radius >= f_lo) & (freq_radius <= f_hi)
    band_mask[0, 0] = False  # exclude DC

    # Compute |X|^2 and Re(conj(X)*Y) in band
    XX = X.real ** 2 + X.imag ** 2
    XY_real = X.real * Y.real + X.imag * Y.imag

    XX_band = XX[band_mask]
    XY_band = XY_real[band_mask]

    if XX_band.numel() == 0:
        return particle.clone(), 0.0

    # Threshold: keep only strong |CTF*P|^2 frequencies
    if 0.0 < power_quantile < 1.0:
        threshold = torch.quantile(XX_band, power_quantile)
        keep_mask = XX_band >= threshold
        if keep_mask.any():
            XX_band = XX_band[keep_mask]
            XY_band = XY_band[keep_mask]

    # Fit: minimize ||Y - a*X||^2
    # Solution: a = sum(Re(conj(X)*Y)) / sum(|X|^2)
    denominator = XX_band.sum()
    if denominator < 1e-20:
        return particle.clone(), 0.0

    a = float(XY_band.sum() / denominator)

    # Subtract in Fourier space: Y_noise = Y - a*X
    Y_noise = Y - a * X
    noise_only = torch.fft.irfft2(Y_noise, s=(H, W))

    return noise_only, a


# ======================== Dataset / DataLoader ======================== #

class ParticlesDatasetLite(Dataset):
    """
    Lightweight dataset wrapper for ParticlesStarSet.

    Multiprocessing-safe: Each worker process initializes its own ParticlesStarSet
    with fresh file handles to avoid deadlocks.
    """

    def __init__(self, star_fname: str, particles_dir: Optional[str], need_images: bool, px_A: float):
        # Store initialization parameters instead of the ParticlesStarSet object
        # Each worker will create its own instance
        self.star_fname = star_fname
        self.particles_dir = particles_dir
        self.need_images = need_images
        self.px_A = px_A

        # These will be initialized lazily per worker
        self._pset = None
        self._length = None

        # Initialize in main process to get length
        temp_pset = ParticlesStarSet(star_fname, particlesDir=particles_dir)
        self._length = len(temp_pset)
        # Don't keep temp_pset reference to avoid file handle issues

    def _ensure_pset(self):
        """Lazy initialization of ParticlesStarSet in each worker process."""
        if self._pset is None:
            self._pset = ParticlesStarSet(self.star_fname, particlesDir=self.particles_dir)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int):
        # Ensure this worker has its own ParticlesStarSet instance
        self._ensure_pset()

        img, md_row = self._pset[idx]
        if self.need_images:
            img = torch.as_tensor(img, dtype=torch.float32)
        else:
            img = None

        # Extract metadata here on CPU during data loading
        # This happens in parallel with GPU computation!
        angles = [float(md_row.get(k, 0.0)) for k in RELION_ANGLES_NAMES]

        dfu = float(md_row.get("rlnDefocusU", 15000.0))
        dfv = float(md_row.get("rlnDefocusV", 15000.0))
        dfang = float(md_row.get("rlnDefocusAngle", 0.0))
        phase_shift = float(md_row.get("rlnPhaseShift", 0.0)) if "rlnPhaseShift" in md_row else 0.0
        bfac_val = md_row.get("rlnCtfBfactor", None)
        bfactor = None if _isnan_or_none(bfac_val) else float(bfac_val)

        sx_A = float(md_row.get(RELION_SHIFTS_NAMES[0], 0.0))
        sy_A = float(md_row.get(RELION_SHIFTS_NAMES[1], 0.0))
        shifts = [sx_A / self.px_A, sy_A / self.px_A]

        return {
            'idx': idx,
            'img': img,
            'angles': angles,
            'ctf': (dfu, dfv, dfang, phase_shift, bfactor),
            'shifts': shifts,
        }


def _collate(batch):
    """Collate function that pre-organizes data into arrays."""
    # Separate data types
    idxs = [item['idx'] for item in batch]
    imgs = [item['img'] for item in batch]

    # Stack metadata into arrays (on CPU)
    angles = np.array([item['angles'] for item in batch], dtype=np.float32)
    shifts = np.array([item['shifts'] for item in batch], dtype=np.float32)

    # CTF params
    ctf_data = [item['ctf'] for item in batch]
    dfu = np.array([c[0] for c in ctf_data], dtype=np.float32)
    dfv = np.array([c[1] for c in ctf_data], dtype=np.float32)
    dfang = np.array([c[2] for c in ctf_data], dtype=np.float32)
    phase_shift = np.array([c[3] for c in ctf_data], dtype=np.float32)
    bfactors = [c[4] for c in ctf_data]

    return {
        'idxs': idxs,
        'imgs': imgs,
        'angles': angles,
        'shifts': shifts,
        'ctf': (dfu, dfv, dfang, phase_shift, bfactors),
    }


# ======================== Simulator Class ======================== #

class CryoEMSimulator:
    """
    Cryo-EM particle simulator with two modes:
    - central_slice: classic projection + CTF + noise
    - noise_additive: subtract ground-truth, add jittered (with proper scaling)
    """

    def __init__(self, volume_mrc: str, device: str = "cpu", normalize_volume: bool = False):
        self.device = torch.device(device)

        # Load volume
        with mrcfile.open(volume_mrc, permissive=True) as mrc:
            vol_np = np.asarray(mrc.data.copy(), dtype=np.float32)
            if vol_np.ndim == 4:
                vol_np = vol_np.squeeze()
            assert vol_np.ndim == 3, "Volume must be 3D"
            assert vol_np.shape[0] == vol_np.shape[1] == vol_np.shape[2], "Volume must be cubic"
            self.voxel_size_A = float(abs(mrc.voxel_size.x)) if mrc.voxel_size.x != 0 else None

        self.vol_size = int(vol_np.shape[0])
        vol = torch.as_tensor(vol_np, device=self.device, dtype=torch.float32)

        if normalize_volume:
            raise NotImplementedError("This will be required for cryoSPARC")

        # Precompute volume FFT
        vol_rfft, vol_shape, _ = compute_dft_3d(vol, pad_length=0)
        self.vol_rfft = vol_rfft.to(torch.complex64)
        self.base_image_shape = (int(vol_shape[-1]), int(vol_shape[-1]))

    def _project(self, R: torch.Tensor) -> torch.Tensor:
        """
        Generate projections from volume.

        Args:
            R: (B, 3, 3) rotation matrices

        Returns:
            (B, H, W) projections
        """
        projs_rfft = extract_central_slices_rfft_3d(
            self.vol_rfft,
            image_shape=self.base_image_shape,
            rotation_matrices=R,
            fftfreq_max=None,
            zyx_matrices=False
        )
        projs_real = _fourier_proj_to_real_2d(projs_rfft, pad_length=None)
        return projs_real

    def _get_shifts_px(self, row: pd.Series, px_A: float) -> Tuple[float, float]:
        """Extract shifts in pixels from STAR row."""
        sx_A = float(row.get(RELION_SHIFTS_NAMES[0], 0.0))
        sy_A = float(row.get(RELION_SHIFTS_NAMES[1], 0.0))
        return (sx_A / px_A, sy_A / px_A)

    @torch.no_grad()
    def run(
            self,
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
            sub_bp_lo_A: float = 40.0,
            sub_bp_hi_A: float = 8.0,
            sub_power_q: float = 0.30,
            random_seed: Optional[int] = None,
            disable_tqdm: bool = False,
    ) -> str:
        """Run simulation and write output stack + STAR file."""

        os.makedirs(out_dir, exist_ok=True)

        # Extract metadata
        parts_df = pset.particles_md
        optics_df = getattr(pset, "optics_md", None)
        optics_row = optics_df.iloc[0] if optics_df is not None else parts_df.iloc[0]
        px_A = _get_px_A_from_optics_row(optics_row)

        volt_kV = float(optics_row.get("rlnVoltage", 300.0))
        cs_mm = float(optics_row.get("rlnSphericalAberration", 2.7))
        amp_contrast = float(optics_row.get("rlnAmplitudeContrast", 0.07))

        # Setup data loader with multiprocessing support
        # Each worker will lazily initialize its own ParticlesStarSet to avoid file handle conflicts
        need_images = (simulation_mode == "noise_additive")
        star_fname = pset.starFname
        particles_dir = getattr(pset, 'particlesDir', None)
        ds = ParticlesDatasetLite(
            star_fname=star_fname,
            particles_dir=particles_dir,
            need_images=need_images,
            px_A=px_A
        )
        # Enable pin_memory for faster CPU->GPU transfer when using CUDA
        use_pin_memory = self.device.type == 'cuda'
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=_collate,
            pin_memory=use_pin_memory, persistent_workers=num_workers > 0
        )

        # RNG for jitter
        gen = torch.Generator(device=self.device)
        if random_seed is not None:
            gen.manual_seed(int(random_seed))

        # Output file handling
        shard_idx = 0
        stack_paths: List[str] = []
        buffer = None
        out_path = os.path.join(out_dir, f"{basename}_{shard_idx:04d}.mrcs")

        def _flush_shard(buf: np.ndarray, path: str):
            with mrcfile.new(path, overwrite=True) as m:
                m.set_data(buf)
                m.voxel_size = (px_A, px_A, px_A)
            stack_paths.append(path)

        # Determine particle size from first image
        H_part = W_part = self.vol_size

        # Progress bar
        total_particles = len(pset)
        pbar = tqdm(total=total_particles, desc="Simulating particles", unit="particle", disable=disable_tqdm)

        for batch_idx, batch_data in enumerate(loader):
            imgs = batch_data['imgs']
            B = len(imgs)

            # Get particle size from first batch with images
            if need_images and imgs[0] is not None:
                H_part = int(imgs[0].shape[-2])
                W_part = int(imgs[0].shape[-1])

            # Convert pre-processed numpy arrays to GPU tensors (fast!)
            euls_gt = torch.from_numpy(batch_data['angles']).to(self.device, dtype=torch.float32)
            shifts_gt_px = torch.from_numpy(batch_data['shifts']).to(self.device, dtype=torch.float32)

            # CTF parameters (already extracted on CPU during data loading)
            dfu_np, dfv_np, dfang_np, phase_shift_np, bfactors = batch_data['ctf']
            dfu_batch = torch.from_numpy(dfu_np).to(self.device, dtype=torch.float32)
            dfv_batch = torch.from_numpy(dfv_np).to(self.device, dtype=torch.float32)
            dfang_batch = torch.from_numpy(dfang_np).to(self.device, dtype=torch.float32)
            phase_shift_batch = torch.from_numpy(phase_shift_np).to(self.device, dtype=torch.float32)

            # Sample jitter
            angle_jit = _sample_angle_jitter(B, angle_jitter_deg, angle_jitter_frac, device=self.device, gen=gen)
            shift_jit_px = _sample_shift_jitter_px(B, shift_jitter_A, shift_jitter_frac, px_A, device=self.device, gen=gen)

            # Generate projections
            if simulation_mode == "central_slice":
                R_sim = _deg_eulers_to_R(euls_gt + angle_jit)
                projs = self._project(R_sim)  # (B, H, W)

                # Apply shifts (ground-truth + jitter) in batch - fully vectorized!
                total_shifts_px = shifts_gt_px + shift_jit_px  # (B, 2)
                # Relion convention: shifts are corrections, so negate
                shift_vectors = torch.stack([-total_shifts_px[:, 1], -total_shifts_px[:, 0]], dim=1)  # (B, 2): [-y, -x]

                # Apply shifts to all projections at once (batch operation)
                projs = fourier_shift_image_2d(projs, shifts=shift_vectors)  # (B, H, W)

                # Apply CTF in batch (if all have same bfactor or all None)
                if apply_ctf:
                    # Check if we can batch CTF (all bfactors are None or a constant)
                    unique_bfactors = set(bfactors)
                    bfactor = list(unique_bfactors)[0] if len(unique_bfactors) == 1 else None

                    # Apply CTF in-place to avoid memory accumulation
                    for i in range(B):
                        bf = bfactor if bfactor is not None else bfactors[i]
                        _, projs[i] = corrupt_with_ctf(
                            image=projs[i], sampling_rate=px_A,
                            dfu=dfu_batch[i], dfv=dfv_batch[i], dfang=dfang_batch[i],
                            volt=volt_kV, cs=cs_mm, w=amp_contrast,
                            phase_shift=phase_shift_batch[i], bfactor=bf, fftshift=True
                        )

                out_imgs = projs

            else:
                # noise_additive mode - still needs per-particle processing for subtraction
                R_gt = _deg_eulers_to_R(euls_gt)
                projs_gt = self._project(R_gt)  # (B, H, W)
                R_jit = _deg_eulers_to_R(euls_gt + angle_jit)
                projs_jit = self._project(R_jit)  # (B, H, W)

                # Pre-allocate output tensor to avoid list building
                out_imgs = torch.zeros((B, H_part, W_part), device=self.device, dtype=torch.float32)

                for i in range(B):
                    particle = torch.as_tensor(imgs[i], device=self.device, dtype=torch.float32)

                    # Get CTF (rfft layout, fftshift=False)
                    ctf = compute_ctf_rfft(
                        image_size=H_part, sampling_rate=px_A,
                        dfu=dfu_batch[i], dfv=dfv_batch[i], dfang=dfang_batch[i],
                        volt=volt_kV, cs=cs_mm, w=amp_contrast,
                        phase_shift=phase_shift_batch[i], bfactor=bfactors[i],
                        fftshift=False, device=self.device
                    )

                    # SUBTRACTION: ground-truth pose and shifts
                    proj_gt = projs_gt[i].clone()
                    if shifts_gt_px[i, 0] != 0 or shifts_gt_px[i, 1] != 0:
                        proj_gt = fourier_shift_image_2d(
                            proj_gt,
                            shifts=torch.tensor([-shifts_gt_px[i, 1], -shifts_gt_px[i, 0]], device=self.device)
                        )

                    # Subtract with T0 scaling
                    noise_only, a_fit = subtract_projection_T0(
                        particle=particle,
                        projection=proj_gt,
                        ctf=ctf,
                        px_A=px_A,
                        band_lo_A=sub_bp_lo_A,
                        band_hi_A=sub_bp_hi_A,
                        power_quantile=sub_power_q,
                    )

                    # ADDITION: jittered pose and shifts
                    total_shift_jit_px = shifts_gt_px[i] + shift_jit_px[i]
                    proj_jit = projs_jit[i].clone()
                    if total_shift_jit_px[0] != 0 or total_shift_jit_px[1] != 0:
                        proj_jit = fourier_shift_image_2d(
                            proj_jit,
                            shifts=torch.tensor([-total_shift_jit_px[1], -total_shift_jit_px[0]], device=self.device)
                        )

                    # Add with same scale factor
                    P_jit = torch.fft.rfft2(proj_jit)
                    X_jit = ctf * P_jit
                    signal_jit = torch.fft.irfft2(a_fit * X_jit, s=(H_part, W_part))

                    # Write directly to pre-allocated tensor
                    out_imgs[i] = noise_only + signal_jit

                    # Explicit cleanup to help garbage collector
                    del particle, ctf, proj_gt, noise_only, proj_jit, P_jit, X_jit, signal_jit

            # Add Gaussian noise to entire batch (vectorized!)
            if snr is not None and snr > 0:
                # Compute variance for each image in the batch
                sig_vars = torch.var(out_imgs, dim=(-2, -1))  # (B,)
                noise_stds = torch.sqrt(sig_vars / snr).clamp(min=1e-6)  # (B,)
                noise = torch.randn_like(out_imgs) * noise_stds.view(B, 1, 1)
                out_imgs += noise

            # Convert entire batch to numpy and add to buffer
            batch_np = out_imgs.detach().cpu().numpy().astype(np.float32)  # (B, H, W)

            # Explicit cleanup of GPU tensors to prevent memory accumulation
            del out_imgs, projs, euls_gt, angle_jit, shift_jit_px
            del dfu_batch, dfv_batch, dfang_batch, phase_shift_batch, shifts_gt_px
            if simulation_mode != "central_slice":
                del projs_gt, projs_jit
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Add batch to buffer
            if buffer is None:
                buffer = batch_np
            else:
                buffer = np.concatenate([buffer, batch_np], axis=0)

            # Flush shards as needed
            while buffer.shape[0] >= images_per_file:
                _flush_shard(buffer[:images_per_file], out_path)
                buffer = buffer[images_per_file:]
                shard_idx += 1
                out_path = os.path.join(out_dir, f"{basename}_{shard_idx:04d}.mrcs")

            pbar.update(B)

        pbar.close()

        # Flush remaining buffer
        if buffer is not None and buffer.shape[0] > 0:
            _flush_shard(buffer, out_path)

        # Write STAR file
        image_names_all: List[str] = []
        for p in stack_paths:
            with mrcfile.open(p, permissive=True, mode="r") as m:
                n_in_file = m.data.shape[0]
            base = os.path.basename(p)
            image_names_all.extend([f"{k + 1}@{base}" for k in range(n_in_file)])

        out_star = os.path.join(out_dir, f"{basename}.star")
        parts_out = parts_df.copy()
        assert len(image_names_all) == len(parts_out), "Mismatch: images vs STAR rows"
        parts_out["rlnImageName"] = image_names_all

        star_dict = {"particles": parts_out}
        if optics_df is not None:
            star_dict["optics"] = optics_df
        starfile.write(star_dict, out_star, overwrite=True)

        return out_star


# ======================== Public API ======================== #

@torch.no_grad()
def run_simulation(
        volume: str,
        in_star: str,
        output_dir: str,
        basename: str = "stack",
        images_per_file: int = 10_000,
        batch_size: int = 128,
        num_workers: int = 0,
        apply_ctf: bool = True,
        snr: Optional[float] = None,
        use_gpu: bool = False,
        device: Optional[str] = None,
        simulation_mode: str = "central_slice",
        angle_jitter_deg: float = 0.0,
        angle_jitter_frac: float = 0.0,
        shift_jitter_A: float = 0.0,
        shift_jitter_frac: float = 0.0,
        sub_bp_lo_A: float = 40.0,
        sub_bp_hi_A: float = 8.0,
        sub_power_q: float = 0.30,
        random_seed: Optional[int] = None,
        disable_tqdm: bool = False,
) -> str:
    """Run particle simulation."""
    if device is None:
        device = "cuda:0" if use_gpu and torch.cuda.is_available() else "cpu"

    pset = ParticlesStarSet(in_star)
    sim = CryoEMSimulator(volume_mrc=volume, device=device)

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
        sub_bp_lo_A=sub_bp_lo_A,
        sub_bp_hi_A=sub_bp_hi_A,
        sub_power_q=sub_power_q,
        random_seed=random_seed,
        disable_tqdm=disable_tqdm,
    )


# Note: CLI has been moved to simulateParticles.py
# This module provides the core simulation functionality