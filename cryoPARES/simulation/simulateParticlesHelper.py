"""
Cryo-EM particle stack simulator with projection subtraction.

Key idea: at low SNR, fit scalar 'a' in a frequency band where
particle ≈ a*(CTF*projection) + noise, then subtract a*(CTF*projection).
"""

import math
import os
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
    fy = torch.fft.fftfreq(h, d=px_A, device=device)
    fx = torch.fft.rfftfreq(w, d=px_A, device=device)
    FY, FX = torch.meshgrid(fy, fx, indexing="ij")
    R = torch.sqrt(FY ** 2 + FX ** 2)
    return FY, FX, R

def fourier_shift_image_2d(img: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    """Shift image by (dy, dx) pixels via Fourier shift theorem."""
    H, W = img.shape
    Y = torch.fft.fftfreq(H, d=1.0, device=img.device)
    X = torch.fft.fftfreq(W, d=1.0, device=img.device)
    FY, FX = torch.meshgrid(Y, X, indexing="ij")
    phase = torch.exp(-2j * math.pi * (shifts[0] * FY + shifts[1] * FX))
    F = torch.fft.fft2(img) * phase
    return torch.fft.ifft2(F).real

# ---------------------
# Projection extraction (central slice)
# ---------------------

def rotation_matrix_from_euler(angles_deg: torch.Tensor) -> torch.Tensor:
    """angles_deg: (B, 3) — RELION ZYZ (Rot, Tilt, Psi). Returns (B, 3, 3)."""
    a, b, g = [torch.deg2rad(angles_deg[:, i]) for i in range(3)]
    ca, sa = torch.cos(a), torch.sin(a)
    cb, sb = torch.cos(b), torch.sin(b)
    cg, sg = torch.cos(g), torch.sin(g)
    Rz1 = torch.stack([
        torch.stack([ca, -sa, torch.zeros_like(a)], dim=-1),
        torch.stack([sa,  ca, torch.zeros_like(a)], dim=-1),
        torch.stack([torch.zeros_like(a), torch.zeros_like(a), torch.ones_like(a)], dim=-1),
    ], dim=-2)
    Ry  = torch.stack([
        torch.stack([cb, torch.zeros_like(b), sb], dim=-1),
        torch.stack([torch.zeros_like(b), torch.ones_like(b), torch.zeros_like(b)], dim=-1),
        torch.stack([-sb, torch.zeros_like(b), cb], dim=-1),
    ], dim=-2)
    Rz2 = torch.stack([
        torch.stack([cg, -sg, torch.zeros_like(g)], dim=-1),
        torch.stack([sg,  cg, torch.zeros_like(g)], dim=-1),
        torch.stack([torch.zeros_like(g), torch.zeros_like(g), torch.ones_like(g)], dim=-1),
    ], dim=-2)
    return Rz1 @ Ry @ Rz2

def extract_central_slices_rfft_3d(vol_rfft: torch.Tensor,
                                   image_shape: Tuple[int, int],
                                   rotation_matrices: torch.Tensor) -> torch.Tensor:
    """Nearest-neighbor central-slice sampling from 3D Fourier volume.
    Returns (B, H, W//2+1) complex RFFT planes.
    """
    B = rotation_matrices.shape[0]
    H, W = image_shape
    device = vol_rfft.device

    kz = torch.fft.fftfreq(vol_rfft.shape[-3], d=1.0, device=device)
    ky = torch.fft.fftfreq(vol_rfft.shape[-2], d=1.0, device=device)
    kx = torch.fft.rfftfreq(vol_rfft.shape[-1], d=1.0, device=device)

    fy = torch.fft.fftfreq(H, d=1.0, device=device)
    fx = torch.fft.rfftfreq(W, d=1.0, device=device)
    FY2, FX2 = torch.meshgrid(fy, fx, indexing="ij")
    FZ2 = torch.zeros_like(FY2)

    planes = []
    for i in range(B):
        R = rotation_matrices[i]
        coords = torch.stack([FZ2, FY2, FX2], dim=-1)  # (H, W/2+1, 3) Z,Y,X
        coords_rot = coords @ R.T
        z_idx = torch.clamp(((coords_rot[..., 0] - kz[0]) / (kz[1] - kz[0])).round().long(), 0, vol_rfft.shape[-3]-1)
        y_idx = torch.clamp(((coords_rot[..., 1] - ky[0]) / (ky[1] - ky[0])).round().long(), 0, vol_rfft.shape[-2]-1)
        x_idx = torch.clamp(((coords_rot[..., 2] - kx[0]) / (kx[1] - kx[0])).round().long(), 0, vol_rfft.shape[-1]-1)
        planes.append(vol_rfft[z_idx, y_idx, x_idx])

    return torch.stack(planes, dim=0)

def _fourier_proj_to_real_2d(proj_rfft: torch.Tensor) -> torch.Tensor:
    return irfft2c(proj_rfft, s=None)

# ---------------------
# CTF Model
# ---------------------

def ctf_2d_rfft(H: int, W: int, px_A: float, defocus_u_A: float, defocus_v_A: float,
                defocus_ang_deg: float, volt: float, cs: float, w: float,
                phase_shift: float, bfactor: float,
                device: torch.device) -> torch.Tensor:
    """CTF on RFFT grid (H, W//2+1). volt in kV, cs in mm, defocus in Å, bfactor in Å^2."""
    FY, FX, R = fftfreq_2d(H, W, px_A, device=device)
    lam = 12.3986 / math.sqrt((volt * 1e3) * (1 + 0.97845e-6 * (volt * 1e3)))  # Å
    cs_A = cs * 1e7  # mm -> Å
    astig = (defocus_u_A + defocus_v_A) / 2 + (defocus_u_A - defocus_v_A) / 2 * torch.cos(
        2 * (torch.atan2(FY, FX) - math.radians(defocus_ang_deg))
    )
    chi = math.pi * lam * astig * (R ** 2) - 0.5 * math.pi * cs_A * (lam ** 3) * (R ** 4) + phase_shift
    c = -(torch.sin(chi) * (1 - w) + torch.cos(chi) * w)  # mixed phase/amp contrast
    env = torch.exp(-bfactor * (R ** 2))
    return c * env

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
        # angles (deg)
        rot  = float(r.get("_rlnAngleRot", 0.0))
        tilt = float(r.get("_rlnAngleTilt", 0.0))
        psi  = float(r.get("_rlnAnglePsi", 0.0))
        # shifts -> pixels (dy, dx)
        sx_A = float(r.get("_rlnOriginXAngst", 0.0))
        sy_A = float(r.get("_rlnOriginYAngst", 0.0))
        dy_px = sy_A / self.px_A
        dx_px = sx_A / self.px_A
        # CTF params
        return {
            "angles_deg": np.array([rot, tilt, psi], dtype=np.float32),   # (3,)
            "shift_px":  np.array([dy_px, dx_px], dtype=np.float32),      # (2,)
            "dfu":  float(r.get("_rlnDefocusU", 15000.0)),
            "dfv":  float(r.get("_rlnDefocusV", 15000.0)),
            "dfa":  float(r.get("_rlnDefocusAngle", 0.0)),
            "volt": float(r.get("_rlnVoltage", 300.0)),
            "cs":   float(r.get("_rlnSphericalAberration", 2.7)),
            "w":    float(r.get("_rlnAmplitudeContrast", 0.07)),
            "phase":float(r.get("_rlnPhaseShift", 0.0)),
            "bfac": float(r.get("_rlnBfactor", 0.0)),
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
        self.vol = torch.from_numpy(vol_np).to(self.device)
        self.vol_size = self.vol.shape[-1]
        if normalize_volume:
            v = self.vol
            self.vol = (v - v.mean()) / (v.std() + 1e-8)
        self.vol_rfft = fft.rfftn(self.vol, dim=(-3, -2, -1))
        self.base_image_shape = (self.vol.shape[-2], self.vol.shape[-1])

    def _project_central_slice(self, R: torch.Tensor) -> torch.Tensor:
        projs_rfft = extract_central_slices_rfft_3d(self.vol_rfft, self.base_image_shape, R)
        return _fourier_proj_to_real_2d(projs_rfft)

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

        shard_idx = 0
        stack_paths: List[str] = []
        buffer: List[np.ndarray] = []
        buffer_size = 0
        out_path = os.path.join(out_dir, f"{basename}_{shard_idx:04d}.mrcs")

        def _flush_shard(buf_list: List[np.ndarray], path: str):
            buf_array = buf_list[0] if len(buf_list) == 1 else np.concatenate(buf_list, axis=0)
            with mrcfile.new(path, overwrite=True) as m:
                m.set_data(buf_array)
                m.voxel_size = (px_A, px_A, px_A)
            stack_paths.append(path)

        H_part = W_part = self.vol_size
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
            R_gt = rotation_matrix_from_euler(euls_gt)
            R_jit = rotation_matrix_from_euler(euls_jit)

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
                        ctf = ctf_2d_rfft(H, W, px_A, dfu, dfv, dfa, volt, cs, w, phase, bfac, device=self.device)
                        img = irfft2c(rfft2c(proj) * ctf, s=(H, W))
                    else:
                        img = proj
                    dy, dx = shifts_gt_px[i]
                    if dy != 0 or dx != 0:
                        img = fourier_shift_image_2d(img, torch.tensor([-dy, -dx], device=self.device))
                    out_imgs.append(img)
                else:
                    particle = projs_gt[i]
                    if apply_ctf:
                        ctf = ctf_2d_rfft(H, W, px_A, dfu, dfv, dfa, volt, cs, w, phase, bfac, device=self.device)
                        particle = irfft2c(rfft2c(particle) * ctf, s=(H, W))
                    else:
                        ctf = torch.ones((H, W // 2 + 1), device=self.device)

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
                        proj_jit = irfft2c(rfft2c(proj_jit) * ctf, s=(H, W))

                    out_imgs.append(noise_only + proj_jit)

            out_imgs = torch.stack(out_imgs, dim=0)  # (B, H, W)

            if snr is not None and snr > 0:
                sig_pow = out_imgs.pow(2).mean(dim=(1, 2), keepdim=True)
                noise_var = (sig_pow / snr).clamp(min=1e-12)
                noise = torch.randn_like(out_imgs) * torch.sqrt(noise_var)
                out_imgs = out_imgs + noise

            win = _hann_2d(H_part, W_part, device=self.device)
            out_imgs = out_imgs * win

            batch_np = out_imgs.detach().cpu().numpy().astype(np.float32)

            if simulation_mode == "central_slice":
                del out_imgs, projs
            else:
                del out_imgs, projs_gt, projs_jit
            del euls_gt, angle_jit, shift_jit_px, shifts_gt_px
            del dfu_batch, dfv_batch, dfang_batch, volt_kV, cs_mm, amp_contrast, phase_shift, bfactors

            buffer.append(batch_np)
            buffer_size += batch_np.shape[0]

            while buffer_size >= images_per_file:
                to_write, remaining = [], images_per_file
                while remaining > 0 and buffer:
                    chunk = buffer[0]
                    n = chunk.shape[0]
                    if n <= remaining:
                        to_write.append(chunk)
                        buffer.pop(0)
                        remaining -= n
                        buffer_size -= n
                    else:
                        to_write.append(chunk[:remaining])
                        buffer[0] = chunk[remaining:]
                        buffer_size -= remaining
                        remaining = 0
                _flush_shard(to_write, out_path)
                shard_idx += 1
                out_path = os.path.join(out_dir, f"{basename}_{shard_idx:04d}.mrcs")

            pbar.update(B)

        pbar.close()

        if buffer_size > 0:
            _flush_shard(buffer, out_path)

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
