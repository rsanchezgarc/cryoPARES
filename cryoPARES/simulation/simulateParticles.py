# cryoem_simulator.py
"""
Cryo-EM particle stack simulator (DataLoader version)
- central_slice: Fourier central-slice projections (+ optional CTF/noise)
- noise_additive: RELION-like simulate
    noise_only = orig - (alpha * CTF(proj_sub) + beta)
    out        = noise_only + scale_add( CTF(proj_add) ) [+ optional noise]

Key simplifications:
- Uses starstack.particlesStar.ParticlesStarSet once at startup.
- Minimal Dataset/DataLoader that yields (index, md_row, [image_if_needed]).
- Batch-wise projection & CTF; per-image shift & scaling.
"""

import math
import os
import warnings
from typing import Optional, Union, List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import mrcfile
import starfile

# --- your stack & helpers ---
from starstack.particlesStar import ParticlesStarSet  # preferred entry point  :contentReference[oaicite:1]{index=1}

from cryoPARES.constants import (
    RELION_ANGLES_NAMES, RELION_EULER_CONVENTION, RELION_SHIFTS_NAMES,
)
from cryoPARES.geometry.convert_angles import euler_angles_to_matrix
from cryoPARES.projmatching.projmatchingUtils.fourierOperations import (
    compute_dft_3d, _fourier_proj_to_real_2d,
)
from torch_fourier_slice.slice_extraction import extract_central_slices_rfft_3d
from torch_fourier_shift import fourier_shift_image_2d
from cryoPARES.datamanager.ctf.rfft_ctf import corrupt_with_ctf


# ---------------------------- Small utilities ---------------------------- #

def _deg_eulers_to_R(eulers_deg: torch.Tensor) -> torch.Tensor:
    return euler_angles_to_matrix(torch.deg2rad(eulers_deg), convention=RELION_EULER_CONVENTION)

def _isnan_or_none(x) -> bool:
    try:
        return x is None or (isinstance(x, float) and math.isnan(x))
    except Exception:
        return False

def _get_px_A_from_optics_row(row) -> float:
    for k in ("rlnImagePixelSize", "rlnPixelSize", "rlnDetectorPixelSize"):
        if k in row and not (isinstance(row[k], float) and math.isnan(row[k])):
            try: return float(row[k])
            except Exception: return float(pd.to_numeric(row[k], errors="coerce"))
    raise ValueError("No pixel size found in optics/particles")

def _match_linear(y: torch.Tensor, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[float, float]:
    """
    Find alpha, beta minimizing || y - (alpha*x + beta) ||_2 over (masked) pixels.
    Returns (alpha, beta) as floats. Robust & fast closed-form.
    """
    if mask is not None:
        y = y[mask]
        x = x[mask]
    xm = x.mean()
    ym = y.mean()
    xv = (x - xm)
    denom = float(torch.sum(xv * xv))
    if denom <= 1e-12:
        return 1.0, float(ym - xm)  # degenerate: fall back
    alpha = float(torch.sum(xv * (y - ym)) / denom)
    beta = float(ym - alpha * xm)
    return alpha, beta

def _circular_background_mask(HW: int, radius_frac: float = 0.5, device=None) -> torch.Tensor:
    """
    Simple circular background mask (outside a radius). Useful to regress scale on background only, if desired.
    radius_frac: fraction of half-size defining the particle radius.
    """
    rmax = HW / 2
    rr = radius_frac * rmax
    ys, xs = torch.meshgrid(
        torch.linspace(-rmax, rmax, HW, device=device, dtype=torch.float32),
        torch.linspace(-rmax, rmax, HW, device=device, dtype=torch.float32),
        indexing="ij"
    )
    rad = torch.sqrt(ys * ys + xs * xs)
    # background = outside radius
    return (rad > rr)


# ------------------------- Dataset / DataLoader ------------------------- #

class ParticlesDatasetLite(Dataset):
    """
    Very basic dataset:
      - Holds a ParticlesStarSet and yields (idx, md_row, image) for noise_additive,
        or (idx, md_row, None) for central_slice.
    """
    def __init__(self, pset: ParticlesStarSet, need_images: bool):
        self.pset = pset
        self.need_images = need_images
        # Do NOT reset_index here; pandas collision can happen if the index name
        # already exists as a column (e.g., 'rlnImageName').
        # We don’t need a separate DataFrame copy anyway.

    def __len__(self) -> int:
        return len(self.pset)

    def __getitem__(self, idx: int):
        # pset[idx] -> (img, md_row) per your particlesDataset.py
        if self.need_images:
            img, md_row = self.pset[idx]
            img = torch.as_tensor(img, dtype=torch.float32)   # HxW
        else:
            _, md_row = self.pset[idx]
            img = None
        return idx, md_row, img


def _collate(batch):
    idxs, md_rows, imgs = zip(*batch)
    # Put md into a list of dicts to avoid heavy conversions
    return list(idxs), list(md_rows), list(imgs)  # imgs may contain None in central_slice


# ------------------------------ Simulator ------------------------------ #

class CryoEMSimulator:
    """
    Simulator with two modes (select via simulation_mode):
      - 'central_slice' (default)
      - 'noise_additive' (RELION-like subtraction + addition with scaling)
    """

    def __init__(self, volume_mrc: str, device: str = "cpu", normalize_volume: bool = True):
        self.device = torch.device(device)
        # Load volume and precompute its RFFT once
        with mrcfile.open(volume_mrc, permissive=True) as mrc:
            vol_np = np.asarray(mrc.data.copy(), dtype=np.float32)
            if vol_np.ndim == 4: vol_np = vol_np.squeeze()
            assert vol_np.ndim == 3 and vol_np.shape[0] == vol_np.shape[1] == vol_np.shape[2], "Volume must be cubic"
            self.N = int(vol_np.shape[0])

        vol = torch.as_tensor(vol_np, device=self.device, dtype=torch.float32)
        if normalize_volume:
            vol = (vol - vol.mean()) / (vol.std() + 1e-6)

        vol_rfft, vol_shape, _ = compute_dft_3d(vol, pad_length=0)
        self.vol_rfft = vol_rfft.to(torch.complex64)
        self.image_shape = (int(vol_shape[-1]), int(vol_shape[-1]))

    def _project(self, R: torch.Tensor) -> torch.Tensor:
        """
        R: (B, 3, 3)
        returns real-space projections (B, H, W) float32
        """
        projs_rfft = extract_central_slices_rfft_3d(
            self.vol_rfft, image_shape=self.image_shape, rotation_matrices=R, fftfreq_max=None, zyx_matrices=False
        )
        projs_real = _fourier_proj_to_real_2d(projs_rfft, pad_length=None)
        return projs_real

    def _batch_angles_R(self, md_rows: List[pd.Series]) -> torch.Tensor:
        euls_deg = torch.tensor(
            [[float(row.get(k, 0.0)) for k in RELION_ANGLES_NAMES] for row in md_rows],
            device=self.device, dtype=torch.float32
        )
        return _deg_eulers_to_R(euls_deg)

    def _row_shifts_px(self, row: pd.Series, px_A: float) -> Tuple[float, float]:
        sxA = float(row.get(RELION_SHIFTS_NAMES[0], 0.0))
        syA = float(row.get(RELION_SHIFTS_NAMES[1], 0.0))
        return (sxA / px_A, syA / px_A)

    def _apply_shift_ctf(self, img: torch.Tensor, row: pd.Series, px_A: float,
                         volt_kV: float, cs_mm: float, amp_contrast: float) -> torch.Tensor:
        # Fourier shift (if any)
        sx, sy = self._row_shifts_px(row, px_A)
        if sx or sy:
            img = fourier_shift_image_2d(img, shifts=torch.tensor([sx, sy], device=self.device))
        # CTF (always needed for noise_additive; optional for central_slice)
        dfU = float(row.get("rlnDefocusU", 15000.0))
        dfV = float(row.get("rlnDefocusV", 15000.0))
        dfAng = float(row.get("rlnDefocusAngle", 0.0))
        phase_shift = float(row.get("rlnPhaseShift", 0.0)) if "rlnPhaseShift" in row else 0.0
        bfac_val = row.get("rlnCtfBfactor", None)
        bfactor = None if _isnan_or_none(bfac_val) else float(bfac_val)
        _, img_ctf = corrupt_with_ctf(
            image=img, sampling_rate=px_A,
            dfu=dfU, dfv=dfV, dfang=dfAng,
            volt=volt_kV, cs=cs_mm, w=amp_contrast,
            phase_shift=phase_shift, bfactor=bfactor, fftshift=True
        )
        return img_ctf

    # ---------------------------- Main loop ---------------------------- #

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
        grayscale_adjust: str,
        reuse_subtraction_scale_for_addition: bool,
        num_workers: int,
    ) -> str:

        os.makedirs(out_dir, exist_ok=True)
        parts_df = pset.particles_md
        optics_df = getattr(pset, "optics_md", None)
        optics_row = (optics_df.iloc[0] if optics_df is not None else parts_df.iloc[0])
        px_A = _get_px_A_from_optics_row(optics_row)

        volt_kV = float(optics_row.get("rlnVoltage", 300.0))
        cs_mm = float(optics_row.get("rlnSphericalAberration", 2.7))
        amp_contrast = float(optics_row.get("rlnAmplitudeContrast", 0.07))

        need_images = (simulation_mode == "noise_additive")
        ds = ParticlesDatasetLite(pset, need_images=need_images)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=_collate, pin_memory=False)

        H, W = self.image_shape
        stack_paths: List[str] = []
        image_names_all: List[str] = []

        # Output shard mgmt
        def _new_shard_writer(shard_idx: int):
            out_path = os.path.join(out_dir, f"{basename}_{shard_idx:04d}.mrcs")
            buf = np.empty((0, H, W), dtype=np.float32)
            return out_path, buf

        shard_idx = 0
        out_path, buffer = _new_shard_writer(shard_idx)

        def _flush_and_rotate_shard(buf: np.ndarray, shard_idx: int) -> Tuple[str, np.ndarray, int]:
            nonlocal out_path
            with mrcfile.new(out_path, overwrite=True) as m:
                m.set_data(buf)
                m.voxel_size = (px_A, px_A, px_A)
            stack_paths.append(out_path)
            return *_new_shard_writer(shard_idx + 1), shard_idx + 1  # type: ignore

        img_counter = 0

        for idxs, md_rows, imgs in loader:
            # 1) Project current batch
            R = self._batch_angles_R(md_rows)
            projs = self._project(R)  # (B,H,W)
            B = projs.shape[0]

            # 2) For each image in the batch, finalize according to mode
            for i in range(B):
                row = md_rows[i]
                proj = projs[i]  # (H,W), device=self.device, float32

                if simulation_mode == "central_slice":
                    out_img = proj
                    if apply_ctf:
                        out_img = self._apply_shift_ctf(out_img, row, px_A, volt_kV, cs_mm, amp_contrast)
                    else:
                        # still apply shift if defined (without CTF)
                        sx, sy = self._row_shifts_px(row, px_A)
                        if sx or sy:
                            out_img = fourier_shift_image_2d(out_img, shifts=torch.tensor([sx, sy], device=self.device))

                else:  # noise_additive
                    assert imgs[i] is not None, "Dataset must provide images in noise_additive mode"
                    orig = imgs[i].to(self.device)  # HxW

                    # Subtraction projection uses current row pose/shifts/ctf
                    proj_sub_ctf = self._apply_shift_ctf(proj, row, px_A, volt_kV, cs_mm, amp_contrast)

                    # ----- SCALE (alpha, beta) so experimental & reference are on same scale -----
                    # Optionally regress on full image, or restrict to background if wanted:
                    bg_mask = None  # you can set to _circular_background_mask(H) if you prefer background-based fit
                    alpha, beta = _match_linear(orig, proj_sub_ctf, mask=bg_mask)

                    noise_only = orig - (alpha * proj_sub_ctf + beta)

                    # Addition: make another projection to add (here we use same R; if you randomize angles upstream,
                    # they will already differ).
                    proj_add_ctf = self._apply_shift_ctf(proj, row, px_A, volt_kV, cs_mm, amp_contrast)

                    # Scale the added projection
                    if reuse_subtraction_scale_for_addition:
                        proj_add_scaled = alpha * proj_add_ctf  # reuse alpha, no extra offset
                    else:
                        if grayscale_adjust == "match_noise_std":
                            s_noise = float(torch.std(noise_only).cpu())
                            scale = s_noise if s_noise > 0 else 1.0
                            proj_add_scaled = (proj_add_ctf - proj_add_ctf.mean()) * scale
                        elif grayscale_adjust == "variance_residual":
                            v_orig = float(torch.var(orig).cpu())
                            v_noise = float(torch.var(noise_only).cpu())
                            s_sig = math.sqrt(max(v_orig - v_noise, 1e-8))
                            proj_add_scaled = (proj_add_ctf - proj_add_ctf.mean()) * s_sig
                        else:  # "none"
                            proj_add_scaled = proj_add_ctf

                    out_img = noise_only + proj_add_scaled

                # Optional final additive white noise
                if snr is not None and snr > 0:
                    sig_var = float(torch.var(out_img).cpu())
                    noise_std = math.sqrt(sig_var / snr) if sig_var > 0 else 1.0
                    out_img = out_img + torch.randn_like(out_img) * noise_std

                # Stash to current shard buffer
                np_img = out_img.detach().cpu().numpy().astype(np.float32)
                if buffer.shape[0] == 0:
                    buffer = np_img[None, ...]
                else:
                    buffer = np.concatenate([buffer, np_img[None, ...]], axis=0)

                img_counter += 1
                # Roll shard if full
                if buffer.shape[0] >= images_per_file:
                    out_path, buffer, shard_idx = _flush_and_rotate_shard(buffer, shard_idx)

                # Build RELION-style name (k@basename.mrcs); k is 1-based within shard
                # We’ll fix names after final write using actual shard files and counts.
                # For now, just track counts.

        # Final flush
        if buffer.shape[0] > 0:
            with mrcfile.new(out_path, overwrite=True) as m:
                m.set_data(buffer)
                m.voxel_size = (px_A, px_A, px_A)
            stack_paths.append(out_path)

        # Build final STAR with proper rlnImageName
        # Collect names from each shard in order
        image_names_all = []
        for p in stack_paths:
            n_in_file = mrcfile.open(p, permissive=True, mode="r").data.shape[0]
            base = os.path.basename(p)
            image_names_all.extend([f"{k+1}@{base}" for k in range(n_in_file)])

        out_star = os.path.join(out_dir, f"{basename}.star")
        parts_out = parts_df.copy()
        assert len(image_names_all) == len(parts_out), "Mismatch between written images and metadata rows"
        parts_out["rlnImageName"] = image_names_all

        star_dict = {"particles": parts_out}
        if hasattr(pset, "optics_md") and pset.optics_md is not None:
            star_dict["optics"] = pset.optics_md
        starfile.write(star_dict, out_star, overwrite=True)
        return out_star


# -------------------------- Public API + CLI -------------------------- #

@torch.no_grad()
def run_simulation(
    volume: str,
    in_star: str,
    output_dir: str,
    basename: str = "stack",
    images_per_file: int = 10_000,
    batch_size: int = 128,
    num_workers: int = 0,  # DataLoader workers (0 = main process; safer with CUDA)
    apply_ctf: bool = True,
    snr: Optional[float] = None,
    use_gpu: bool = False,
    device: Optional[str] = None,   # allow explicit 'cuda:0'
    randomize_angles_frac: float = 0.0,  # (kept for parity; apply upstream if needed)
    randomize_angles_max_deg: float = 0.0,
    randomize_shifts_frac: float = 0.0,
    randomize_shifts_max_A: float = 0.0,
    random_seed: Optional[int] = None,
    simulation_mode: str = "central_slice",  # "central_slice" | "noise_additive"
    grayscale_adjust: str = "match_noise_std",  # used if not reusing subtraction alpha
    reuse_subtraction_scale_for_addition: bool = True,
) -> str:
    """
    Minimal high-level entry point.

    Notes:
      - Randomization knobs kept to match your previous interface, but this simplified
        script does not mutate the STAR. If you need randomization, do it before calling.
      - num_workers defaults to 0 to avoid CUDA+fork issues; raise if CPU-only.
    """
    if device is None:
        device = "cuda:0" if use_gpu and torch.cuda.is_available() else "cpu"

    # Prepare ParticlesStarSet (single place for images+metadata)
    pset = ParticlesStarSet(in_star)  # matches your usage pattern  :contentReference[oaicite:3]{index=3}

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
        grayscale_adjust=grayscale_adjust,
        reuse_subtraction_scale_for_addition=reuse_subtraction_scale_for_addition,
        num_workers=num_workers,
    )


def main():
    import argparse
    p = argparse.ArgumentParser(description="Simulate cryo-EM particle stack from volume+STAR (DataLoader version)")
    p.add_argument("--volume", required=True)
    p.add_argument("--in_star", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--basename", default="stack")
    p.add_argument("--images_per_file", type=int, default=10_000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--apply_ctf", dest="apply_ctf", action="store_true")
    p.add_argument("--NOT_apply_ctf", dest="apply_ctf", action="store_false")
    p.set_defaults(apply_ctf=True)
    p.add_argument("--snr", type=float, default=None)

    p.add_argument("--use_gpu", action="store_true")
    p.add_argument("--device", default=None)

    p.add_argument("--simulation_mode", choices=["central_slice", "noise_additive"], default="central_slice")
    p.add_argument("--grayscale_adjust", choices=["none", "match_noise_std", "variance_residual"],
                   default="match_noise_std")
    p.add_argument("--reuse_subtraction_scale_for_addition", action="store_true")
    p.add_argument("--no_reuse_subtraction_scale_for_addition", dest="reuse_subtraction_scale_for_addition",
                   action="store_false")
    p.set_defaults(reuse_subtraction_scale_for_addition=True)

    # (randomization flags kept for API parity; not applied here)
    p.add_argument("--randomize_angles_frac", type=float, default=0.0)
    p.add_argument("--randomize_angles_max_deg", type=float, default=0.0)
    p.add_argument("--randomize_shifts_frac", type=float, default=0.0)
    p.add_argument("--randomize_shifts_max_A", type=float, default=0.0)
    p.add_argument("--random_seed", type=int, default=None)

    args = p.parse_args()

    out_star = run_simulation(
        volume=args.volume,
        in_star=args.in_star,
        output_dir=args.output_dir,
        basename=args.basename,
        images_per_file=args.images_per_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        apply_ctf=args.apply_ctf,
        snr=args.snr,
        use_gpu=args.use_gpu,
        device=args.device,
        randomize_angles_frac=args.randomize_angles_frac,
        randomize_angles_max_deg=args.randomize_angles_max_deg,
        randomize_shifts_frac=args.randomize_shifts_frac,
        randomize_shifts_max_A=args.randomize_shifts_max_A,
        random_seed=args.random_seed,
        simulation_mode=args.simulation_mode,
        grayscale_adjust=args.grayscale_adjust,
        reuse_subtraction_scale_for_addition=args.reuse_subtraction_scale_for_addition,
    )
    print(f"Wrote STAR: {out_star}")


if __name__ == "__main__":
    main()
