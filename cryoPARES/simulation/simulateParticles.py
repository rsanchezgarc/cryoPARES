"""
Cryo-EM particle stack simulator (Fourier central-slice pipeline)

Given a 3D volume (MRC) and a RELION-style STAR file with per-particle
poses (Rot/Tilt/Psi), optional XY shifts, and CTF/optics parameters,
this module synthesizes a 2D particle stack and a matching STAR.

CLI example
-----------
```bash
python cryoem_simulator.py \
  --volume /path/map.mrc \
  --in_star /path/particles.star \
  --output_dir /path/sim_out \
  --basename sim \
  --images_per_file 1000 \
  --num_workers 4 \
  --apply_ctf \
  --use_gpu \
  --gpus 0,1,2 \
  --batch_size 128 \
  --snr 0.1 \
  --randomize_angles_frac 0.2 \
  --randomize_angles_max_deg 5 \
  --randomize_shifts_frac 0.1 \
  --randomize_shifts_max_A 2.0 \
  --random_seed 123
```
"""
from __future__ import annotations

import math
import os
import warnings
from typing import Optional, Tuple

import numpy as np
import torch
import starfile
import mrcfile

from cryoPARES.constants import (
    RELION_ANGLES_NAMES,
    RELION_EULER_CONVENTION,
    RELION_SHIFTS_NAMES,
)
from cryoPARES.geometry.convert_angles import euler_angles_to_matrix
from cryoPARES.datamanager.ctf.rfft_ctf import corrupt_with_ctf
from cryoPARES.projmatching.fourierOperations import compute_dft_3d, _fourier_proj_to_real_2d
from torch_fourier_slice.slice_extraction import extract_central_slices_rfft_3d
from torch_fourier_shift import fourier_shift_image_2d


# ----------------------------- STAR I/O ----------------------------- #

def load_star_with_optics(star_path: str):
    data = starfile.read(star_path)
    if isinstance(data, dict):
        particles = data.get("particles")
        optics = data.get("optics")
    else:
        particles = data
        optics = None
    if particles is None:
        raise ValueError("STAR file has no 'particles' table")

    if optics is not None and "rlnOpticsGroup" in particles.columns and "rlnOpticsGroup" in optics.columns:
        particles = particles.merge(
            optics,
            how="left",
            on="rlnOpticsGroup",
            suffixes=("", "_optics"),
        )
    return particles, optics


def get_optics_pixel_size(row) -> float:
    for k in ("rlnImagePixelSize", "rlnPixelSize", "rlnDetectorPixelSize"):
        if k in row and not (isinstance(row[k], float) and math.isnan(row[k])):
            return float(row[k])
    raise ValueError("Could not find pixel size in STAR (optics).")

# ----------------------------- Helpers ------------------------------ #

def relion_eulers_to_rotmats_deg(eulers_deg: torch.Tensor) -> torch.Tensor:
    """(B,3) Rot/Tilt/Psi degrees -> (B,3,3) rotation matrices using RELION convention."""
    return euler_angles_to_matrix(torch.deg2rad(eulers_deg), convention=RELION_EULER_CONVENTION)


def _apply_randomizations(
    parts_df,
    angles_frac: float,
    angles_max_deg: float,
    shifts_frac: float,
    shifts_max_A: float,
    seed: Optional[int] = None,
):
    """In-place randomization of a fraction of angles and/or shifts.

    - Angles: add independent uniform noise in [-angles_max_deg, +angles_max_deg]
      to each of (Rot,Tilt,Psi) on a random subset of size ~angles_frac * N.
    - Shifts: add independent uniform noise in [-shifts_max_A, +shifts_max_A]
      to rlnOriginXAngst, rlnOriginYAngst on a random subset of size ~shifts_frac * N.
    """
    import pandas as pd
    rng = np.random.default_rng(seed)

    N = len(parts_df)

    if angles_frac and angles_frac > 0:
        k = max(1, int(round(angles_frac * N)))
        idx = rng.choice(N, size=k, replace=False)
        noise = rng.uniform(-angles_max_deg, angles_max_deg, size=(k, 3)).astype(np.float32)
        for j, col in enumerate(RELION_ANGLES_NAMES):
            if col in parts_df.columns:
                parts_df.iloc[idx, parts_df.columns.get_loc(col)] = (
                    parts_df.iloc[idx][col].values.astype(np.float32) + noise[:, j]
                )

    if shifts_frac and shifts_frac > 0:
        k = max(1, int(round(shifts_frac * N)))
        idx = rng.choice(N, size=k, replace=False)
        noise = rng.uniform(-shifts_max_A, shifts_max_A, size=(k, 2)).astype(np.float32)
        for j, col in enumerate(RELION_SHIFTS_NAMES[:2]):
            if col in parts_df.columns:
                parts_df.iloc[idx, parts_df.columns.get_loc(col)] = (
                    parts_df.iloc[idx][col].values.astype(np.float32) + noise[:, j]
                )


# ------------------------------ Core -------------------------------- #

class CryoEMFourierSimulator:
    def __init__(self, volume_mrc: str, device: str = "cpu", normalize_volume: bool = True):
        self.volume_mrc = volume_mrc
        self.device = torch.device(device)
        with mrcfile.open(volume_mrc, permissive=True) as mrc:
            vol_np = np.asarray(mrc.data.copy(), dtype=np.float32)
            if vol_np.ndim == 4:
                vol_np = vol_np.squeeze()
            assert vol_np.ndim == 3 and vol_np.shape[0] == vol_np.shape[1] == vol_np.shape[2], "Volume must be cubic"
            self.n = int(vol_np.shape[0])
            self.voxel_size_A = float(abs(mrc.voxel_size.x)) if mrc.voxel_size.x != 0 else None
        vol = torch.as_tensor(vol_np, device=self.device, dtype=torch.float32)
        if normalize_volume:
            vol = (vol - vol.mean()) / (vol.std() + 1e-6)

        # Precompute fftshifted RFFT of volume; keep padding_factor=0 for direct size
        vol_rfft, vol_shape, _ = compute_dft_3d(vol, pad_length=0)
        self.vol_rfft = vol_rfft.to(torch.complex64)  # (D, D, D//2+1)
        self.vol_shape = tuple(int(x) for x in vol_shape)

    @staticmethod
    def _process_shard(
        volume_path: str,
        device: str,
        star_csv: "pandas.DataFrame",
        start: int,
        end: int,
        optics_row: dict,
        px_A: float,
        out_path: str,
        batch_size: int,
        apply_ctf: bool,
        snr: Optional[float],
    ) -> list[str]:
        import pandas as pd
        sim = CryoEMFourierSimulator(volume_mrc=volume_path, device=device)
        parts_df = star_csv.iloc[start:end]

        voltage_kV = float(optics_row.get("rlnVoltage", 300.0))
        cs_mm = float(optics_row.get("rlnSphericalAberration", 2.7))
        amp_contrast = float(optics_row.get("rlnAmplitudeContrast", 0.07))

        H = W = sim.vol_shape[-1]
        N = len(parts_df)
        stack = np.empty((N, H, W), dtype=np.float32)

        def row_eulers(row):
            return [float(row.get(k, 0.0)) for k in RELION_ANGLES_NAMES]

        def row_shifts_px(row):
            sxA = float(row.get(RELION_SHIFTS_NAMES[0], 0.0))
            syA = float(row.get(RELION_SHIFTS_NAMES[1], 0.0))
            return (sxA / px_A, syA / px_A)

        i_local = 0
        while i_local < N:
            j_local = min(i_local + batch_size, N)
            batch = parts_df.iloc[i_local:j_local]

            euls = torch.tensor([row_eulers(r) for _, r in batch.iterrows()], device=sim.device, dtype=torch.float32)
            R = relion_eulers_to_rotmats_deg(euls)

            projs_rfft = extract_central_slices_rfft_3d(
                sim.vol_rfft, image_shape=sim.vol_shape, rotation_matrices=R, fftfreq_max=None, zyx_matrices=False
            )
            projs_real = _fourier_proj_to_real_2d(projs_rfft, pad_length=None)

            for b, ((_, row), img) in enumerate(zip(batch.iterrows(), projs_real)):
                sx_px, sy_px = row_shifts_px(row)
                img_shifted = (
                    fourier_shift_image_2d(image=img, shifts=torch.tensor([sx_px, sy_px], device=sim.device))
                    if (sx_px or sy_px) else img
                )
                if apply_ctf:
                    dfU = float(row.get("rlnDefocusU", 15000.0))
                    dfV = float(row.get("rlnDefocusV", 15000.0))
                    dfAng = float(row.get("rlnDefocusAngle", 0.0))
                    phase_shift = float(row.get("rlnPhaseShift", 0.0)) if "rlnPhaseShift" in row else 0.0
                    bfactor = float(row.get("rlnCtfBfactor", float("nan"))) if "rlnCtfBfactor" in row else None
                    _, img_corr = corrupt_with_ctf(
                        image=img_shifted,
                        sampling_rate=px_A,
                        dfu=dfU,
                        dfv=dfV,
                        dfang=dfAng,
                        volt=voltage_kV,
                        cs=cs_mm,
                        w=amp_contrast,
                        phase_shift=phase_shift,
                        bfactor=bfactor,
                        fftshift=True,
                    )
                    out_img = img_corr
                else:
                    out_img = img_shifted

                if snr is not None and snr > 0:
                    sig_var = float(torch.var(out_img).cpu())
                    noise_std = math.sqrt(sig_var / snr) if sig_var > 0 else 1.0
                    noise = torch.randn_like(out_img) * noise_std
                    out_img = (out_img + noise).to(out_img.dtype)

                stack[i_local + b] = out_img.detach().cpu().numpy()

            i_local = j_local

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with mrcfile.new(out_path, overwrite=True) as m:
            m.set_data(stack)
            m.voxel_size = (px_A, px_A, px_A)

        base = os.path.basename(out_path)
        names = [f"{k+1}@{base}" for k in range(N)]
        return names

    @torch.no_grad()
    def simulate_to_dir(
        self,
        star_in: str,
        output_dir: str,
        basename: str = "stack",
        images_per_file: int = 1000,
        batch_size: int = 128,
        num_workers: int = 1,
        apply_ctf: bool = True,
        snr: Optional[float] = None,
        use_gpu: bool = False,
        gpus: Optional[list[int]] = None,
        randomize_angles_frac: float = 0.0,
        randomize_angles_max_deg: float = 0.0,
        randomize_shifts_frac: float = 0.0,
        randomize_shifts_max_A: float = 0.0,
        random_seed: Optional[int] = None,
    ) -> str:
        import pandas as pd
        import multiprocessing as mp

        parts_df, optics_df = load_star_with_optics(star_in)

        # Apply requested randomizations IN-PLACE before sharding
        _apply_randomizations(
            parts_df,
            angles_frac=randomize_angles_frac,
            angles_max_deg=randomize_angles_max_deg,
            shifts_frac=randomize_shifts_frac,
            shifts_max_A=randomize_shifts_max_A,
            seed=random_seed,
        )

        optics_row = (optics_df.iloc[0] if optics_df is not None else parts_df.iloc[0])
        px_A = get_optics_pixel_size(optics_row)

        N = len(parts_df)
        os.makedirs(output_dir, exist_ok=True)

        # Build shard specs
        shard_specs = []
        shard_idx = 0

        gpu_devices: list[str] = []
        if use_gpu:
            if gpus is None or len(gpus) == 0:
                gpu_devices = ["cuda:0"]
            else:
                gpu_devices = [f"cuda:{gi}" for gi in gpus]
        else:
            gpu_devices = ["cpu"]

        for start in range(0, N, images_per_file):
            end = min(start + images_per_file, N)
            out_path = os.path.join(output_dir, f"{basename}_{shard_idx:04d}.mrcs")
            dev_for_shard = gpu_devices[shard_idx % len(gpu_devices)]
            shard_specs.append((
                self.volume_mrc,
                dev_for_shard,
                parts_df,
                start,
                end,
                optics_row,
                px_A,
                out_path,
                batch_size,
                apply_ctf,
                snr,
            ))
            shard_idx += 1

        if num_workers > 1 and len(shard_specs) > 1:
            try:
                mp.set_start_method("spawn", force=False)
            except RuntimeError:
                pass
            with mp.Pool(processes=num_workers) as pool:
                names_per_shard = pool.starmap(CryoEMFourierSimulator._process_shard, shard_specs)
        else:
            names_per_shard = [CryoEMFourierSimulator._process_shard(*spec) for spec in shard_specs]

        out_star = os.path.join(output_dir, f"{basename}.star")
        particles_out = parts_df.copy()
        all_names = [name for names in names_per_shard for name in names]
        assert len(all_names) == len(particles_out)
        particles_out["rlnImageName"] = all_names

        star_dict = {"particles": particles_out}
        if optics_df is not None:
            star_dict["optics"] = optics_df
        starfile.write(star_dict, out_star, overwrite=True)

        return out_star

# -------------------------------- CLI ------------------------------- #

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Simulate cryo-EM particle stack from volume+STAR (Fourier slices)")
    p.add_argument("--volume", required=True, help="Input MRC volume (cubic)")
    p.add_argument("--in_star", required=True, help="Input STAR with particles (+optics)")
    p.add_argument("--output_dir", required=True, help="Directory to write .mrcs shards and the STAR")
    p.add_argument("--basename", default="stack", help="Basename prefix for shard files and STAR")
    p.add_argument("--images_per_file", type=int, default=10_000, help="Max images per .mrcs shard")
    p.add_argument("--batch_size", type=int, default=128, help="In-memory batch size per worker")
    p.add_argument("--num_workers", type=int, default=1, help="Number of parallel worker processes")
    p.add_argument("--apply_ctf", dest="apply_ctf", action="store_true")
    p.add_argument("--NOT_apply_ctf", dest="apply_ctf", action="store_false")
    p.set_defaults(apply_ctf=True)
    p.add_argument("--snr", type=float, default=None, help="Target SNR for additive Gaussian noise (per-image)")
    p.add_argument("--use_gpu", dest="use_gpu", action="store_true", help="Use GPU(s) for simulation")
    p.add_argument("--NOT_use_gpu", dest="use_gpu", action="store_false", help="Disable GPU; force CPU")
    p.set_defaults(use_gpu=False)
    p.add_argument("--gpus", default="0", help="Comma-separated GPU IDs (e.g., '0,1,3'). Default: '0'")
    p.add_argument("--randomize_angles_frac", type=float, default=0.0, help="Fraction [0,1] of particles to randomize angles")
    p.add_argument("--randomize_angles_max_deg", type=float, default=0.0, help="Max |Δ| in degrees for Rot/Tilt/Psi")
    p.add_argument("--randomize_shifts_frac", type=float, default=0.0, help="Fraction [0,1] of particles to randomize shifts")
    p.add_argument("--randomize_shifts_max_A", type=float, default=0.0, help="Max |Δ| in Å for rlnOriginX/YAngst")
    p.add_argument("--random_seed", type=int, default=None, help="Seed for reproducible randomization")

    args = p.parse_args()

    # Parse GPUs
    gpu_list = None
    if args.gpus:
        try:
            gpu_list = [int(x) for x in str(args.gpus).split(',') if x.strip() != '']
        except ValueError:
            raise SystemExit("--gpus must be a comma-separated list of integers, e.g., '0,1,2'")

    sim = CryoEMFourierSimulator(volume_mrc=args.volume, device=("cuda:0" if args.use_gpu else "cpu"))
    star_path = sim.simulate_to_dir(
        star_in=getattr(args, "in_star"),
        output_dir=args.output_dir,
        basename=args.basename,
        images_per_file=args.images_per_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        apply_ctf=args.apply_ctf,
        snr=args.snr,
        use_gpu=args.use_gpu,
        gpus=gpu_list,
        randomize_angles_frac=args.randomize_angles_frac,
        randomize_angles_max_deg=args.randomize_angles_max_deg,
        randomize_shifts_frac=args.randomize_shifts_frac,
        randomize_shifts_max_A=args.randomize_shifts_max_A,
        random_seed=args.random_seed,
    )
    print(f"Wrote STAR: {star_path}")


"""
python -m cryoPARES.simulation.simulateParticles --volume ~/cryo/data/preAlignedParticles/EMPIAR-10166/data/donwsampled/output_volume.mrc  --in_star ~/cryo/data/preAlignedParticles/EMPIAR-10166/data/donwsampled/down1000particles.star  --output_dir /tmp/simulate/  --snr 0.01
"""