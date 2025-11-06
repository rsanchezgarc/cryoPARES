"""
Cryo-EM particle simulator CLI - supports single and multi-GPU execution.

This script provides a unified interface for particle simulation that automatically
handles both single-GPU and multi-GPU execution based on the --num_gpus parameter.
"""

import os
import argparse
from typing import Optional
import torch
import torch.multiprocessing as mp

import starfile
from starstack.particlesStar import ParticlesStarSet

# Import core functionality from helper module
from cryoPARES.simulation.simulateParticlesHelper import run_simulation


def _worker_process(
        gpu_id: int,
        particle_range: tuple,
        volume: str,
        in_star: str,
        output_dir: str,
        basename: str,
        images_per_file: int,
        batch_size: int,
        num_workers: int,
        apply_ctf: bool,
        snr: Optional[float],
        simulation_mode: str,
        angle_jitter_deg: float,
        angle_jitter_frac: float,
        shift_jitter_A: float,
        shift_jitter_frac: float,
        sub_bp_lo_A: float,
        sub_bp_hi_A: float,
        sub_power_q: float,
        random_seed: Optional[int],
        disable_tqdm: bool,
):
    """Worker process for a single GPU in multi-GPU mode."""
    # Set GPU device
    device = f"cuda:{gpu_id}"

    # Load full particle set and create shard
    pset_full = ParticlesStarSet(in_star)
    start_idx, end_idx = particle_range

    # Create a subset STAR file for this shard
    shard_star = os.path.join(output_dir, f".shard_{gpu_id}.star")
    particles_df = pset_full.particles_md.iloc[start_idx:end_idx].copy()

    star_dict = {"particles": particles_df}
    if hasattr(pset_full, "optics_md") and pset_full.optics_md is not None:
        star_dict["optics"] = pset_full.optics_md
    starfile.write(star_dict, shard_star, overwrite=True)

    # Run simulation on this shard
    shard_basename = f"{basename}_gpu{gpu_id}"
    out_star = run_simulation(
        volume=volume,
        in_star=shard_star,
        output_dir=output_dir,
        basename=shard_basename,
        images_per_file=images_per_file,
        batch_size=batch_size,
        num_workers=num_workers,
        apply_ctf=apply_ctf,
        snr=snr,
        device=device,
        simulation_mode=simulation_mode,
        angle_jitter_deg=angle_jitter_deg,
        angle_jitter_frac=angle_jitter_frac,
        shift_jitter_A=shift_jitter_A,
        shift_jitter_frac=shift_jitter_frac,
        sub_bp_lo_A=sub_bp_lo_A,
        sub_bp_hi_A=sub_bp_hi_A,
        sub_power_q=sub_power_q,
        random_seed=random_seed + gpu_id if random_seed is not None else None,
        disable_tqdm=disable_tqdm or (gpu_id != 0),  # Only show progress on GPU 0
    )

    # Clean up temporary shard star file
    os.remove(shard_star)

    print(f"[GPU {gpu_id}] Complete: {out_star}")


def run_simulation_multi_gpu(
        volume: str,
        in_star: str,
        output_dir: str,
        basename: str,
        images_per_file: int,
        batch_size: int,
        num_workers: int,
        apply_ctf: bool,
        snr: Optional[float],
        num_gpus: int,
        simulation_mode: str,
        angle_jitter_deg: float,
        angle_jitter_frac: float,
        shift_jitter_A: float,
        shift_jitter_frac: float,
        sub_bp_lo_A: float,
        sub_bp_hi_A: float,
        sub_power_q: float,
        random_seed: Optional[int],
        disable_tqdm: bool,
) -> str:
    """
    Run particle simulation across multiple GPUs using data parallelism.

    Args:
        num_gpus: Number of GPUs to use
        ... (other args same as run_simulation)

    Returns:
        Path to merged output STAR file
    """
    import pandas as pd

    print(f"Running simulation across {num_gpus} GPUs")
    os.makedirs(output_dir, exist_ok=True)

    # Load particle set to determine sharding
    pset = ParticlesStarSet(in_star)
    total_particles = len(pset)
    particles_per_gpu = (total_particles + num_gpus - 1) // num_gpus

    # Calculate particle ranges for each GPU
    particle_ranges = []
    for i in range(num_gpus):
        start_idx = i * particles_per_gpu
        end_idx = min((i + 1) * particles_per_gpu, total_particles)
        if start_idx < total_particles:
            particle_ranges.append((start_idx, end_idx))

    # Spawn worker processes
    mp.set_start_method('spawn', force=True)
    processes = []
    for gpu_id, (start_idx, end_idx) in enumerate(particle_ranges):
        p = mp.Process(
            target=_worker_process,
            args=(
                gpu_id, (start_idx, end_idx), volume, in_star, output_dir,
                basename, images_per_file, batch_size, num_workers, apply_ctf, snr,
                simulation_mode, angle_jitter_deg, angle_jitter_frac,
                shift_jitter_A, shift_jitter_frac, sub_bp_lo_A, sub_bp_hi_A,
                sub_power_q, random_seed, disable_tqdm,
            )
        )
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()

    # Merge output STAR files
    print("Merging results from all GPUs...")
    merged_particles = []
    optics_df = None

    for gpu_id in range(len(particle_ranges)):
        gpu_star = os.path.join(output_dir, f"{basename}_gpu{gpu_id}.star")
        if os.path.exists(gpu_star):
            star_data = starfile.read(gpu_star)
            if isinstance(star_data, dict):
                merged_particles.append(star_data["particles"])
                if "optics" in star_data and optics_df is None:
                    optics_df = star_data["optics"]
            else:
                merged_particles.append(star_data)

    # Write merged STAR file
    merged_df = pd.concat(merged_particles, ignore_index=True)
    out_star = os.path.join(output_dir, f"{basename}.star")

    star_dict = {"particles": merged_df}
    if optics_df is not None:
        star_dict["optics"] = optics_df
    starfile.write(star_dict, out_star, overwrite=True)

    print(f"Merged {len(merged_df)} particles into: {out_star}")
    return out_star


def main():
    parser = argparse.ArgumentParser(
        description="Cryo-EM particle simulator with single/multi-GPU support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument("--volume", required=True, help="Input volume MRC file")
    parser.add_argument("--in_star", required=True, help="Input STAR file with particles")
    parser.add_argument("--output_dir", required=True, help="Output directory")

    # Output options
    parser.add_argument("--basename", default="stack", help="Output basename")
    parser.add_argument("--images_per_file", type=int, default=10_000,
                        help="Number of images per output MRC file")

    # Processing options
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for GPU processing")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of CPU workers for data loading")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs to use (1=single GPU, >1=multi-GPU)")

    # CTF and noise options
    parser.add_argument("--apply_ctf", action="store_true", default=True,
                        help="Apply CTF to projections")
    parser.add_argument("--no_ctf", dest="apply_ctf", action="store_false",
                        help="Do not apply CTF")
    parser.add_argument("--snr", type=float, default=None,
                        help="Signal-to-noise ratio (add Gaussian noise)")

    # Simulation mode
    parser.add_argument("--simulation_mode", choices=["central_slice", "noise_additive"],
                        default="central_slice",
                        help="Simulation mode: central_slice (forward model) or noise_additive (subtraction)")

    # Jitter options
    parser.add_argument("--angle_jitter_deg", type=float, default=0.0,
                        help="Maximum angular jitter in degrees")
    parser.add_argument("--angle_jitter_frac", type=float, default=0.0,
                        help="Fraction of particles to apply angular jitter")
    parser.add_argument("--shift_jitter_A", type=float, default=0.0,
                        help="Maximum shift jitter in Angstroms")
    parser.add_argument("--shift_jitter_frac", type=float, default=0.0,
                        help="Fraction of particles to apply shift jitter")

    # Subtraction parameters (for noise_additive mode)
    parser.add_argument("--sub_bp_lo_A", type=float, default=40.0,
                        help="Subtraction bandpass low-resolution cutoff (Angstroms)")
    parser.add_argument("--sub_bp_hi_A", type=float, default=8.0,
                        help="Subtraction bandpass high-resolution cutoff (Angstroms)")
    parser.add_argument("--sub_power_q", type=float, default=0.30,
                        help="Subtraction power quantile threshold")

    # Other options
    parser.add_argument("--random_seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--disable_tqdm", action="store_true",
                        help="Disable progress bar")

    args = parser.parse_args()

    # Determine execution mode
    if args.num_gpus < 0:
        raise ValueError(f"num_gpus must be non-negative, got {args.num_gpus}")

    available_gpus = torch.cuda.device_count()

    # Force CPU mode if num_gpus=0
    if args.num_gpus == 0:
        print("Running on CPU (num_gpus=0)")
        use_gpu = False
        device = "cpu"
    elif available_gpus == 0:
        print("No GPUs available, running on CPU")
        args.num_gpus = 1
        use_gpu = False
        device = "cpu"
    elif args.num_gpus == 1:
        print("Running on single GPU")
        use_gpu = True
        device = "cuda:0"
    else:
        if args.num_gpus > available_gpus:
            print(f"Warning: Requested {args.num_gpus} GPUs but only {available_gpus} available. Using {available_gpus}.")
            args.num_gpus = available_gpus
        print(f"Running on {args.num_gpus} GPUs in parallel")
        use_gpu = True
        device = None

    # Run simulation
    if args.num_gpus > 1 and available_gpus > 1:
        # Multi-GPU mode
        out_star = run_simulation_multi_gpu(
            volume=args.volume,
            in_star=args.in_star,
            output_dir=args.output_dir,
            basename=args.basename,
            images_per_file=args.images_per_file,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            apply_ctf=args.apply_ctf,
            snr=args.snr,
            num_gpus=args.num_gpus,
            simulation_mode=args.simulation_mode,
            angle_jitter_deg=args.angle_jitter_deg,
            angle_jitter_frac=args.angle_jitter_frac,
            shift_jitter_A=args.shift_jitter_A,
            shift_jitter_frac=args.shift_jitter_frac,
            sub_bp_lo_A=args.sub_bp_lo_A,
            sub_bp_hi_A=args.sub_bp_hi_A,
            sub_power_q=args.sub_power_q,
            random_seed=args.random_seed,
            disable_tqdm=args.disable_tqdm,
        )
    else:
        # Single GPU/CPU mode
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
            use_gpu=use_gpu,
            device=device,
            simulation_mode=args.simulation_mode,
            angle_jitter_deg=args.angle_jitter_deg,
            angle_jitter_frac=args.angle_jitter_frac,
            shift_jitter_A=args.shift_jitter_A,
            shift_jitter_frac=args.shift_jitter_frac,
            sub_bp_lo_A=args.sub_bp_lo_A,
            sub_bp_hi_A=args.sub_bp_hi_A,
            sub_power_q=args.sub_power_q,
            random_seed=args.random_seed,
            disable_tqdm=args.disable_tqdm,
        )

    print(f"\nSimulation complete. Output STAR: {out_star}")


if __name__ == "__main__":
    main()