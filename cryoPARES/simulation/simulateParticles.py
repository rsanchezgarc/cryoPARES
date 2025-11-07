"""
Cryo-EM particle simulator CLI — single or multi-GPU.

This script provides a unified interface for particle simulation that
automatically handles single-GPU/CPU or multi-GPU execution based on --num_gpus.
"""

import os
import argparse
from typing import Optional, List

import numpy as np
import pandas as pd
import torch
import starfile

# Use the APIs exposed by the helper you integrated earlier
# (both single-run and multi-GPU sharded run)
from cryoPARES.simulation.simulateParticlesHelper import (
    run_simulation,
    run_simulation_sharded,
    ParticlesStarSet,
)


def _validate_args(args: argparse.Namespace) -> None:
    if args.images_per_file <= 0:
        raise ValueError("--images_per_file must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be > 0")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be >= 0")
    if args.num_gpus < 0:
        raise ValueError("--num_gpus must be >= 0")
    if not os.path.exists(args.volume):
        raise FileNotFoundError(f"Volume not found: {args.volume}")
    if not os.path.exists(args.in_star):
        raise FileNotFoundError(f"STAR not found: {args.in_star}")
    os.makedirs(args.output_dir, exist_ok=True)


def _resolve_device(num_gpus: int) -> (str, List[int]):
    """Return (device_string_for_single, gpu_id_list_for_multi)."""
    available = torch.cuda.device_count()
    if num_gpus == 0 or available == 0:
        # CPU path
        return "cpu", []
    if num_gpus == 1:
        return "cuda:0", []
    # Multi-GPU
    use = min(num_gpus, available)
    return "", list(range(use))  # device not used in sharded path


def _maybe_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_output_star(
    stack_paths: List[str],
    output_dir: str,
    basename: str,
    in_star: str,
    images_per_file: int,
) -> str:
    """
    Create output STAR file referencing the generated MRC stacks.

    Args:
        stack_paths: List of paths to generated MRC stacks
        output_dir: Output directory
        basename: Base name for output
        in_star: Original input STAR file
        images_per_file: Number of images per MRC file

    Returns:
        Path to created STAR file
    """
    # Load original STAR to get particle count and optics
    pset = ParticlesStarSet.load(in_star)
    total_particles = len(pset.particles_md)

    # Create particle entries
    particles_data = []
    particle_idx = 0

    for stack_idx, stack_path in enumerate(sorted(stack_paths)):
        # Get relative path from output_dir
        rel_path = os.path.relpath(stack_path, output_dir)

        # Determine how many particles are in this stack
        if stack_idx < len(stack_paths) - 1:
            # Full stack
            n_particles_in_stack = images_per_file
        else:
            # Last stack - might be partial
            n_particles_in_stack = total_particles - (stack_idx * images_per_file)

        # Create entries for particles in this stack
        for img_idx in range(n_particles_in_stack):
            # Copy original particle metadata
            orig_row = pset.particles_md.iloc[particle_idx]

            # Create new row with updated image name
            new_row = orig_row.copy()
            # Note: starfile library reads "_rlnImageName" as "rlnImageName" (without underscore)
            new_row['rlnImageName'] = f"{img_idx + 1:06d}@{rel_path}"

            particles_data.append(new_row)
            particle_idx += 1

    # Create output dataframe
    particles_df = pd.DataFrame(particles_data)

    # Prepare output dictionary
    output_data = {'particles': particles_df}
    if pset.optics_md is not None:
        output_data['optics'] = pset.optics_md

    # Write output STAR file
    out_star_path = os.path.join(output_dir, f"{basename}.star")
    starfile.write(output_data, out_star_path, overwrite=True)

    return out_star_path


def main():
    parser = argparse.ArgumentParser(
        description="Cryo-EM particle simulator with single/multi-GPU support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument("--volume", required=True, help="Input volume MRC file")
    parser.add_argument("--in_star", required=True, help="Input STAR file with particles")
    parser.add_argument("--output_dir", required=True, help="Output directory")

    # Output options
    parser.add_argument("--basename", default="stack", help="Output basename")
    parser.add_argument(
        "--images_per_file",
        type=int,
        default=2_000,
        help="Number of images per output MRC file",
    )

    # Processing options
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use (0=CPU, 1=single GPU, >1=multi-GPU sharded)",
    )

    # CTF and noise options
    parser.add_argument(
        "--apply_ctf",
        action="store_true",
        default=True,
        help="Apply CTF to projections",
    )
    parser.add_argument(
        "--no_ctf", dest="apply_ctf", action="store_false", help="Do not apply CTF"
    )
    parser.add_argument(
        "--snr",
        type=float,
        default=None,
        help="Signal-to-noise ratio (Gaussian noise added if set)",
    )

    # Simulation mode
    parser.add_argument(
        "--simulation_mode",
        choices=["central_slice", "noise_additive"],
        default="central_slice",
        help="Forward model ('central_slice') or subtraction ('noise_additive')",
    )

    # Jitter options
    parser.add_argument(
        "--angle_jitter_deg", type=float, default=0.0, help="Max angular jitter (deg)"
    )
    parser.add_argument(
        "--angle_jitter_frac",
        type=float,
        default=0.0,
        help="Fractional angular jitter scale",
    )
    parser.add_argument(
        "--shift_jitter_A", type=float, default=0.0, help="Max shift jitter (Å)"
    )
    parser.add_argument(
        "--shift_jitter_frac",
        type=float,
        default=0.0,
        help="Fractional shift jitter scale",
    )

    # Subtraction parameters (for noise_additive mode)
    parser.add_argument(
        "--sub_bp_lo_A",
        type=float,
        default=8.0,
        help="Subtraction bandpass low-resolution cutoff (Å)",
    )
    parser.add_argument(
        "--sub_bp_hi_A",
        type=float,
        default=20.0,
        help="Subtraction bandpass high-resolution cutoff (Å)",
    )
    parser.add_argument(
        "--sub_power_q",
        type=float,
        default=0.85,
        help="Subtraction power quantile threshold (0-1)",
    )

    # Misc
    parser.add_argument("--px_A", type=float, default=1.0, help="Pixel size (Å)")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device for single-run (e.g., 'cuda:0' or 'cpu'). "
             "Ignored for multi-GPU.",
    )
    parser.add_argument("--random_seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--disable_tqdm", action="store_true", help="Disable progress bar"
    )

    args = parser.parse_args()
    _validate_args(args)
    _maybe_seed(args.random_seed)

    # Decide device(s)
    auto_device, gpu_ids = _resolve_device(args.num_gpus)
    device = args.device if args.device is not None else auto_device

    # Dispatch
    if len(gpu_ids) > 1:
        print(f"Running on {len(gpu_ids)} GPUs in parallel: {gpu_ids}")
        # Use the helper's sharded multi-GPU implementation
        out_paths = run_simulation_sharded(
            volume=args.volume,
            in_star=args.in_star,
            output_dir=args.output_dir,
            basename=args.basename,
            images_per_file=args.images_per_file,
            batch_size=args.batch_size,
            simulation_mode=args.simulation_mode,
            apply_ctf=args.apply_ctf,
            snr=args.snr,
            angle_jitter_deg=args.angle_jitter_deg,
            angle_jitter_frac=args.angle_jitter_frac,
            shift_jitter_A=args.shift_jitter_A,
            shift_jitter_frac=args.shift_jitter_frac,
            bandpass_lo_A=6.0,             # kept for API symmetry; unused in helper core currently
            bandpass_hi_A=25.0,            # kept for API symmetry; unused in helper core currently
            sub_bp_lo_A=args.sub_bp_lo_A,
            sub_bp_hi_A=args.sub_bp_hi_A,
            sub_power_q=args.sub_power_q,
            px_A=args.px_A,
            gpus=gpu_ids,
            normalize_volume=False,
            disable_tqdm=args.disable_tqdm,
        )
        # print(f"\nGenerated {len(out_paths)} MRC stacks:")
        # for p in out_paths:
        #     print("  ", p)

        # Write output STAR file
        out_star = _write_output_star(
            stack_paths=out_paths,
            output_dir=args.output_dir,
            basename=args.basename,
            in_star=args.in_star,
            images_per_file=args.images_per_file,
        )
        print(f"\nOutput STAR file: {out_star}")
        return

    # Single GPU or CPU
    if device:
        print(f"Running single-run on device: {device}")
    else:
        # Should not happen, but keep safe default
        device = "cpu"
        print("Running single-run on device: cpu (fallback)")

    out_paths = run_simulation(
        volume=args.volume,
        in_star=args.in_star,
        output_dir=args.output_dir,
        basename=args.basename,
        images_per_file=args.images_per_file,
        batch_size=args.batch_size,
        simulation_mode=args.simulation_mode,
        apply_ctf=args.apply_ctf,
        snr=args.snr,
        num_workers=args.num_workers,
        angle_jitter_deg=args.angle_jitter_deg,
        angle_jitter_frac=args.angle_jitter_frac,
        shift_jitter_A=args.shift_jitter_A,
        shift_jitter_frac=args.shift_jitter_frac,
        sub_bp_lo_A=args.sub_bp_lo_A,
        sub_bp_hi_A=args.sub_bp_hi_A,
        sub_power_q=args.sub_power_q,
        px_A=args.px_A,
        device=device,
        normalize_volume=False,
        disable_tqdm=args.disable_tqdm,
    )

    # print(f"\nGenerated {len(out_paths)} MRC stacks:")
    # for p in out_paths:
    #     print("  ", p)

    # Write output STAR file
    out_star = _write_output_star(
        stack_paths=out_paths,
        output_dir=args.output_dir,
        basename=args.basename,
        in_star=args.in_star,
        images_per_file=args.images_per_file,
    )
    print(f"\nOutput STAR file: {out_star}")


if __name__ == "__main__":
    main()

"""

python -m cryoPARES.simulation.simulateParticles --volume ~/cryo/data/preAlignedParticles/EMPIAR-10166/data/allparticles_reconstruct.mrc  --in_star ~/cryo/data/preAlignedParticles/EMPIAR-10166/data/allparticles.star  --output_dir /tmp/simulation/

"""