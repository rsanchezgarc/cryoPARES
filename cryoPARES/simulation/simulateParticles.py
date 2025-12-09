"""
Cryo-EM particle simulator CLI — single or multi-GPU.

This script provides a unified interface for particle simulation that
automatically handles single-GPU/CPU or multi-GPU execution based on --num_gpus.
"""

import os
from typing import List, Optional

import mrcfile
import starfile
import torch

from autoCLI_config import CONFIG_PARAM, inject_defaults_from_config, inject_docs_from_config_params
from cryoPARES.configs.mainConfig import main_config

# Use the APIs exposed by the helper you integrated earlier
# (both single-run and multi-GPU sharded run)
from cryoPARES.simulation.simulateParticlesHelper import (
    ParticlesStarSet,
    run_simulation,
    run_simulation_sharded,
)


def _write_output_star(
    stack_paths: List[str],
    output_dir: str,
    basename: str,
    in_star: str,
    images_per_file: int,
    n_first_particles: Optional[int] = None,
) -> str:
    """
    Write output STAR file with references to simulated particle images.

    Args:
        stack_paths: List of MRC stack file paths
        output_dir: Output directory
        basename: Base name for output
        in_star: Input STAR file path
        images_per_file: Number of images per file
        n_first_particles: Optional: Limit to the first N particles

    Returns:
        Path to written STAR file
    """
    # Load input star to get metadata
    pset = ParticlesStarSet(in_star)
    parts_df = pset.particles_md.copy()
    if n_first_particles is not None:
        parts_df = parts_df.head(n_first_particles)
    optics_df = pset.optics_md if hasattr(pset, "optics_md") else None

    # Generate image names for all stacks
    # Optimization: avoid opening files - we know the count from images_per_file
    total_particles = len(parts_df)
    image_names_all: List[str] = []
    particles_remaining = total_particles

    for p in stack_paths:
        # All files except potentially the last have images_per_file images
        n_in_file = min(images_per_file, particles_remaining)
        base = os.path.basename(p)
        image_names_all.extend([f"{k + 1}@{base}" for k in range(n_in_file)])
        particles_remaining -= n_in_file

    # Update particle dataframe with new image names
    out_star = os.path.join(output_dir, f"{basename}.star")
    assert len(image_names_all) == len(parts_df), f"Mismatch: images ({len(image_names_all)}) vs STAR rows ({len(parts_df)})"
    parts_df["rlnImageName"] = image_names_all

    # Write output star file
    star_dict = {"particles": parts_df}
    if optics_df is not None:
        star_dict["optics"] = optics_df
    starfile.write(star_dict, out_star, overwrite=True)

    return out_star


@inject_docs_from_config_params
@inject_defaults_from_config(main_config.train, update_config_with_args=True)
def simulate_particles_cli(
        volume: str,
        in_star: str,
        output_dir: str,
        basename: str = "stack",
        images_per_file: int = 2000,
        batch_size: int = CONFIG_PARAM(),
        num_dataworkers: int = CONFIG_PARAM(config=main_config.datamanager),
        n_gpus_for_simulation: int = CONFIG_PARAM(),
        apply_ctf: bool = True,
        snr_for_simulation: Optional[float] = CONFIG_PARAM(),
        simulation_mode: str = "central_slice",
        angle_jitter_deg: float = 0.0,
        angle_jitter_frac: float = 0.0,
        shift_jitter_A: float = 0.0,
        shift_jitter_frac: float = 0.0,
        sub_bp_lo_A: float = 8.0,
        sub_bp_hi_A: float = 20.0,
        sub_power_q: float = 0.85,
        px_A: float = 1.0,
        device: Optional[str] = None,
        random_seed: Optional[int] = None,
        disable_tqdm: bool = False,
        n_first_particles: Optional[int] = None,
):
    """
    Simulate cryo-EM particle projections from a 3D volume.

    Args:
        volume: Path to input volume MRC file
        in_star: Path to input STAR file with particle metadata
        output_dir: Output directory for simulated particles
        basename: Base name for output MRC stacks
        images_per_file: Number of images per output MRC file
        batch_size: {batch_size}
        num_dataworkers: {num_dataworkers}
        n_gpus_for_simulation: {n_gpus_for_simulation}
        apply_ctf: Apply CTF to projections
        snr_for_simulation: {snr_for_simulation}
        simulation_mode: Simulation mode: 'central_slice' or 'noise_additive'
        angle_jitter_deg: Maximum angular jitter in degrees
        angle_jitter_frac: Fractional angular jitter scale
        shift_jitter_A: Maximum shift jitter in Ångströms
        shift_jitter_frac: Fractional shift jitter scale
        sub_bp_lo_A: Subtraction bandpass low-resolution cutoff (Å)
        sub_bp_hi_A: Subtraction bandpass high-resolution cutoff (Å)
        sub_power_q: Subtraction power quantile threshold (0-1)
        px_A: Pixel size in Ångströms
        device: Override device for single-run (e.g., 'cuda:0' or 'cpu'). Ignored for multi-GPU
        random_seed: Random seed for reproducibility
        disable_tqdm: Disable progress bar
        n_first_particles: If set, only process the first N particles from the input STAR file.
    """
    # Validation
    if images_per_file <= 0:
        raise ValueError("images_per_file must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if num_dataworkers < 0:
        raise ValueError("num_dataworkers must be >= 0")
    if n_first_particles is not None and n_first_particles <= 0:
        raise ValueError("n_first_particles must be > 0")
    if not os.path.exists(volume):
        raise FileNotFoundError(f"Volume not found: {volume}")
    if not os.path.exists(in_star):
        raise FileNotFoundError(f"STAR not found: {in_star}")
    os.makedirs(output_dir, exist_ok=True)

    # Set random seed
    if random_seed is not None:
        import numpy as np
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)

    # Resolve GPU configuration
    num_gpus = n_gpus_for_simulation
    if num_gpus == -1:
        num_gpus = torch.cuda.device_count()

    available_gpus = torch.cuda.device_count()
    if num_gpus == 0 or available_gpus == 0:
        # CPU mode
        auto_device = "cpu"
        gpu_ids = []
    elif num_gpus == 1:
        # Single GPU
        auto_device = "cuda:0"
        gpu_ids = []
    else:
        # Multi-GPU
        use_gpus = min(num_gpus, available_gpus)
        auto_device = ""
        gpu_ids = list(range(use_gpus))

    device = device if device is not None else auto_device

    # Dispatch
    if len(gpu_ids) > 1:
        print(f"Running on {len(gpu_ids)} GPUs in parallel: {gpu_ids}")
        out_paths = run_simulation_sharded(
            volume=volume,
            in_star=in_star,
            output_dir=output_dir,
            basename=basename,
            images_per_file=images_per_file,
            batch_size=batch_size,
            simulation_mode=simulation_mode,
            apply_ctf=apply_ctf,
            snr=snr_for_simulation,
            angle_jitter_deg=angle_jitter_deg,
            angle_jitter_frac=angle_jitter_frac,
            shift_jitter_A=shift_jitter_A,
            shift_jitter_frac=shift_jitter_frac,
            bandpass_lo_A=6.0,
            bandpass_hi_A=25.0,
            sub_bp_lo_A=sub_bp_lo_A,
            sub_bp_hi_A=sub_bp_hi_A,
            sub_power_q=sub_power_q,
            px_A=px_A,
            gpus=gpu_ids,
            normalize_volume=False,
            disable_tqdm=disable_tqdm,
            n_first_particles=n_first_particles,
        )
    else:
        # Single GPU or CPU
        print(f"Running single-run on device: {device}")
        out_paths = run_simulation(
            volume=volume,
            in_star=in_star,
            output_dir=output_dir,
            basename=basename,
            images_per_file=images_per_file,
            batch_size=batch_size,
            simulation_mode=simulation_mode,
            apply_ctf=apply_ctf,
            snr=snr_for_simulation,
            num_workers=num_dataworkers,
            angle_jitter_deg=angle_jitter_deg,
            angle_jitter_frac=angle_jitter_frac,
            shift_jitter_A=shift_jitter_A,
            shift_jitter_frac=shift_jitter_frac,
            sub_bp_lo_A=sub_bp_lo_A,
            sub_bp_hi_A=sub_bp_hi_A,
            sub_power_q=sub_power_q,
            px_A=px_A,
            device=device,
            normalize_volume=False,
            disable_tqdm=disable_tqdm,
            n_first_particles=n_first_particles,
        )

    # Write output STAR file
    out_star = _write_output_star(
        stack_paths=out_paths,
        output_dir=output_dir,
        basename=basename,
        in_star=in_star,
        images_per_file=images_per_file,
        n_first_particles=n_first_particles,
    )
    print(f"\nOutput STAR file: {out_star}")


def main():
    import sys
    from autoCLI_config import ConfigArgumentParser

    print('---------------------------------------')
    print(' '.join(sys.argv))
    print('---------------------------------------')

    parser = ConfigArgumentParser(
        prog='simulate_particles',
        description='Cryo-EM particle simulator with single/multi-GPU support',
        config_obj=main_config
    )
    parser.add_args_from_function(simulate_particles_cli)
    args, config_args = parser.parse_args()

    # Call the CLI function with parsed arguments
    simulate_particles_cli(**vars(args))


if __name__ == "__main__":
    main()

"""

python -m cryoPARES.simulation.simulateParticles --volume ~/cryo/data/preAlignedParticles/EMPIAR-10166/data/allparticles_reconstruct.mrc  --in_star ~/cryo/data/preAlignedParticles/EMPIAR-10166/data/allparticles.star  --output_dir /tmp/simulation/

"""