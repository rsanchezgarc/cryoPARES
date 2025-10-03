from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Projmatching_config:
    """Projection matching configuration parameters."""

    # Centralized parameter documentation
    PARAM_DOCS = {
        # Config parameters
        'grid_distance_degs': 'Maximum angular distance in degrees for local refinement search. Grid ranges from -grid_distance_degs to +grid_distance_degs around predicted pose',
        'grid_step_degs': 'Angular step size in degrees for grid search during local refinement',
        'top_k_poses_localref': 'Number of best matching poses to keep after local refinement',

        # CLI-exposed parameters (used in projmatching_starfile)
        'reference_vol': 'Path to reference 3D volume (.mrc file) for generating projection templates',
        'particles_star_fname': 'Path to input STAR file with particle metadata',
        'out_fname': 'Path for output STAR file with aligned particle poses',
        'particles_dir': 'Root directory for particle image paths. If provided, overrides paths in the .star file',
        'mask_radius_angs': 'Radius of circular mask in Angstroms applied to particle images',
        'return_top_k_poses': 'Number of top matching poses to save per particle',
        'filter_resolution_angst': 'Low-pass filter resolution in Angstroms applied to reference volume before matching',
        'n_jobs': 'Number of parallel worker processes for distributed projection matching',
        'num_dataworkers': 'Number of CPU workers per PyTorch DataLoader for data loading',
        'batch_size': 'Number of particles to process simultaneously per job',
        'use_cuda': 'Enable GPU acceleration. If False, runs on CPU only',
        'verbose': 'Enable progress logging and status messages',
        'float32_matmul_precision': 'PyTorch float32 matrix multiplication precision mode (highest/high/medium). Higher is more accurate but slower',
        'gpu_id': 'Specific GPU device ID to use (if multiple GPUs available)',
        'n_first_particles': 'Process only the first N particles from dataset (for testing or validation)',
        'correct_ctf': 'Apply CTF correction during processing',
        'halfmap_subset': 'Select half-map subset (1 or 2) for half-map validation',
    }

    grid_distance_degs: float = 6.0 #maximum angular distance from the original pose. Grid will go from -grid_distance_degs to grid_distance_degs
    grid_step_degs: float = 2.0
    max_resolution_A: float = 6.
    max_shift_fraction: float = 0.2
    correct_ctf: bool = True
    verbose: bool = False
    top_k_poses_localref: int = 1

    disable_compile_projectVol: bool = False
    compile_projectVol_mode: Optional[str] = "max-autotune"

    disable_compile_correlate_dft_2d: bool = True #at the moment, inductor does not support complex numbers
    compile_correlate_dft_2d_mode: Optional[str] = "max-autotune"

    disable_compile_analyze_cc: bool = False
    compile_analyze_cc_mode: Optional[str] = "max-autotune" #None #"reduce-overhead" #"max-autotune"

    float32_matmul_precision: str = "high"
