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
        'max_resolution_A': 'Maximum resolution in Angstroms for projection matching computations',
        'max_shift_fraction': 'Maximum allowed in-plane shift as fraction of box size during local refinement',
        'correct_ctf': 'Apply CTF correction during projection matching',
        'verbose': 'Enable verbose logging output',
        'disable_compile_projectVol': 'Disable torch.compile optimization for volume projection',
        'compile_projectVol_mode': 'Compilation mode for volume projection: "default" or "max-autotune" (does not work with dynamic batches)',
        'disable_compile_correlate_dft_2d': 'Disable torch.compile for 2D correlation (currently required as inductor does not support complex numbers)',
        'compile_correlate_dft_2d_mode': 'Compilation mode for 2D correlation if enabled',
        'disable_compile_analyze_cc': 'Disable torch.compile optimization for cross-correlation analysis',
        'compile_analyze_cc_mode': 'Compilation mode for CC analysis: "default" or "max-autotune" (does not work with dynamic batches)',
        'float32_matmul_precision': 'PyTorch float32 matrix multiplication precision mode ("highest", "high", or "medium")',

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
        'gpu_id': 'Specific GPU device ID to use (if multiple GPUs available)',
        'n_first_particles': 'Process only the first N particles from dataset (for testing or validation)',
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
    compile_projectVol_mode: Optional[str] = "default" #"max-autotune" does not work with dynamic size batches (zscore)

    disable_compile_correlate_dft_2d: bool = True #at the moment, inductor does not support complex numbers
    compile_correlate_dft_2d_mode: Optional[str] = "default"

    disable_compile_analyze_cc: bool = False
    compile_analyze_cc_mode: Optional[str] = "default"  #"max-autotune" does not work with dynamic size batches (zscore)

    float32_matmul_precision: str = "high"
