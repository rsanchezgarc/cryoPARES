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
        'disable_inductor_shape_padding': 'Disable the inductor shape-padding (pad_mm) pass that OOMs on large rotation grids with torch>=2.6',
        'compile_projectVol_mode': 'Compilation mode for volume projection: "default" or "max-autotune" (does not work with dynamic batches)',
        'disable_compile_correlate_dft_2d': 'Disable torch.compile for 2D correlation (currently required as inductor does not support complex numbers)',
        'compile_correlate_dft_2d_mode': 'Compilation mode for 2D correlation if enabled',
        'disable_compile_analyze_cc': 'Disable torch.compile optimization for cross-correlation analysis',
        'compile_analyze_cc_mode': 'Compilation mode for CC analysis: "default" or "max-autotune" (does not work with dynamic batches)',
        'float32_matmul_precision': 'PyTorch float32 matrix multiplication precision mode ("highest", "high", or "medium")',
        'use_subpixel_shifts': 'Use parabolic sub-pixel interpolation to refine the CC peak location beyond integer-pixel resolution (Change #1)',
        'zero_dc': 'Zero the DC component of both particle and projection DFTs before correlation, preventing low-frequency bias (Change #2a)',
        'spectral_whitening': 'Apply particle-adaptive spectral whitening to projections: estimates the per-shell amplitude from the first particle batch and uses it to upweight high-frequency features in templates, analogous to per-shell SNR normalization in RELION (Change #2b)',
        'whitening_warmup_batches': 'Number of particle batches to average when estimating the spectral whitening filter. More batches → smoother estimate that averages out CTF oscillations across defocus groups; 1 = single-batch (legacy behavior). Only affects the first N batches of each align_star() call.',
        'fftfreq_min': 'High-pass cutoff frequency as fraction of Nyquist [0, 0.5]; excludes low-frequency ring from CC (Change #5)',
        'rotation_composition': 'How to combine the SO(3) delta grid with each predicted pose. "euler_add" (default, legacy): adds Euler angles directly — fast but approximate near poles. "pre_multiply": R_total = R_delta @ R_current (delta in lab frame). "post_multiply": R_total = R_current @ R_delta (delta in body frame). Both rotation-matrix modes are exact SO(3) composition.',
        'use_fibo_grid': 'Use the geodesic Fibonacci (ω-ball tangent) SO(3) grid instead of the Cartesian Euler grid. The Fibonacci grid has uniform coverage in SO(3) (no polar clustering) with ~half the point count of the Cartesian grid at the same step size. Requires rotation_composition != "euler_add"; if euler_add is set it is automatically switched to pre_multiply.',
        'use_two_stage_search': ('Two-pass coarse-to-fine search. Coarse pass uses grid_distance_degs/'
            'grid_step_degs; fine pass uses fine_grid_distance_degs/fine_grid_step_degs around the top '
            'fine_top_k coarse winners. Automatically switches rotation_composition to pre_multiply. '
            'Example: coarse 6°/2° (209 pts) + fine 1.5°/0.5° × K=5 (1249 pts total) vs 1638 for flat '
            '6°/1° — similar or better accuracy at lower cost. Default: False.'),
        'fine_grid_distance_degs': ('Radius (degrees) of fine-pass Fibonacci ω-ball around each coarse '
            'winner. Should be ≥ coarse grid_step_degs to guarantee full coverage. '
            'Only used when use_two_stage_search=True. Point count ≈ 7.7×(distance/step)³ '
            '(e.g. 1.5°/0.5° ≈ 208 pts, 1°/0.5° ≈ 63 pts).'),
        'fine_grid_step_degs': ('Step (degrees) of fine-pass Fibonacci grid. '
            'Only used when use_two_stage_search=True.'),
        'fine_top_k': ('Number of coarse-pass winners fed into the fine pass. Must be ≥ '
            'top_k_poses_localref. Only used when use_two_stage_search=True.'),
        'use_so3_interpolation': ('Parabolic sub-step SO(3) interpolation of the winning grid point '
            'after single-stage Cartesian (euler_add) search. Fits a 1D parabola through the winner '
            'and its two axis-aligned neighbors for each of the 3 Euler axes, giving a sub-step '
            'correction analogous to sub-pixel shift refinement. Only active when '
            'rotation_composition=euler_add and top_k_poses_localref=1. (Change #7)'),

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
    max_batch_rotations: int = 10000  # max product of (batch_size × n_rotations) per forward pass; align_star auto-reduces batch_size to respect this limit
    max_resolution_A: float = 6.
    max_shift_fraction: float = 0.2
    correct_ctf: bool = True
    verbose: bool = False
    top_k_poses_localref: int = 1

    disable_compile_projectVol: bool = False
    compile_projectVol_mode: Optional[str] = "default" #"max-autotune" does not work with dynamic size batches (zscore)
    disable_inductor_shape_padding: bool = True  # disables inductor pad_mm/shape-padding pass that OOMs during compilation with torch>=2.6 (safe to disable, no accuracy impact)

    disable_compile_correlate_dft_2d: bool = True #at the moment, inductor does not support complex numbers
    compile_correlate_dft_2d_mode: Optional[str] = "default"

    disable_compile_analyze_cc: bool = False
    compile_analyze_cc_mode: Optional[str] = "default"  #"max-autotune" does not work with dynamic size batches (zscore)

    float32_matmul_precision: str = "high"

    # Accuracy improvement flags (Change #1, #2a, #2b, #5)
    use_subpixel_shifts: bool = True   # parabolic sub-pixel peak interpolation (Change #1)
    zero_dc: bool = True               # zero DC component before correlation (Change #2a)
    spectral_whitening: bool = True    # particle-adaptive spectral whitening applied to projections (Change #2b)
    whitening_warmup_batches: int = 8  # number of batches to warm-up average the whitening filter (1 = single-batch)
    fftfreq_min: float = 0.0           # high-pass cutoff as fraction of Nyquist; 0=disabled (benchmarks showed it hurts) (Change #5)
    rotation_composition: str = "euler_add"  # "euler_add" (legacy), "pre_multiply" (R_delta@R), "post_multiply" (R@R_delta)
    use_fibo_grid: bool = False              # geodesic ω-ball grid (uniform, ~half point count vs Cartesian at same step)

    # Two-stage coarse-to-fine search (#6)
    use_two_stage_search: bool = False   # enable two-pass search (coarse then fine)
    fine_grid_distance_degs: float = 1.5 # fine-pass ball radius (1.5°/0.5° ≈ 208 pts/candidate)
    fine_grid_step_degs: float = 0.5     # fine-pass step size
    fine_top_k: int = 5                  # coarse-pass candidates handed to fine pass

    # SO(3) sub-step pose interpolation (#7)
    use_so3_interpolation: bool = False  # parabolic sub-step angular refinement after grid search (Change #7)
