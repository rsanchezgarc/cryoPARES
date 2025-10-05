from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Reconstruct_config:
    """Reconstruction configuration parameters."""

    # Centralized parameter documentation
    PARAM_DOCS = {
        # Config parameters
        'eps': 'Regularization constant for reconstruction (ideally set to 1/SNR). Prevents division by zero and stabilizes reconstruction',
        'weight_with_confidence': 'Apply per-particle confidence weighting during backprojection. If True, particles with higher confidence contribute more to reconstruction',
        'correct_ctf': 'Apply CTF correction during reconstruction',
        'float32_matmul_precision': 'PyTorch float32 matrix multiplication precision mode (highest/high/medium). Higher is more accurate but slower',

        # CLI-exposed parameters (used in reconstruct_starfile)
        'particles_star_fname': 'Path to input STAR file with particle metadata and poses to reconstruct',
        'symmetry': 'Point group symmetry of the volume for reconstruction (e.g., C1, D2, I, O, T)',
        'output_fname': 'Path for output reconstructed 3D volume (.mrc file)',
        'particles_dir': 'Root directory for particle image paths. If provided, overrides paths in the .star file',
        'n_jobs': 'Number of parallel worker processes for distributed reconstruction',
        'num_dataworkers': 'Number of CPU workers per PyTorch DataLoader for data loading',
        'batch_size': 'Number of particles to backproject simultaneously per job',
        'use_cuda': 'Enable GPU acceleration for reconstruction. If False, runs on CPU only',
        'min_denominator_value': 'Minimum value for denominator to prevent numerical instabilities during reconstruction',
        'use_only_n_first_batches': 'Reconstruct using only first N batches (for testing or quick validation)',
        'halfmap_subset': 'Select half-map subset (1 or 2) for half-map reconstruction and validation',
    }

    disable_compile_insert_central_slices_rfft_3d_multichannel: bool = False
    compile_insert_central_slices_rfft_3d_multichanne_mode: Optional[str] = "default" #"max-autotune" does not work with dynamic size batches (zscore)

    eps: float = 1e-3 #epsilon in the denominator. ~ tikhonov regularization
    weight_with_confidence: bool = False
    correct_ctf: bool = True

    float32_matmul_precision: str = "high"
