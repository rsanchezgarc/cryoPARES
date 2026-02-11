from dataclasses import dataclass, field
from typing import Optional

from cryoPARES.constants import RELION_PRED_POSE_CONFIDENCE_NAME


@dataclass
class Reconstruct_config:
    """Reconstruction configuration parameters."""

    # Centralized parameter documentation
    PARAM_DOCS = {
        # Config parameters
        'eps': 'Regularization mode and strength. Sign selects mode: eps >= 0 uses Tikhonov regularization, eps < 0 uses RELION-style radial averaging. Magnitude sets scale: for Tikhonov, eps is the regularization constant (ideally 1/SNR); for radial averaging, abs(eps) is the divisor for radial weights (RELION uses 1000). Recommended: -1000 for radial averaging, 1e-3 for Tikhonov',
        'min_denominator_value': 'Minimum denominator threshold for numerical stability (prevents division by zero). Applied as final safety clamp regardless of regularization mode. RELION uses 1e-6',
        'weight_with_confidence': f'Apply per-particle confidence weighting during backprojection. If True, particles with higher confidence contribute more to reconstruction. It reads the confidence from the metadata label "{RELION_PRED_POSE_CONFIDENCE_NAME}"',
        'correct_ctf': 'Apply CTF correction during reconstruction',
        'float32_matmul_precision': 'PyTorch float32 matrix multiplication precision mode ("highest", "high", or "medium")',
        'disable_compile_insert_central_slices_rfft_3d_multichannel': 'Disable torch.compile optimization for central slice insertion',
        'compile_insert_central_slices_rfft_3d_multichanne_mode': 'Compilation mode for central slice insertion: "default" or "max-autotune" (does not work with dynamic batches)',

        # CLI-exposed parameters (used in reconstruct_starfile)
        'particles_star_fname': 'Path to input STAR file with particle metadata and poses to reconstruct',
        'symmetry': 'Point group symmetry of the volume for reconstruction (e.g., C1, D2, I, O, T)',
        'output_fname': 'Path for output reconstructed 3D volume (.mrc file)',
        'particles_dir': 'Root directory for particle image paths. If provided, overrides paths in the .star file',
        'n_jobs': 'Number of parallel worker processes for distributed reconstruction',
        'num_dataworkers': 'Number of CPU workers per PyTorch DataLoader for data loading',
        'batch_size': 'Number of particles to backproject simultaneously per job',
        'use_cuda': 'Enable GPU acceleration for reconstruction. If False, runs on CPU only',
        'min_denominator_value': 'Minimum denominator threshold for numerical stability (prevents division by zero). Applied as final safety clamp regardless of regularization mode. RELION uses 1e-6',
        'use_only_n_first_batches': 'Reconstruct using only first N batches (for testing or quick validation)',
        'halfmap_subset': 'Select half-map subset (1 or 2) for half-map reconstruction and validation',
        'apply_soft_mask': 'Apply soft spherical masking after reconstruction to reduce edge artifacts (RELION-style)',
        'mask_radius_pix': 'Radius for soft mask in pixels. If negative, defaults to box_size/2 ',
        'mask_edge_width': 'Width of cosine falloff edge in pixels',
    }

    disable_compile_insert_central_slices_rfft_3d_multichannel: bool = False
    compile_insert_central_slices_rfft_3d_multichanne_mode: Optional[str] = "default" #"max-autotune" does not work with dynamic size batches (zscore)

    eps: float = -1000.0  # Negative = radial averaging with scale abs(eps) (RELION default: 1000)
                          # Positive = Tikhonov regularization constant (e.g., 1e-3)
    min_denominator_value: float = 1e-6  # Numerical stability floor (matches RELION's 1e-6)
    weight_with_confidence: bool = False
    correct_ctf: bool = True

    float32_matmul_precision: str = "high"

    # Soft masking parameters (RELION-style)
    apply_soft_mask: bool = True
    mask_radius_pix: float = -1.0  # Negative = auto (box_size/2), matching RELION
    mask_edge_width: int = 3  # RELION default
