from dataclasses import dataclass
from enum import Enum
from typing import Optional

class ImgNormalizationType(str, Enum):
    none = "none"
    noiseStats = "noiseStats"
    subtractMean = "subtractMean"

class CtfCorrectionType(str, Enum):
    none = "none"
    phase_flip = "phase_flip"
    concat_phase_flip = "concat_phase_flip"
    ctf_multiply = "ctf_multiply"

@dataclass
class ParticlesDataset_config():
    """Particle dataset configuration parameters."""

    # Centralized parameter documentation
    PARAM_DOCS = {
        'sampling_rate_angs_for_nnet': 'Target sampling rate in Angstroms/pixel for neural network input. Particle images are first rescaled to this sampling rate before processing',
        'image_size_px_for_nnet': 'Target image size in pixels for neural network input. After rescaling to target sampling rate, images are cropped or padded to this size',
        'mask_radius_angs': 'Radius of circular mask in Angstroms applied to particle images. If not provided, defaults to half the box size',
    }

    sampling_rate_angs_for_nnet: float = 1.5
    image_size_px_for_nnet: int = 160
    store_data_in_memory: bool = True
    mask_radius_angs: Optional[float] = None #If None, use a circular mask of radius Box/2
    apply_mask_to_img: bool = True # If True, apply the mask to the the image, otherwise, use it only for computing normalization stats
    min_maxProb: Optional[float] = None #Particles with maxProb smaller than this number will be ruled out
    perImg_normalization: ImgNormalizationType = ImgNormalizationType.noiseStats
    ctf_correction: CtfCorrectionType = CtfCorrectionType.concat_phase_flip
    reduce_symmetry_in_label: bool = True
