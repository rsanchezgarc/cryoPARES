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
    desired_sampling_rate_angs: float = 1.5
    desired_image_size_px: int = 160
    store_data_in_memory: bool = False
    mask_radius_angs: Optional[float] = None #If None, use a circular mask of radius Box/2
    apply_mask_to_img: bool = True # If True, apply the mask to the the image, otherwise, use it only for computing normalization stats
    min_maxProb: Optional[float] = None #Particles with maxProb smaller than this number will be ruled out
    perImg_normalization: ImgNormalizationType = ImgNormalizationType.noiseStats
    ctf_correction: CtfCorrectionType = CtfCorrectionType.concat_phase_flip
    reduce_symmetry_in_label: bool = True
