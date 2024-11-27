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

@dataclass
class ParticlesDataset_config():
    desired_sampling_rate_angs: float = 1.5
    desired_image_size_px: int = 160
    store_data_in_memory: bool = False
    mask_radius_angs: Optional[float] = None
    min_maxProb: Optional[float] = None
    perImg_normalization: ImgNormalizationType = ImgNormalizationType.noiseStats
    ctf_correction: CtfCorrectionType = CtfCorrectionType.concat_phase_flip