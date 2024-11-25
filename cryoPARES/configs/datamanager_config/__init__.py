from dataclasses import dataclass
from enum import Enum


class CTFCorrectionMode(str, Enum):
    none = "none"
    concat_phase_flip = "concat_phase_flip"
    concat_wiener = "concat_wiener"
    use_phase_flip = "use_phase_flip"
    use_wiener = "use_wiener"

@dataclass
class Datamanager_fields:
    desired_image_size_pixels: int = 160
    desired_sampling_rate_angs: float = 1.5
    ctf_correction_mode: CTFCorrectionMode = CTFCorrectionMode.concat_phase_flip
