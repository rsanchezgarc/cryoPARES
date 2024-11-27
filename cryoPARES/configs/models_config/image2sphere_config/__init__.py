from dataclasses import dataclass


@dataclass
class Image2Sphere_fields:
    lmax: int = 12
    label_smoothing: float = 0.05
