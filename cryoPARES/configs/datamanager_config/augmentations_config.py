from dataclasses import dataclass, field
from typing import List, Optional, Tuple

#Note that Individual operation configs are not named *_config not to be added at the same level as augmentator_config. #TODO: move them to submodule
@dataclass
class RandomGaussNoiseConfig:
    scale: float = 0.5
    p: float = 0.1

@dataclass
class RandomUnifNoiseConfig:
    scale: float = 2.0
    p: float = 0.2

@dataclass
class GaussianBlurConfig:
    scale: float = 2
    p: float = 0.2 #TODO: change it to .1, not to make things too easy. Or perhaps add a scheduler here. .2 seems to work better

@dataclass
class SizePerturbationConfig:
    maxSizeFraction: float = 0.05
    p: float = 0.2 #TODO: change it to .1, since this is not a naturally ocurring problem


@dataclass
class ErasingConfig:
    ratio: Tuple[float, float] = (0.3, 2.3)
    scale: Tuple[float, float] = (0.01, 0.02)
    p: float = 0.1

@dataclass
class RandomElasticConfig:
    kernel_size_fraction: float = 0.2
    sigma_fraction: float = 0.1
    alpha: float = 0.1
    p: float = 0.1


@dataclass
class InPlaneRotations90Config:
    p: float = 1.0

@dataclass
class InPlaneRotationsConfig:
    maxDegrees: float = 20
    p: float = 0.5

@dataclass
class InPlaneShiftsConfig:
    maxShiftFraction: float = 0.05
    p: float = 0.5



"""
This is an example for a probScheduler:
"probScheduler": {"type": "linear_down",
                   "kwargs": {
                       "scheduler_steps": int(3e6), # 1e6 is ~10 epochs assuming 1e5 particles
                       "min_prob": .1}
                 }
"""

@dataclass
class Operations:
    randomGaussNoise: RandomGaussNoiseConfig = field(default_factory=RandomGaussNoiseConfig)
    randomUnifNoise: RandomUnifNoiseConfig = field(default_factory=RandomUnifNoiseConfig)
    sizePerturbation: SizePerturbationConfig = field(default_factory=SizePerturbationConfig)
    gaussianBlur: GaussianBlurConfig = field(default_factory=GaussianBlurConfig)
    erasing: ErasingConfig = field(default_factory=ErasingConfig)
    randomElastic: RandomElasticConfig = field(default_factory=RandomElasticConfig)
    inPlaneRotations90: InPlaneRotations90Config = field(default_factory=InPlaneRotations90Config)
    inPlaneRotations: InPlaneRotationsConfig = field(default_factory=InPlaneRotationsConfig)
    inPlaneShifts: InPlaneShiftsConfig = field(default_factory=InPlaneShiftsConfig)

@dataclass
class Augmenter_config:
    min_n_augm_per_img: int = 1
    max_n_augm_per_img: int = 8 #TODO: We probably want to do 4 since, in the original code we have the swap between particle and proj
    operations: Operations = field(default_factory=Operations)
