from dataclasses import dataclass, asdict, field, MISSING
from typing import Optional

import numpy as np


@dataclass
class I2SProjector_config:
    sphere_fdim: int = 128 #512
    hp_order: int = 3
    coverage: float = 0.9
    sigma: float = 0.2
    max_beta: float = np.radians(90)
    taper_beta: float = np.radians(75)
    rand_fraction_points_to_project: Optional[float] = 0.5

@dataclass
class S2Conv_config:
    f_out: int = 16 #64
    hp_order: int = 3 #4

@dataclass
class SO3Activation_config:
    so3_act_resolution: int = 10

@dataclass
class SO3Conv_config:
    f_out = 1
    max_rads: float = np.pi/12
    n_angles: int = 8

@dataclass
class SO3OuptutGrid_config:
    hp_order: int = 3 #4


@dataclass
class So3Components_config:
    i2sprojector: I2SProjector_config = field(default_factory=I2SProjector_config)
    s2conv: S2Conv_config = field(default_factory=S2Conv_config)
    so3activation: SO3Activation_config = field(default_factory=SO3Activation_config)
    so3conv: SO3Conv_config = field(default_factory=SO3Conv_config)
    so3outputgrid: SO3OuptutGrid_config = field(default_factory=SO3OuptutGrid_config)