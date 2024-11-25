from dataclasses import dataclass, asdict, field, MISSING

import numpy as np


@dataclass
class I2SProjector_config:
    coverage: float = 0.9
    sigma: float = 0.2
    max_beta: float = np.radians(90)
    taper_beta: float = np.radians(75)
    rand_fraction_points_to_project: float = 0.5

@dataclass
class S2Conv_config:
   pass

@dataclass
class SO3Conv_config:
    max_rads: float = np.pi/12 # np.pi/12== 15º
    n_angles: int = 8

@dataclass
class SO3Grid_config:
    pass
