from dataclasses import dataclass
from typing import List, Optional

from omegaconf import MISSING


@dataclass
class GaussianFilterBank_config:
    in_channels: Optional[int]= MISSING
    sigma_values: List[float] = (0., 1., 3., 5., 10, 15.)
    kernel_sizes: Optional[List[int]] = None
    out_dim: Optional[int] = MISSING
