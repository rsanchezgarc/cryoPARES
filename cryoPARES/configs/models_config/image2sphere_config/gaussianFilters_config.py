from dataclasses import dataclass
from typing import List, Optional


@dataclass
class GaussianFilters_config:
    sigma_values: List[float] = (0., 1., 3., 5., 10., 15.)
    kernel_sizes: Optional[List[int]] = None
