from dataclasses import dataclass
from typing import List, Optional


@dataclass
class GaussianFilters_config:
    """Gaussian filter bank configuration for multi-scale image processing."""

    # Centralized parameter documentation
    PARAM_DOCS = {
        'sigma_values': 'List of Gaussian blur sigma values for multi-scale processing. Higher values create stronger blur',
        'kernel_sizes': 'Optional list of kernel sizes for each Gaussian filter. If None, automatically computed from sigma values',
    }

    sigma_values: List[float] = (0., 1., 3., 5., 10., 15.)
    kernel_sizes: Optional[List[int]] = None
