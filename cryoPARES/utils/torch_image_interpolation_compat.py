"""
Stable import for the locally implemented sampling-coordinate conversion.

Preserved signature:
  array_to_grid_sample(array_coordinates, array_shape)
"""

from cryoPARES.utils.fourier_grid import array_to_grid_sample  # noqa: F401 – re-exported

__all__ = [
    "array_to_grid_sample",
]
