"""
Compatibility wrappers for torch-image-interpolation API changes.

All imports from torch_image_interpolation are centralised here.

Current signatures (v0.0.9):
  array_to_grid_sample(array_coordinates, array_shape)
"""

from torch_image_interpolation.grid_sample_utils import array_to_grid_sample  # noqa: F401 – re-exported

__all__ = [
    "array_to_grid_sample",
]
