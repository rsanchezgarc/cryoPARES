"""
Compatibility wrappers for torch-grid-utils API changes.

All imports from torch_grid_utils are centralised here.

Observed inconsistency across the codebase:
  - `circle` was imported from both `torch_grid_utils` (top-level) and
    `torch_grid_utils.shapes_2d` — these are the same function.
    The top-level import is the canonical one going forward.

Current signatures (v0.1.0):
  circle(radius, image_shape, center=None, smoothing_radius=0, device=None)
  fftfreq_grid(image_shape, rfft, fftshift=False, spacing=1, norm=False, device=None)
"""

from torch_grid_utils import circle, fftfreq_grid  # noqa: F401 – re-exported

__all__ = [
    "circle",
    "fftfreq_grid",
]
