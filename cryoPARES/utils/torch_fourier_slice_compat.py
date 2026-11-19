"""
Compatibility wrappers for torch-fourier-slice API changes.

External slice extraction imports remain centralized here. Central-slice
frequency grids are implemented locally.

extract_central_slices_rfft_3d
  v0.3.x: (..., image_shape, ...)  – image_shape required
  v0.4.0: image_shape removed      – inferred from volume_rfft.shape

_central_slice_fftfreq_grid
  Legacy name retained as an alias of our local central_slice_fftfreq_grid.

Wrappers are built once at import time by inspecting the installed signature,
so callers can always pass image_shape= and the right thing happens.
"""

import inspect

from torch_fourier_slice.slice_extraction import (
    extract_central_slices_rfft_3d as _extract_central_slices_rfft_3d,
)
from cryoPARES.utils.fourier_grid import central_slice_fftfreq_grid as _central_slice_fftfreq_grid

# ---------------------------------------------------------------------------
# extract_central_slices_rfft_3d
# ---------------------------------------------------------------------------

_has_image_shape = "image_shape" in inspect.signature(_extract_central_slices_rfft_3d).parameters

if _has_image_shape:
    # torch-fourier-slice <= 0.3.x: forward image_shape to the library
    def extract_central_slices_rfft_3d(volume_rfft, rotation_matrices, image_shape=None, **kw):
        return _extract_central_slices_rfft_3d(volume_rfft=volume_rfft, rotation_matrices=rotation_matrices,
                                               image_shape=image_shape, **kw)
else:
    # torch-fourier-slice >= 0.4.0: image_shape inferred from volume_rfft; drop silently
    def extract_central_slices_rfft_3d(volume_rfft, rotation_matrices, image_shape=None, **kw):
        return _extract_central_slices_rfft_3d(volume_rfft=volume_rfft, rotation_matrices=rotation_matrices, **kw)


__all__ = [
    "extract_central_slices_rfft_3d",
    "_central_slice_fftfreq_grid",
]
