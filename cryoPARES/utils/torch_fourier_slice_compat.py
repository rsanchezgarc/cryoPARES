"""
Compatibility wrappers for torch-fourier-slice API changes.

All imports from torch_fourier_slice are centralised here so that version-specific
differences are handled in a single place.

extract_central_slices_rfft_3d
  v0.3.x: (..., image_shape, ...)  – image_shape required
  v0.4.0: image_shape removed      – inferred from volume_rfft.shape

_central_slice_fftfreq_grid
  Signature unchanged across known versions; re-exported here to avoid
  direct imports of the private torch_fourier_slice._grids module.

Wrappers are built once at import time by inspecting the installed signature,
so callers can always pass image_shape= and the right thing happens.
"""

import inspect

from torch_fourier_slice.slice_extraction import (
    extract_central_slices_rfft_3d as _extract_central_slices_rfft_3d,
)
from torch_fourier_slice._grids import _central_slice_fftfreq_grid  # noqa: F401 – re-exported

# ---------------------------------------------------------------------------
# extract_central_slices_rfft_3d
# ---------------------------------------------------------------------------

_has_image_shape = "image_shape" in inspect.signature(_extract_central_slices_rfft_3d).parameters

if _has_image_shape:
    # torch-fourier-slice <= 0.3.x: forward image_shape to the library
    def extract_central_slices_rfft_3d(volume_rfft, rotation_matrices, image_shape=None, **kw):
        return _extract_central_slices_rfft_3d(volume_rfft, rotation_matrices,
                                               image_shape=image_shape, **kw)
else:
    # torch-fourier-slice >= 0.4.0: image_shape inferred from volume_rfft; drop silently
    def extract_central_slices_rfft_3d(volume_rfft, rotation_matrices, image_shape=None, **kw):
        return _extract_central_slices_rfft_3d(volume_rfft, rotation_matrices, **kw)


__all__ = [
    "extract_central_slices_rfft_3d",
    "_central_slice_fftfreq_grid",
]
