"""
Compatibility wrappers for torch-fourier-shift API changes.

All imports from torch_fourier_shift are centralised here.

Known signatures:

  Old (v0.0.6):
    fourier_shift_image_2d(image, shifts)
    fourier_shift_dft_2d(dft, image_shape, shifts, rfft, fftshifted)

  New (teamtomo monorepo):
    fourier_shift_image_2d(image, shifts, cache_intermediates=False)
    fourier_shift_dft_2d(dft, image_shape, shifts, rfft, fftshifted, cache_intermediates=False)

Wrappers are built once at import time by inspecting the installed signature,
so callers can always pass cache_intermediates= and the right thing happens.
The new API caches the fftfreq grid; cache size is controlled by the
TORCH_FOURIER_SHIFT_CACHE_SIZE environment variable.

Watch: fourier_shift_dft_2d still takes `image_shape` in the new API. If a
future version removes it (as happened with torch-fourier-slice v0.4.0), add
an image_shape-aware wrapper here following the pattern in
torch_fourier_slice_compat.py.
"""

import inspect

from torch_fourier_shift import (
    fourier_shift_image_2d as _fourier_shift_image_2d,
    fourier_shift_dft_2d as _fourier_shift_dft_2d,
)

_image_2d_has_cache = "cache_intermediates" in inspect.signature(_fourier_shift_image_2d).parameters
_dft_2d_has_cache = "cache_intermediates" in inspect.signature(_fourier_shift_dft_2d).parameters

if _image_2d_has_cache:
    # New API: forward cache_intermediates
    def fourier_shift_image_2d(image, shifts, cache_intermediates=False, **kw):
        return _fourier_shift_image_2d(image=image, shifts=shifts,
                                       cache_intermediates=cache_intermediates, **kw)
else:
    # Old API (v0.0.6): drop cache_intermediates silently
    def fourier_shift_image_2d(image, shifts, cache_intermediates=False, **kw):
        return _fourier_shift_image_2d(image, shifts, **kw)

if _dft_2d_has_cache:
    # New API: forward cache_intermediates
    def fourier_shift_dft_2d(dft, image_shape, shifts, rfft, fftshifted,
                              cache_intermediates=False, **kw):
        return _fourier_shift_dft_2d(dft=dft, image_shape=image_shape, shifts=shifts,
                                     rfft=rfft, fftshifted=fftshifted,
                                     cache_intermediates=cache_intermediates, **kw)
else:
    # Old API (v0.0.6): drop cache_intermediates silently
    def fourier_shift_dft_2d(dft, image_shape, shifts, rfft, fftshifted,
                              cache_intermediates=False, **kw):
        return _fourier_shift_dft_2d(dft, image_shape, shifts, rfft, fftshifted, **kw)


__all__ = [
    "fourier_shift_image_2d",
    "fourier_shift_dft_2d",
]
