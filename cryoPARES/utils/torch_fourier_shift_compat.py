"""
Compatibility wrappers for torch-fourier-shift API changes.

All imports from torch_fourier_shift are centralised here.

Current signatures (v0.0.6):
  fourier_shift_image_2d(image, shifts)
  fourier_shift_dft_2d(dft, image_shape, shifts, rfft, fftshifted)

Watch: fourier_shift_dft_2d takes `image_shape` — a parameter pattern that has
previously been dropped in sibling packages (e.g. torch-fourier-slice v0.4.0).
If a future version removes it, add a version-aware wrapper here following the
same inspect-based pattern used in torch_fourier_slice_compat.py.
"""

from torch_fourier_shift import fourier_shift_image_2d, fourier_shift_dft_2d  # noqa: F401 – re-exported

__all__ = [
    "fourier_shift_image_2d",
    "fourier_shift_dft_2d",
]
