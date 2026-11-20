"""Local periodic image translations using the Fourier shift theorem."""

import torch

from cryoPARES.utils.fourier_grid import fftfreq_grid


def fourier_shift_dft_2d(
    dft: torch.Tensor,
    image_shape: tuple[int, int],
    shifts: torch.Tensor,
    rfft: bool,
    fftshifted: bool,
    cache_intermediates: bool = False,
) -> torch.Tensor:
    """Translate a full or real 2D DFT by ``(dy, dx)`` pixels.

    ``image_shape`` is the original real-space ``(height, width)``. Inputs have
    shapes ``(..., height, frequency_width)`` and ``(..., 2)``; their leading
    dimensions broadcast normally. Positive shifts move real-space features
    toward increasing indices with periodic wraparound. Shifted rFFTs have only
    their row-frequency axis centered; their reduced column axis is unshifted.

    Frequency grids use PyTorch's default floating dtype, preserving the previous
    helper's precision and promotion rules. Inputs must be on the same device.
    Neither input is modified, and gradients propagate through both inputs.
    ``cache_intermediates`` is accepted for caller compatibility; no tensors are
    retained between calls, including tensors connected to an autograd graph.
    """
    if len(image_shape) != 2:
        raise ValueError("image_shape must contain height and width")
    if shifts.ndim < 1 or shifts.shape[-1] != 2:
        raise ValueError("shifts must have shape (..., 2) in (dy, dx) order")
    height, width = image_shape
    expected_shape = (height, width // 2 + 1 if rfft else width)
    if dft.ndim < 2 or tuple(dft.shape[-2:]) != expected_shape:
        raise ValueError(f"DFT spatial dimensions must be {expected_shape} for image_shape={image_shape}")

    frequencies = fftfreq_grid(image_shape, rfft=rfft, fftshift=fftshifted, device=shifts.device)
    # Keep the existing float32 rounding order as well as the shift convention.
    angles = ((-2 * torch.pi * frequencies) * shifts[..., None, None, :]).sum(dim=-1)
    phase = torch.complex(angles.cos(), angles.sin())
    return dft * phase


def fourier_shift_image_2d(
    image: torch.Tensor,
    shifts: torch.Tensor,
    cache_intermediates: bool = False,
) -> torch.Tensor:
    """Periodically translate real images of shape ``(..., height, width)``.

    Shifts use ``(..., 2)`` in pixel ``(dy, dx)`` order; leading dimensions
    broadcast with the image batch. Fractional translations interpolate the
    periodic signal in Fourier space. On even-sized axes, Nyquist modes are
    projected back to a real image by the inverse rFFT, so fractional shifts
    need not preserve their energy or invert exactly.

    Original dimensions are passed explicitly to the inverse FFT, including
    odd widths and singleton axes. See ``fourier_shift_dft_2d`` for dtype,
    device, gradient, and cache-hint behavior.
    """
    image_shape = tuple(image.shape[-2:])
    # Expanded views have zero strides, which some CPU FFT backends reject.
    spectrum = torch.fft.rfft2(image.contiguous())
    shifted = fourier_shift_dft_2d(
        spectrum, image_shape, shifts, rfft=True, fftshifted=False,
        cache_intermediates=cache_intermediates,
    )
    return torch.fft.irfft2(shifted, s=image_shape)
