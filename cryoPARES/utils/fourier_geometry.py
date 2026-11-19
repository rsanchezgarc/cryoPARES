"""Coordinate conversions for centered discrete Fourier arrays."""

from collections.abc import Sequence

import torch


def fftfreq_to_dft_coordinates(
    frequencies: torch.Tensor, image_shape: Sequence[int], rfft: bool,
) -> torch.Tensor:
    """Map cycles/pixel in array order to indices in an fftshifted DFT.

    Full axes have index ``frequency * length + length // 2``. An rFFT's final
    axis is unshifted and has index ``frequency * original_length``. Using the
    original lengths preserves odd-size spacing and the exact DC location.
    Coordinates outside the stored array are retained for the sampling policy.
    Dtype, device, leading dimensions and gradients follow ``frequencies``.
    """
    if frequencies.ndim < 1 or frequencies.shape[-1] != len(image_shape):
        raise ValueError("One image dimension is required per frequency component")
    sizes = frequencies.new_tensor(image_shape)
    centers = [n // 2 for n in image_shape]
    if rfft:
        centers[-1] = 0
    return frequencies * sizes + frequencies.new_tensor(centers)
