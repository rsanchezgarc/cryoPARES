"""Local frequency grids and sampling coordinates.

Coordinates follow array axis order (YX or ZYX). Only the final conversion to
``grid_sample`` reverses that order. These helpers require no TeamTomo packages.
"""

from collections.abc import Sequence

import torch


def fftfreq_grid(
    image_shape: tuple[int, int] | tuple[int, int, int],
    rfft: bool,
    fftshift: bool = False,
    spacing: float | tuple[float, ...] = 1,
    norm: bool = False,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return a 2D/3D grid of frequencies in cycles per spatial unit.

    ``image_shape`` is the real-space shape, including for an rFFT. Components
    follow the input axes; the component dimension is last. For an rFFT only the
    final axis is truncated, and ``fftshift`` shifts only the other axes. A scalar
    ``spacing`` applies to every axis; a sequence specifies each axis separately.
    ``norm=True`` returns the Euclidean frequency magnitude instead of components.

    Output uses PyTorch's default floating dtype and the requested/default device.
    Each call owns its storage, so callers may modify the result safely.
    """
    ndim = len(image_shape)
    if ndim not in (2, 3):
        raise NotImplementedError("Frequency grids support only 2D and 3D images.")
    steps = tuple(spacing) if isinstance(spacing, Sequence) else (spacing,) * ndim
    if len(steps) != ndim:
        raise ValueError("spacing must provide one value per image axis")

    axes = []
    for axis, (size, step) in enumerate(zip(image_shape, steps)):
        half_axis = rfft and axis == ndim - 1
        frequencies = (torch.fft.rfftfreq if half_axis else torch.fft.fftfreq)(
            size, d=step, device=device
        )
        if fftshift and not half_axis:
            frequencies = torch.fft.fftshift(frequencies)
        axes.append(frequencies)

    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    return grid.square().sum(dim=-1).sqrt() if norm else grid


def central_slice_fftfreq_grid(
    volume_shape: tuple[int, int, int],
    rfft: bool,
    fftshift: bool = False,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return the central XY Fourier plane as ``(height, frequency_width, 3)``.

    Components use array order ``(kz, ky, kx)`` with ``kz=0`` throughout. The
    real-space volume's height and width determine the plane's sampling; depth
    does not affect it. Frequencies are in cycles/pixel. For an rFFT the final
    axis contains nonnegative frequencies and is never shifted. Other axes,
    dtype, device and storage ownership follow ``fftfreq_grid``.
    """
    if len(volume_shape) != 3:
        raise ValueError("volume_shape must contain depth, height and width")
    plane = fftfreq_grid(tuple(volume_shape[-2:]), rfft=rfft, fftshift=fftshift, device=device)
    return torch.cat((torch.zeros_like(plane[..., :1]), plane), dim=-1)


def array_to_grid_sample(
    array_coordinates: torch.Tensor, array_shape: Sequence[int]
) -> torch.Tensor:
    """Normalize array coordinates for ``grid_sample(align_corners=True)``.

    Input has shape ``(..., ndim)`` in YX/ZYX order. Index 0 maps to -1 and
    index N-1 maps to +1, then axes reverse to XY/XYZ order for PyTorch.
    Out-of-bounds coordinates are retained for the caller's padding policy.
    Floating dtype, device, leading dimensions and gradients are preserved.

    As in the previous implementation, singleton axes have undefined normalized
    coordinates (division by zero). Callers must use axis lengths greater than 1
    when passing these coordinates to ``grid_sample``.
    """
    shape = torch.as_tensor(
        array_shape, dtype=array_coordinates.dtype, device=array_coordinates.device
    )
    half_extent = (shape - 1) * 0.5
    return (array_coordinates / half_extent - 1).flip(-1)
