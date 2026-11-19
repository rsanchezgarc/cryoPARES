"""Coordinate conversions for centered discrete Fourier arrays."""

from collections.abc import Sequence

import torch
from torch import nn

from cryoPARES.utils.fourier_grid import central_slice_fftfreq_grid


def fftfreq_to_dft_coordinates(
    frequencies: torch.Tensor, image_shape: Sequence[int], rfft: bool,
) -> torch.Tensor:
    """Map cycles/pixel in array order to indices in an fftshifted DFT.

    Full axes have index ``frequency * length + length // 2``. An rFFT's final
    axis is unshifted and has index ``frequency * original_length``. Using the
    original lengths preserves odd-size spacing and the exact DC location.
    Roundoff immediately outside array edges is snapped to the edge. Other
    outside coordinates are retained for the sampling policy.
    Dtype, device, leading dimensions and gradients follow ``frequencies``.
    """
    if frequencies.ndim < 1 or frequencies.shape[-1] != len(image_shape):
        raise ValueError("One image dimension is required per frequency component")
    sizes = frequencies.new_tensor(image_shape)
    centers = [n // 2 for n in image_shape]
    if rfft:
        centers[-1] = 0
    upper = [n - 1 for n in image_shape]
    if rfft:
        upper[-1] = image_shape[-1] // 2
    upper = frequencies.new_tensor(upper)
    tolerance = 2 * torch.finfo(frequencies.dtype).eps * sizes
    return _coordinates_from_constants(frequencies, sizes, frequencies.new_tensor(centers), upper, tolerance)


def _coordinates_from_constants(frequencies, sizes, centers, upper, tolerance):
    """Tensor-only coordinate kernel; constants may be owned by a geometry."""
    coordinates = frequencies * sizes + centers
    # Native float32 rfftfreq can place an odd-width final sample a fraction of
    # an index beyond the last voxel. It is still an in-bounds Fourier sample.
    coordinates = torch.where((coordinates < 0) & (coordinates >= -tolerance), 0., coordinates)
    return torch.where((coordinates > upper) & (coordinates <= upper + tolerance), upper, coordinates)


class FourierGeometry(nn.Module):
    """Own the constant tensors for one Fourier volume and central-plane cutoff.

    Construction fixes the shape, layout and inclusive frequency cutoff. Reuse
    the object for every batch; create another object for different geometry.
    Derived buffers are omitted from state dictionaries and rebuilt on explicit
    dtype/device moves, so ``geometry.double()`` has native double precision.
    No input-dependent values or autograd graphs are retained.
    """

    def __init__(self, volume_shape, fftfreq_max=None, *, rfft=True, fftshift=True,
                 device=None, dtype=None):
        super().__init__()
        shape = tuple(volume_shape)
        if len(shape) != 3 or any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in shape):
            raise ValueError("volume_shape must contain three positive integers")
        self._volume_shape = shape
        self._fftfreq_max = fftfreq_max
        self._rfft = rfft
        self._fftshift = fftshift
        self._build(device, dtype or torch.get_default_dtype())

    @property
    def volume_shape(self):
        return self._volume_shape

    @property
    def fftfreq_max(self):
        return self._fftfreq_max

    @property
    def rfft(self):
        return self._rfft

    @property
    def fftshift(self):
        return self._fftshift

    @property
    def dft_shape(self):
        return (*self.volume_shape[:-1], self.volume_shape[-1] // 2 + 1) if self.rfft else self.volume_shape

    @property
    def plane_shape(self):
        return self.dft_shape[-2:]

    @property
    def coordinate_constants(self):
        return self.sizes, self.centers, self.upper, self.tolerance

    @property
    def sampling_constants(self):
        return self.upper, self.sampling_extent, self.sampling_half_extent, self.singleton_axes

    @property
    def insertion_constants(self):
        return self.corner_offsets, self.sampling_sizes

    def _build(self, device, dtype):
        if dtype not in (torch.float32, torch.float64):
            raise TypeError("Fourier geometry requires float32 or float64")
        plane = central_slice_fftfreq_grid(self.volume_shape, self.rfft, self.fftshift,
                                           device=device, dtype=dtype)
        mask = torch.ones(self.plane_shape, device=plane.device, dtype=torch.bool)
        if self.fftfreq_max is not None:
            mask = plane.square().sum(-1).sqrt() <= self.fftfreq_max
        indices = torch.nonzero(mask.reshape(-1), as_tuple=False).squeeze(-1)
        centers = [n // 2 if self.fftshift else 0 for n in self.volume_shape]
        if self.rfft:
            centers[-1] = 0
        sizes = plane.new_tensor(self.volume_shape)
        buffers = {
            'plane': plane, 'frequency_mask': mask, 'valid_indices': indices,
            'valid_coords': plane.reshape(-1, 3).index_select(0, indices).unsqueeze(-1),
            'row_indices': indices // self.plane_shape[-1],
            'column_indices': indices % self.plane_shape[-1],
            'sizes': sizes, 'centers': plane.new_tensor(centers),
            'upper': plane.new_tensor(self.dft_shape) - 1,
            'tolerance': 2 * torch.finfo(dtype).eps * sizes,
            'sampling_sizes': plane.new_tensor(self.dft_shape),
            'sampling_extent': (plane.new_tensor(self.dft_shape) - 1).clamp_min(1),
            'sampling_half_extent': (plane.new_tensor(self.dft_shape) - 1).clamp_min(1) * 0.5,
            'singleton_axes': plane.new_tensor(self.dft_shape) == 1,
            'corner_offsets': plane.new_tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                                                [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]),
        }
        for name, value in buffers.items():
            self.register_buffer(name, value, persistent=False)

    def _apply(self, fn, recurse=True):
        probe = fn(self.plane.new_empty(0))
        if probe.dtype not in (torch.float32, torch.float64):
            raise TypeError("Fourier geometry requires float32 or float64")
        super()._apply(fn, recurse=recurse)
        self._build(probe.device, probe.dtype)
        return self

    def to_dft_coordinates(self, frequencies):
        return _coordinates_from_constants(frequencies, *self.coordinate_constants)
