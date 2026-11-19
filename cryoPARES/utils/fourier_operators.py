"""Reusable Fourier operators with explicit ownership of invariant geometry.

Volumes, images and rotations remain forward inputs. Only geometry and fixed
frequency factors are retained, so repeated differentiable calls do not retain
an earlier input's autograd graph. Pipeline owners keep their reference volumes
and accumulation buffers as before.
"""

import torch
from torch import nn

from cryoPARES.utils.fourier_geometry import FourierGeometry
from cryoPARES.utils.fourier_grid import fftfreq_grid


class FourierProjector(nn.Module):
    """Project many rotation batches with one shape/cutoff configuration.

    ``representation='complex'`` accepts a complex (or real) 3D Fourier volume.
    ``'channels'`` accepts a real volume with leading [real, imaginary] channels.
    Move/cast the operator explicitly to match its volume inputs. Rotations are
    XYZ by default; pass ``zyx_matrices=True`` for array-order rotations.
    """

    def __init__(self, volume_shape, fftfreq_max=None, *, representation='complex',
                 compiled=False, device=None, dtype=None):
        super().__init__()
        if representation not in ('complex', 'channels'):
            raise ValueError("representation must be 'complex' or 'channels'")
        self.geometry = FourierGeometry(volume_shape, fftfreq_max, device=device, dtype=dtype)
        self.representation = representation
        if representation == 'channels':
            from cryoPARES.projmatching.projmatchingUtils.extract_central_slices_as_real import (
                _extract_central_slices_rfft_3d_multichannel_precomputed,
                _compiled_extract_central_slices_rfft_3d_multichannel,
            )
            self._kernel = (_compiled_extract_central_slices_rfft_3d_multichannel if compiled else
                            _extract_central_slices_rfft_3d_multichannel_precomputed)
        else:
            from cryoPARES.utils.fourier_slice import _extract_precomputed
            self._kernel = torch.compile(_extract_precomputed, fullgraph=True, dynamic=True) if compiled else _extract_precomputed

    def forward(self, volume, rotation_matrices, *, zyx_matrices=False):
        geometry = self.geometry
        expected = ((2,) if self.representation == 'channels' else ()) + geometry.dft_shape
        if tuple(volume.shape) != expected:
            raise ValueError(f"volume_rfft shape must be {expected} for image_shape={geometry.volume_shape}")
        _check_inputs(volume, rotation_matrices, geometry)
        rotations = rotation_matrices.to(dtype=geometry.plane.dtype)
        # Keep the layout conversion outside compiled workers (Triton dynamic
        # flip kernels can otherwise generate invalid launch sizes).
        if not zyx_matrices:
            rotations = rotations.flip((-2, -1))
        if self.representation == 'channels':
            if volume.is_complex():
                raise TypeError("channels representation requires real-valued channels")
            return self._kernel(volume, geometry.volume_shape, rotations,
                geometry.plane_shape, geometry.valid_coords,
                (geometry.row_indices, geometry.column_indices), True,
                geometry.coordinate_constants, geometry.sampling_constants)
        return self._kernel(volume, rotations, geometry.plane_shape, geometry.valid_coords,
                            geometry.valid_indices, geometry.coordinate_constants, geometry.sampling_constants)


class FourierBackprojector(nn.Module):
    """Own the insertion geometry for real/imaginary and optional weight channels.

    Each call returns a batch contribution and interpolation weights. The
    enclosing reconstructor owns and accumulates these contributions, allowing
    callers to inspect, combine or differentiate individual batches.
    """

    def __init__(self, volume_shape, fftfreq_max=None, *, compiled=False, device=None, dtype=None):
        super().__init__()
        self.geometry = FourierGeometry(volume_shape, fftfreq_max, device=device, dtype=dtype)
        from cryoPARES.reconstruction.insert_central_slices_rfft_3d import (
            _worker_insert_central_slices_rfft_3d_multichannel,
            _compiled_worker_insert_central_slices_rfft_3d_multichannel,
        )
        self._kernel = (_compiled_worker_insert_central_slices_rfft_3d_multichannel if compiled else
                        _worker_insert_central_slices_rfft_3d_multichannel)

    def forward(self, slices, rotation_matrices, *, zyx_matrices=False):
        geometry = self.geometry
        if slices.ndim < 3 or tuple(slices.shape[-2:]) != geometry.plane_shape or slices.shape[-3] < 2:
            raise ValueError(f"slices must have shape (..., channels>=2, {geometry.plane_shape})")
        if tuple(slices.shape[:-3]) != tuple(rotation_matrices.shape[:-2]):
            raise ValueError("slice and rotation batch dimensions must match")
        if slices.is_complex():
            raise TypeError("slices require real-valued real/imaginary channels")
        _check_inputs(slices, rotation_matrices, geometry)
        rotations = rotation_matrices.to(dtype=geometry.plane.dtype)
        if not zyx_matrices:
            rotations = rotations.flip((-2, -1))
        return self._kernel(slices, geometry.volume_shape, rotations, geometry.valid_coords,
                            geometry.valid_indices, True, geometry.coordinate_constants, geometry.insertion_constants)


def _check_inputs(values, rotations, geometry):
    if values.dtype not in (torch.float32, torch.float64, torch.complex64, torch.complex128):
        raise TypeError("Fourier inputs require float32/64 or complex64/128 dtype")
    if rotations.ndim < 2 or rotations.shape[-2:] != (3, 3):
        raise ValueError("rotation_matrices must have shape (..., 3, 3)")
    if values.device != geometry.plane.device or rotations.device != values.device:
        raise RuntimeError("Fourier operator, inputs and rotations must share a device; move the operator explicitly")
    if values.real.dtype != geometry.plane.dtype:
        raise TypeError("Fourier input dtype must match the operator's geometry; cast the operator explicitly")


class FourierShift2D(nn.Module):
    """Own frequency factors for repeated translations of one image layout."""

    def __init__(self, image_shape, *, rfft=True, fftshifted=False, device=None, dtype=None):
        super().__init__()
        self._image_shape = tuple(image_shape)
        if len(self.image_shape) != 2 or any(n < 1 for n in self.image_shape):
            raise ValueError("image_shape must contain two positive dimensions")
        self._rfft = rfft
        self._fftshifted = fftshifted
        self._build(device, dtype or torch.get_default_dtype())

    def _build(self, device, dtype):
        if dtype not in (torch.float32, torch.float64):
            raise TypeError("Fourier shifts require float32 or float64 frequency factors")
        frequencies = fftfreq_grid(self.image_shape, self.rfft, self.fftshifted, device=device, dtype=dtype)
        self.register_buffer('radians', -2 * torch.pi * frequencies, persistent=False)

    def _apply(self, fn, recurse=True):
        probe = fn(self.radians.new_empty(0))
        if probe.dtype not in (torch.float32, torch.float64):
            raise TypeError("Fourier shifts require float32 or float64 frequency factors")
        super()._apply(fn, recurse=recurse)
        self._build(self.radians.device, self.radians.dtype)
        return self

    @property
    def image_shape(self):
        return self._image_shape

    @property
    def rfft(self):
        return self._rfft

    @property
    def fftshifted(self):
        return self._fftshifted

    def forward(self, dft, shifts):
        expected = (self.image_shape[0], self.image_shape[1] // 2 + 1 if self.rfft else self.image_shape[1])
        if dft.ndim < 2 or tuple(dft.shape[-2:]) != expected:
            raise ValueError(f"DFT spatial dimensions must be {expected}")
        if shifts.ndim < 1 or shifts.shape[-1] != 2:
            raise ValueError("shifts must have shape (..., 2) in (dy, dx) order")
        if dft.device != self.radians.device or shifts.device != dft.device:
            raise RuntimeError("Fourier shifter and inputs must share a device")
        angles = (self.radians * shifts[..., None, None, :]).sum(-1)
        return dft * torch.complex(angles.cos(), angles.sin())

    def shift_image(self, image, shifts):
        if not self.rfft or self.fftshifted:
            raise ValueError("image translation requires an unshifted rFFT layout")
        if tuple(image.shape[-2:]) != self.image_shape:
            raise ValueError("image shape must match the Fourier shifter")
        return torch.fft.irfft2(self(torch.fft.rfft2(image.contiguous()), shifts), s=self.image_shape)
