"""
Central slice extraction for projection matching.

This module contains code derived from the torch-fourier-slice package:
https://github.com/teamtomo/torch-fourier-slice

Original code Copyright (c) 2023 Alister Burt
Licensed under BSD 3-Clause License

Modifications Copyright (c) 2025 CryoPARES developers
Licensed under GPL-3.0

The original torch-fourier-slice license is included in THIRD-PARTY-LICENSES.
"""

import einops
from typing import Tuple, Sequence

import torch
import torch.nn.functional as F
from cryoPARES.utils.torch_image_interpolation_compat import array_to_grid_sample
from cryoPARES.utils.fourier_grid import central_slice_fftfreq_grid as _central_slice_fftfreq_grid
from cryoPARES.utils.fourier_geometry import (
    FourierGeometry, _coordinates_from_constants,
    fftfreq_to_dft_coordinates as _fftfreq_to_dft_coordinates,
)

from cryoPARES.configs.mainConfig import main_config


def _get_freq_grid_valid_coords_and_freq_grid_mask(image_shape, device:torch.device,
                                                   fftfreq_max: float | None):
    """Compatibility builder; repeated callers should own a FourierProjector."""
    geometry = FourierGeometry(image_shape, fftfreq_max, device=device)
    return geometry.plane_shape, geometry.valid_coords, (geometry.row_indices, geometry.column_indices)


def extract_central_slices_rfft_3d_multichannel(
    volume_rfft: torch.Tensor,  # (c, d, d, d//2+1)
    image_shape: tuple[int, int, int],
    rotation_matrices: torch.Tensor,  # (..., 3, 3)
    fftfreq_max: float | None = None,
    zyx_matrices: bool = False,
) -> torch.Tensor:  # (..., c, h, w)
    """Extract central slice from an fftshifted rfft."""
    from cryoPARES.utils.fourier_operators import FourierProjector
    projector = FourierProjector(image_shape, fftfreq_max, representation='channels',
                                 device=volume_rfft.device, dtype=volume_rfft.dtype)
    return projector(volume_rfft, rotation_matrices, zyx_matrices=zyx_matrices)

def compiled_extract_central_slices_rfft_3d_multichannel(
    volume_rfft: torch.Tensor,  # (c, d, d, d//2+1)
    image_shape: tuple[int, int, int],
    rotation_matrices: torch.Tensor,  # (..., 3, 3)
    fftfreq_max: float | None = None,
    zyx_matrices: bool = False,
) -> torch.Tensor:  # (..., c, h, w)
    """Extract central slice from an fftshifted rfft."""
    from cryoPARES.utils.fourier_operators import FourierProjector
    projector = FourierProjector(image_shape, fftfreq_max, representation='channels', compiled=True,
                                 device=volume_rfft.device, dtype=volume_rfft.dtype)
    return projector(volume_rfft, rotation_matrices, zyx_matrices=zyx_matrices)

#TODO: This might be redundant if compilation becomes possible for complex numbers, becuase it is just being as a replacement of extract_central_slices_rfft_3d
def _extract_central_slices_rfft_3d_multichannel_precomputed(
        volume_rfft: torch.Tensor,
        image_shape: tuple[int, int, int],
        rotation_matrices: torch.Tensor,
        rfft_shape: tuple[int, int],
        valid_coords: torch.Tensor,
        freq_mask_indices: torch.Tensor,
        zyx_matrices: bool = False,
        coordinate_constants=None,
        sampling_constants=None,
) -> torch.Tensor:
    """Version with pre-computed indices - fastest for repeated calls."""

    # ... [same initial processing as above] ...
    rotation_matrices = rotation_matrices.to(volume_rfft.dtype)
    channels = volume_rfft.shape[0]
    stack_shape = tuple(rotation_matrices.shape[:-2])
    output_shape = (*stack_shape, channels, *rfft_shape)

    if not zyx_matrices:
        rotation_matrices = torch.flip(rotation_matrices, dims=(-2, -1))

    # Single efficient GEMM: (*batch, 3, 3) @ (3, n_valid) → (*batch, n_valid, 3)
    # Avoids broadcasting 53M tiny (3,3)×(3,1) matvecs (63 ms → 0.4 ms at B=32, nCand=343).
    rotated_coords = (rotation_matrices @ valid_coords[:, :, 0].T).transpose(-1, -2)

    conjugate_mask = rotated_coords[..., 2] < 0
    rotated_coords = torch.where(conjugate_mask.unsqueeze(-1), -rotated_coords, rotated_coords)

    rotated_coords = (_fftfreq_to_dft_coordinates(rotated_coords, image_shape, rfft=True)
                      if coordinate_constants is None else
                      _coordinates_from_constants(rotated_coords, *coordinate_constants))

    samples = sample_image_3d(image=volume_rfft, coordinates=rotated_coords,
                             sampling_constants=sampling_constants)

    real_part = samples[..., 0]
    imag_part = samples[..., 1]
    imag_conjugated = torch.where(conjugate_mask, -imag_part, imag_part)
    samples = torch.stack([real_part, imag_conjugated], dim=-1)
    samples = einops.rearrange(samples, "... hw c -> ... c hw")


    # zeros, not empty: positions outside the fftfreq_max band are never written below, and
    # band_mask does NOT cover all of them — uninitialised memory would leak into the correlation.
    projection_image_dfts = torch.zeros(
        output_shape, device=volume_rfft.device, dtype=volume_rfft.dtype
    )
    projection_image_dfts[..., freq_mask_indices[0], freq_mask_indices[1]] = samples

    return projection_image_dfts

_compiled_extract_central_slices_rfft_3d_multichannel = torch.compile(
    _extract_central_slices_rfft_3d_multichannel_precomputed, dynamic=True,
    # torch.flip is applied *outside* this compiled function (in the wrapper above) to avoid
    # Triton generating XBLOCK=8192 > max=4096 from worst-case size_hints with dynamic=True.
    mode=main_config.projmatching.compile_projectVol_mode, fullgraph=True)


def sample_image_3d(
        image: torch.Tensor,
        coordinates: torch.Tensor,
        sampling_constants=None,
) -> torch.Tensor:
    """Compilation-friendly version of sample_image_3d that avoids in-place operations."""

    device = coordinates.device
    if image.ndim != 4:
        raise ValueError("Wrong input shape")
    # setup coordinates for sampling image with torch.nn.functional.grid_sample
    # shape (..., 3) -> (b, 3)
    coordinates, ps = einops.pack([coordinates], pattern='* zyx')
    n_samples = coordinates.shape[0]

    # torch.nn.functional.grid_sample setup
    image = einops.repeat(image, 'c d h w -> b c d h w', b=n_samples)
    coordinates = einops.rearrange(coordinates, 'b zyx -> b 1 1 1 zyx')

    # take the samples
    if sampling_constants is None:
        normalized = array_to_grid_sample(coordinates, array_shape=image.shape[-3:])
    else:
        upper, _, half_extent, singleton = sampling_constants
        normalized = (coordinates / half_extent) - 1
        if singleton is not None:
            normalized = torch.where(singleton, 0., normalized)
        normalized = normalized.flip(-1)
    samples = F.grid_sample(
        input=image,
        grid=normalized,
        mode='bilinear',
        padding_mode='border',
        align_corners=True,
    )
    del normalized


    samples = einops.rearrange(samples, 'b c 1 1 1 -> b c')

    # Replace in-place operation with functional equivalent to allow for compilation
    coordinates = einops.rearrange(coordinates, 'b 1 1 1 zyx -> b zyx')
    upper = (torch.as_tensor(image.shape[-3:], device=device) - 1
             if sampling_constants is None else sampling_constants[0])
    inside = torch.logical_and(coordinates >= 0, coordinates <= upper)
    inside = torch.all(inside, dim=-1)

    # Use torch.where instead of in-place multiplication
    samples = torch.where(inside.unsqueeze(-1), samples, torch.zeros_like(samples))

    # pack samples back into the expected shape
    [samples] = einops.unpack(samples, pattern='* c', packed_shapes=ps)

    return samples


def _rfft_shape_tuple(input_shape: Sequence[int]) -> tuple[int, ...]:
    """Get the output shape of an rfft on an input with input_shape."""
    rfft_shape = list(input_shape)
    rfft_shape[-1] = int((rfft_shape[-1] / 2) + 1)
    return tuple(rfft_shape)

def _rfft_shape_tensor(input_shape: Sequence[int], device:torch.tensor, dtype:torch.dtype=torch.long) -> torch.Tensor:
    rfft_shape = torch.as_tensor(input_shape, device=device, dtype=dtype)
    rfft_shape[-1] = ((rfft_shape[-1] / 2) + 1).long()
    return rfft_shape

def _dft_center(
    image_shape: tuple[int, ...],
    rfft: bool,
    fftshifted: bool,
    device: torch.device | None = None,
) -> torch.LongTensor:
    """Return the position of the DFT center for a given input shape."""
    fft_center = torch.zeros(size=(len(image_shape),), device=device)
    image_shape = torch.as_tensor(image_shape, device=device).float()
    if rfft is True:
        image_shape = _rfft_shape_tensor(image_shape, device=device)
    if fftshifted is True:
        fft_center = torch.divide(image_shape, 2, rounding_mode="floor")
    if rfft is True:
        fft_center[-1] = 0
    return fft_center.long()


if __name__ == "__main__":
    image_shape = (128,128,128)
    shape = _rfft_shape_tuple(image_shape)
    volume_rfft = torch.rand((2,) + shape)
    rotation_matrices = torch.eye(3).unsqueeze(0)
    extract_central_slices_rfft_3d_multichannel(volume_rfft, image_shape=image_shape,
                                                rotation_matrices=rotation_matrices, fftfreq_max=None,
                                                zyx_matrices=False)
    print("DONE!!")
