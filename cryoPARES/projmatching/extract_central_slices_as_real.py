from functools import lru_cache
import einops
from typing import Tuple

import torch
import torch.nn.functional as F
from torch_image_interpolation.grid_sample_utils import array_to_grid_sample
from torch_image_interpolation import utils
from torch_image_interpolation import sample_image_3d
from torch_fourier_slice._dft_utils import _fftfreq_to_dft_coordinates
from  torch_fourier_slice._grids import _central_slice_fftfreq_grid

@lru_cache(1) #TODO: The cache is not working with compilation
def _get_freq_grid_valid_coords_and_freq_grid_mask(image_shape, device:torch.device,
                                                   fftfreq_max: float | None):

    freq_grid = _central_slice_fftfreq_grid(
        volume_shape=image_shape,
        rfft=True,
        fftshift=True,
        device=device,
    )  # (h, w, 3) zyx coords
    rfft_shape = freq_grid.shape[-3], freq_grid.shape[-2]
    if fftfreq_max is not None:
        normed_grid = (
            einops.reduce(freq_grid**2, "h w zyx -> h w", reduction="sum") ** 0.5
        )
        freq_grid_mask = normed_grid <= fftfreq_max
        valid_coords = freq_grid[freq_grid_mask, ...]  # (b, zyx)
    else:
        freq_grid_mask = torch.ones(
            size=rfft_shape, dtype=torch.bool, device=device
        )
        valid_coords = einops.rearrange(freq_grid, "h w zyx -> (h w) zyx")
    valid_coords = einops.rearrange(valid_coords, "b zyx -> b zyx 1")
    return freq_grid, rfft_shape, valid_coords, freq_grid_mask


# @torch.compile(fullgraph=False)
def extract_central_slices_rfft_3d_multichannel(
    volume_rfft: torch.Tensor,  # (c, d, d, d)
    image_shape: tuple[int, int, int],
    rotation_matrices: torch.Tensor,  # (..., 3, 3)
    fftfreq_max: float | None = None,
    zyx_matrices: bool = False,
) -> torch.Tensor:  # (..., c, h, w)
    """Extract central slice from an fftshifted rfft."""
    rotation_matrices = rotation_matrices.to(torch.float32)

    # generate grid of DFT sample frequencies for a central slice spanning the xy-plane
    # freq_grid = _central_slice_fftfreq_grid(
    #     volume_shape=image_shape,
    #     rfft=True,
    #     fftshift=True,
    #     device=volume_rfft.device,
    # )  # (h, w, 3) zyx coords

    freq_grid, rfft_shape, valid_coords, freq_grid_mask = _get_freq_grid_valid_coords_and_freq_grid_mask(image_shape, volume_rfft.device, fftfreq_max)

    # keep track of some shapes
    channels = volume_rfft.shape[0]
    stack_shape = tuple(rotation_matrices.shape[:-2])
    # rfft_shape = freq_grid.shape[-3], freq_grid.shape[-2]
    output_shape = (*stack_shape, channels, *rfft_shape)

    # # get (b, 3, 1) array of zyx coordinates to rotate
    # if fftfreq_max is not None:
    #     normed_grid = (
    #         einops.reduce(freq_grid**2, "h w zyx -> h w", reduction="sum") ** 0.5
    #     )
    #     freq_grid_mask = normed_grid <= fftfreq_max
    #     valid_coords = freq_grid[freq_grid_mask, ...]  # (b, zyx)
    # else:
    #     freq_grid_mask = torch.ones(
    #         size=rfft_shape, dtype=torch.bool, device=volume_rfft.device
    #     )
    #     valid_coords = einops.rearrange(freq_grid, "h w zyx -> (h w) zyx")
    # valid_coords = einops.rearrange(valid_coords, "hw zyx -> hw zyx 1")

    # rotation matrices rotate xyz coordinates, make them rotate zyx coordinates
    # xyz:
    # [a b c] [x]   [ax + by + cz]   [x']
    # [d e f] [y]   [dx + ey + fz]   [y']
    # [g h i] [z] = [gx + hy + iz] = [z']
    #
    # zyx:
    # [i h g] [z]   [gx + hy + iz]   [z']
    # [f e d] [y]   [dx + ey + fz]   [y']
    # [c b a] [x] = [ax + by + cz] = [x']
    if not zyx_matrices:
        rotation_matrices = torch.flip(rotation_matrices, dims=(-2, -1))
    # add extra dim to rotation matrices for broadcasting
    rotation_matrices = einops.rearrange(rotation_matrices, "... i j -> ... 1 i j")

    # rotate all valid coordinates by each rotation matrix
    rotated_coords = rotation_matrices @ valid_coords  # (..., b, zyx, 1)

    # remove last dim of size 1
    rotated_coords = einops.rearrange(rotated_coords, "... hw zyx 1 -> ... hw zyx")

    # flip coordinates that ended up in redundant half transform after rotation
    conjugate_mask = rotated_coords[..., 2] < 0
    rotated_coords[conjugate_mask, ...] *= -1
    # rotated_coords = torch.where(conjugate_mask.unsqueeze(-1), -rotated_coords, rotated_coords)

    # convert frequencies to array coordinates in fftshifted DFT
    rotated_coords = _fftfreq_to_dft_coordinates(
        frequencies=rotated_coords, image_shape=image_shape, rfft=True
    )
    samples = sample_image_3d_compiled(
        image=volume_rfft, coordinates=rotated_coords, interpolation="trilinear"
    )  # shape is (..., c)

    # take complex conjugate of values from redundant half transform
    half_transform = samples[conjugate_mask]
    half_transform[..., 1] *= -1
    # half_transform = torch.stack([half_transform[..., 0], -half_transform[..., 1]], dim=-1)
    samples[conjugate_mask] = half_transform
    samples = einops.rearrange(samples, "... hw c -> ... c hw")

    # insert samples back into DFTs
    projection_image_dfts = torch.zeros(
        output_shape, device=volume_rfft.device, dtype=volume_rfft.dtype
    )
    projection_image_dfts[..., freq_grid_mask] = samples

    return projection_image_dfts


def sample_image_3d_compiled(
        image: torch.Tensor,
        coordinates: torch.Tensor,
        interpolation: str = 'trilinear',
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
    interpolation_mode = 'bilinear' if interpolation == 'trilinear' else interpolation
    samples = F.grid_sample(
        input=image,
        grid=array_to_grid_sample(coordinates, array_shape=image.shape[-3:]),
        mode=interpolation_mode,
        padding_mode='border',
        align_corners=True,
    )


    samples = einops.rearrange(samples, 'b c 1 1 1 -> b c')

    # Replace in-place operation with functional equivalent to allow for compilation
    coordinates = einops.rearrange(coordinates, 'b 1 1 1 zyx -> b zyx')
    volume_shape = torch.as_tensor(image.shape[-3:], device=device)
    inside = torch.logical_and(coordinates >= 0, coordinates <= volume_shape - 1)
    inside = torch.all(inside, dim=-1)

    # Use torch.where instead of in-place multiplication
    samples = torch.where(inside.unsqueeze(-1), samples, torch.zeros_like(samples))

    # pack samples back into the expected shape
    [samples] = einops.unpack(samples, pattern='* c', packed_shapes=ps)

    return samples