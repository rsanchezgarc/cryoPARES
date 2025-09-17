from functools import lru_cache
import einops
from typing import Tuple, Sequence

import torch
import torch.nn.functional as F
from torch_image_interpolation.grid_sample_utils import array_to_grid_sample
from  torch_fourier_slice._grids import _central_slice_fftfreq_grid

from cryoPARES.configs.mainConfig import main_config


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
    return rfft_shape, valid_coords, freq_grid_mask


def extract_central_slices_rfft_3d_multichannel(
    volume_rfft: torch.Tensor,  # (c, d, d, d//2+1)
    image_shape: tuple[int, int, int],
    rotation_matrices: torch.Tensor,  # (..., 3, 3)
    fftfreq_max: float | None = None,
    zyx_matrices: bool = False,
) -> torch.Tensor:  # (..., c, h, w)
    """Extract central slice from an fftshifted rfft."""
    rotation_matrices = rotation_matrices.to(torch.float32)


    rfft_shape, valid_coords, freq_grid_mask = _get_freq_grid_valid_coords_and_freq_grid_mask(image_shape, volume_rfft.device, fftfreq_max)
    return _extract_central_slices_rfft_3d_multichannel(
                                                        volume_rfft,
                                                        image_shape,
                                                        rotation_matrices,  # (..., 3, 3)
                                                        rfft_shape,
                                                        valid_coords,
                                                        freq_grid_mask,
                                                        zyx_matrices)
def _extract_central_slices_rfft_3d_multichannel(
    volume_rfft: torch.Tensor,  # (c, d, d, d)
    image_shape: tuple[int, int, int],
    rotation_matrices: torch.Tensor,  # (..., 3, 3)
    rfft_shape: tuple[int, int],
    valid_coords: torch.Tensor,
    freq_grid_mask: torch.Tensor,
    zyx_matrices: bool = False,
) -> torch.Tensor:  # (..., c, h, w)
    """Extract central slice from an fftshifted rfft."""

    # freq_grid, rfft_shape, valid_coords, freq_grid_mask = _get_freq_grid_valid_coords_and_freq_grid_mask(image_shape, volume_rfft.device, fftfreq_max)

    rotation_matrices = rotation_matrices.to(torch.float32)

    # keep track of some shapes
    channels = volume_rfft.shape[0]
    stack_shape = tuple(rotation_matrices.shape[:-2])
    output_shape = (*stack_shape, channels, *rfft_shape)

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
    # rotated_coords[conjugate_mask, ...] *= -1
    rotated_coords = torch.where(conjugate_mask.unsqueeze(-1), -rotated_coords, rotated_coords)

    # convert frequencies to array coordinates in fftshifted DFT
    rotated_coords = _fftfreq_to_dft_coordinates(
        frequencies=rotated_coords, image_shape=image_shape, rfft=True
    )
    samples = sample_image_3d_compiled(
        image=volume_rfft, coordinates=rotated_coords
    )  # shape is (..., c)

    # # take complex conjugate of values from redundant half transform

    # For complex numbers, conjugate means flipping the sign of imaginary part
    # samples has shape (..., hw, 2) where last dim is [real, imag]
    real_part = samples[..., 0]  # Real part stays the same
    imag_part = samples[..., 1]  # Imaginary part

    # Apply conjugate where needed: flip imaginary part sign
    imag_conjugated = torch.where(conjugate_mask, -imag_part, imag_part)

    # Reconstruct the complex samples
    samples = torch.stack([real_part, imag_conjugated], dim=-1)

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
    samples = F.grid_sample(
        input=image,
        grid=array_to_grid_sample(coordinates, array_shape=image.shape[-3:]),
        mode='bilinear',
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


def _rfft_shape_tuple(input_shape: Sequence[int]) -> tuple[int, ...]:
    """Get the output shape of an rfft on an input with input_shape."""
    rfft_shape = list(input_shape)
    rfft_shape[-1] = int((rfft_shape[-1] / 2) + 1)
    return tuple(rfft_shape)

def _rfft_shape_tensor(input_shape: Sequence[int], device:torch.tensor, dtype:torch.dtype=torch.long) -> torch.Tensor:
    rfft_shape = torch.as_tensor(input_shape, device=device, dtype=dtype)
    rfft_shape[-1] = ((rfft_shape[-1] / 2) + 1).long()
    return rfft_shape

def _fftfreq_to_dft_coordinates(
    frequencies: torch.Tensor, image_shape: tuple[int, ...], rfft: bool
) -> torch.Tensor:
    """Convert DFT sample frequencies into array coordinates in a fftshifted DFT.

    Parameters
    ----------
    frequencies: torch.Tensor
        `(..., d)` array of multidimensional DFT sample frequencies
    image_shape: tuple[int, ...]
        Length `d` array of image dimensions.
    rfft: bool
        Whether output should be compatible with an rfft (`True`) or a
        full DFT (`False`)

    Returns
    -------
    coordinates: torch.Tensor
        `(..., d)` array of coordinates into a fftshifted DFT.
    """
    image_shape = torch.as_tensor(
        image_shape, device=frequencies.device, dtype=frequencies.dtype
    )
    rfft_shape = _rfft_shape_tensor(image_shape, device=frequencies.device, dtype=frequencies.dtype)
    # define step size in each dimension
    delta_fftfreq = 1 / image_shape

    # calculate total width of DFT interval in cycles/sample per dimension
    # last dim is only non-redundant half in rfft case
    fftfreq_interval_width = 1 - delta_fftfreq
    if rfft is True:
        fftfreq_interval_width[-1] = 0.5

    # allocate for continuous output dft sample coordinates
    coordinates = torch.empty_like(frequencies)

    # transform frequency coordinates into array coordinates
    if rfft is True:
        # full dimensions span `[-0.5, 0.5 - delta_fftfreq]`
        coordinates[..., :-1] = (frequencies[..., :-1] + 0.5) / fftfreq_interval_width[:-1]
        coordinates[..., :-1] = coordinates[..., :-1] * (image_shape[:-1] - 1)

        # half transform dimension (interval width 0.5)
        coordinates[..., -1] = (frequencies[..., -1] * 2) * (rfft_shape[-1] - 1)
    else:
        # all dims are full and span `[-0.5, 0.5 - delta_fftfreq]`
        coordinates[..., :] = (frequencies[..., :] + 0.5) / fftfreq_interval_width
        coordinates[..., :] = coordinates[..., :] * (image_shape - 1)
    return coordinates

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