import functools
from functools import lru_cache
from typing import Literal

import torch
from einops import einops
from torch_fourier_slice._grids import _central_slice_fftfreq_grid
from torch_image_interpolation import insert_into_image_3d

# from torch_image_interpolation import insert_into_image_3d

from cryoPARES.projmatching.extract_central_slices_as_real import _fftfreq_to_dft_coordinates, _rfft_shape_tuple


@lru_cache(1)
def _initialize_insertion(volume_shape, device, fftfreq_max, image_rfft_shape):
    # generate grid of DFT sample frequencies for a central slice spanning the xy-plane
    freq_grid = _central_slice_fftfreq_grid(
        volume_shape=volume_shape,
        rfft=True,
        fftshift=True,
        device=device,
    )  # (d, d, d, 3)

    # get (b, 3, 1) array of zyx coordinates to rotate (up to fftfreq_max)
    if fftfreq_max is not None:
        normed_grid = (
                einops.reduce(freq_grid ** 2, "h w zyx -> h w", reduction="sum") ** 0.5
        )
        freq_grid_mask = normed_grid <= fftfreq_max
        valid_coords = freq_grid[freq_grid_mask, ...]  # (b, zyx)
    else:
        freq_grid_mask = torch.ones(
            size=image_rfft_shape[-2:], dtype=torch.bool, device=device
        )
        valid_coords = einops.rearrange(freq_grid, "h w zyx -> (h w) zyx")

    valid_coords = einops.rearrange(valid_coords, "hw zyx -> hw zyx 1")

    return valid_coords, freq_grid_mask


def insert_central_slices_rfft_3d_multichannel(
        image_rfft: torch.Tensor,  # fftshifted rfft of (..., c, d, d) 2d image
        volume_shape: tuple[int, int, int],
        rotation_matrices: torch.Tensor,  # (..., 3, 3) dims need to match rfft
        fftfreq_max: float | None = None,
        zyx_matrices: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotation_matrices = rotation_matrices.to(torch.float32)
    ft_dtype = image_rfft.dtype
    device = image_rfft.device
    channels = image_rfft.shape[-3]

    valid_coords, freq_grid_mask = _initialize_insertion(volume_shape, device, fftfreq_max, image_rfft.shape)

    # get (..., b) array of data at each coordinate from image rffts
    valid_data = image_rfft[..., freq_grid_mask]

    # rotation matrices rotate xyz coordinates, make them rotate zyx coordinates
    # xyz:
    # [a b c] [x]    [ax + by + cz]
    # [d e f] [y]  = [dx + ey + fz]
    # [g h i] [z]    [gx + hy + iz]
    #
    # zyx:
    # [i h g] [z]    [gx + hy + iz]
    # [f e d] [y]  = [dx + ey + fz]
    # [c b a] [x]    [ax + by + cz]
    if not zyx_matrices:
        rotation_matrices = torch.flip(rotation_matrices, dims=(-2, -1))

    # add extra dim to rotation matrices for broadcasting
    rotation_matrices = einops.rearrange(rotation_matrices, "... i j -> ... 1 i j")

    # rotate all valid coordinates by each rotation matrix and remove last dim
    rotated_coordinates = einops.rearrange(
        rotation_matrices @ valid_coords, pattern="... hw zyx 1 -> ... hw zyx"
    )

    # flip coordinates in redundant half transform and take conjugate value
    conjugate_mask = rotated_coordinates[..., 2] < 0
    rotated_coordinates[conjugate_mask] *= -1
    # switch channel to end for torch-image-interpolation
    valid_data = einops.rearrange(valid_data, "... c hw -> ... hw c")
    # valid_data[conjugate_mask] = torch.conj(valid_data[conjugate_mask])
    #valid_data.shape ->  torch.Size([96, 56784, 3]) Last dim are [real, imagin, ctf**2]

    data_to_conjugate = valid_data[conjugate_mask]
    data_to_conjugate[..., 1] = -data_to_conjugate[..., 1]
    valid_data[conjugate_mask] = data_to_conjugate

    # calculate positions to sample in DFT array from fftfreq coordinates
    rotated_coordinates = _fftfreq_to_dft_coordinates(
        rotated_coordinates, image_shape=volume_shape, rfft=True
    )

    # initialise output volume and volume for keeping track of weights
    volume_dft_shape = _rfft_shape_tuple(volume_shape)
    dft_3d = torch.zeros(
        size=(channels, *volume_dft_shape),
        dtype=ft_dtype,
        device=device,
    )
    weights = torch.zeros(
        size=volume_dft_shape,
        dtype=torch.float32,
        device=device,
    )

    # insert data into 3D DFT
    dft_3d, weights = insert_into_image_3d(
        values=valid_data,
        coordinates=rotated_coordinates,
        image=dft_3d,
        weights=weights,
    )
    return dft_3d, weights


def __insert_into_image_3d(
        values: torch.Tensor,
        coordinates: torch.Tensor,
        image: torch.Tensor,
        weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Insert values into a 3D image with specified interpolation.
    (torch.compile compatible)
    """
    input_image_is_multichannel = image.ndim == 4
    d, h, w = image.shape[-3:]

    # --- Validate inputs ---
    values_shape = values.shape[:-1] if input_image_is_multichannel else values.shape
    coordinates_shape, coordinates_ndim = coordinates.shape[:-1], coordinates.shape[-1]
    if values_shape != coordinates_shape:
        raise ValueError('One coordinate triplet is required for each value in data.')
    if coordinates_ndim != 3:
        raise ValueError('Coordinates must be 3D with shape (..., 3).')
    if image.dtype != values.dtype:
        raise ValueError('Image and values must have the same dtype.')

    # --- Prepare inputs for processing ---
    if weights is None:
        weights = torch.zeros(size=(d, h, w), dtype=torch.float32, device=image.device)

    image_c = image.unsqueeze(0) if not input_image_is_multichannel else image
    values_c = values.unsqueeze(-1) if not input_image_is_multichannel else values

    # Flatten inputs for processing
    values_flat = values_c.reshape(-1, values_c.shape[-1])
    coords_flat = coordinates.reshape(-1, 3).float()

    # Filter out coordinates that are outside the image
    image_shape = torch.tensor((d, h, w), device=image.device, dtype=torch.float32)
    upper_bound = image_shape - 1
    idx_inside = torch.all((coords_flat >= 0) & (coords_flat <= upper_bound), dim=-1)

    values_in = values_flat[idx_inside]
    coords_in = coords_flat[idx_inside]

    if values_in.shape[0] == 0:  # Early exit if no values are inside the image
        return image, weights

    image_c, weights = _insert_linear_3d(values_in, coords_in, image_c, weights)

    # --- Finalize output shape ---
    # (1, d, h, w) -> (d, h, w) if input was single-channel
    image_out = image_c.squeeze(0) if not input_image_is_multichannel else image_c

    return image_out, weights


def _insert_linear_3d(
        data,  # (b, c)
        coordinates,  # (b, 3)
        image,  # (c, d, h, w)
        weights  # (d, h, w)
):

    c, d, h, w = image.shape
    num_points = data.shape[0]
    if num_points == 0:
        return image, weights

    # --- 1. Calculate corner coordinates and trilinear weights ---
    coords_floor = torch.floor(coordinates)
    t = coordinates - coords_floor

    offsets = torch.tensor(
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
         [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
        device=image.device, dtype=torch.long
    )

    # Vectorized calculation of all 8 corner coordinates for all points
    corner_coords = coords_floor.unsqueeze(1) + offsets  # Shape: (num_points, 8, 3)

    # Vectorized calculation of all 8 trilinear weights for all points
    tz, ty, tx = t.T.unsqueeze(1)
    w_z = torch.cat([(1 - tz), tz], dim=0)
    w_y = torch.cat([(1 - ty), ty], dim=0)
    w_x = torch.cat([(1 - tx), tx], dim=0)
    corner_weights = (w_z[offsets[:, 0]] * w_y[offsets[:, 1]] * w_x[offsets[:, 2]]).T  # Shape: (num_points, 8)

    # --- 2. Filter out corners that fall outside the image boundaries ---
    image_dims = torch.tensor([d, h, w], device=image.device, dtype=corner_coords.dtype)
    valid_mask = torch.all((corner_coords >= 0) & (corner_coords < image_dims), dim=2)  # Shape: (num_points, 8)

    # --- 3. Gather all valid data for a single scatter operation ---
    # Get the original point index (0 to num_points-1) for each valid corner
    point_indices = torch.arange(num_points, device=image.device, dtype=torch.long).unsqueeze(1)
    valid_point_indices = point_indices.expand(-1, 8)[valid_mask]

    if valid_point_indices.shape[0] == 0:
        return image, weights

    # Gather weights and coordinates for valid corners ONLY
    valid_corner_weights = corner_weights[valid_mask]
    valid_corner_coords = corner_coords[valid_mask].long()

    # Calculate the flat 1D indices into the volume
    idx_z, idx_y, idx_x = valid_corner_coords.T
    flat_indices = idx_z * h * w + idx_y * w + idx_x

    # --- 4. Perform scatter-add for weights and image in a fully vectorized way ---
    # Update weights tensor
    weights.flatten().index_add_(0, flat_indices, valid_corner_weights)

    # Gather the source data corresponding to each valid corner
    valid_data = data[valid_point_indices]

    # Calculate the final data values to add to the image grid
    weighted_data = valid_data * valid_corner_weights.to(data.dtype).unsqueeze(1)

    # Perform a single, vectorized index_add_ for all channels
    image.view(c, -1).index_add_(1, flat_indices, weighted_data.T)

    return image, weights