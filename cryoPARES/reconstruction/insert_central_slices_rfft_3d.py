"""
Central slice insertion for 3D reconstruction.

This module contains code derived from the torch-fourier-slice package:
https://github.com/teamtomo/torch-fourier-slice

Original code Copyright (c) 2023 Alister Burt
Licensed under BSD 3-Clause License

Modifications Copyright (c) 2025 CryoPARES developers
Licensed under GPL-3.0

The original torch-fourier-slice license is included in THIRD-PARTY-LICENSES.
"""

from typing import Tuple

import torch
import einops

from cryoPARES.configs.mainConfig import main_config
from cryoPARES.projmatching.projmatchingUtils.extract_central_slices_as_real import (
    _fftfreq_to_dft_coordinates,
    _rfft_shape_tuple,
)
from cryoPARES.utils.fourier_grid import central_slice_fftfreq_grid as _central_slice_fftfreq_grid
from cryoPARES.utils.fourier_geometry import _coordinates_from_constants


def _initialize_insertion(volume_shape, device, fftfreq_max, image_rfft_shape, *,
                          rfft=True, fftshift=True):
    """Compatibility builder; repeated callers should own a FourierBackprojector."""
    from cryoPARES.utils.fourier_geometry import FourierGeometry
    geometry = FourierGeometry(volume_shape, fftfreq_max, rfft=rfft, fftshift=fftshift, device=device)
    if tuple(image_rfft_shape[-2:]) != geometry.plane_shape:
        raise ValueError("image shape does not match Fourier insertion geometry")
    return geometry.valid_coords, geometry.valid_indices


def insert_central_slices_rfft_3d_multichannel(image_rfft, volume_shape, rotation_matrices,
                                             fftfreq_max=None, zyx_matrices=False):
    from cryoPARES.utils.fourier_operators import FourierBackprojector
    operator = FourierBackprojector(volume_shape, fftfreq_max, device=image_rfft.device,
                                    dtype=image_rfft.dtype)
    return operator(image_rfft, rotation_matrices, zyx_matrices=zyx_matrices)


def compiled_insert_central_slices_rfft_3d_multichannel(image_rfft, volume_shape, rotation_matrices,
                                                      fftfreq_max=None, zyx_matrices=False):
    from cryoPARES.utils.fourier_operators import FourierBackprojector
    operator = FourierBackprojector(volume_shape, fftfreq_max, compiled=True,
                                    device=image_rfft.device, dtype=image_rfft.dtype)
    return operator(image_rfft, rotation_matrices, zyx_matrices=zyx_matrices)


# ---------------------------
# Core worker (compile-friendly)
# ---------------------------

def _worker_insert_central_slices_rfft_3d_multichannel(
    image_rfft: torch.Tensor,                 # (..., C, H, W)
    volume_shape: Tuple[int, int, int],
    rotation_matrices: torch.Tensor,          # (..., 3, 3)
    valid_coords: torch.Tensor,               # (HW_sel, 3, 1)
    valid_idx: torch.Tensor,                  # (HW_sel,)
    zyx_matrices: bool = False,
    coordinate_constants=None,
    insertion_constants=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    No boolean advanced indexing or in-place masked writes. All shapes inside are static for a given
    (volume_shape, fftfreq_max) pair; `valid_idx` ensures fixed gather size HW_sel.
    """
    rotation_matrices = rotation_matrices.to(image_rfft.dtype)
    ft_dtype = image_rfft.dtype
    device = image_rfft.device

    # image_rfft: (..., C, H, W) -> (..., C, HW)
    img_flat = einops.rearrange(image_rfft, "... c H W -> ... c (H W)")
    # Gather fixed number of positions (no dynamic nonzero inside graph)
    gathered = torch.index_select(img_flat, dim=-1, index=valid_idx)       # (..., C, HW_sel)
    # For interpolation we want (..., HW_sel, C)
    valid_data = einops.rearrange(gathered, "... c b -> ... b c")          # (..., HW_sel, C)

    # Convert rotation matrices to zyx convention if requested
    if not zyx_matrices:
        # flip both matrix dims reverses rows and cols; static op (no data-dependent shape)
        rotation_matrices = torch.flip(rotation_matrices, dims=(-2, -1))
    # Add broadcast dim for matmul with (HW_sel, 3, 1)
    R = einops.rearrange(rotation_matrices, "... i j -> ... 1 i j")        # (..., 1, 3, 3)

    # Rotate coordinates: (..., 1, 3, 3) @ (HW_sel, 3, 1) -> (..., HW_sel, 3, 1) -> (..., HW_sel, 3)
    rotated_coordinates = R @ valid_coords
    rotated_coordinates = einops.rearrange(rotated_coordinates, "... hw zyx 1 -> ... hw zyx")

    # Determine sign for conjugate symmetry: flip coords with negative frequency along z
    conj_mask = rotated_coordinates[..., 2] < 0                            # (..., HW_sel)
    sign = torch.where(conj_mask, -1.0, 1.0).to(rotated_coordinates.dtype) # (..., HW_sel)

    # Flip coordinates by sign (broadcast over last axis=3)
    rotated_coordinates = rotated_coordinates * sign.unsqueeze(-1)         # (..., HW_sel, 3)

    # Conjugate the imaginary component of the data where needed (channels assumed: [real, imag, ctf^2, ...])
    # valid_data: (..., HW_sel, C)
    # real stays same; imag *= sign
    imag = valid_data[..., 1] * sign.to(valid_data.dtype)                  # (..., HW_sel)
    valid_data = torch.cat(
        (valid_data[..., :1], imag.unsqueeze(-1), valid_data[..., 2:]),
        dim=-1
    )                                                                       # (..., HW_sel, C)

    # Map fftfreq coords to DFT voxel coordinates (rfft=True)
    rotated_coordinates = (_fftfreq_to_dft_coordinates(rotated_coordinates, volume_shape, rfft=True)
                           if coordinate_constants is None else
                           _coordinates_from_constants(rotated_coordinates, *coordinate_constants))                                                                       # (..., HW_sel, 3)

    # Output volumes
    vol_dft_shape = _rfft_shape_tuple(volume_shape)                         # (Dz, Dy, Dx_rfft)
    C = image_rfft.shape[-3]
    dft_3d = torch.zeros((C, *vol_dft_shape), dtype=ft_dtype, device=device)
    weights = torch.zeros(vol_dft_shape, dtype=rotated_coordinates.dtype, device=device)

    # Insert data into 3D DFT
    dft_3d, weights = insert_into_image_3d(
        values=valid_data,                  # (..., HW_sel, C)
        coordinates=rotated_coordinates,    # (..., HW_sel, 3)
        image=dft_3d,                       # (C, Dz, Dy, Dx_rfft)
        weights=weights,                    # (Dz, Dy, Dx_rfft)
        insertion_constants=insertion_constants,
    )
    return dft_3d, weights


_compiled_worker_insert_central_slices_rfft_3d_multichannel = torch.compile(
    _worker_insert_central_slices_rfft_3d_multichannel,
    fullgraph=True,
    dynamic=True,
    mode=main_config.reconstruct.compile_insert_central_slices_rfft_3d_multichannel_mode,
)


# -------------------------------------------------------------
# Compile-friendly trilinear inserter (vectorized, no bool idx)
# -------------------------------------------------------------

def insert_into_image_3d(
    values: torch.Tensor,                  # (..., HW_sel, C)
    coordinates: torch.Tensor,             # (..., HW_sel, 3) in DFT voxel coords (z,y,x)
    image: torch.Tensor,                   # (C, D, H, W)
    weights: torch.Tensor | None = None,   # (D, H, W)
    insertion_constants=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compile-friendly trilinear insertion:
      - No data-dependent branching
      - No boolean advanced indexing
      - Fixed-shape scatter-add (adds zeros for out-of-bounds corners)
    """
    C, D, H, W = image.shape
    dev = image.device
    dtype = coordinates.dtype
    vdtype = image.dtype

    if weights is None:
        weights = torch.zeros((D, H, W), dtype=torch.float32, device=dev)

    # Flatten any leading batch dims
    vals = values.reshape(-1, values.shape[-1]).to(vdtype)    # (B, C)
    coords = coordinates.reshape(-1, 3).to(dtype)             # (B, 3)

    # Base voxel + fractional offsets
    base = torch.floor(coords)                                # (B, 3)
    tz, ty, tx = (coords - base).unbind(-1)                   # each (B,)

    # 8 corner offsets (z,y,x) in {0,1}
    offs = torch.tensor(
        [[0,0,0],[0,0,1],[0,1,0],[0,1,1],
         [1,0,0],[1,0,1],[1,1,0],[1,1,1]],
        device=dev, dtype=dtype
    ) if insertion_constants is None else insertion_constants[0]  # (8, 3)

    # Corner voxel coordinates, unclamped (B, 8, 3)
    corner = base.unsqueeze(1) + offs                         # (B, 8, 3)

    # Validity mask per corner (B, 8) — keep shape, no compression
    dims = (torch.tensor([D, H, W], device=dev, dtype=dtype)
            if insertion_constants is None else insertion_constants[1])
    valid = ((corner >= 0) & (corner < dims)).all(dim=-1)     # (B, 8)
    # All 8 corners must be in-bounds for the pixel to contribute (matches RELION behavior).
    # Partial insertions (some corners out-of-bounds) use a truncated kernel that the
    # sinc² gridding correction cannot properly compensate, adding high-frequency noise.
    pixel_valid = valid.all(dim=-1, keepdim=True)             # (B, 1)
    valid_f = pixel_valid.expand_as(valid).to(dtype)          # (B, 8), float

    # Clamp to bounds so indices are always valid
    corner_clamped = torch.minimum(
        torch.maximum(corner, torch.zeros_like(corner)),
        (dims - 1).unsqueeze(0).unsqueeze(0)
    )                                                         # (B, 8, 3)

    # Trilinear weights per corner (B, 8)
    # For each axis: choose t or (1-t) based on offset bit
    offs_f = offs                                              # (8, 3) in float dtype
    wz = offs_f[:, 0] * tz.unsqueeze(1) + (1 - offs_f[:, 0]) * (1 - tz.unsqueeze(1))
    wy = offs_f[:, 1] * ty.unsqueeze(1) + (1 - offs_f[:, 1]) * (1 - ty.unsqueeze(1))
    wx = offs_f[:, 2] * tx.unsqueeze(1) + (1 - offs_f[:, 2]) * (1 - tx.unsqueeze(1))
    cw = wz * wy * wx                                         # (B, 8)
    cw = cw * valid_f                                         # zero-out invalid corners

    # Flat indices for scatter-add
    zyx = corner_clamped.to(torch.long)                        # (B, 8, 3)
    z, y, x = zyx.unbind(-1)                                   # each (B, 8)
    flat_idx = z * (H * W) + y * W + x                         # (B, 8)

    # Update weights
    weights_flat = weights.view(-1)                            # (D*H*W,)
    weights_flat.index_add_(0, flat_idx.reshape(-1), cw.reshape(-1))

    # Update image across all channels, vectorized
    vals_expand = vals.unsqueeze(1).expand(-1, 8, -1)          # (B, 8, C)
    weighted_vals = vals_expand * cw.unsqueeze(-1).to(vdtype)  # (B, 8, C)
    image_flat = image.view(C, -1)                             # (C, D*H*W)
    image_flat.index_add_(1, flat_idx.reshape(-1), weighted_vals.reshape(-1, C).t())

    return image, weights


__all__ = [
    "insert_central_slices_rfft_3d_multichannel",
    "compiled_insert_central_slices_rfft_3d_multichannel",
]
