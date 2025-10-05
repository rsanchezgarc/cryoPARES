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

from functools import lru_cache
from typing import Tuple

import torch
import einops

from cryoPARES.configs.mainConfig import main_config
from cryoPARES.projmatching.projmatchingUtils.extract_central_slices_as_real import (
    _fftfreq_to_dft_coordinates,
    _rfft_shape_tuple,
)
from torch_fourier_slice._grids import _central_slice_fftfreq_grid


# ---------------------------
# Utilities / Cached builders
# ---------------------------

def _device_key(device: torch.device) -> str:
    # lru_cache keys must be hashable; store the device as a string
    return f"{device.type}:{device.index if device.index is not None else -1}"


@lru_cache(maxsize=16)
def _initialize_insertion_cached(
    volume_shape: Tuple[int, int, int],
    device_key: str,
    fftfreq_max: float | None,
    image_hw: Tuple[int, int],
    rfft: bool,
    fftshift: bool,
):
    """
    Cached helper that constructs:
      - valid_coords: (HW_sel, 3, 1) fftfreq coordinates to rotate
      - valid_idx:    (HW_sel,) linear indices of selected (h, w) grid points
    Everything is device-specific; the cache key includes device_key.
    """
    # Recreate device from key
    device_type, index_str = device_key.split(":")
    device_index = int(index_str)
    device = torch.device(device_type if device_index < 0 else f"{device_type}:{device_index}")

    # freq_grid: (H, W, 3) fftfreq coords for a central slice (xy-plane)
    freq_grid = _central_slice_fftfreq_grid(
        volume_shape=volume_shape,
        rfft=rfft,
        fftshift=fftshift,
        device=device,
    )  # (H, W, 3)

    H, W = image_hw

    if fftfreq_max is not None:
        # radial norm on grid (H, W)
        # einops.reduce over last dim (zyx/xyz), then sqrt
        normed = (einops.reduce(freq_grid ** 2, "h w zyx -> h w", reduction="sum") ** 0.5)
        # Boolean mask on (H, W). We must NOT use it inside compiled code,
        # but we can use it here in the cached builder safely.
        mask = normed <= fftfreq_max

        # valid coordinates (HW_sel, 3), flatten H*W using mask on CPU/GPU here
        valid_coords = freq_grid[mask, ...]
        # linear indices of True entries (HW_sel,)
        valid_idx = torch.nonzero(mask.reshape(-1), as_tuple=False).squeeze(1)
    else:
        # take full grid
        valid_coords = einops.rearrange(freq_grid, "h w zyx -> (h w) zyx")
        valid_idx = torch.arange(H * W, device=device)

    # add trailing singleton for matmul (..., 3, 1)
    valid_coords = einops.rearrange(valid_coords, "hw zyx -> hw zyx 1")

    return valid_coords, valid_idx


def _initialize_insertion(
    volume_shape: Tuple[int, int, int],
    device: torch.device,
    fftfreq_max: float | None,
    image_rfft_shape: torch.Size,
    *,
    rfft: bool = True,
    fftshift: bool = True,
):
    H, W = image_rfft_shape[-2], image_rfft_shape[-1]
    return _initialize_insertion_cached(
        volume_shape=volume_shape,
        device_key=_device_key(device),
        fftfreq_max=fftfreq_max,
        image_hw=(H, W),
        rfft=rfft,
        fftshift=fftshift,
    )


# ---------------------------
# Public API
# ---------------------------

def insert_central_slices_rfft_3d_multichannel(
    image_rfft: torch.Tensor,                 # fftshifted rfft of (..., C, H, W) 2D image
    volume_shape: Tuple[int, int, int],
    rotation_matrices: torch.Tensor,          # (..., 3, 3) dims must match leading dims of image_rfft
    fftfreq_max: float | None = None,
    zyx_matrices: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      dft_3d: (C, Dz, Dy, Dx_rfft) complex/real tensor (matches input dtype)
      weights: (Dz, Dy, Dx_rfft) float32 tensor
    """
    rotation_matrices = rotation_matrices.to(torch.float32)
    device = image_rfft.device

    valid_coords, valid_idx = _initialize_insertion(volume_shape, device, fftfreq_max, image_rfft.shape)
    dft_3d, weights = _worker_insert_central_slices_rfft_3d_multichannel(
        image_rfft, volume_shape, rotation_matrices, valid_coords, valid_idx, zyx_matrices
    )
    return dft_3d, weights


def compiled_insert_central_slices_rfft_3d_multichannel(
    image_rfft: torch.Tensor,                 # fftshifted rfft of (..., C, H, W) 2D image
    volume_shape: Tuple[int, int, int],
    rotation_matrices: torch.Tensor,          # (..., 3, 3) dims must match leading dims of image_rfft
    fftfreq_max: float | None = None,
    zyx_matrices: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotation_matrices = rotation_matrices.to(torch.float32)
    device = image_rfft.device

    valid_coords, valid_idx = _initialize_insertion(volume_shape, device, fftfreq_max, image_rfft.shape)
    dft_3d, weights = _compiled_worker_insert_central_slices_rfft_3d_multichannel(
        image_rfft, volume_shape, rotation_matrices, valid_coords, valid_idx, zyx_matrices
    )
    return dft_3d, weights


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
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    No boolean advanced indexing or in-place masked writes. All shapes inside are static for a given
    (volume_shape, fftfreq_max) pair; `valid_idx` ensures fixed gather size HW_sel.
    """
    rotation_matrices = rotation_matrices.to(torch.float32)
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
    rotated_coordinates = _fftfreq_to_dft_coordinates(
        rotated_coordinates, image_shape=volume_shape, rfft=True
    )                                                                       # (..., HW_sel, 3)

    # Output volumes
    vol_dft_shape = _rfft_shape_tuple(volume_shape)                         # (Dz, Dy, Dx_rfft)
    C = image_rfft.shape[-3]
    dft_3d = torch.zeros((C, *vol_dft_shape), dtype=ft_dtype, device=device)
    weights = torch.zeros(vol_dft_shape, dtype=torch.float32, device=device)

    # Insert data into 3D DFT
    dft_3d, weights = insert_into_image_3d(
        values=valid_data,                  # (..., HW_sel, C)
        coordinates=rotated_coordinates,    # (..., HW_sel, 3)
        image=dft_3d,                       # (C, Dz, Dy, Dx_rfft)
        weights=weights,                    # (Dz, Dy, Dx_rfft)
    )
    return dft_3d, weights


_compiled_worker_insert_central_slices_rfft_3d_multichannel = torch.compile(
    _worker_insert_central_slices_rfft_3d_multichannel,
    fullgraph=True,
    dynamic=True,
    mode=main_config.reconstruct.compile_insert_central_slices_rfft_3d_multichanne_mode,
)


# -------------------------------------------------------------
# Compile-friendly trilinear inserter (vectorized, no bool idx)
# -------------------------------------------------------------

def insert_into_image_3d(
    values: torch.Tensor,                  # (..., HW_sel, C)
    coordinates: torch.Tensor,             # (..., HW_sel, 3) in DFT voxel coords (z,y,x)
    image: torch.Tensor,                   # (C, D, H, W)
    weights: torch.Tensor | None = None,   # (D, H, W)
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
    )                                                         # (8, 3)

    # Corner voxel coordinates, unclamped (B, 8, 3)
    corner = base.unsqueeze(1) + offs                         # (B, 8, 3)

    # Validity mask per corner (B, 8) — keep shape, no compression
    dims = torch.tensor([D, H, W], device=dev, dtype=dtype)
    valid = ((corner >= 0) & (corner < dims)).all(dim=-1)     # (B, 8)
    valid_f = valid.to(dtype)                                 # (B, 8), float

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
