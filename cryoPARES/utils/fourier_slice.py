"""Central Fourier projections implemented with PyTorch trilinear sampling."""

import torch
import torch.nn.functional as F

from cryoPARES.utils.fourier_geometry import fftfreq_to_dft_coordinates, _coordinates_from_constants
from cryoPARES.utils.fourier_grid import central_slice_fftfreq_grid


def extract_central_slices_rfft_3d(
    volume_rfft: torch.Tensor,
    rotation_matrices: torch.Tensor,
    image_shape: tuple[int, int, int] | None = None,
    fftfreq_max: float | None = None,
    zyx_matrices: bool = False,
) -> torch.Tensor:
    """Sample rotated central XY planes from a centered 3D rFFT.

    ``volume_rfft`` has shape ``(depth, height, width // 2 + 1)`` with its full
    Z/Y frequency axes fftshifted and its reduced X axis unshifted. ``image_shape``
    is the original 3D real-space shape. When omitted, an even width is inferred
    (a single stored column implies width one); odd widths require an explicit
    shape. No FFT or interpolation-compensation filter is applied here.

    Rotation matrices have shape ``(..., 3, 3)`` and actively rotate the frequency
    plane's XYZ coordinates, or ZYX when ``zyx_matrices=True``. Output has shape
    ``(..., height, width // 2 + 1)``. Negative X frequencies are recovered by
    Hermitian conjugation. Trilinear interpolation returns zero outside the
    stored array; no periodic wrapping is applied at the full-axis boundaries.
    ``fftfreq_max`` is an inclusive radial cutoff in cycles/pixel on the plane.

    Volume dtype/device are preserved; rotations must share the device and are
    converted to the volume's real dtype. Gradients propagate through the volume
    and rotations away from interpolation, conjugation and mask boundaries.
    """
    if volume_rfft.ndim != 3:
        raise ValueError("volume_rfft must have shape (depth, height, rfft_width)")
    if volume_rfft.dtype not in (torch.float32, torch.float64, torch.complex64, torch.complex128):
        raise TypeError("volume_rfft must have float32/64 or complex64/128 dtype")
    if rotation_matrices.ndim < 2 or rotation_matrices.shape[-2:] != (3, 3):
        raise ValueError("rotation_matrices must have shape (..., 3, 3)")
    if image_shape is None:
        image_shape = (*volume_rfft.shape[:2], max(1, 2 * (volume_rfft.shape[-1] - 1)))
    image_shape = tuple(image_shape)
    if len(image_shape) != 3 or any(n < 1 for n in image_shape):
        raise ValueError("image_shape must contain three positive real-space dimensions")
    expected_shape = (*image_shape[:2], image_shape[2] // 2 + 1)
    if tuple(volume_rfft.shape) != expected_shape:
        raise ValueError(f"volume_rfft shape must be {expected_shape} for image_shape={image_shape}")

    real_dtype = volume_rfft.real.dtype
    plane = central_slice_fftfreq_grid(
        image_shape, rfft=True, fftshift=True, device=volume_rfft.device, dtype=real_dtype,
    )
    matrices = rotation_matrices.to(dtype=real_dtype)
    if not zyx_matrices:
        matrices = matrices.flip((-2, -1))
    rotated = (matrices @ plane.reshape(-1, 3).T).transpose(-1, -2)
    rotated = rotated.reshape(*matrices.shape[:-2], *plane.shape)
    conjugate = rotated[..., 2] < 0
    folded = torch.where(conjugate[..., None], -rotated, rotated)
    coordinates = fftfreq_to_dft_coordinates(folded, image_shape, rfft=True)

    sizes = coordinates.new_tensor(volume_rfft.shape)
    inside = ((coordinates >= 0) & (coordinates <= sizes - 1)).all(dim=-1)
    if fftfreq_max is not None:
        inside = inside & (plane.square().sum(dim=-1).sqrt() <= fftfreq_max)
    normalized = 2 * coordinates / (sizes - 1).clamp_min(1) - 1
    normalized = torch.where(sizes == 1, 0., normalized).flip(-1)
    # One volume and one sampling grid: rotation batches become output points,
    # avoiding an input volume batch with one entry per interpolation point.
    channels = torch.stack((volume_rfft.real, volume_rfft.imag)) if volume_rfft.is_complex() else volume_rfft[None]
    samples = F.grid_sample(
        channels[None], normalized.reshape(1, 1, 1, -1, 3),
        mode="bilinear", padding_mode="border", align_corners=True,
    ).reshape(channels.shape[0], *rotated.shape[:-1])
    if volume_rfft.is_complex():
        values = torch.complex(samples[0], torch.where(conjugate, -samples[1], samples[1]))
    else:
        values = samples[0]
    return torch.where(inside, values, torch.zeros_like(values))


def _extract_precomputed(volume_rfft, matrices, plane_shape, valid_coords, valid_indices,
                         coordinate_constants, sampling_constants):
    """Tensor-only kernel. Geometry and XYZ-to-ZYX conversion belong to the caller."""
    rotated = (matrices @ valid_coords[..., 0].T).transpose(-1, -2)
    conjugate = rotated[..., 2] < 0
    folded = torch.where(conjugate[..., None], -rotated, rotated)
    coordinates = _coordinates_from_constants(folded, *coordinate_constants)
    upper, extent, _, singleton = sampling_constants
    inside = ((coordinates >= 0) & (coordinates <= upper)).all(dim=-1)
    normalized = 2 * coordinates / extent - 1
    normalized = torch.where(singleton, 0., normalized).flip(-1)
    # One volume and one sampling grid: rotation batches become output points,
    # avoiding an input volume batch with one entry per interpolation point.
    channels = torch.stack((volume_rfft.real, volume_rfft.imag)) if volume_rfft.is_complex() else volume_rfft[None]
    samples = F.grid_sample(
        channels[None], normalized.reshape(1, 1, 1, -1, 3),
        mode="bilinear", padding_mode="border", align_corners=True,
    ).reshape(channels.shape[0], *rotated.shape[:-1])
    if volume_rfft.is_complex():
        values = torch.complex(samples[0], torch.where(conjugate, -samples[1], samples[1]))
    else:
        values = samples[0]
    values = torch.where(inside, values, torch.zeros_like(values))
    output = values.new_zeros((*matrices.shape[:-2], plane_shape[0] * plane_shape[1]))
    return output.index_copy(-1, valid_indices, values).reshape(*matrices.shape[:-2], *plane_shape)
