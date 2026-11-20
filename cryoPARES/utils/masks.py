"""Local image masks with the numerical conventions used by projection matching."""

import torch
from scipy.ndimage import distance_transform_edt


def circle(
    radius: float,
    image_shape: tuple[int, int] | int,
    center: tuple[float, float] | None = None,
    smoothing_radius: float = 0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return a float32 circular mask on the requested/default PyTorch device.

    The default YX center is ``(height // 2, width // 2)``, matching the shifted
    FFT origin. Pixels strictly inside ``radius`` are one. A nonzero smoothing
    radius adds an outward cosine edge according to Euclidean distance to the
    nearest *foreground pixel*, not distance to the ideal circular perimeter.

    Edge distances are computed on CPU using SciPy, an existing dependency, then
    transferred to the output device. This preserves the previous implementation's
    float32 rounding, including tiny negative values at the end of the cosine.
    For compatibility, an empty disk retains SciPy's implicit exterior site at
    (-1, 0), which can produce a corner halo when smoothing is enabled.
    """
    height, width = (image_shape, image_shape) if isinstance(image_shape, int) else image_shape
    if center is None:
        center = (height // 2, width // 2)
    origin = torch.as_tensor(center, dtype=torch.float32, device=device)
    y = torch.arange(height, dtype=torch.float32, device=origin.device) - origin[0]
    x = torch.arange(width, dtype=torch.float32, device=origin.device) - origin[1]
    distance_from_center = (y[:, None].square() + x[None, :].square()).sqrt()
    foreground = distance_from_center < radius
    mask = foreground.to(torch.float32)

    edge_width = torch.as_tensor(smoothing_radius, dtype=torch.float32, device=mask.device)
    if edge_width == 0:
        return mask

    distances = torch.as_tensor(
        distance_transform_edt((~foreground).cpu().numpy()),
        dtype=torch.float32,
        device=mask.device,
    )
    edge = (distances > 0) & (distances <= edge_width)
    mask[edge] = torch.cos((torch.pi / 2) * (distances[edge] / edge_width))
    return mask
