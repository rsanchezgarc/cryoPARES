"""
Auto-mask generation for cryo-EM post-processing.

Two modes:
  A) Spherical — reuses Reconstructor.get_soft_mask()
  B) Density-threshold — Otsu or user threshold, dilation, cosine soft edge
"""
import numpy as np
import torch
from scipy import ndimage


def otsu_threshold(vol: np.ndarray, n_bins: int = 256) -> float:
    """
    Compute Otsu's threshold from a 256-bin histogram (pure numpy).

    Returns
    -------
    threshold : float
    """
    flat = vol.ravel()
    lo, hi = float(flat.min()), float(flat.max())
    if hi <= lo:
        return float(lo)

    counts, bin_edges = np.histogram(flat, bins=n_bins, range=(lo, hi))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    total = counts.sum()
    if total == 0:
        return float(lo)

    best_var = -1.0
    best_t = float(bin_centers[0])
    w0 = 0.0
    mu0_sum = 0.0
    mu_total = (counts * bin_centers).sum() / total

    for i in range(len(counts)):
        w0 += counts[i] / total
        mu0_sum += counts[i] * bin_centers[i] / total
        w1 = 1.0 - w0
        if w0 < 1e-10 or w1 < 1e-10:
            continue
        mu0 = mu0_sum / w0
        mu1 = (mu_total - w0 * mu0) / w1
        var = w0 * w1 * (mu0 - mu1) ** 2
        if var > best_var:
            best_var = var
            best_t = float(bin_centers[i])

    return best_t


def make_spherical_mask(shape: tuple, px_A: float,
                        radius_pix: float = -1.0,
                        edge_width: int = 6) -> np.ndarray:
    """
    Generate a soft spherical mask using Reconstructor.get_soft_mask().

    Parameters
    ----------
    shape : (D, H, W)
    px_A : float — pixel size (not used for spherical, kept for API symmetry)
    radius_pix : float — sphere radius in pixels; if <0 defaults to box_size/2
    edge_width : int — cosine fall-off width in pixels

    Returns
    -------
    mask : np.ndarray (D, H, W), float32, values in [0, 1]
    """
    from cryoPARES.reconstruction.reconstructor import Reconstructor
    mask_t = Reconstructor.get_soft_mask(
        tuple(shape), device="cpu",
        radius_pix=float(radius_pix),
        edge_width=int(edge_width))
    return mask_t.numpy().astype(np.float32)


def make_threshold_mask(avg_map: np.ndarray, px_A: float,
                        threshold: float = None,
                        dilation_A: float = 8.0,
                        edge_width_px: int = 6,
                        lowpass_A: float = 15.0) -> np.ndarray:
    """
    Generate a density-threshold soft mask.

    Steps
    -----
    1. Low-pass filter avg_map at *lowpass_A* Å.
    2. Threshold (Otsu or user-supplied).
    3. Binary dilation by *dilation_A* / *px_A* pixels.
    4. Binary closing (2 iterations).
    5. Cosine soft edge of width *edge_width_px* pixels via EDT.

    Parameters
    ----------
    avg_map : np.ndarray (D, H, W)
    px_A : float
    threshold : float, optional — if None, use Otsu
    dilation_A : float — dilation radius in Å
    edge_width_px : int — soft-edge width in pixels
    lowpass_A : float — low-pass filter resolution before thresholding (Å)

    Returns
    -------
    mask : np.ndarray (D, H, W), float32, values in [0, 1]
    """
    from cryoPARES.projmatching.projmatchingUtils.filterToResolution import low_pass_filter

    # 1. Low-pass filter
    vol_t = torch.from_numpy(avg_map.astype(np.float32))
    vol_lp = low_pass_filter(vol_t, resolution=lowpass_A,
                             sampling_rate=px_A).numpy()

    # 2. Threshold
    if threshold is None:
        threshold = otsu_threshold(vol_lp)
        print(f"Auto-mask Otsu threshold: {threshold:.6g}")
    binary = vol_lp > threshold

    # 3. Dilation
    dilation_px = max(1, round(dilation_A / px_A))
    binary = ndimage.binary_dilation(binary, iterations=dilation_px)

    # 4. Closing
    binary = ndimage.binary_closing(binary, iterations=2)

    # 5. Soft edge via EDT
    # Distance from the exterior of the binary mask to its surface.
    # Pixels inside the mask have distance 0 from the exterior.
    dist = ndimage.distance_transform_edt(~binary)   # distance outside

    mask = np.zeros_like(dist, dtype=np.float32)
    inside = binary
    mask[inside] = 1.0

    transition = (~binary) & (dist < edge_width_px)
    d_trans = dist[transition]
    mask[transition] = (0.5 + 0.5 * np.cos(np.pi * d_trans / edge_width_px)).astype(np.float32)

    return mask


def generate_mask(avg_map: np.ndarray, px_A: float,
                  spherical: bool = False,
                  threshold: float = None,
                  dilation_A: float = 8.0,
                  edge_width_px: int = 6) -> np.ndarray:
    """
    Unified auto-mask generator.

    Parameters
    ----------
    avg_map : np.ndarray (D, H, W)
    px_A : float
    spherical : bool — if True, produce spherical mask
    threshold : float, optional — density threshold for mode B
    dilation_A : float — dilation radius (mode B only)
    edge_width_px : int — soft-edge width in pixels

    Returns
    -------
    mask : np.ndarray (D, H, W), float32, values in [0, 1]
    """
    if spherical:
        return make_spherical_mask(avg_map.shape, px_A,
                                   edge_width=edge_width_px)
    return make_threshold_mask(avg_map, px_A,
                               threshold=threshold,
                               dilation_A=dilation_A,
                               edge_width_px=edge_width_px)
