"""
Fourier-space B-factor sharpening and FSC-weighting for cryo-EM maps.
"""
import numpy as np
import torch


def _get_device(device):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device) if not isinstance(device, torch.device) else device


def _torch_interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """1D piecewise-linear interpolation on device (equivalent to np.interp, clamps at edges)."""
    idx = torch.searchsorted(xp.contiguous(), x.contiguous()).clamp(1, len(xp) - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    y0, y1 = fp[idx - 1], fp[idx]
    t = ((x - x0) / (x1 - x0 + 1e-30)).clamp(0.0, 1.0)
    return y0 + t * (y1 - y0)


def radial_freq_grid_3d(shape: tuple, px_A: float, device=None) -> torch.Tensor:
    """
    Build a 3D radial spatial-frequency grid for an rFFT layout on *device*.

    Parameters
    ----------
    shape : (D, H, W)
    px_A  : float — pixel size in Å
    device : torch.device or None — auto-selects GPU if available

    Returns
    -------
    freq_grid : torch.Tensor, shape (D, H, W//2+1), float32, values in 1/Å
    """
    dev = _get_device(device)
    D, H, W = shape
    fz = (torch.fft.fftfreq(D, device=dev)  / px_A).reshape(-1, 1, 1)
    fy = (torch.fft.fftfreq(H, device=dev)  / px_A).reshape(1, -1, 1)
    fx = (torch.fft.rfftfreq(W, device=dev) / px_A).reshape(1, 1, -1)
    return torch.sqrt(fz**2 + fy**2 + fx**2)  # (D, H, W//2+1), float32


def fsc_weight_curve(fsc_corrected: np.ndarray) -> np.ndarray:
    """
    Compute the FSC-based amplitude weight per shell.

        W(s) = sqrt(2 * FSC(s) / (1 + FSC(s)))

    Clipped to [0, 1]; shells with FSC <= 0 receive weight 0.
    """
    fsc = np.clip(fsc_corrected, 0.0, 1.0)
    denom = 1.0 + fsc
    denom = np.where(denom < 1e-6, 1e-6, denom)
    w = np.sqrt(2.0 * fsc / denom)
    return np.clip(w.astype(np.float32), 0.0, 1.0)


def apply_bfactor_and_fsc_weight(vol_np: np.ndarray,
                                  fsc_corrected: np.ndarray,
                                  spatial_freq: np.ndarray,
                                  bfactor: float,
                                  px_A: float,
                                  lowpass_A: float = None,
                                  device=None) -> np.ndarray:
    """
    Apply FSC-based weighting and B-factor sharpening to a cryo-EM map.

    The per-shell filter is:
        H(s) = W(s) * exp(-B * s² / 4)

    where W(s) is the FSC figure-of-merit weight and B is the sharpening
    B-factor (typically negative for sharpening).

    Matching RELION's relion_postprocess: a hard Fourier cutoff (zeroing all
    shells beyond the cutoff) is applied when *lowpass_A* is given.  This
    matches RELION's behaviour of zeroing shells beyond the first-zero-crossing
    of fsc_corrected, which prevents noise amplification at very high
    frequency where the B-factor boost would otherwise be enormous.

    Parameters
    ----------
    vol_np : np.ndarray (D, H, W) — average of two half-maps
    fsc_corrected : 1D array — phase-corrected FSC per shell
    spatial_freq : 1D array (1/Å) — spatial freq for each FSC shell
    bfactor : float — B-factor in Å² (negative → sharpening)
    px_A : float — pixel size in Å
    lowpass_A : float, optional — hard Fourier cutoff resolution in Å;
                all shells beyond 1/lowpass_A are zeroed.  Default: the
                first-zero-crossing of fsc_corrected (computed internally).
    device : torch.device or None — GPU if available, else CPU

    Returns
    -------
    np.ndarray (D, H, W), dtype float32
    """
    dev = _get_device(device)

    # Build per-shell weight curve (small 1D op, stay numpy)
    w_curve = fsc_weight_curve(fsc_corrected)

    # Move 1D arrays to device
    freq_t  = torch.as_tensor(spatial_freq, dtype=torch.float32, device=dev)
    w_t     = torch.as_tensor(w_curve,      dtype=torch.float32, device=dev)
    fsc_t   = torch.as_tensor(fsc_corrected, dtype=torch.float32, device=dev)

    # Radial spatial-frequency grid in rFFT layout
    freq_grid = radial_freq_grid_3d(vol_np.shape, px_A, device=dev)  # (D, H, W//2+1)

    # Interpolate W(s) onto the full 3D frequency grid
    w_3d = _torch_interp(freq_grid.reshape(-1), freq_t, w_t).reshape(freq_grid.shape)

    # B-factor weight: exp(-B * s² / 4)
    bfac_3d = torch.exp(-bfactor * freq_grid ** 2 / 4.0)

    # Combined per-voxel weight
    weight_3d = (w_3d * bfac_3d).to(torch.float32)

    # Apply in Fourier space
    vol_t = torch.as_tensor(vol_np, dtype=torch.float32, device=dev)
    ft = torch.fft.rfftn(vol_t)
    ft = ft * weight_3d

    # Determine hard cutoff
    if lowpass_A is None:
        below = torch.where(fsc_t < 0.0001)[0]
        if below.numel() > 0:
            cutoff_freq = float(freq_t[below[0]])
            ft[freq_grid > cutoff_freq] = 0.0
    else:
        ft[freq_grid > 1.0 / lowpass_A] = 0.0

    sharpened = torch.fft.irfftn(ft, s=vol_np.shape).to(torch.float32)
    return sharpened.cpu().numpy()
