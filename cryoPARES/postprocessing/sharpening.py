"""
Fourier-space B-factor sharpening and FSC-weighting for cryo-EM maps.
"""
import numpy as np


def radial_freq_grid_3d(shape: tuple, px_A: float) -> np.ndarray:
    """
    Build a 3D radial spatial-frequency grid for an rFFT layout.

    Parameters
    ----------
    shape : (D, H, W)
    px_A : float — pixel size in Å

    Returns
    -------
    freq_grid : np.ndarray, shape (D, H, W//2+1), values in 1/Å
    """
    D, H, W = shape
    fz = np.fft.fftfreq(D)   / px_A  # (D,)
    fy = np.fft.fftfreq(H)   / px_A  # (H,)
    fx = np.fft.rfftfreq(W)  / px_A  # (W//2+1,)

    gz, gy, gx = np.meshgrid(fz, fy, fx, indexing='ij')
    freq_grid = np.sqrt(gz**2 + gy**2 + gx**2)
    return freq_grid.astype(np.float32)


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
                                  lowpass_A: float = None) -> np.ndarray:
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

    Returns
    -------
    np.ndarray (D, H, W), dtype float32
    """
    # Build the per-shell weight curve
    w_curve = fsc_weight_curve(fsc_corrected)  # shape: (n_shells,)

    # Radial spatial-frequency grid in rFFT layout
    freq_grid = radial_freq_grid_3d(vol_np.shape, px_A)  # (D, H, W//2+1)

    # Interpolate W(s) onto the full 3D frequency grid
    w_3d = np.interp(freq_grid.ravel(), spatial_freq, w_curve).reshape(freq_grid.shape)

    # B-factor weight: exp(-B * s² / 4)
    bfac_3d = np.exp(-bfactor * freq_grid**2 / 4.0)

    # Combined per-voxel weight
    weight_3d = (w_3d * bfac_3d).astype(np.float32)

    # Apply in Fourier space
    ft = np.fft.rfftn(vol_np.astype(np.float32))
    ft *= weight_3d

    # Determine hard cutoff: default is the first shell where fsc_corrected < 0.0001
    # (matches RELION's applyFscWeighting threshold; shells beyond that are noise
    # and would be massively amplified by the B-factor boost).
    if lowpass_A is None:
        cutoff_freq = float("nan")
        for i in range(len(spatial_freq)):
            if fsc_corrected[i] < 0.0001:
                cutoff_freq = float(spatial_freq[i])
                break
        if np.isfinite(cutoff_freq):
            ft[freq_grid > cutoff_freq] = 0.0
    else:
        # User-specified hard cutoff
        ft[freq_grid > 1.0 / lowpass_A] = 0.0

    sharpened = np.fft.irfftn(ft, s=vol_np.shape, axes=(0, 1, 2)).astype(np.float32)
    return sharpened
