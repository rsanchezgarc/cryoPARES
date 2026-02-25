"""
Guinier-plot B-factor estimation for cryo-EM maps.

Convention used throughout:
  - Amplitude in Fourier shell s (1/Å): A(s) ~ A0 * exp(B * s² / 4)
  - Guinier plot: x = s², y = ln(A(s))
  - Linear fit slope = B / 4  →  B = 4 * slope
  - Negative B corresponds to a decaying amplitude (radiation damage);
    applying exp(−B·s²/4) with B < 0 then SHARPENS the map.
"""
import numpy as np
from typing import Optional

from cryoPARES.postprocessing.fsc_utils import compute_shell_amplitudes


def build_guinier_plot(amplitudes: np.ndarray, spatial_freq: np.ndarray,
                       res_lo_A: float, res_hi_A: float):
    """
    Build Guinier plot arrays for the frequency range [1/res_lo_A, 1/res_hi_A].

    Parameters
    ----------
    amplitudes : 1D array — mean |FFT| per shell
    spatial_freq : 1D array (1/Å)
    res_lo_A : float — lower resolution bound (Å); only shells at higher resolution used
    res_hi_A : float — upper resolution bound (Å); only shells at lower resolution used

    Returns
    -------
    x : 1D array — s² values (1/Å²)
    y : 1D array — ln(amplitude) values
    valid_mask : bool 1D array — True for shells within [1/res_lo_A, 1/res_hi_A]
    """
    s = spatial_freq
    lo_freq = 1.0 / res_lo_A  # lower freq bound (coarser res)
    hi_freq = 1.0 / res_hi_A  # upper freq bound (finer res)

    valid_mask = (s >= lo_freq) & (s <= hi_freq) & (amplitudes > 0)

    x = s**2
    # Guard against log(0)
    amp_safe = np.where(amplitudes > 0, amplitudes, 1.0)
    y = np.log(amp_safe)

    return x.astype(np.float64), y.astype(np.float64), valid_mask


def weighted_linear_fit(x: np.ndarray, y: np.ndarray,
                        weights: np.ndarray):
    """
    Weighted least-squares linear fit: y = slope * x + intercept.

    Parameters
    ----------
    x, y, weights : 1D arrays of equal length

    Returns
    -------
    slope, intercept, r2 : floats
    """
    w = np.asarray(weights, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    w_sum   = w.sum()
    wx_sum  = (w * x).sum()
    wy_sum  = (w * y).sum()
    wxx_sum = (w * x * x).sum()
    wxy_sum = (w * x * y).sum()

    denom = w_sum * wxx_sum - wx_sum**2
    if abs(denom) < 1e-30:
        return 0.0, float(np.average(y, weights=w)), 0.0

    slope     = (w_sum * wxy_sum - wx_sum * wy_sum) / denom
    intercept = (wy_sum - slope * wx_sum) / w_sum

    # R²
    y_pred = slope * x + intercept
    ss_res = (w * (y - y_pred)**2).sum()
    ss_tot = (w * (y - np.average(y, weights=w))**2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return float(slope), float(intercept), float(r2)


def estimate_bfactor(fsc_curves: dict, avg_map: np.ndarray, px_A: float,
                     guinier_lo_A: float = 10.0,
                     adhoc_bfac: Optional[float] = None):
    """
    Estimate the B-factor from a FOM-weighted Guinier plot of the average map.

    Matches RELION's relion_postprocess workflow:
      1. Apply FOM weight W(s) = sqrt(2·FSC/(1+FSC)) to each Fourier shell
         amplitude — this suppresses the noise floor at high resolution and
         confines the effective fit range to signal-dominated shells.
      2. Exclude shells where corrected FSC < 0.0001 (matches RELION's
         applyFscWeighting threshold; FOM ≈ 0, log undefined).
      3. OLS fit of ln(amplitude × W) vs s²;  B = 4 × slope.
      4. The first shell where fsc_corrected < 0.0001 defines the sharpening
         cutoff (same criterion as the Guinier range upper bound), matching
         RELION's hard Fourier zeroing at that shell.

    Parameters
    ----------
    fsc_curves : dict — output of run_gold_standard_fsc()
    avg_map : np.ndarray (D, H, W) — average of half1 and half2
    px_A : float — pixel size in Å
    guinier_lo_A : float — coarsest resolution to include in fit (Å)
    adhoc_bfac : float, optional — if given, skip fit and return directly

    Returns
    -------
    bfactor : float
    slope, intercept : float — Guinier fit parameters (or 0, 0 if adhoc)
    x_guin, y_guin : 1D arrays — Guinier plot x / ln(FOM·amplitude) values
    valid_mask : bool array — shells actually used in the fit
    cutoff_A : float — sharpening hard-cutoff resolution in Å (first shell
               where fsc_corrected < 0.0001); nan if FSC stays above threshold.
    """
    fsc_corr = fsc_curves["fsc_corrected"]
    sf_fsc   = fsc_curves["spatial_freq"]

    # First-zero-crossing of fsc_corrected — used for the hard Fourier cutoff
    # in both the Guinier fit range and the sharpening step, matching RELION.
    nyquist_A = 2.0 * px_A
    lo_freq   = 1.0 / guinier_lo_A
    hi_freq   = 1.0 / nyquist_A
    cutoff_freq = float("nan")
    for idx in range(len(sf_fsc)):
        if sf_fsc[idx] < lo_freq:
            continue
        if sf_fsc[idx] > hi_freq:
            break
        if fsc_corr[idx] < 0.0001:
            cutoff_freq = float(sf_fsc[idx])
            break
    cutoff_A = (1.0 / cutoff_freq) if np.isfinite(cutoff_freq) else float("nan")

    if adhoc_bfac is not None:
        empty = np.array([], dtype=np.float32)
        return float(adhoc_bfac), 0.0, 0.0, empty, empty, empty, cutoff_A

    amplitudes, spatial_freq = compute_shell_amplitudes(avg_map, px_A)

    # FOM weight per shell: W(s) = sqrt(2·FSC/(1+FSC)), clipped to [0,1].
    # Interpolated from the corrected FSC curve onto our shell grid.
    fsc_at_shell = np.interp(spatial_freq, sf_fsc, fsc_corr)
    fsc_pos      = np.clip(fsc_at_shell, 0.0, 1.0)
    fom          = np.sqrt(2.0 * fsc_pos / (1.0 + fsc_pos + 1e-9))

    amp_fom = amplitudes * fom          # FOM-weighted amplitude
    # y = ln(amp_fom); exclude shells where FOM=0 (log undefined)
    safe_amp = np.where(amp_fom > 0, amp_fom, np.nan)
    y_fom    = np.log(safe_amp)         # NaN where fom=0

    # Guinier fit range: [guinier_lo_A → first shell where fsc_corrected < 0.0001].
    # Matches RELION's applyFscWeighting threshold: break on first shell where
    # FSC < 0.0001 (includes negative FSC and very-near-zero FSC).  The cutoff
    # is seed-dependent, making the B-factor stochastic — same as RELION.
    amp_cutoff_idx = len(spatial_freq)
    for idx in range(len(spatial_freq)):
        if spatial_freq[idx] < lo_freq:
            continue
        if spatial_freq[idx] > hi_freq:
            break
        if fsc_at_shell[idx] < 0.0001:
            amp_cutoff_idx = idx
            break
    shell_idx = np.arange(len(spatial_freq))
    valid_mask = ((spatial_freq >= lo_freq) & (spatial_freq <= hi_freq)
                  & (shell_idx < amp_cutoff_idx) & np.isfinite(y_fom))

    x_guin = spatial_freq**2
    y_guin = y_fom

    if valid_mask.sum() < 3:
        print("Warning: too few shells for Guinier fit; returning B=0")
        return 0.0, 0.0, 0.0, x_guin, y_guin, valid_mask, cutoff_A

    # OLS fit (uniform weights within the valid region, matching RELION)
    weights = valid_mask.astype(np.float64)
    slope, intercept, r2 = weighted_linear_fit(
        x_guin[valid_mask], y_guin[valid_mask], weights[valid_mask])

    bfactor = 4.0 * slope  # B = 4 × slope
    print(f"Guinier fit (FOM-weighted): slope={slope:.4f}, "
          f"intercept={intercept:.4f}, R²={r2:.4f}, "
          f"shells={valid_mask.sum()}")
    return float(bfactor), float(slope), float(intercept), x_guin, y_guin, valid_mask, cutoff_A
