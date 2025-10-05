"""
GMM-based Score Distribution Analysis and Threshold Estimation
================================================================

This module provides tools for analyzing and comparing score distributions between
"good" and "bad" particle populations in cryo-EM datasets, with robust automatic
threshold estimation using Gaussian Mixture Models (GMMs).

Overview
--------
The primary use case is to:
1. Compare directional z-score distributions between aligned (good) and misaligned (bad) particles
2. Automatically estimate an optimal threshold to filter out low-quality particles
3. Visualize the distributions and fitted GMMs to assess separation quality

Key Features
------------
- **Robust GMM fitting**: Outlier-resistant fitting using initial filtering
- **Adaptive component selection**: Automatically tries 2-4 components based on data structure
- **Intelligent component pairing**: Selects appropriate components for threshold estimation
- **ROC-based fallback**: When GMM fails, falls back to ROC curve analysis
- **Quality warnings**: Alerts when separation is poor (d-prime < 1.0) or components are mismatched
- **Configurable percentile clipping**: Handles long-tailed distributions

Usage Modes
-----------
The `compare_prob_hists()` function supports three modes:

1. **GOOD + ALL mode**: Provide good particles and all particles; bad = all \\ good
   ```python
   compare_prob_hists(
       fname_good=["aligned.star"],
       fname_all=["all_particles.star"]
   )
   ```

2. **GOOD + BAD mode**: Provide good and bad particles directly
   ```python
   compare_prob_hists(
       fname_good=["aligned.star"],
       fname_bad=["misaligned.star"]
   )
   ```

3. **Symmetry-based inference**: Infer good/bad from angular error
   ```python
   compare_prob_hists(
       fname_good=["particles_with_poses.star"],
       symmetry="C1",
       degs_error_thr=3.0  # particles with error < 3° are "good"
   )
   ```

GMM Threshold Algorithm
-----------------------
The threshold estimation follows this strategy:

1. **Outlier clipping**: Remove extreme outliers using percentiles (default: 1-99%)
   to prevent long tails from dominating the fit

2. **Adaptive component selection**:
   - First try 2-component GMM
   - If smallest component has weight < 0.1 (likely outlier cluster), try 3-4 components
   - Select best model using BIC (Bayesian Information Criterion)

3. **Robust component pairing**:
   - **Good distribution**: Select higher-mean component when weights comparable (>0.3 each)
   - **Bad distribution**: Select lower-mean component when weights comparable
   - Otherwise select dominant component (highest weight)
   - Filter out tiny components (weight < 0.1) likely representing outliers

4. **Threshold computation**:
   - Find intersection of selected weighted Gaussian components
   - If intersection fails, use midpoint of component means
   - If GMM fails entirely, fall back to ROC-based threshold (weighted Youden's J)

5. **Quality assessment**:
   - Compute d-prime separation: d' = |μ_good - μ_bad| / (σ_good + σ_bad)
   - Warn if d' < 1.0 (poor separation, threshold unreliable)
   - Warn if selected components have inverted means (bad > good)

Key Parameters
--------------
- `low_pct`, `up_pct`: Percentile bounds for outlier clipping (default: 2.5, 97.5)
- `fn_cost`: Cost multiplier for false negatives (default: 2.0)
  - fn_cost > 1 means we prefer keeping good particles (conservative threshold)
  - Used in ROC fallback: weighted Youden's J = TPR - (FPR / fn_cost)
- `fallback_method`: What to do if GMM fails
  - "auto": Use ROC-based threshold (recommended)
  - "manual": Return None (requires manual inspection)
  - "none": Raise exception

Quality Metrics
---------------
- **d-prime (d')**: Signal detection theory metric measuring separation
  - d' > 2.0: Excellent separation
  - d' > 1.0: Good separation
  - d' < 1.0: Poor separation (warning issued)

- **Component weights**: GMM mixture weights
  - Components with weight < 0.1 considered outlier clusters
  - Comparable weights (both > 0.3) trigger special selection logic

Warnings and Edge Cases
------------------------
The code issues warnings for:
- Long tails: >10% of data clipped by percentiles
- Small GMM components: Smallest component weight < 0.1
- Poor separation: d-prime < 1.0
- Inverted means: Bad component mean > good component mean
- GMM fitting failures: Falls back to ROC method

CLI Usage
---------
Run as a script with automatic CLI generation via argParseFromDoc:

```bash
python -m cryoPARES.scripts.gmm_hists \
    --fname_good aligned.star \
    --fname_all all_particles.star \
    --plot_fname results/distributions.png \
    --low_pct 1.0 \
    --up_pct 99.0 \
    --fn_cost 2.0
```

Output files:
- `results/distributions.png`: Main histogram plot (bad vs good overlay)
- `results/distributions_gmm.png`: GMM components and threshold visualization

See Also
--------
- `cryoPARES.inference.infer`: Uses directional z-scores for particle filtering
- `cryoPARES.datamanager.particlesDataset`: Dataset class that applies thresholds
"""

import os
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import starfile
import torch

from scipy import stats, optimize
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_curve
import warnings

from cryoPARES.constants import (
    DIRECTIONAL_ZSCORE_NAME,
    RELION_EULER_CONVENTION,
    RELION_ANGLES_NAMES,
)
from cryoPARES.geometry.metrics_angles import rotation_error_with_sym


PRIMARY_KEYS = ["rlnMicrographName", "rlnCoordinateX", "rlnCoordinateY"]


def _normalize_micrograph_paths(df: pd.DataFrame, root_dir: Optional[str]) -> pd.DataFrame:
    """Normalize rlnMicrographName to improve joining across stacks."""
    def _to_rel_or_base(p):
        if not isinstance(p, str):
            return p
        if root_dir:
            try:
                return os.path.relpath(p, root_dir)
            except Exception:
                return p
        return os.path.basename(p)

    df = df.copy()
    df["rlnMicrographName"] = df["rlnMicrographName"].map(_to_rel_or_base)
    return df


def _read_particles_star(paths: List[str], root_dir: Optional[str], label: str) -> pd.DataFrame:
    out = []
    for p in paths:
        assert os.path.isfile(p), f"{label}: file not found: {p}"
        tab = starfile.read(p)["particles"]
        tab = _normalize_micrograph_paths(tab, root_dir)
        for k in PRIMARY_KEYS:
            if k not in tab.columns:
                raise KeyError(f"Missing column '{k}' in {label} ({p})")
        out.append(tab)
    df = pd.concat(out, ignore_index=True)
    return df

def _clip_percentiles(x: np.ndarray, low_pct: float = 1, up_pct: float=99) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo = np.percentile(x, low_pct)
    hi = np.percentile(x, up_pct)
    return np.clip(x, lo, hi)


def _find_intersection(mean1, std1, w1, mean2, std2, w2) -> Optional[float]:
    """Intersection of two weighted Gaussians."""
    def diff(x):
        return w1 * stats.norm.pdf(x, mean1, std1) - w2 * stats.norm.pdf(x, mean2, std2)
    x0 = (mean1 + mean2) / 2.0
    try:
        r = optimize.root_scalar(diff, x0=x0, bracket=[min(mean1, mean2), max(mean1, mean2)])
        if r.converged:
            return float(r.root)
    except Exception:
        pass
    return None


def _fit_gmm_robust(data: np.ndarray, n_components: int, random_state: int = 42) -> GaussianMixture:
    """
    Fit GMM with outlier-resistant strategy.

    Strategy:
    1. Fit 1-component GMM to identify extreme outliers
    2. Remove bottom 5% of points by log-likelihood
    3. Fit final n-component GMM on cleaned data

    This two-stage approach downweights extreme outliers without requiring
    sample_weight parameter (which is not available in older sklearn versions).

    Note: sklearn 1.7.2 and earlier don't support sample_weight in GaussianMixture.fit().
    This implementation uses outlier removal instead.

    Returns fitted GaussianMixture model
    """
    data = data.reshape(-1, 1) if data.ndim == 1 else data

    # Initial fit to identify outliers
    gmm_init = GaussianMixture(n_components=1, random_state=random_state)
    gmm_init.fit(data)

    # Compute log probabilities
    log_prob = gmm_init.score_samples(data)

    # Remove extreme outliers (bottom 5% by likelihood)
    threshold_prob = np.percentile(log_prob, 5)
    mask_keep = log_prob >= threshold_prob
    data_cleaned = data[mask_keep]

    # Fit final GMM on cleaned data
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(data_cleaned)

    return gmm


def _select_best_gmm(data: np.ndarray, n_components_options: List[int] = [2, 3, 4],
                     low_pct: float = 1, up_pct: float = 99,
                     min_component_weight: float = 0.1) -> Tuple[GaussianMixture, int]:
    """
    Try fitting GMM with different numbers of components, select best by BIC.

    Strategy:
    1. Check for long tails: if >10% of data clipped by percentiles, limit to 2 components
       (avoids overfitting sparse outlier regions)
    2. First try 2-component GMM
    3. If smallest component has weight < min_component_weight (default 0.1):
       - Suggests outlier cluster that's small but not negligible
       - Try 3-4 components to model outliers separately
    4. Use BIC (Bayesian Information Criterion) to select best among tried options
       - Lower BIC = better fit with appropriate complexity
    5. Require at least 10 data points per component

    Warnings issued for:
    - Long tails detected (>10% clipped)
    - Small component in 2-GMM (triggers multi-component search)
    - Selected model has >2 components (indicates outlier structure)
    - GMM fitting failures for specific component counts

    Returns (best_gmm, n_components_selected)
    """
    # Check if long tails warrant trying multiple components
    data_clipped = _clip_percentiles(data, low_pct, up_pct)
    fraction_kept = len(data_clipped) / max(1, len(data))

    if fraction_kept < 0.9:
        warnings.warn(f"Long tails detected ({(1-fraction_kept)*100:.1f}% of data clipped). "
                     "Limiting to 2-component GMM to avoid overfitting tails.")
        n_components_options = [2]

    data = data.reshape(-1, 1) if data.ndim == 1 else data

    # Always try 2 components first
    try:
        gmm_2 = _fit_gmm_robust(data, n_components=2)
        bic_2 = gmm_2.bic(data)
        weights_2 = gmm_2.weights_.flatten()
        min_weight_2 = weights_2.min()

        # If smallest component is too small, it likely represents outliers
        # Try 3-4 components to better model the outlier tail
        if min_weight_2 < min_component_weight and len(data) >= 40:
            warnings.warn(f"2-component GMM has small component (w={min_weight_2:.3f} < {min_component_weight}). "
                         f"Trying 3-4 components to model outliers separately.")
            n_components_to_try = [2, 3, 4]
        else:
            # 2 components look balanced, stick with it
            n_components_to_try = [2]

    except Exception as e:
        warnings.warn(f"Failed to fit 2-component GMM: {e}")
        n_components_to_try = [2, 3, 4]  # Try everything as fallback

    # Now fit all candidates and pick best by BIC
    best_gmm = None
    best_bic = np.inf
    best_n = None

    for n in n_components_to_try:
        if len(data) < n * 10:  # Need at least 10 points per component
            continue
        try:
            gmm = _fit_gmm_robust(data, n_components=n)
            bic = gmm.bic(data)
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
                best_n = n
        except Exception as e:
            warnings.warn(f"Failed to fit {n}-component GMM: {e}")
            continue

    if best_gmm is None:
        raise ValueError("Could not fit GMM with any number of components")

    if best_n != 2:
        warnings.warn(f"Selected {best_n}-component GMM (BIC={best_bic:.1f}). "
                     f"Data has outlier structure requiring extra components.")

    return best_gmm, best_n


def _select_components_robust(gmm_bad: GaussianMixture, gmm_good: GaussianMixture) -> Tuple[int, int, str]:
    """
    Robustly select which component from each GMM to use for threshold.

    Strategy:
    - **Good distribution**: Select higher-mean component when weights comparable (both > 0.3)
      Otherwise select dominant component (highest weight)
    - **Bad distribution**: Select lower-mean component when weights comparable (both > 0.3)
      Otherwise select dominant component (highest weight)
    - **Rationale**: Good particles should have higher scores than bad particles.
      When both components are substantial, we want the "main" distribution, not outlier tails.
      For good: higher mean captures true positives. For bad: lower mean captures contamination.
    - **Edge cases**: Filter out tiny components (weight < 0.1) before selection

    Quality checks:
    - Warns if good distribution has lower weighted mean than bad (suggests data issues)
    - Warns if selected bad component has higher mean than good (threshold unreliable)
    - Computes d-prime separation and warns if < 1.0 (poor discriminability)

    Returns (idx_bad, idx_good, method_description)
    """
    m_b = gmm_bad.means_.flatten()
    s_b = np.sqrt(gmm_bad.covariances_.flatten())
    w_b = gmm_bad.weights_.flatten()

    m_g = gmm_good.means_.flatten()
    s_g = np.sqrt(gmm_good.covariances_.flatten())
    w_g = gmm_good.weights_.flatten()

    # Sort by mean
    ob, og = np.argsort(m_b), np.argsort(m_g)
    m_b, s_b, w_b = m_b[ob], s_b[ob], w_b[ob]
    m_g, s_g, w_g = m_g[og], s_g[og], w_g[og]

    # Sanity check: good should have higher mean overall
    mean_bad_weighted = np.sum(m_b * w_b)
    mean_good_weighted = np.sum(m_g * w_g)

    if mean_good_weighted < mean_bad_weighted:
        warnings.warn(f"Good distribution has LOWER weighted mean ({mean_good_weighted:.2f}) "
                     f"than bad ({mean_bad_weighted:.2f})! Data may be mislabeled or scores inverted.")

    # Select good component
    # If weights are comparable (both > 0.3), prefer higher mean
    # Otherwise use dominant component
    n_comp_g = len(m_g)
    if n_comp_g == 2:
        if w_g[0] > 0.3 and w_g[1] > 0.3:
            idx_good = 1  # Higher mean (since sorted)
            method_good = f"higher_mean(w0={w_g[0]:.2f},w1={w_g[1]:.2f})"
        else:
            idx_good = np.argmax(w_g)
            method_good = f"dominant(w={w_g[idx_good]:.2f})"
    else:
        # For 3+ components, use dominant
        idx_good = np.argmax(w_g)
        method_good = f"dominant_{n_comp_g}comp(w={w_g[idx_good]:.2f})"

    # Select bad component: higher mean if weights comparable, otherwise dominant
    # Same logic as good, but we expect bad to have lower mean overall
    min_weight_threshold = 0.1  # Components below this are likely outliers
    n_comp_b = len(m_b)

    if n_comp_b == 2:
        # Filter out tiny components (likely outliers)
        valid_components = w_b >= min_weight_threshold

        if valid_components.sum() == 2:
            # Both components have sufficient weight
            if w_b[0] > 0.3 and w_b[1] > 0.3:
                # Comparable weights - pick lower mean (bad should have lower scores than good)
                idx_bad = 0
                method_bad = f"lower_mean(w0={w_b[0]:.2f},w1={w_b[1]:.2f})"
            else:
                # One dominant - use it
                idx_bad = np.argmax(w_b)
                method_bad = f"dominant(w={w_b[idx_bad]:.2f})"
        elif valid_components.sum() == 1:
            # One component is tiny outlier - use the valid one
            idx_bad = np.where(valid_components)[0][0]
            method_bad = f"only_valid(w={w_b[idx_bad]:.2f})"
        else:
            # Both components are tiny?! Use dominant anyway
            idx_bad = np.argmax(w_b)
            method_bad = f"dominant_despite_small(w={w_b[idx_bad]:.2f})"
    else:
        # For 3+ components, use dominant among valid ones
        valid_components = w_b >= min_weight_threshold
        if valid_components.sum() > 0:
            valid_indices = np.where(valid_components)[0]
            idx_bad = valid_indices[np.argmax(w_b[valid_components])]
            method_bad = f"dominant_{n_comp_b}comp(w={w_b[idx_bad]:.2f})"
        else:
            idx_bad = np.argmax(w_b)
            method_bad = f"dominant_{n_comp_b}comp_all_small(w={w_b[idx_bad]:.2f})"

    # Warning if selected components don't make sense
    if m_b[idx_bad] > m_g[idx_good]:
        warnings.warn(f"Selected bad component (μ={m_b[idx_bad]:.2f}) has HIGHER mean than "
                     f"good component (μ={m_g[idx_good]:.2f})! Threshold may be unreliable.")

    # Compute separation quality (d-prime)
    separation = abs(m_g[idx_good] - m_b[idx_bad]) / (s_g[idx_good] + s_b[idx_bad] + 1e-9)

    if separation < 1.0:
        warnings.warn(f"Poor d-prime separation between selected components (d'={separation:.2f}). "
                     f"Threshold may be unreliable. Consider manual inspection.")

    # Convert back to original indices
    idx_bad_orig = ob[idx_bad]
    idx_good_orig = og[idx_good]

    method_desc = f"good:{method_good}|bad:{method_bad}|sep={separation:.2f}"

    return idx_bad_orig, idx_good_orig, method_desc


def _fallback_roc_threshold(score_bad: np.ndarray, score_good: np.ndarray,
                            fn_cost: float = 2.0) -> Tuple[float, float, str]:
    """
    Fallback method: Find optimal threshold using ROC curve analysis.

    This method is used when GMM fitting fails. It finds the threshold that maximizes
    a weighted Youden's J statistic, which balances true positive rate and false positive rate.

    Algorithm:
    1. Combine bad (label=0) and good (label=1) scores
    2. Compute ROC curve (TPR, FPR, thresholds)
    3. Find threshold maximizing: J_weighted = TPR - (FPR / fn_cost)
       - fn_cost > 1: More conservative, prioritizes keeping good particles
       - fn_cost = 1: Standard Youden's J (TPR - FPR)
       - fn_cost < 1: More aggressive, willing to lose good particles to reject bad ones

    Parameters:
    - score_bad: Array of scores for bad particles
    - score_good: Array of scores for good particles
    - fn_cost: Cost multiplier for false negatives (default: 2.0)

    Returns (threshold, quality_metric, method_description)
    """
    # Create binary labels
    y_true = np.concatenate([
        np.zeros(len(score_bad)),  # 0 = bad
        np.ones(len(score_good))   # 1 = good
    ])
    y_scores = np.concatenate([score_bad, score_good])

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Weighted Youden's J: TPR - (fn_cost * FPR)
    # fn_cost > 1 means we care more about keeping good particles
    J_weighted = tpr - (fpr / fn_cost)
    idx_best = np.argmax(J_weighted)

    threshold = thresholds[idx_best]
    quality = J_weighted[idx_best]

    method = f"ROC_Youden_weighted(fn_cost={fn_cost:.1f},J={quality:.3f})"

    return float(threshold), float(quality), method


def _plot_gmms(dist_bad, dist_good, threshold, gmms, gmm_labels, method, show_plots, out_base: Optional[str]):
    plt.figure(figsize=(15, 6))

    # Plot histograms and capture their max density for y-limit calculation
    n_bad, _, _ = plt.hist(dist_bad, bins=50, alpha=0.5, density=True, label="Bad", color="orange")
    n_good, _, _ = plt.hist(dist_good, bins=50, alpha=0.5, density=True, label="Good", color="blue")
    hist_max_density = max(n_bad.max(), n_good.max())

    # Compute intelligent x-axis limits based on GMM components
    # Focus on the region where the main distributions are, not outliers
    all_means = []
    all_stds = []
    for gmm in gmms:
        means = gmm.means_.flatten()
        stds = np.sqrt(gmm.covariances_.flatten())
        all_means.extend(means)
        all_stds.extend(stds)

    all_means = np.array(all_means)
    all_stds = np.array(all_stds)

    # X-axis: span from min(μ - 3σ) to max(μ + 3σ) of all components
    # This covers ~99.7% of each Gaussian, focusing on the main distributions
    x_min = np.min(all_means - 3 * all_stds)
    x_max = np.max(all_means + 3 * all_stds)

    # Add small padding around the focused region
    x_range = x_max - x_min
    x_min -= 0.1 * x_range
    x_max += 0.1 * x_range

    # Make sure threshold is visible
    x_min = min(x_min, threshold - 0.5 * x_range)
    x_max = max(x_max, threshold + 0.5 * x_range)

    X = np.linspace(x_min, x_max, 1000)

    styles = ["--r", "--b"]
    for i, (gmm, label) in enumerate(zip(gmms, gmm_labels)):
        means = gmm.means_.flatten()
        stds = np.sqrt(gmm.covariances_.flatten())
        weights = gmm.weights_.flatten()
        order = np.argsort(means)
        means, stds, weights = means[order], stds[order], weights[order]
        for j, (m, s, w) in enumerate(zip(means, stds, weights)):
            plt.plot(X, w * stats.norm.pdf(X, m, s), styles[i], alpha=0.7,
                     label=f"{label} Comp {j+1} (μ={m:.2f}, σ={s:.2f}, w={w:.2f})")

    plt.axvline(threshold, color="g", label=f"Threshold {threshold:.2f} ({method.split("(")[0]})")
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.title("GMM fitting on score distributions")

    # Set x and y limits intelligently
    plt.xlim(x_min, x_max)
    plt.ylim(0, hist_max_density * 1.5)

    plt.legend()
    plt.grid(True)

    if out_base:
        plt.savefig(f"{out_base}_gmm.png", bbox_inches="tight", dpi=150)
    if show_plots:
        plt.show()
    plt.close()


def separate_gmm_threshold(
    dist_bad: np.ndarray,
    dist_good: np.ndarray,
    show_plots: bool = True,
    out_base: Optional[str] = None,
    low_pct: float = 1,
    up_pct: float = 99,
    fallback_method: str = "auto",
    fn_cost: float = 2.0,
    n_components_options: List[int] = [2, 3, 4],
) -> Tuple[float, Tuple[Optional[GaussianMixture], Optional[GaussianMixture]], str]:
    """
    Estimate threshold between good/bad particle distributions using robust GMM fitting.

    This is the main threshold estimation function. It fits separate GMMs to bad and good
    distributions, intelligently selects appropriate components, and finds their intersection
    as the optimal threshold.

    Algorithm:
    1. **Outlier clipping**: Clip data to [low_pct, up_pct] percentiles to handle long tails
    2. **Adaptive GMM fitting**: Try 2-4 components per distribution, select best by BIC
       - Start with 2 components
       - If smallest component weight < 0.1, try 3-4 (suggests outliers need separate model)
    3. **Robust component selection**:
       - Good: higher-mean component if weights comparable, else dominant
       - Bad: lower-mean component if weights comparable, else dominant
    4. **Threshold computation**: Find weighted Gaussian intersection, or use midpoint
    5. **Fallback**: If GMM fails, use ROC-based threshold (weighted Youden's J)

    Quality checks:
    - Requires at least 20 samples per distribution
    - Warns if d-prime separation < 1.0
    - Warns if component means are inverted (bad > good)
    - Warns if GMM fitting fails (triggers fallback)

    Visualization:
    - Histogram overlay of bad (orange) and good (blue) distributions
    - GMM component curves with parameters (μ, σ, w)
    - Threshold line with method description
    - Intelligent x-axis limits (μ ± 3σ of components, not full data range)
    - Y-axis limited to prevent tiny-sigma components from dominating

    @param dist_bad: 1D array of scores for BAD particles
    @param dist_good: 1D array of scores for GOOD particles
    @param show_plots: If True, display GMM plot interactively
    @param out_base: If provided, save GMM plot to "<out_base>_gmm.png"
    @param low_pct: Lower percentile for clipping outliers (default: 1)
    @param up_pct: Upper percentile for clipping outliers (default: 99)
    @param fallback_method: Behavior when GMM fails:
        - "auto": Use ROC-based threshold (recommended)
        - "manual": Return None, requiring manual inspection
        - "none": Raise exception
    @param fn_cost: False negative cost multiplier (default: 2.0)
        - fn_cost > 1: Conservative, prioritizes keeping good particles
        - Used in ROC fallback: J_weighted = TPR - (FPR / fn_cost)
    @param n_components_options: GMM component counts to try (default: [2,3,4])
        Note: Long tails (>10% clipped) override this to [2] only

    @returns: (threshold, (gmm_bad, gmm_good), method_description)
        - threshold: Optimal score threshold (float)
        - gmm_bad, gmm_good: Fitted GaussianMixture objects (or None, None if fallback used)
        - method_description: String describing how threshold was computed

    @raises ValueError: If fewer than 20 samples in either distribution (and fallback_method="none")

    Example:
        >>> thr, (gmm_b, gmm_g), method = separate_gmm_threshold(
        ...     score_bad, score_good,
        ...     low_pct=2.5, up_pct=97.5,
        ...     fallback_method="auto", fn_cost=2.0
        ... )
        >>> print(f"Threshold: {thr:.4f}, Method: {method}")
    """
    db = np.asarray(dist_bad, dtype=float)
    dg = np.asarray(dist_good, dtype=float)
    db = db[np.isfinite(db)]
    dg = dg[np.isfinite(dg)]

    if len(db) < 20 or len(dg) < 20:
        raise ValueError(f"Not enough points for GMM thresholding (bad={len(db)}, good={len(dg)}). Need at least 20 each.")

    # Data for plotting (percentile clipped)
    db_plot = _clip_percentiles(db, low_pct, up_pct)
    dg_plot = _clip_percentiles(dg, low_pct, up_pct)

    # Data for GMM fitting (percentile clipped only - no absolute thresholds!)
    db_fit = _clip_percentiles(db, low_pct, up_pct)
    dg_fit = _clip_percentiles(dg, low_pct, up_pct)

    try:
        # Try to fit GMMs with robust component selection
        print(f"Fitting GMM to bad distribution ({len(db_fit)} samples after clipping)...")
        gmm_b, n_comp_b = _select_best_gmm(db_fit, n_components_options, low_pct, up_pct)

        print(f"Fitting GMM to good distribution ({len(dg_fit)} samples after clipping)...")
        gmm_g, n_comp_g = _select_best_gmm(dg_fit, n_components_options, low_pct, up_pct)

        print(f"Selected {n_comp_b}-component GMM for bad, {n_comp_g}-component GMM for good")

        # Robustly select which components to use for intersection
        idx_bad, idx_good, method_desc = _select_components_robust(gmm_b, gmm_g)

        # Extract selected components
        m_b = gmm_b.means_.flatten()[idx_bad]
        s_b = np.sqrt(gmm_b.covariances_.flatten()[idx_bad])
        w_b = gmm_b.weights_.flatten()[idx_bad]

        m_g = gmm_g.means_.flatten()[idx_good]
        s_g = np.sqrt(gmm_g.covariances_.flatten()[idx_good])
        w_g = gmm_g.weights_.flatten()[idx_good]

        # Find intersection
        x_int = _find_intersection(m_b, s_b, w_b, m_g, s_g, w_g)

        if x_int is not None:
            thr = x_int
            method = f"GMM_intersection({method_desc})"
        else:
            # Fallback within GMM: use midpoint
            thr = (m_b + m_g) / 2.0
            method = f"GMM_midpoint({method_desc})"
            warnings.warn("GMM intersection failed, using midpoint of selected components")

        # Determine labels for plotting
        gmm_labels = ("Bad", "Good")
        gmms = (gmm_b, gmm_g)

        if show_plots or out_base:
            _plot_gmms(db_plot, dg_plot, thr, gmms, gmm_labels, method, show_plots, out_base)

        return float(thr), gmms, method

    except Exception as e:
        # GMM failed - use fallback
        warnings.warn(f"GMM threshold estimation failed: {e}")

        if fallback_method == "auto":
            warnings.warn("Using fallback: ROC-based threshold")
            thr, quality, method = _fallback_roc_threshold(db, dg, fn_cost=fn_cost)
            print(f"Fallback threshold: {thr:.4f} (quality={quality:.3f})")

            # Still plot histograms even without GMM
            if show_plots or out_base:
                plt.figure(figsize=(15, 6))
                plt.hist(db_plot, bins=50, alpha=0.5, density=True, label="Bad", color="orange")
                plt.hist(dg_plot, bins=50, alpha=0.5, density=True, label="Good", color="blue")
                plt.axvline(thr, color="g", label=f"Threshold {thr:.2f} ({method})")
                plt.xlabel("Score")
                plt.ylabel("Density")
                plt.title("Score distributions (GMM failed, using ROC)")
                plt.legend()
                plt.grid(True)
                if out_base:
                    plt.savefig(f"{out_base}_fallback.png", bbox_inches="tight", dpi=150)
                if show_plots:
                    plt.show()
                plt.close()

            return float(thr), (None, None), method

        elif fallback_method == "manual":
            warnings.warn("GMM failed and fallback_method='manual'. Returning None - manual inspection required.")
            return None, (None, None), "manual_required"

        else:  # fallback_method == "none"
            raise


def compare_prob_hists(
    fname_good: List[str],
    fname_all: Optional[List[str]] = None,
    fname_bad: Optional[List[str]] = None,
    fname_ignore: Optional[List[str]] = None,
    score_name: str = DIRECTIONAL_ZSCORE_NAME,
    plot_fname: Optional[str] = None,
    show_plots: bool = True,
    symmetry: Optional[str] = None,
    degs_error_thr: float = 3.0,
    bins: int = 200,
    fraction_to_sample_from_all: Optional[float] = None,
    compute_gmm: bool = True,
    root_dir: Optional[str] = None,
    low_pct: float = 2.5,
    up_pct: float = 97.5,
    fallback_method: str = "auto",
    fn_cost: float = 2.0,
):
    """
    Compare score distributions for GOOD vs BAD particles and (optionally) compute a GMM threshold.

    Modes:
      1) GOOD + ALL  -> BAD = ALL \\ GOOD (set difference on primary keys).
      2) GOOD + BAD  -> BAD is provided directly (no set difference).
      3) GOOD only + symmetry -> infer GOOD by angular error < degs_error_thr; BAD = complement.

    @param fname_good: list of .star files with GOOD particles
    @param fname_all: list of .star files with ALL particles (superset of GOOD). Mutually exclusive with fname_bad
    @param fname_bad: list of .star files with BAD particles (disjoint from GOOD). Mutually exclusive with fname_all
    @param fname_ignore: list of .star files to be removed from the background set (ALL or BAD) via primary keys
    @param score_name: name of the score column to analyze (default: DIRECTIONAL_ZSCORE_NAME)
    @param plot_fname: path to save the main histogram figure; if compute_gmm=True, also saves "<plot_fname>_gmm.png"
    @param show_plots: if True, show matplotlib figures interactively
    @param symmetry: required if neither fname_all nor fname_bad is provided (for angle-based inference)
    @param degs_error_thr: angular error threshold (degrees) to define GOOD when inferring by symmetry
    @param bins: number of histogram bins
    @param fraction_to_sample_from_all: fraction to subsample from the background set for speed (0 < f < 1)
    @param compute_gmm: if True, fit separate 2-component GMMs and compute a threshold
    @param root_dir: normalize rlnMicrographName as relative to this directory; if None, use basename
    @param low_pct: lower percentile for clipping outliers in histograms and GMM fitting
    @param up_pct: upper percentile for clipping outliers in histograms and GMM fitting
    @param fallback_method: fallback if GMM fails - "auto" (ROC-based), "manual" (return None), "none" (raise error)
    @param fn_cost: cost multiplier for false negatives vs false positives (>1 = more conservative, keeps more good particles)
    @returns: If compute_gmm and enough samples, returns (threshold, (gmm_bad, gmm_good), method); otherwise None
    """

    if fname_all is not None and fname_bad is not None:
        raise ValueError("Provide only one of fname_all or fname_bad.")

    # ==================================================================================
    # STEP 1: Read GOOD particles (always required)
    # ==================================================================================
    parts_good = _read_particles_star(fname_good, root_dir, "GOOD")
    if score_name not in parts_good.columns:
        raise KeyError(f"GOOD missing score column '{score_name}'")

    # ==================================================================================
    # STEP 2: Build background dataset (parts_bg) - determines operating mode
    # ==================================================================================
    # Three modes:
    #   MODE 1 (GOOD + ALL):   parts_bg = ALL particles, later compute BAD = ALL \ GOOD
    #   MODE 2 (GOOD + BAD):   parts_bg = BAD particles (provided directly)
    #   MODE 3 (GOOD + symmetry): parts_bg = original GOOD file, infer GOOD/BAD by angular error

    if fname_all is not None:
        # MODE 1: User provided ALL particles (superset containing both good and bad)
        parts_bg = _read_particles_star(fname_all, root_dir, "ALL")
    elif fname_bad is not None:
        # MODE 2: User provided BAD particles directly (disjoint from GOOD)
        parts_bg = _read_particles_star(fname_bad, root_dir, "BAD")
    else:
        # MODE 3: No background provided - infer GOOD/BAD by comparing predicted vs ground-truth poses
        if symmetry is None:
            raise ValueError("symmetry is required when neither fname_all nor fname_bad is provided.")
        from scipy.spatial.transform import Rotation as R
        for c in RELION_ANGLES_NAMES + [x + "_ori" for x in RELION_ANGLES_NAMES]:
            if c not in parts_good.columns:
                raise KeyError(f"Missing column '{c}' required for angle-based inference.")

        # MODE 3: Compute angular errors between predicted and ground-truth poses
        rots  = R.from_euler(RELION_EULER_CONVENTION, parts_good[RELION_ANGLES_NAMES], degrees=True)
        gtRot = R.from_euler(RELION_EULER_CONVENTION, parts_good[[x + "_ori" for x in RELION_ANGLES_NAMES]], degrees=True)

        ang = rotation_error_with_sym(
            torch.as_tensor(rots.as_matrix(), dtype=torch.float32),
            torch.as_tensor(gtRot.as_matrix(), dtype=torch.float32),
            symmetry=symmetry,
        )
        ang_deg = torch.rad2deg(ang).numpy()
        mask_good = ang_deg < float(degs_error_thr)

        # MODE 3: Split original data into GOOD (error < threshold) and BAD (error >= threshold)
        parts_bg = parts_good.copy()
        parts_good = parts_good.loc[mask_good]

    # ==================================================================================
    # STEP 3: Apply optional filtering to background
    # ==================================================================================
    # Remove ignored particles (anti-join on primary keys)
    if fname_ignore:
        parts_ign = _read_particles_star(fname_ignore, root_dir, "IGNORE")
        key = PRIMARY_KEYS
        parts_bg = parts_bg.merge(parts_ign[key].drop_duplicates(), on=key, how="left", indicator=True)
        parts_bg = parts_bg.loc[parts_bg["_merge"] == "left_only"].drop(columns=["_merge"])

    # Optionally subsample background for speed (useful for large datasets)
    if fraction_to_sample_from_all is not None and 0.0 < fraction_to_sample_from_all < 1.0:
        parts_bg = parts_bg.sample(frac=fraction_to_sample_from_all, random_state=42)

    # ==================================================================================
    # STEP 4: Build final BAD dataset (mode-specific logic)
    # ==================================================================================
    if fname_all is not None:
        # MODE 1: BAD = ALL \ GOOD (set difference on primary keys)
        # This is the only mode that requires a merge to compute the set difference
        key = PRIMARY_KEYS
        merged = parts_bg.merge(parts_good[key].drop_duplicates(), on=key, how="left", indicator=True)
        parts_bad = merged.loc[merged["_merge"] == "left_only"].drop(columns=["_merge"])
        if len(parts_bad) == 0:
            raise ValueError("ALL and GOOD are identical on primary keys; no BAD could be derived.")
    elif fname_bad is not None:
        # MODE 2: BAD provided directly (already in parts_bg)
        parts_bad = parts_bg
    else:
        # MODE 3: BAD = complement of GOOD within the original file (error >= threshold)
        # Re-read original file to get all particles, then exclude GOOD subset
        key = PRIMARY_KEYS
        orig = _read_particles_star(fname_good, root_dir, "GOOD(for BAD complement)")
        parts_bad = orig.merge(parts_good[key].drop_duplicates(), on=key, how="left", indicator=True)
        parts_bad = parts_bad.loc[parts_bad["_merge"] == "left_only"].drop(columns=["_merge"])
        if len(parts_bad) == 0:
            raise ValueError("Could not infer BAD from angular error (all passed the threshold?).")

    # ==================================================================================
    # STEP 5: Extract scores and print diagnostics
    # ==================================================================================
    def _scores(df: pd.DataFrame, col: str) -> np.ndarray:
        """Extract score column as finite numpy array."""
        if col not in df.columns:
            raise KeyError(f"Missing score column '{col}'.")
        x = df[col].to_numpy()
        return x[np.isfinite(x)]

    score_good = _scores(parts_good, score_name)
    score_bad  = _scores(parts_bad,  score_name)

    # Print dataset statistics
    print(f"GOOD n={len(parts_good)}")
    print(parts_good[score_name].describe())
    print("BAD n={len(parts_bad)}")
    print(parts_bad[score_name].describe())
    print(" BG n={len(parts_bg)}")

    # Compute background quantile if applicable (useful diagnostic for MODE 1)
    if len(parts_bg) > 0 and len(parts_good) > 0:
        # Only compute background quantile if we have both datasets
        fraction = len(parts_good) / len(parts_bg)
        if 0 < fraction < 1:
            # Only compute quantile if fraction is valid (not 0 or >= 1)
            # This shows the score threshold that would select |good| particles from background
            q = np.quantile(parts_bg[score_name], 1 - fraction)
            print(f"Background quantile at |good|/|bg|={fraction:.3f}: {q:.6f}")

    # ==================================================================================
    # STEP 6: Plot histogram comparison (BAD vs GOOD)
    # ==================================================================================
    fig, ax1 = plt.subplots(figsize=(10, 6))
    # Use twin y-axes since GOOD and BAD may have very different counts
    ax1.hist(_clip_percentiles(score_bad, low_pct, up_pct), bins=bins, alpha=0.5, color="orange", label="bad")
    ax1.set_xlabel("Score")
    ax1.set_ylabel("# bad particles")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.hist(_clip_percentiles(score_good, low_pct, up_pct), bins=bins, alpha=0.5, color="blue", label="good")
    ax2.set_ylabel("# good particles")
    ax2.legend(loc="upper right")

    # Set x-axis limits to clipped data range
    xmin = min(_clip_percentiles(score_bad, low_pct, up_pct).min(), _clip_percentiles(score_good, low_pct, up_pct).min())
    xmax = max(_clip_percentiles(score_bad, low_pct, up_pct).max(), _clip_percentiles(score_good, low_pct, up_pct).max())
    plt.xlim(xmin, xmax)
    plt.title("Score distributions: BAD vs GOOD")

    if plot_fname:
        d = os.path.dirname(plot_fname)
        if d:
            os.makedirs(d, exist_ok=True)
        plt.savefig(plot_fname, bbox_inches="tight", dpi=150)
    if show_plots:
        plt.show()
    plt.close()

    # ==================================================================================
    # STEP 7: Optional GMM threshold estimation
    # ==================================================================================
    if compute_gmm and len(score_bad) > 10 and len(score_good) > 10:
        base_for_gmm = os.path.splitext(plot_fname)[0] if plot_fname else None
        thr, gmms, method = separate_gmm_threshold(
            score_bad, score_good, show_plots=show_plots, out_base=base_for_gmm,
            low_pct=low_pct, up_pct=up_pct, fallback_method=fallback_method, fn_cost=fn_cost
        )
        if thr is not None:
            print(f"GMM threshold: {thr:.6f} (method={method})")
        return thr, gmms, method

    return None


if __name__ == "__main__":
    from argParseFromDoc import parse_function_and_call
    parse_function_and_call(compare_prob_hists)
