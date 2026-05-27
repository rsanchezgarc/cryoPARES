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
- **Rematched star file output**: Optionally writes _good.star / _bad.star with optics blocks

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
       degs_error_thr=3.0  # particles with error < 3 deg are "good"
   )
   ```

Output star files (--out_star_prefix, MODE 1 only)
---------------------------------------------------
When --out_star_prefix is provided, the script writes:
  <prefix>_good.star  -- rows from ALL that matched GOOD (optics block preserved)
  <prefix>_bad.star   -- rows from ALL that did NOT match GOOD (optics block preserved)

Both files use rows sourced from the ALL file so column layout and optics metadata
are internally consistent.  If the ALL file has an "optics" block it is written
verbatim into both output files.

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
   - Compute d-prime separation: d' = |mu_good - mu_bad| / (sigma_good + sigma_bad)
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
python -m cryoPARES.scripts.gmm_hists \\
    --fname_good aligned.star \\
    --fname_all all_particles.star \\
    --plot_fname results/distributions.png \\
    --low_pct 1.0 \\
    --up_pct 99.0 \\
    --fn_cost 2.0 \\
    --out_star_prefix results/split
```

Output files:
- `results/distributions.png`: Main histogram plot (bad vs good overlay)
- `results/distributions_gmm.png`: GMM components and threshold visualization
- `results/split_good.star`: Good particles (rows from ALL, with optics block)
- `results/split_bad.star`: Bad particles (rows from ALL, with optics block)

See Also
--------
- `cryoPARES.inference.infer`: Uses directional z-scores for particle filtering
- `cryoPARES.datamanager.particlesDataset`: Dataset class that applies thresholds
"""
import json
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
_MICROGRAPH_COL_CANDIDATES = ["rlnMicrographName", "rlnImageName"]
_COORD_COL_CANDIDATES = [
    ("rlnCoordinateX", "rlnCoordinateY"),
    ("rlnMicrographCoordinatesX", "rlnMicrographCoordinatesY"),
]


# ---------------------------------------------------------------------------
# Micrograph path normalisation
# ---------------------------------------------------------------------------

def _micrograph_basename(series: pd.Series) -> pd.Series:
    """
    Normalise micrograph identifiers to a bare filename, handling:
      - full paths, relative paths, bare names
      - stack references:  000006@raw_data/.../Foo.mrcs  ->  Foo.mrcs
    """
    def _norm(val) -> str:
        s = str(val)
        if "@" in s:
            s = s.split("@", 1)[1]
        return os.path.basename(s)
    return series.map(_norm)


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
    if "rlnMicrographName" in df.columns:
        df["rlnMicrographName"] = df["rlnMicrographName"].map(_to_rel_or_base)
    return df


# ---------------------------------------------------------------------------
# I/O  (read and write, preserving optics)
# ---------------------------------------------------------------------------

def _read_particles_star(
    paths: List[str], root_dir: Optional[str], label: str
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Read one or more star files and return (particles_df, optics_df).

    optics_df is None when no "optics" block is present.  When multiple
    files are concatenated their optics tables are also concatenated and
    de-duplicated on rlnOpticsGroup (keeping the first occurrence).
    """
    parts_list = []
    optics_list = []

    for p in paths:
        assert os.path.isfile(p), f"{label}: file not found: {p}"
        raw = starfile.read(p)

        if isinstance(raw, dict):
            # RELION 3.1+ multi-block format
            if "particles" not in raw:
                raise KeyError(f"{label}: no 'particles' block in {p}. Keys: {list(raw.keys())}")
            tab = raw["particles"]
            if "optics" in raw:
                optics_list.append(raw["optics"])
        else:
            # Older single-block format
            tab = raw

        tab = _normalize_micrograph_paths(tab, root_dir)
        for k in PRIMARY_KEYS:
            if k not in tab.columns:
                raise KeyError(f"Missing column '{k}' in {label} ({p})")
        parts_list.append(tab)

    assert len(parts_list) > 0, f"Error reading paths {paths}"
    particles = pd.concat(parts_list, ignore_index=True)

    optics = None
    if optics_list:
        combined = pd.concat(optics_list, ignore_index=True)
        # De-duplicate on rlnOpticsGroup if present
        if "rlnOpticsGroup" in combined.columns:
            combined = combined.drop_duplicates(subset=["rlnOpticsGroup"], keep="first")
        optics = combined.reset_index(drop=True)

    return particles, optics


def _write_star(
    df: pd.DataFrame,
    optics: Optional[pd.DataFrame],
    path: str,
) -> None:
    """
    Write a particles dataframe as a RELION-style star file.
    If optics is not None, writes a two-block file {"optics": ..., "particles": ...}
    matching the RELION 3.1+ format.  Otherwise writes a single-block file.
    """
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

    if optics is not None:
        # Keep only optics groups that are actually referenced by this particle subset
        if "rlnOpticsGroup" in df.columns and "rlnOpticsGroup" in optics.columns:
            used_groups = df["rlnOpticsGroup"].unique()
            optics_out = optics[optics["rlnOpticsGroup"].isin(used_groups)].reset_index(drop=True)
        else:
            optics_out = optics

        starfile.write({"optics": optics_out, "particles": df}, path, overwrite=True)
    else:
        starfile.write({"particles": df}, path, overwrite=True)

    print(f"Wrote {len(df)} particles -> {path}")


# ---------------------------------------------------------------------------
# Coordinate-based matching helpers (ported from compare_star)
# ---------------------------------------------------------------------------

def _detect_coord_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    mg_col = next((c for c in _MICROGRAPH_COL_CANDIDATES if c in df.columns), None)
    if mg_col is None:
        raise ValueError(f"No micrograph column found. Tried: {_MICROGRAPH_COL_CANDIDATES}")
    coord_pair = next(
        ((x, y) for x, y in _COORD_COL_CANDIDATES if x in df.columns and y in df.columns),
        None,
    )
    if coord_pair is None:
        raise ValueError(f"No coordinate columns found. Tried: {_COORD_COL_CANDIDATES}")
    return mg_col, coord_pair[0], coord_pair[1]


def _prepare_coord_frame(
    df: pd.DataFrame,
    mg_col: str,
    x_col: str,
    y_col: str,
    scale: float,
    bin_size: float,
) -> pd.DataFrame:
    out = df[[mg_col, x_col, y_col]].copy()
    out = out.rename(columns={mg_col: "__mg", x_col: "__x_raw", y_col: "__y_raw"})
    out["__mg_key"] = _micrograph_basename(out["__mg"])
    out["__x"] = pd.to_numeric(out["__x_raw"], errors="coerce") * scale
    out["__y"] = pd.to_numeric(out["__y_raw"], errors="coerce") * scale
    if out["__x"].isna().any() or out["__y"].isna().any():
        raise ValueError("Non-numeric coordinates encountered.")
    out["__row_id"] = np.arange(len(out), dtype=np.int64)
    out["__bin_x"] = np.floor(out["__x"] / bin_size).astype(np.int64)
    out["__bin_y"] = np.floor(out["__y"] / bin_size).astype(np.int64)
    return out


def _coord_based_intersection_indices(
    df_all: pd.DataFrame,
    df_good: pd.DataFrame,
    margin: float = 5.0,
    scale_all: float = 1.0,
    scale_good: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (idx_in_all_that_are_good, idx_in_good_matched) using the
    binned spatial join from compare_star.  Both index arrays are integer
    row indices into the respective reset-index dataframes.
    """
    mg_all, x_all, y_all = _detect_coord_columns(df_all)
    mg_good, x_good, y_good = _detect_coord_columns(df_good)

    bin_size = max(1.0, float(margin))

    a = _prepare_coord_frame(df_all,  mg_all,  x_all,  y_all,  scale_all,  bin_size)
    b = _prepare_coord_frame(df_good, mg_good, x_good, y_good, scale_good, bin_size)

    common_mgs = set(a["__mg_key"].unique()) & set(b["__mg_key"].unique())
    if not common_mgs:
        raise ValueError(
            "Coordinate fallback: no shared micrograph basenames. "
            "Check that both files refer to the same dataset."
        )

    # Expand b into 3x3 neighbourhood of each bin
    b_parts = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            tmp = b.copy()
            tmp["__bin_x"] += dx
            tmp["__bin_y"] += dy
            b_parts.append(tmp)
    b_exp = pd.concat(b_parts, ignore_index=True)

    merged = a.merge(b_exp, on=["__mg_key", "__bin_x", "__bin_y"],
                     how="inner", suffixes=("_a", "_b"))

    if merged.empty:
        raise ValueError("Coordinate fallback: no candidate pairs after binning.")

    within = merged[
        ((merged["__x_a"] - merged["__x_b"]).abs() <= margin)
        & ((merged["__y_a"] - merged["__y_b"]).abs() <= margin)
    ].copy()

    if within.empty:
        raise ValueError(
            f"Coordinate fallback: no pairs within margin={margin} px."
        )

    # Closest match in good for each row in all
    within["__dist"] = np.hypot(
        within["__x_a"] - within["__x_b"],
        within["__y_a"] - within["__y_b"],
    )
    best = (
        within.sort_values("__dist")
        .drop_duplicates(subset="__row_id_a", keep="first")
    )
    return best["__row_id_a"].to_numpy(), best["__row_id_b"].to_numpy()


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _diagnose_merge_failure(
    parts_all: pd.DataFrame,
    parts_good: pd.DataFrame,
    n_sample: int = 5,
) -> None:
    """
    Print diagnostic information to understand why primary-key merge failed.
    Checks micrograph name overlap and coordinate ranges.
    """
    mg_col_all  = "rlnMicrographName" if "rlnMicrographName" in parts_all.columns  else None
    mg_col_good = "rlnMicrographName" if "rlnMicrographName" in parts_good.columns else None

    print("\n--- Merge failure diagnostics ---")

    if mg_col_all and mg_col_good:
        raw_all  = parts_all[mg_col_all].unique()
        raw_good = parts_good[mg_col_good].unique()
        base_all  = set(_micrograph_basename(pd.Series(raw_all)))
        base_good = set(_micrograph_basename(pd.Series(raw_good)))
        common = base_all & base_good

        print(f"  Unique micrograph paths in ALL : {len(raw_all)}")
        print(f"  Unique micrograph paths in GOOD: {len(raw_good)}")
        print(f"  Shared basenames               : {len(common)} / {min(len(base_all), len(base_good))}")
        print(f"  Sample ALL  raw paths : {list(raw_all[:n_sample])}")
        print(f"  Sample GOOD raw paths : {list(raw_good[:n_sample])}")
        print(f"  Sample ALL  basenames : {list(list(base_all)[:n_sample])}")
        print(f"  Sample GOOD basenames : {list(list(base_good)[:n_sample])}")

        if len(common) == 0:
            print("  *** NO shared micrograph basenames — path normalisation is insufficient ***")
            print("--- end diagnostics ---\n")
            return

        # Pick a micrograph present in both and compare coordinate ranges
        example_mg = list(common)[0]
        base_all_series  = _micrograph_basename(parts_all[mg_col_all])
        base_good_series = _micrograph_basename(parts_good[mg_col_good])

        rows_all  = parts_all.loc[base_all_series  == example_mg]
        rows_good = parts_good.loc[base_good_series == example_mg]

        for tag, df in (("ALL", rows_all), ("GOOD", rows_good)):
            try:
                _, x_col, y_col = _detect_coord_columns(df)
                xs = pd.to_numeric(df[x_col], errors="coerce").dropna()
                ys = pd.to_numeric(df[y_col], errors="coerce").dropna()
                print(f"  {tag} coords for '{example_mg}': "
                      f"X=[{xs.min():.1f}, {xs.max():.1f}]  "
                      f"Y=[{ys.min():.1f}, {ys.max():.1f}]  "
                      f"n={len(xs)}")
            except Exception as e:
                print(f"  {tag} coord detection failed: {e}")

    print("--- end diagnostics ---\n")


# ---------------------------------------------------------------------------
# Set-difference with exact merge + coordinate fallback
# ---------------------------------------------------------------------------

def _set_difference_all_minus_good(
    parts_all: pd.DataFrame,
    parts_good: pd.DataFrame,
    coord_margin: float = 5.0,
    scale_all: float = 1.0,
    scale_good: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute ALL \\ GOOD with an exact primary-key merge, then a coordinate fallback.

    Returns (parts_bad, parts_good_rematched).

    parts_good_rematched contains the rows of ALL that matched GOOD — it uses
    data from ALL so that column layout and optics metadata are internally
    consistent with parts_bad.

    The coordinate fallback auto-expands the margin up to 100 px and warns
    if fewer than 80% of GOOD particles were matched.
    """
    key = PRIMARY_KEYS

    # --- attempt 1: exact merge on normalised primary keys ---
    indicator = parts_all.merge(
        parts_good[key].drop_duplicates(), on=key, how="left", indicator=True
    )
    n_matched_exact = (indicator["_merge"] == "both").sum()

    if n_matched_exact > 0:
        good_mask = (indicator["_merge"] == "both").to_numpy()
        parts_bad_out  = parts_all.loc[~good_mask].reset_index(drop=True)
        parts_good_out = parts_all.loc[ good_mask].reset_index(drop=True)
        if len(parts_bad_out) == 0:
            raise ValueError("ALL and GOOD are identical on primary keys; no BAD could be derived.")
        print(f"Exact merge: matched {n_matched_exact} good particles, {len(parts_bad_out)} BAD remain.")
        return parts_bad_out, parts_good_out

    # --- attempt 2: coordinate proximity with auto-expanding margin ---
    warnings.warn(
        "Exact primary-key merge found 0 matches between ALL and GOOD "
        "(likely full-path vs basename mismatch in rlnMicrographName). "
        "Falling back to coordinate-based matching."
    )
    _diagnose_merge_failure(parts_all, parts_good)

    parts_all_r  = parts_all.reset_index(drop=True)
    parts_good_r = parts_good.reset_index(drop=True)

    expected_min = int(0.80 * len(parts_good_r))
    margins_to_try = sorted(set([coord_margin, 10.0, 25.0, 50.0, 100.0]))
    last_exc = None

    for margin in margins_to_try:
        try:
            idx_good_in_all, _ = _coord_based_intersection_indices(
                parts_all_r, parts_good_r,
                margin=margin,
                scale_all=scale_all,
                scale_good=scale_good,
            )
        except ValueError as e:
            last_exc = e
            warnings.warn(f"Coordinate fallback with margin={margin} px failed: {e}")
            continue

        n_matched = len(idx_good_in_all)
        print(f"  Coordinate fallback margin={margin} px: matched {n_matched} / {len(parts_good_r)} good particles.")

        if n_matched >= expected_min:
            warnings.warn(
                f"Coordinate fallback matched {n_matched} good particles "
                f"(margin={margin} px, scale_all={scale_all}, scale_good={scale_good})."
            )
            mask_good = np.zeros(len(parts_all_r), dtype=bool)
            mask_good[idx_good_in_all] = True
            parts_bad_out  = parts_all_r.loc[~mask_good].reset_index(drop=True)
            parts_good_out = parts_all_r.loc[ mask_good].reset_index(drop=True)
            if len(parts_bad_out) == 0:
                raise ValueError("Coordinate fallback matched all particles as GOOD; no BAD remains.")
            return parts_bad_out, parts_good_out

        warnings.warn(
            f"Coordinate fallback with margin={margin} px matched only {n_matched} / {len(parts_good_r)} "
            f"good particles (< 80% threshold). This suggests a coordinate scale mismatch. "
            f"Consider passing --scale_all or --scale_good."
        )

    raise ValueError(
        f"Coordinate fallback could not match >=80% of GOOD particles into ALL, "
        f"even with margin up to {margins_to_try[-1]} px. "
        f"Last error: {last_exc}\n"
        f"Hints:\n"
        f"  - If files use different pixel sizes, pass --scale_all or --scale_good\n"
        f"  - If files use Angstrom coordinates, the margin needs to be much larger\n"
        f"  - Run with --coord_margin 200 as a first test\n"
        f"  - Check diagnostic output above for coordinate ranges"
    )


# ---------------------------------------------------------------------------
# GMM utilities
# ---------------------------------------------------------------------------

def _clip_percentiles(x: np.ndarray, low_pct: float = 1, up_pct: float = 99) -> np.ndarray:
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
    """
    data = data.reshape(-1, 1) if data.ndim == 1 else data

    gmm_init = GaussianMixture(n_components=1, random_state=random_state)
    gmm_init.fit(data)
    log_prob = gmm_init.score_samples(data)
    threshold_prob = np.percentile(log_prob, 5)
    mask_keep = log_prob >= threshold_prob
    data_cleaned = data[mask_keep]

    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(data_cleaned)
    return gmm


def _select_best_gmm(
    data: np.ndarray,
    n_components_options: List[int] = [2, 3, 4],
    low_pct: float = 1,
    up_pct: float = 99,
    min_component_weight: float = 0.1,
) -> Tuple[GaussianMixture, int]:
    """
    Try fitting GMM with different numbers of components, select best by BIC.
    """
    data_clipped = _clip_percentiles(data, low_pct, up_pct)
    fraction_kept = len(data_clipped) / max(1, len(data))

    if fraction_kept < 0.9:
        warnings.warn(
            f"Long tails detected ({(1-fraction_kept)*100:.1f}% of data clipped). "
            "Limiting to 2-component GMM to avoid overfitting tails."
        )
        n_components_options = [2]

    data = data.reshape(-1, 1) if data.ndim == 1 else data

    try:
        gmm_2 = _fit_gmm_robust(data, n_components=2)
        weights_2 = gmm_2.weights_.flatten()
        min_weight_2 = weights_2.min()

        if min_weight_2 < min_component_weight and len(data) >= 40:
            warnings.warn(
                f"2-component GMM has small component (w={min_weight_2:.3f} < {min_component_weight}). "
                "Trying 3-4 components to model outliers separately."
            )
            n_components_to_try = [2, 3, 4]
        else:
            n_components_to_try = [2]

    except Exception as e:
        warnings.warn(f"Failed to fit 2-component GMM: {e}")
        n_components_to_try = [2, 3, 4]

    best_gmm = None
    best_bic = np.inf
    best_n = None

    for n in n_components_to_try:
        if len(data) < n * 10:
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
        warnings.warn(
            f"Selected {best_n}-component GMM (BIC={best_bic:.1f}). "
            "Data has outlier structure requiring extra components."
        )

    return best_gmm, best_n


def _select_components_robust(
    gmm_bad: GaussianMixture, gmm_good: GaussianMixture
) -> Tuple[int, int, str]:
    """
    Robustly select which component from each GMM to use for threshold.
    """
    m_b = gmm_bad.means_.flatten()
    s_b = np.sqrt(gmm_bad.covariances_.flatten())
    w_b = gmm_bad.weights_.flatten()

    m_g = gmm_good.means_.flatten()
    s_g = np.sqrt(gmm_good.covariances_.flatten())
    w_g = gmm_good.weights_.flatten()

    ob, og = np.argsort(m_b), np.argsort(m_g)
    m_b, s_b, w_b = m_b[ob], s_b[ob], w_b[ob]
    m_g, s_g, w_g = m_g[og], s_g[og], w_g[og]

    mean_bad_weighted  = np.sum(m_b * w_b)
    mean_good_weighted = np.sum(m_g * w_g)

    if mean_good_weighted < mean_bad_weighted:
        warnings.warn(
            f"Good distribution has LOWER weighted mean ({mean_good_weighted:.2f}) "
            f"than bad ({mean_bad_weighted:.2f})! Data may be mislabeled or scores inverted."
        )

    # Select good component
    n_comp_g = len(m_g)
    if n_comp_g == 2 and w_g[0] > 0.3 and w_g[1] > 0.3:
        idx_good = 1
        method_good = f"higher_mean(w0={w_g[0]:.2f},w1={w_g[1]:.2f})"
    else:
        idx_good = int(np.argmax(w_g))
        method_good = f"dominant(w={w_g[idx_good]:.2f})"

    # Select bad component
    min_weight_threshold = 0.1
    n_comp_b = len(m_b)

    if n_comp_b == 2:
        valid = w_b >= min_weight_threshold
        if valid.sum() == 2 and w_b[0] > 0.3 and w_b[1] > 0.3:
            idx_bad = 0
            method_bad = f"lower_mean(w0={w_b[0]:.2f},w1={w_b[1]:.2f})"
        elif valid.sum() >= 1:
            idx_bad = int(np.where(valid)[0][np.argmax(w_b[valid])])
            method_bad = f"dominant(w={w_b[idx_bad]:.2f})"
        else:
            idx_bad = int(np.argmax(w_b))
            method_bad = f"dominant_despite_small(w={w_b[idx_bad]:.2f})"
    else:
        valid = w_b >= min_weight_threshold
        if valid.sum() > 0:
            valid_indices = np.where(valid)[0]
            idx_bad = int(valid_indices[np.argmax(w_b[valid])])
            method_bad = f"dominant_{n_comp_b}comp(w={w_b[idx_bad]:.2f})"
        else:
            idx_bad = int(np.argmax(w_b))
            method_bad = f"dominant_{n_comp_b}comp_all_small(w={w_b[idx_bad]:.2f})"

    if m_b[idx_bad] > m_g[idx_good]:
        warnings.warn(
            f"Selected bad component (mu={m_b[idx_bad]:.2f}) has HIGHER mean than "
            f"good component (mu={m_g[idx_good]:.2f})! Threshold may be unreliable."
        )

    separation = abs(m_g[idx_good] - m_b[idx_bad]) / (s_g[idx_good] + s_b[idx_bad] + 1e-9)

    if separation < 1.0:
        warnings.warn(
            f"Poor d-prime separation between selected components (d'={separation:.2f}). "
            "Threshold may be unreliable. Consider manual inspection."
        )

    idx_bad_orig  = int(ob[idx_bad])
    idx_good_orig = int(og[idx_good])
    method_desc = f"good:{method_good}|bad:{method_bad}|sep={separation:.2f}"

    return idx_bad_orig, idx_good_orig, method_desc


def _fallback_roc_threshold(
    score_bad: np.ndarray,
    score_good: np.ndarray,
    fn_cost: float = 2.0,
) -> Tuple[float, float, str]:
    """
    Fallback: find optimal threshold using weighted Youden's J on an ROC curve.
    """
    y_true = np.concatenate([
        np.zeros(len(score_bad)),
        np.ones(len(score_good)),
    ])
    y_scores = np.concatenate([score_bad, score_good])

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    J_weighted = tpr - (fpr / fn_cost)
    idx_best = np.argmax(J_weighted)

    threshold = thresholds[idx_best]
    quality   = J_weighted[idx_best]
    method    = f"ROC_Youden_weighted(fn_cost={fn_cost:.1f},J={quality:.3f})"

    return float(threshold), float(quality), method


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_gmms(
    dist_bad,
    dist_good,
    threshold,
    gmms,
    gmm_labels,
    method,
    show_plots: bool,
    out_base: Optional[str],
) -> None:
    plt.figure(figsize=(15, 6))

    n_bad,  _, _ = plt.hist(dist_bad,  bins=50, alpha=0.5, density=True, label="Bad",  color="orange")
    n_good, _, _ = plt.hist(dist_good, bins=50, alpha=0.5, density=True, label="Good", color="blue")
    hist_max_density = max(n_bad.max(), n_good.max())

    all_means = np.concatenate([g.means_.flatten()                      for g in gmms])
    all_stds  = np.concatenate([np.sqrt(g.covariances_.flatten())       for g in gmms])

    x_min = np.min(all_means - 3 * all_stds)
    x_max = np.max(all_means + 3 * all_stds)
    x_range = x_max - x_min
    x_min -= 0.1 * x_range
    x_max += 0.1 * x_range
    x_min = min(x_min, threshold - 0.5 * x_range)
    x_max = max(x_max, threshold + 0.5 * x_range)

    X = np.linspace(x_min, x_max, 1000)

    styles = ["--r", "--b"]
    for i, (gmm, label) in enumerate(zip(gmms, gmm_labels)):
        means   = gmm.means_.flatten()
        stds    = np.sqrt(gmm.covariances_.flatten())
        weights = gmm.weights_.flatten()
        order   = np.argsort(means)
        means, stds, weights = means[order], stds[order], weights[order]
        for j, (m, s, w) in enumerate(zip(means, stds, weights)):
            plt.plot(
                X, w * stats.norm.pdf(X, m, s), styles[i], alpha=0.7,
                label=f"{label} Comp {j+1} (mu={m:.2f}, sigma={s:.2f}, w={w:.2f})",
            )

    plt.axvline(threshold, color="g",
                label=f"Threshold {threshold:.2f} ({method.split('(')[0]})")
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.title("GMM fitting on score distributions")
    plt.xlim(x_min, x_max)
    plt.ylim(0, hist_max_density * 1.5)
    plt.legend()
    plt.grid(True)

    if out_base:
        plt.savefig(f"{out_base}_gmm.png", bbox_inches="tight", dpi=150)
        with open(f"{out_base}_directional_zscore_threshold.json", "w") as f:
            json.dump({"threshold": threshold, "method": method}, f)
    if show_plots:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Main threshold estimation
# ---------------------------------------------------------------------------

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

    @param dist_bad: 1D array of scores for BAD particles
    @param dist_good: 1D array of scores for GOOD particles
    @param show_plots: If True, display GMM plot interactively
    @param out_base: If provided, save GMM plot to "<out_base>_gmm.png"
    @param low_pct: Lower percentile for clipping outliers (default: 1)
    @param up_pct: Upper percentile for clipping outliers (default: 99)
    @param fallback_method: "auto" (ROC), "manual" (return None), "none" (raise)
    @param fn_cost: False negative cost multiplier (default: 2.0)
    @param n_components_options: GMM component counts to try (default: [2,3,4])

    @returns: (threshold, (gmm_bad, gmm_good), method_description)
    """
    db = np.asarray(dist_bad,  dtype=float)
    dg = np.asarray(dist_good, dtype=float)
    db = db[np.isfinite(db)]
    dg = dg[np.isfinite(dg)]

    if len(db) < 20 or len(dg) < 20:
        raise ValueError(
            f"Not enough points for GMM thresholding (bad={len(db)}, good={len(dg)}). "
            "Need at least 20 each."
        )

    db_plot = _clip_percentiles(db, low_pct, up_pct)
    dg_plot = _clip_percentiles(dg, low_pct, up_pct)
    db_fit  = _clip_percentiles(db, low_pct, up_pct)
    dg_fit  = _clip_percentiles(dg, low_pct, up_pct)

    try:
        print(f"Fitting GMM to bad  distribution ({len(db_fit)} samples after clipping)...")
        gmm_b, n_comp_b = _select_best_gmm(db_fit, n_components_options, low_pct, up_pct)

        print(f"Fitting GMM to good distribution ({len(dg_fit)} samples after clipping)...")
        gmm_g, n_comp_g = _select_best_gmm(dg_fit, n_components_options, low_pct, up_pct)

        print(f"Selected {n_comp_b}-component GMM for bad, {n_comp_g}-component GMM for good")

        idx_bad, idx_good, method_desc = _select_components_robust(gmm_b, gmm_g)

        m_b = gmm_b.means_.flatten()[idx_bad]
        s_b = np.sqrt(gmm_b.covariances_.flatten()[idx_bad])
        w_b = gmm_b.weights_.flatten()[idx_bad]

        m_g = gmm_g.means_.flatten()[idx_good]
        s_g = np.sqrt(gmm_g.covariances_.flatten()[idx_good])
        w_g = gmm_g.weights_.flatten()[idx_good]

        x_int = _find_intersection(m_b, s_b, w_b, m_g, s_g, w_g)

        if x_int is not None:
            thr    = x_int
            method = f"GMM_intersection({method_desc})"
        else:
            thr    = (m_b + m_g) / 2.0
            method = f"GMM_midpoint({method_desc})"
            warnings.warn("GMM intersection failed, using midpoint of selected components")

        if show_plots or out_base:
            _plot_gmms(db_plot, dg_plot, thr, (gmm_b, gmm_g), ("Bad", "Good"),
                       method, show_plots, out_base)

        return float(thr), (gmm_b, gmm_g), method

    except Exception as e:
        warnings.warn(f"GMM threshold estimation failed: {e}")

        if fallback_method == "auto":
            warnings.warn("Using fallback: ROC-based threshold")
            thr, quality, method = _fallback_roc_threshold(db, dg, fn_cost=fn_cost)
            print(f"Fallback threshold: {thr:.4f} (quality={quality:.3f})")

            if show_plots or out_base:
                plt.figure(figsize=(15, 6))
                plt.hist(db_plot, bins=50, alpha=0.5, density=True, label="Bad",  color="orange")
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
            warnings.warn(
                "GMM failed and fallback_method='manual'. "
                "Returning None - manual inspection required."
            )
            return None, (None, None), "manual_required"

        else:  # fallback_method == "none"
            raise


# ---------------------------------------------------------------------------
# Top-level comparison function
# ---------------------------------------------------------------------------

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
    coord_margin: float = 5.0,
    scale_all: float = 1.0,
    scale_good: float = 1.0,
    out_star_prefix: Optional[str] = None,
):
    """
    Compare score distributions for GOOD vs BAD particles and (optionally) compute a GMM threshold.

    Modes:
      1) GOOD + ALL  -> BAD = ALL \\ GOOD (set difference on primary keys, with coordinate fallback).
      2) GOOD + BAD  -> BAD is provided directly (no set difference).
      3) GOOD only + symmetry -> infer GOOD by angular error < degs_error_thr; BAD = complement.

    @param fname_good: list of .star files with GOOD particles
    @param fname_all: list of .star files with ALL particles (superset of GOOD).
        Mutually exclusive with fname_bad.
    @param fname_bad: list of .star files with BAD particles (disjoint from GOOD).
        Mutually exclusive with fname_all.
    @param fname_ignore: list of .star files to be removed from the background set
        (ALL or BAD) via primary keys
    @param score_name: name of the score column to analyze (default: DIRECTIONAL_ZSCORE_NAME)
    @param plot_fname: path to save the main histogram figure; if compute_gmm=True, also saves
        "<plot_fname>_gmm.png"
    @param show_plots: if True, show matplotlib figures interactively
    @param symmetry: required if neither fname_all nor fname_bad is provided
        (for angle-based inference)
    @param degs_error_thr: angular error threshold (degrees) to define GOOD when inferring
        by symmetry
    @param bins: number of histogram bins
    @param fraction_to_sample_from_all: fraction to subsample from the background set for
        speed (0 < f < 1)
    @param compute_gmm: if True, fit separate GMMs and compute a threshold
    @param root_dir: normalize rlnMicrographName as relative to this directory;
        if None, use basename
    @param low_pct: lower percentile for clipping outliers in histograms and GMM fitting
    @param up_pct: upper percentile for clipping outliers in histograms and GMM fitting
    @param fallback_method: fallback if GMM fails -
        "auto" (ROC-based), "manual" (return None), "none" (raise error)
    @param fn_cost: cost multiplier for false negatives vs false positives
        (>1 = more conservative, keeps more good particles)
    @param coord_margin: coordinate proximity margin in pixels for the set-difference fallback
        (MODE 1). Auto-expanded up to 100 px if needed.
    @param scale_all: multiply ALL coordinates by this factor before proximity matching
    @param scale_good: multiply GOOD coordinates by this factor before proximity matching
    @param out_star_prefix: if provided, write rematched particles to
        '<out_star_prefix>_good.star' and '<out_star_prefix>_bad.star' (MODE 1 only).
        Both output files preserve the optics block from the ALL file.
        The GOOD file contains the rows from ALL matched to GOOD; the BAD file contains
        the complement.  If the prefix path contains directories they are created
        automatically.
    @returns: If compute_gmm and enough samples, returns (threshold, (gmm_bad, gmm_good), method);
        otherwise None
    """
    if fname_all is not None and fname_bad is not None:
        raise ValueError("Provide only one of fname_all or fname_bad.")

    # =========================================================================
    # STEP 1: Read GOOD particles
    # =========================================================================
    parts_good, optics_good = _read_particles_star(fname_good, root_dir, "GOOD")
    if score_name not in parts_good.columns:
        raise KeyError(f"GOOD missing score column '{score_name}'")

    # =========================================================================
    # STEP 2: Build background dataset
    # =========================================================================
    optics_bg = None

    if fname_all is not None:
        parts_bg, optics_bg = _read_particles_star(fname_all, root_dir, "ALL")
    elif fname_bad is not None:
        parts_bg, optics_bg = _read_particles_star(fname_bad, root_dir, "BAD")
    else:
        # MODE 3: infer GOOD/BAD from angular error
        if symmetry is None:
            raise ValueError(
                "symmetry is required when neither fname_all nor fname_bad is provided."
            )
        from scipy.spatial.transform import Rotation as R
        for c in RELION_ANGLES_NAMES + [x + "_ori" for x in RELION_ANGLES_NAMES]:
            if c not in parts_good.columns:
                raise KeyError(f"Missing column '{c}' required for angle-based inference.")

        rots  = R.from_euler(RELION_EULER_CONVENTION,
                             parts_good[RELION_ANGLES_NAMES], degrees=True)
        gtRot = R.from_euler(RELION_EULER_CONVENTION,
                             parts_good[[x + "_ori" for x in RELION_ANGLES_NAMES]],
                             degrees=True)

        ang = rotation_error_with_sym(
            torch.as_tensor(rots.as_matrix(),  dtype=torch.float32),
            torch.as_tensor(gtRot.as_matrix(), dtype=torch.float32),
            symmetry=symmetry,
        )
        ang_deg = torch.rad2deg(ang).numpy()
        mask_good = ang_deg < float(degs_error_thr)

        parts_bg   = parts_good.copy()
        optics_bg  = optics_good
        parts_good = parts_good.loc[mask_good]

    # =========================================================================
    # STEP 3: Apply optional ignore filtering to background
    # =========================================================================
    if fname_ignore:
        parts_ign, _ = _read_particles_star(fname_ignore, root_dir, "IGNORE")
        parts_bg = parts_bg.merge(
            parts_ign[PRIMARY_KEYS].drop_duplicates(),
            on=PRIMARY_KEYS, how="left", indicator=True,
        )
        parts_bg = parts_bg.loc[parts_bg["_merge"] == "left_only"].drop(columns=["_merge"])

    if fraction_to_sample_from_all is not None and 0.0 < fraction_to_sample_from_all < 1.0:
        parts_bg = parts_bg.sample(frac=fraction_to_sample_from_all, random_state=42)

    # =========================================================================
    # STEP 4: Build final BAD dataset (and rematched GOOD where applicable)
    # =========================================================================
    parts_good_rematched = None  # only populated in MODE 1

    if fname_all is not None:
        # MODE 1: exact merge + coordinate fallback
        parts_bad, parts_good_rematched = _set_difference_all_minus_good(
            parts_bg, parts_good,
            coord_margin=coord_margin,
            scale_all=scale_all,
            scale_good=scale_good,
        )
        # Use the rematched version so scores come from the ALL file
        # (consistent column layout with parts_bad)
        parts_good = parts_good_rematched

    elif fname_bad is not None:
        # MODE 2: BAD provided directly
        parts_bad = parts_bg

    else:
        # MODE 3: BAD = complement within the original file
        orig, _ = _read_particles_star(fname_good, root_dir, "GOOD(for BAD complement)")
        parts_bad = orig.merge(
            parts_good[PRIMARY_KEYS].drop_duplicates(),
            on=PRIMARY_KEYS, how="left", indicator=True,
        )
        parts_bad = parts_bad.loc[parts_bad["_merge"] == "left_only"].drop(columns=["_merge"])
        if len(parts_bad) == 0:
            raise ValueError(
                "Could not infer BAD from angular error (all passed the threshold?)."
            )

    # =========================================================================
    # STEP 5: Optionally write rematched star files (MODE 1 only)
    # =========================================================================
    if out_star_prefix is not None:
        if parts_good_rematched is not None:
            _write_star(parts_good_rematched, optics_bg, f"{out_star_prefix}_good.star")
            _write_star(parts_bad,            optics_bg, f"{out_star_prefix}_bad.star")
        else:
            warnings.warn(
                "--out_star_prefix is only supported in MODE 1 (fname_all provided). "
                "No star files written."
            )

    # =========================================================================
    # STEP 6: Extract scores and print diagnostics
    # =========================================================================
    def _scores(df: pd.DataFrame, col: str) -> np.ndarray:
        if col not in df.columns:
            raise KeyError(f"Missing score column '{col}'.")
        x = df[col].to_numpy()
        return x[np.isfinite(x)]

    score_good = _scores(parts_good, score_name)
    score_bad  = _scores(parts_bad,  score_name)

    print(f"GOOD n={len(parts_good)}")
    print(parts_good[score_name].describe())
    print(f"BAD n={len(parts_bad)}")
    print(parts_bad[score_name].describe())
    print(f" BG n={len(parts_bg)}")

    if len(parts_bg) > 0 and len(parts_good) > 0:
        fraction = len(parts_good) / len(parts_bg)
        if 0 < fraction < 1:
            q = np.quantile(parts_bg[score_name], 1 - fraction)
            print(f"Background quantile at |good|/|bg|={fraction:.3f}: {q:.6f}")

    # =========================================================================
    # STEP 7: Plot histogram comparison
    # =========================================================================
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.hist(
        _clip_percentiles(score_bad,  low_pct, up_pct),
        bins=bins, alpha=0.5, color="orange", label="bad",
    )
    ax1.set_xlabel("Score")
    ax1.set_ylabel("# bad particles")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.hist(
        _clip_percentiles(score_good, low_pct, up_pct),
        bins=bins, alpha=0.5, color="blue", label="good",
    )
    ax2.set_ylabel("# good particles")
    ax2.legend(loc="upper right")

    xmin = min(
        _clip_percentiles(score_bad,  low_pct, up_pct).min(),
        _clip_percentiles(score_good, low_pct, up_pct).min(),
    )
    xmax = max(
        _clip_percentiles(score_bad,  low_pct, up_pct).max(),
        _clip_percentiles(score_good, low_pct, up_pct).max(),
    )
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

    # =========================================================================
    # STEP 8: Optional GMM threshold estimation
    # =========================================================================
    if compute_gmm and len(score_bad) > 10 and len(score_good) > 10:
        base_for_gmm = os.path.splitext(plot_fname)[0] if plot_fname else None
        thr, gmms, method = separate_gmm_threshold(
            score_bad, score_good,
            show_plots=show_plots,
            out_base=base_for_gmm,
            low_pct=low_pct,
            up_pct=up_pct,
            fallback_method=fallback_method,
            fn_cost=fn_cost,
        )
        if thr is not None:
            print(
                f"\n--------------------------------------------\n"
                f"GMM threshold: {thr:.5f} (method={method}).\n"
                f"You can use {thr:.5f} as the --directional_zscore_thr when running "
                f"inference to perform automatic particle pruning."
            )
        return thr, gmms, method

    return None


if __name__ == "__main__":
    from argParseFromDoc import parse_function_and_call
    parse_function_and_call(compare_prob_hists)
