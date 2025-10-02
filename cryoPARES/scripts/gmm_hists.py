import os
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import starfile
import torch

from scipy import stats, optimize
from sklearn.mixture import GaussianMixture

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
    return pd.concat(out, ignore_index=True)


def _clip_percentiles(x: np.ndarray, low_pct: float = 0.1) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo = np.percentile(x, low_pct)
    hi = np.percentile(x, 100 - low_pct)
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


def _plot_gmms(dist_bad, dist_good, threshold, gmms, method, show_plots, out_base: Optional[str]):
    plt.figure(figsize=(15, 6))
    plt.hist(dist_bad, bins=50, alpha=0.5, density=True, label="Bad")
    plt.hist(dist_good, bins=50, alpha=0.5, density=True, label="Good")

    X = np.linspace(min(dist_bad.min(), dist_good.min()),
                    max(dist_bad.max(), dist_good.max()), 1000)

    styles = ["--r", "--b"]
    for i, gmm in enumerate(gmms):
        means = gmm.means_.flatten()
        stds = np.sqrt(gmm.covariances_.flatten())
        weights = gmm.weights_.flatten()
        order = np.argsort(means)
        means, stds, weights = means[order], stds[order], weights[order]
        for j, (m, s, w) in enumerate(zip(means, stds, weights)):
            plt.plot(X, w * stats.norm.pdf(X, m, s), styles[i], alpha=0.7,
                     label=f"Dist {i+1} Comp {j+1} (μ={m:.2f}, σ={s:.2f}, w={w:.2f})")

    plt.axvline(threshold, color="g", label=f"Threshold {threshold:.2f} ({method})")
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.title("GMM fitting on score distributions")
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
) -> Tuple[float, Tuple[GaussianMixture, GaussianMixture], str]:
    """
    @param dist_bad: 1D array of scores for BAD particles
    @param dist_good: 1D array of scores for GOOD particles
    @param show_plots: If True, show GMM plot
    @param out_base: If provided, save GMM plot to "<out_base>_gmm.png"
    @returns: (threshold, (gmm_small, gmm_large), method)
    """
    db = np.asarray(dist_bad, dtype=float)
    dg = np.asarray(dist_good, dtype=float)
    db = db[np.isfinite(db)]
    dg = dg[np.isfinite(dg)]
    if len(db) < 5 or len(dg) < 5:
        raise ValueError("Not enough points for GMM thresholding.")

    db_plot = _clip_percentiles(db)
    dg_plot = _clip_percentiles(dg)

    db_fit = np.clip(db, a_min=None, a_max=np.percentile(dg, 99))[:, None]
    dg_fit = np.clip(dg, a_min=np.percentile(db, 1), a_max=None)[:, None]

    gmm_b = GaussianMixture(n_components=2, random_state=42).fit(db_fit)
    gmm_g = GaussianMixture(n_components=2, random_state=42).fit(dg_fit)

    m_b = gmm_b.means_.flatten(); s_b = np.sqrt(gmm_b.covariances_.flatten()); w_b = gmm_b.weights_.flatten()
    m_g = gmm_g.means_.flatten(); s_g = np.sqrt(gmm_g.covariances_.flatten()); w_g = gmm_g.weights_.flatten()
    ob, og = np.argsort(m_b), np.argsort(m_g)
    m_b, s_b, w_b = m_b[ob], s_b[ob], w_b[ob]
    m_g, s_g, w_g = m_g[og], s_g[og], w_g[og]

    larger_mean = max(m_b[np.argmax(w_b)], m_g[np.argmax(w_g)])
    if larger_mean == m_b[np.argmax(w_b)]:
        smaller_means, smaller_stds, smaller_w = m_g, s_g, w_g
        larger_means,  larger_stds,  larger_w  = m_b, s_b, w_b
        gmms = (gmm_g, gmm_b)
    else:
        smaller_means, smaller_stds, smaller_w = m_b, s_b, w_b
        larger_means,  larger_stds,  larger_w  = m_g, s_g, w_g
        gmms = (gmm_b, gmm_g)

    x_int = _find_intersection(smaller_means[0], smaller_stds[0], smaller_w[0],
                               larger_means[1],  larger_stds[1],  larger_w[1])
    if x_int is not None:
        thr, method = x_int, "intersection"
    else:
        thr, method = (smaller_means[0] + larger_means[1]) / 2.0, "midpoint"

    if show_plots or out_base:
        _plot_gmms(db_plot, dg_plot, thr, gmms, method, show_plots, out_base)

    return float(thr), gmms, method


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
    @returns: If compute_gmm and enough samples, returns (threshold, (gmm_small, gmm_large), method); otherwise None
    """

    if fname_all is not None and fname_bad is not None:
        raise ValueError("Provide only one of fname_all or fname_bad.")

    # Read GOOD
    parts_good = _read_particles_star(fname_good, root_dir, "GOOD")
    if score_name not in parts_good.columns:
        raise KeyError(f"GOOD missing score column '{score_name}'")

    # Build background (ALL or BAD or inferred via symmetry)
    if fname_all is not None:
        parts_bg = _read_particles_star(fname_all, root_dir, "ALL")
    elif fname_bad is not None:
        parts_bg = _read_particles_star(fname_bad, root_dir, "BAD")
    else:
        if symmetry is None:
            raise ValueError("symmetry is required when neither fname_all nor fname_bad is provided.")
        from scipy.spatial.transform import Rotation as R
        for c in RELION_ANGLES_NAMES + [x + "_ori" for x in RELION_ANGLES_NAMES]:
            if c not in parts_good.columns:
                raise KeyError(f"Missing column '{c}' required for angle-based inference.")

        rots  = R.from_euler(RELION_EULER_CONVENTION, parts_good[RELION_ANGLES_NAMES], degrees=True)
        gtRot = R.from_euler(RELION_EULER_CONVENTION, parts_good[[x + "_ori" for x in RELION_ANGLES_NAMES]], degrees=True)

        ang = rotation_error_with_sym(
            torch.as_tensor(rots.as_matrix(), dtype=torch.float32),
            torch.as_tensor(gtRot.as_matrix(), dtype=torch.float32),
            symmetry=symmetry,
        )
        ang_deg = torch.rad2deg(ang).numpy()
        mask_good = ang_deg < float(degs_error_thr)

        parts_bg = parts_good.copy()
        parts_good = parts_good.loc[mask_good]

    # Apply ignore to BACKGROUND (anti-join on primary keys)
    if fname_ignore:
        parts_ign = _read_particles_star(fname_ignore, root_dir, "IGNORE")
        key = PRIMARY_KEYS
        parts_bg = parts_bg.merge(parts_ign[key].drop_duplicates(), on=key, how="left", indicator=True)
        parts_bg = parts_bg.loc[parts_bg["_merge"] == "left_only"].drop(columns=["_merge"])

    # Optional subsample of BACKGROUND
    if fraction_to_sample_from_all is not None and 0.0 < fraction_to_sample_from_all < 1.0:
        parts_bg = parts_bg.sample(frac=fraction_to_sample_from_all, random_state=42)

    # Build BAD set
    if fname_all is not None:
        # BAD = ALL \ GOOD (only case that needs a set-difference merge)
        key = PRIMARY_KEYS
        merged = parts_bg.merge(parts_good[key].drop_duplicates(), on=key, how="left", indicator=True)
        parts_bad = merged.loc[merged["_merge"] == "left_only"].drop(columns=["_merge"])
        if len(parts_bad) == 0:
            raise ValueError("ALL and GOOD are identical on primary keys; no BAD could be derived.")
    elif fname_bad is not None:
        # BAD provided directly
        parts_bad = parts_bg
    else:
        # Inferred mode: BAD = complement of GOOD within the original GOOD table
        key = PRIMARY_KEYS
        orig = _read_particles_star(fname_good, root_dir, "GOOD(for BAD complement)")
        parts_bad = orig.merge(parts_good[key].drop_duplicates(), on=key, how="left", indicator=True)
        parts_bad = parts_bad.loc[parts_bad["_merge"] == "left_only"].drop(columns=["_merge"])
        if len(parts_bad) == 0:
            raise ValueError("Could not infer BAD from angular error (all passed the threshold?).")

    # Extract scores
    def _scores(df: pd.DataFrame, col: str) -> np.ndarray:
        if col not in df.columns:
            raise KeyError(f"Missing score column '{col}'.")
        x = df[col].to_numpy()
        return x[np.isfinite(x)]

    score_good = _scores(parts_good, score_name)
    score_bad  = _scores(parts_bad,  score_name)

    print(f"GOOD n={len(parts_good)}; BAD n={len(parts_bad)}; BG n={len(parts_bg)}")
    if len(parts_bg) and len(parts_good):
        q = np.quantile(parts_bg[score_name], 1 - len(parts_good) / max(1, len(parts_bg)))
        print(f"Background quantile at |good|/|bg|: {q:.6f}")

    # Plot overlaid histograms
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.hist(_clip_percentiles(score_bad), bins=bins, alpha=0.5, color="orange", label="bad")
    ax1.set_xlabel("Score")
    ax1.set_ylabel("# bad particles")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.hist(_clip_percentiles(score_good), bins=bins, alpha=0.5, color="blue", label="good")
    ax2.set_ylabel("# good particles")
    ax2.legend(loc="upper right")

    xmin = min(_clip_percentiles(score_bad).min(), _clip_percentiles(score_good).min())
    xmax = max(_clip_percentiles(score_bad).max(), _clip_percentiles(score_good).max())
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

    # Optional GMM threshold
    if compute_gmm and len(score_bad) > 10 and len(score_good) > 10:
        base_for_gmm = os.path.splitext(plot_fname)[0] if plot_fname else None
        thr, gmms, method = separate_gmm_threshold(
            score_bad, score_good, show_plots=show_plots, out_base=base_for_gmm
        )
        print(f"GMM threshold: {thr:.6f} (method={method})")
        return thr, gmms, method

    return None


if __name__ == "__main__":
    from argParseFromDoc import parse_function_and_call
    parse_function_and_call(compare_prob_hists)
