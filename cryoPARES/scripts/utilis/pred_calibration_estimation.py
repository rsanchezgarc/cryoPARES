#!/usr/bin/env python3
"""
pred_calibration_estimation.py

Calibration / reliability analysis for CryoPARES confidence proxies using STAR files.

Aligned with compare_poses.py
-----------------------------
✅ Angular error uses RELION_EULER_CONVENTION (cryoPARES.constants)
✅ Angular error is symmetry-aware exactly like compare_poses.py:
      err = min_{S in symmetry} angle( S * (R_pred R_gt^T) )

Naming / notation (consistent)
------------------------------
Raw scores (NOT probabilities):
  - s_nnet  := raw network score from STAR (default: rlnParticleFigureOfMerit)
  - Z_dir   := robust directional score from STAR (default: rlnDirectionalZscore)
             (not a true z-score; robust median/MAD-based normalization)

Logistic probabilities (calibrated probabilities of correctness):
  - LP_nnet := sigmoid(a*s_nnet + b)
  - LP_dir  := sigmoid(a*Z_dir  + b)

Important consistency rule for plots in this script:
  - If the x-axis is a raw score, we label it with s_nnet or \tilde{Z}_dir
  - If the x-axis is a calibrated probability, we label it with LP_nnet or LP_dir

Cone usage (optional, via --per_cone_stats)
-------------------------------------------
By default, cones are used only to write per_cone.csv (means/medians by cone).

If you pass --per_cone_stats, we additionally compute useful cone-conditional diagnostics:
  - per-cone calibration: ECE_cone(LP_nnet), ECE_cone(LP_dir)
  - per-cone mean predicted probability vs empirical accuracy (over/under-confidence by direction)
  - per-cone AUC using LP_* (classification per cone)
  - plots:
      * mollview maps (Healpix) of per-cone accuracy, mean error, and mean LP_dir (if available)
      * scatter: per-cone accuracy vs mean LP_dir (and/or LP_nnet)
      * per-cone ECE distribution

Outputs
-------
Always:
  - <out_dir>/<out_prefix>.per_particle.csv
  - <out_dir>/<out_prefix>.per_cone.csv
  - <out_dir>/<out_prefix>.<key>.binned.csv      where key in {s_nnet, Z_dir}

If --per_cone_stats:
  - <out_dir>/<out_prefix>.per_cone_stats.csv

If --plots:
  - <out_prefix>.<key>.acc_vs_score.png
  - <out_prefix>.<key>.reliability.png

If --plots --compare_plots:
  - <out_prefix>.compare.acc_vs_score.png
  - <out_prefix>.compare.reliability.png
  - <out_prefix>.compare.scatter_s_nnet_vs_Z_dir.png

If --plots --per_cone_stats:
  - <out_prefix>.cone_map.acc.png
  - <out_prefix>.cone_map.err_mean.png
  - <out_prefix>.cone_map.LP_dir_mean.png (if LP_dir exists)
  - <out_prefix>.cone_map.LP_nnet_mean.png (if LP_nnet exists)
  - <out_prefix>.per_cone.acc_vs_LP_dir.png (if LP_dir exists)
  - <out_prefix>.per_cone.acc_vs_LP_nnet.png (if LP_nnet exists)
  - <out_prefix>.per_cone.ECE_dir_hist.png (if LP_dir exists)
  - <out_prefix>.per_cone.ECE_nnet_hist.png (if LP_nnet exists)

Example
-------
python cryoPARES/scripts/pred_calibration_estimation.py \
  --single /path/to/aligned_particles.star \
  --symmetry D3 --hp_order 2 --correct_deg 5 \
  --plots --compare_plots \
  --per_cone_stats --min_particles_per_cone 100 \
  --zdir_plot_min -5 --zdir_winsor_q 0.005 --zdir_winsor_for fit,plots \
  --out_dir /tmp/calib --out_prefix gdh_g2
"""

import argparse
import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd

import torch
import healpy as hp
import starfile

from cryoPARES.constants import RELION_EULER_CONVENTION
from scipy.spatial.transform import Rotation

try:
    from cryoPARES.models.image2sphere.so3Components import SO3OutputGrid
except Exception as e:
    raise ImportError(
        "Could not import cryoPARES SO3OutputGrid. Ensure cryoPARES is installed and importable.\n"
        f"Original error: {e}"
    )

# -------------------------
# Plot style constants (distinct colors)
# -------------------------
COLOR_SNNET = "tab:blue"
COLOR_ZDIR = "tab:orange"
MARKER_SNNET = "o"
MARKER_ZDIR = "s"

# -------------------------
# Labels (consistent)
# -------------------------
LABELS = {
    "s_nnet": {
        "raw": r"$s_{\mathrm{nnet}}$",
        "lp":  r"$LP_{\mathrm{nnet}}$",
        "raw_long": r"$s_{\mathrm{nnet}}$ (raw network score)",
        "lp_long":  r"$LP_{\mathrm{nnet}}$ (logistic $P(\mathrm{correct})$)",
    },
    "Z_dir": {
        "raw": r"$\tilde{Z}_{\mathrm{dir}}$",
        "lp":  r"$LP_{\mathrm{dir}}$",
        "raw_long": r"$\tilde{Z}_{\mathrm{dir}}$ (robust directional score)",
        "lp_long":  r"$LP_{\mathrm{dir}}$ (logistic $P(\mathrm{correct})$)",
    },
}

# -------------------------
# STAR reading helpers (starfile)
# -------------------------
def read_star_as_df(path: str) -> pd.DataFrame:
    obj = starfile.read(path)
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, dict):
        return obj["particles"]
    raise TypeError(f"Unsupported starfile.read return type: {type(obj)}")


def require_cols(df: pd.DataFrame, cols: List[str], where: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {where}: {missing}\nAvailable: {list(df.columns)[:80]}...")


def merge_gt_pred(gt: pd.DataFrame, pred: pd.DataFrame, merge_key: str) -> pd.DataFrame:
    if merge_key in gt.columns and merge_key in pred.columns:
        return pred.merge(gt, on=merge_key, suffixes=("", "_gt"), how="inner")

    if len(gt) != len(pred):
        raise ValueError(
            f"GT and pred have different lengths and merge key '{merge_key}' not found in both. "
            f"GT={len(gt)}, pred={len(pred)}"
        )
    df = pred.copy()
    for c in ["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]:
        if c not in gt.columns:
            raise KeyError(f"GT file missing '{c}' and merge by row order was required.")
        df[c + "_gt"] = gt[c].to_numpy()
    return df


# -------------------------
# RELION Euler -> rotation matrix (MATCHES compare_poses.py)
# -------------------------
def eulers_to_rotmats_relion(angles_deg: np.ndarray) -> np.ndarray:
    return Rotation.from_euler(RELION_EULER_CONVENTION, angles_deg, degrees=True).as_matrix().astype(np.float32)


def get_symmetry_matrices(sym: str) -> np.ndarray:
    sym = sym.upper()
    if sym == "C1":
        return np.eye(3, dtype=np.float32)[None, ...]
    try:
        rg = Rotation.create_group(sym)
        return rg.as_matrix().astype(np.float32)  # (K,3,3)
    except ValueError as e:
        raise ValueError(f"Invalid symmetry group: {sym}. Error: {str(e)}")


def angular_errors_with_sym_all(R_gt: np.ndarray, R_pred: np.ndarray, sym_mats: np.ndarray) -> np.ndarray:
    # Rdiff = R_pred @ R_gt^T
    Rdiff = np.einsum("nij,nkj->nik", R_pred, R_gt)  # (N,3,3)
    # Apply symmetry on left: S @ Rdiff
    R_diffs = np.einsum("kij,njl->nkil", sym_mats, Rdiff)  # (N,K,3,3)
    traces = np.einsum("nkii->nk", R_diffs)                # (N,K)
    angles = np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0))
    return np.degrees(np.min(angles, axis=1)).astype(np.float32)


# -------------------------
# cryoPARES cone assignment (exact mapping)
# -------------------------
@dataclass
class CryoPARESConeAssigner:
    symmetry: str
    hp_order: int
    device: str = "cpu"

    def __post_init__(self):
        self.symmetry = self.symmetry.upper()
        self.so3_grid = SO3OutputGrid(lmax=1, hp_order=self.hp_order, symmetry=self.symmetry)
        self.n_so3_pixels = int(self.so3_grid.output_rotmats.shape[0])

        self.n_cones = int(hp.nside2npix(hp.order2nside(self.hp_order)))
        if self.n_so3_pixels % self.n_cones != 0:
            raise ValueError(
                f"n_so3_pixels ({self.n_so3_pixels}) not divisible by n_cones ({self.n_cones}). "
                "Cone mapping so3_index // n_psi would be invalid."
            )
        self.n_psi = int(self.n_so3_pixels // self.n_cones)

    def rotmats_to_cone_id(self, rotmats_np: np.ndarray) -> np.ndarray:
        rotmats = torch.from_numpy(rotmats_np).to(self.device)
        with torch.no_grad():
            _, so3_indices = self.so3_grid.nearest_rotmat_idx(rotmats, reduce_sym=True)
        so3_indices = so3_indices.view(-1).cpu().numpy().astype(np.int64)
        return so3_indices // self.n_psi


# -------------------------
# Robust helper: winsorization
# -------------------------
def winsorize(x: np.ndarray, q: float) -> Tuple[np.ndarray, float, float]:
    if q <= 0:
        return x, float("nan"), float("nan")
    lo = float(np.nanquantile(x, q))
    hi = float(np.nanquantile(x, 1.0 - q))
    return np.clip(x, lo, hi), lo, hi


# -------------------------
# Metrics
# -------------------------
def spearman_corr(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if len(x) < 3:
        return None
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    if denom == 0.0:
        return None
    return float((rx @ ry) / denom)


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(p)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        if not np.any(m):
            continue
        conf = float(np.mean(p[m]))
        acc = float(np.mean(y[m]))
        w = float(np.sum(m)) / n
        ece += w * abs(acc - conf)
    return float(ece)


def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def logistic_calibrate(scores: np.ndarray, y: np.ndarray, seed: int = 0):
    """
    Fit 1D logistic mapping score -> P(correct).
    Returns dict with auc/ece and test-set (p,y) for reliability plots.
    If scikit-learn isn't available, returns None.

    Also returns fitted (a,b) as coef/intercept.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score
    except Exception:
        return None

    X = scores.reshape(-1, 1)
    strat = y if len(np.unique(y)) > 1 else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=strat)

    clf = LogisticRegression(solver="lbfgs")
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]

    auc = None
    if len(np.unique(yte)) > 1:
        auc = float(roc_auc_score(yte, p))
    ece = expected_calibration_error(p, yte, n_bins=15)

    a = float(clf.coef_.ravel()[0])
    b = float(clf.intercept_.ravel()[0])

    return {"auc": auc, "ece": ece, "p": p, "y": yte, "a": a, "b": b}


def logistic_predict(scores: np.ndarray, a: float, b: float) -> np.ndarray:
    z = a * scores + b
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def bin_stats_quantile(scores: np.ndarray, err_deg: np.ndarray, correct: np.ndarray, n_bins: int) -> pd.DataFrame:
    df = pd.DataFrame({"score": scores, "err_deg": err_deg, "correct": correct.astype(int)})
    df = df[np.isfinite(df["score"]) & np.isfinite(df["err_deg"])]
    if len(df) == 0:
        return pd.DataFrame(columns=["n", "score_mean", "err_median", "err_mean", "acc"])

    q = min(n_bins, max(2, len(df) // 200))
    df["bin"] = pd.qcut(df["score"], q=q, duplicates="drop")
    g = df.groupby("bin", observed=True)
    out = g.agg(
        n=("score", "size"),
        score_mean=("score", "mean"),
        err_median=("err_deg", "median"),
        err_mean=("err_deg", "mean"),
        acc=("correct", "mean"),
    ).reset_index(drop=True)
    return out


# -------------------------
# Plotting helpers
# -------------------------
def reliability_points_quantile(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.DataFrame({"p": p, "y": y})
    df = df[np.isfinite(df["p"]) & np.isfinite(df["y"])]
    if len(df) == 0:
        return np.array([]), np.array([])

    q = min(n_bins, max(2, len(df) // 200))
    df["bin"] = pd.qcut(df["p"], q=q, duplicates="drop")
    g = df.groupby("bin", observed=True)
    conf = g["p"].mean().to_numpy()
    acc = g["y"].mean().to_numpy()
    return conf, acc


def save_score_plots(out_dir: str, prefix: str, key: str, binned: pd.DataFrame,
                     lp_test: Optional[np.ndarray], y_test: Optional[np.ndarray],
                     reliability_bins: int):
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    if key == "s_nnet":
        color, marker = COLOR_SNNET, MARKER_SNNET
    else:
        color, marker = COLOR_ZDIR, MARKER_ZDIR

    # Accuracy vs raw score
    plt.figure()
    plt.plot(binned["score_mean"], binned["acc"], marker=marker, color=color)
    plt.xlabel(f"Mean {LABELS[key]['raw']} (bin)")
    plt.ylabel(r"Empirical $P(\mathrm{correct})$")
    plt.title(f"Accuracy vs {LABELS[key]['raw']}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}.{key}.acc_vs_score.png"), dpi=200)
    plt.close()

    # Reliability: x-axis must be LP_*
    if lp_test is not None and y_test is not None:
        confs, accs = reliability_points_quantile(lp_test, y_test, n_bins=reliability_bins)
        plt.figure()
        plt.plot([0, 1], [0, 1], linestyle="--", label="ideal", color="gray")
        if len(confs) > 0:
            plt.plot(confs, accs, marker=marker, color=color, label=LABELS[key]["lp"])
        plt.xlabel(f"Mean predicted {LABELS[key]['lp']}")
        plt.ylabel(r"Empirical $P(\mathrm{correct})$")
        plt.title(f"Reliability: {LABELS[key]['lp']}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}.{key}.reliability.png"), dpi=200)
        plt.close()


def save_compare_plots(out_dir: str, prefix: str, per_particle: pd.DataFrame,
                       n_bins: int, reliability_bins: int,
                       zdir_plot_min: Optional[float], zdir_plot_winsor_q: float,
                       logistic_cache: Optional[dict]):
    import matplotlib.pyplot as plt

    if "s_nnet" not in per_particle.columns or "Z_dir" not in per_particle.columns:
        print("[WARN] compare_plots requested but s_nnet and/or Z_dir missing; skipping.")
        return

    s_nnet = pd.to_numeric(per_particle["s_nnet"], errors="coerce").to_numpy(dtype=float)
    zdir = pd.to_numeric(per_particle["Z_dir"], errors="coerce").to_numpy(dtype=float)
    correct = per_particle["correct"].to_numpy(dtype=int)

    # binned curves
    def quantile_binned_acc(scores: np.ndarray, correct_: np.ndarray, n_bins_: int) -> pd.DataFrame:
        df = pd.DataFrame({"score": scores, "correct": correct_.astype(int)})
        df = df[np.isfinite(df["score"])]
        if len(df) < 10:
            return pd.DataFrame(columns=["n", "score_mean", "acc"])
        q = min(n_bins_, max(2, len(df) // 200))
        df["bin"] = pd.qcut(df["score"], q=q, duplicates="drop")
        g = df.groupby("bin", observed=True)
        return g.agg(n=("score", "size"), score_mean=("score", "mean"), acc=("correct", "mean")).reset_index(drop=True)

    b_s = quantile_binned_acc(s_nnet, correct, n_bins)

    z_plot = zdir.copy()
    m_z = np.isfinite(z_plot)
    if zdir_plot_winsor_q > 0:
        z_plot_m, _, _ = winsorize(z_plot[m_z], zdir_plot_winsor_q)
        z_plot[m_z] = z_plot_m
    if zdir_plot_min is not None:
        m_z = m_z & (z_plot > zdir_plot_min)
    b_z = quantile_binned_acc(z_plot[m_z], correct[m_z], n_bins)

    # (A) accuracy vs raw score with two x-axes (bottom Z_dir, top s_nnet)
    fig, ax_bottom = plt.subplots()
    if len(b_z) > 0:
        lab = LABELS["Z_dir"]["raw"]
        if zdir_plot_min is not None:
            lab += f" (>{zdir_plot_min:g})"
        ax_bottom.plot(b_z["score_mean"], b_z["acc"], marker=MARKER_ZDIR, color=COLOR_ZDIR, label=lab)
        ax_bottom.set_xlim(float(np.min(b_z["score_mean"])), float(np.max(b_z["score_mean"])))
    ax_bottom.set_xlabel(LABELS["Z_dir"]["raw"])
    ax_bottom.set_ylabel(r"Empirical $P(\mathrm{correct})$")
    ax_bottom.grid(True)

    ax_top = ax_bottom.twiny()
    if len(b_s) > 0:
        ax_top.plot(b_s["score_mean"], b_s["acc"], marker=MARKER_SNNET, color=COLOR_SNNET, label=LABELS["s_nnet"]["raw"])
        ax_top.set_xlim(float(np.min(b_s["score_mean"])), float(np.max(b_s["score_mean"])))
    ax_top.set_xlabel(LABELS["s_nnet"]["raw"])

    hb, lb = ax_bottom.get_legend_handles_labels()
    ht, lt = ax_top.get_legend_handles_labels()
    if (len(hb) + len(ht)) > 0:
        ax_bottom.legend(hb + ht, lb + lt, loc="best")

    fig.suptitle("Accuracy vs raw score (separate x-scales)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}.compare.acc_vs_score.png"), dpi=200)
    plt.close(fig)

    # (B) reliability overlay: LP_nnet and LP_dir
    if logistic_cache is not None and "s_nnet" in logistic_cache and "Z_dir" in logistic_cache:
        ps, ys = logistic_cache["s_nnet"]["p"], logistic_cache["s_nnet"]["y"]
        pz, yz = logistic_cache["Z_dir"]["p"], logistic_cache["Z_dir"]["y"]
        cs, as_ = reliability_points_quantile(ps, ys, n_bins=reliability_bins)
        cz, az = reliability_points_quantile(pz, yz, n_bins=reliability_bins)

        plt.figure()
        plt.plot([0, 1], [0, 1], linestyle="--", label="ideal", color="gray")
        if len(cs) > 0:
            plt.plot(cs, as_, marker=MARKER_SNNET, color=COLOR_SNNET, label=LABELS["s_nnet"]["lp"])
        if len(cz) > 0:
            plt.plot(cz, az, marker=MARKER_ZDIR, color=COLOR_ZDIR, label=LABELS["Z_dir"]["lp"])
        plt.xlabel("Mean predicted logistic probability")
        plt.ylabel(r"Empirical $P(\mathrm{correct})$")
        plt.title("Reliability (after logistic mapping)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}.compare.reliability.png"), dpi=200)
        plt.close()

    # (C) scatter raw-score relationship colored by error
    if "err_deg" in per_particle.columns:
        err = pd.to_numeric(per_particle["err_deg"], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(s_nnet) & np.isfinite(zdir) & np.isfinite(err)
        xs, yz_, ce = s_nnet[m], zdir[m], err[m]
        max_pts = 50000
        if len(xs) > max_pts:
            idx = np.random.RandomState(0).choice(len(xs), size=max_pts, replace=False)
            xs, yz_, ce = xs[idx], yz_[idx], ce[idx]

        plt.figure()
        sc = plt.scatter(xs, yz_, c=ce, s=3, cmap="viridis")
        plt.xlabel(LABELS["s_nnet"]["raw"])
        plt.ylabel(LABELS["Z_dir"]["raw"])
        plt.title("Raw-score relationship (colored by angular error)")
        plt.grid(True)
        plt.colorbar(sc, label=r"$\theta$ (deg)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}.compare.scatter_s_nnet_vs_Z_dir.png"), dpi=200)
        plt.close()


# -------------------------
# Per-cone stats (useful diagnostics)
# -------------------------
def compute_per_cone_stats(per_particle: pd.DataFrame,
                           min_particles_per_cone: int,
                           ece_bins: int = 15) -> pd.DataFrame:
    """
    Requires per_particle to contain:
      cone_id, correct, err_deg
    Optionally uses:
      LP_nnet, LP_dir

    Produces per-cone calibration diagnostics:
      n, acc, err_mean, err_median,
      LP_*_mean, ECE_*_cone, Brier_*_cone, AUC_*_cone
    """
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        roc_auc_score = None

    out_rows = []
    for cone_id, g in per_particle.groupby("cone_id", observed=True):
        n = int(len(g))
        if n < min_particles_per_cone:
            continue

        y = g["correct"].to_numpy(dtype=int)
        err = pd.to_numeric(g["err_deg"], errors="coerce").to_numpy(dtype=float)

        row = {
            "cone_id": int(cone_id),
            "n": n,
            "acc": float(np.mean(y)),
            "err_mean": float(np.mean(err[np.isfinite(err)])) if np.any(np.isfinite(err)) else np.nan,
            "err_median": float(np.median(err[np.isfinite(err)])) if np.any(np.isfinite(err)) else np.nan,
        }

        for lp_name, lp_col in [("nnet", "LP_nnet"), ("dir", "LP_dir")]:
            if lp_col in g.columns:
                p = pd.to_numeric(g[lp_col], errors="coerce").to_numpy(dtype=float)
                m = np.isfinite(p)
                if np.sum(m) >= max(10, min_particles_per_cone // 10):
                    p_ = p[m]
                    y_ = y[m]
                    row[f"{lp_col}_mean"] = float(np.mean(p_))
                    row[f"ECE_{lp_name}_cone"] = expected_calibration_error(p_, y_, n_bins=ece_bins)
                    row[f"Brier_{lp_name}_cone"] = brier_score(p_, y_)
                    if roc_auc_score is not None and len(np.unique(y_)) > 1:
                        row[f"AUC_{lp_name}_cone"] = float(roc_auc_score(y_, p_))
                    else:
                        row[f"AUC_{lp_name}_cone"] = np.nan
                else:
                    row[f"{lp_col}_mean"] = np.nan
                    row[f"ECE_{lp_name}_cone"] = np.nan
                    row[f"Brier_{lp_name}_cone"] = np.nan
                    row[f"AUC_{lp_name}_cone"] = np.nan

        out_rows.append(row)

    if len(out_rows) == 0:
        return pd.DataFrame(columns=["cone_id", "n", "acc", "err_mean", "err_median"])

    df = pd.DataFrame(out_rows).sort_values("n", ascending=False).reset_index(drop=True)
    return df


def plot_per_cone_stats(out_dir: str, prefix: str,
                        hp_order: int, per_cone_stats: pd.DataFrame):
    """
    Useful cone plots:
      - Mollweide maps (Healpix) for acc, err_mean, LP_dir_mean/LP_nnet_mean
      - Scatter acc vs mean LP_* (over/underconfidence by direction)
      - Histograms of per-cone ECE
    """
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    nside = hp.order2nside(hp_order)
    npix = hp.nside2npix(nside)

    def make_map(values_by_cone: pd.Series, fill=np.nan) -> np.ndarray:
        m = np.full((npix,), fill, dtype=float)
        for cid, val in values_by_cone.items():
            if 0 <= int(cid) < npix:
                m[int(cid)] = float(val)
        return m

    # Maps: acc, err_mean
    acc_map = make_map(per_cone_stats.set_index("cone_id")["acc"])
    err_map = make_map(per_cone_stats.set_index("cone_id")["err_mean"])

    # Healpy mollview saves into current figure; use plt.figure() to ensure separated output
    plt.figure()
    hp.mollview(acc_map, title="Per-cone accuracy (empirical P(correct))", unit="acc", hold=True)
    plt.savefig(os.path.join(out_dir, f"{prefix}.cone_map.acc.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    hp.mollview(err_map, title="Per-cone mean angular error", unit="deg", hold=True)
    plt.savefig(os.path.join(out_dir, f"{prefix}.cone_map.err_mean.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # Maps: mean LP_dir / LP_nnet if present
    if "LP_dir_mean" in per_cone_stats.columns:
        lp_dir_map = make_map(per_cone_stats.set_index("cone_id")["LP_dir_mean"])
        plt.figure()
        hp.mollview(lp_dir_map, title="Per-cone mean LP_dir", unit="LP_dir", hold=True)
        plt.savefig(os.path.join(out_dir, f"{prefix}.cone_map.LP_dir_mean.png"), dpi=200, bbox_inches="tight")
        plt.close()

        # scatter acc vs mean LP_dir
        df = per_cone_stats[["acc", "LP_dir_mean", "n"]].dropna()
        if len(df) > 0:
            plt.figure()
            plt.scatter(df["LP_dir_mean"], df["acc"], s=np.clip(df["n"] / 20.0, 5, 80), alpha=0.7)
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.xlabel(LABELS["Z_dir"]["lp"])
            plt.ylabel(r"Empirical $P(\mathrm{correct})$ (cone)")
            plt.title("Per-cone calibration summary: acc vs mean LP_dir\n(point size ∝ n)")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{prefix}.per_cone.acc_vs_LP_dir.png"), dpi=200)
            plt.close()

    if "LP_nnet_mean" in per_cone_stats.columns:
        lp_nnet_map = make_map(per_cone_stats.set_index("cone_id")["LP_nnet_mean"])
        plt.figure()
        hp.mollview(lp_nnet_map, title="Per-cone mean LP_nnet", unit="LP_nnet", hold=True)
        plt.savefig(os.path.join(out_dir, f"{prefix}.cone_map.LP_nnet_mean.png"), dpi=200, bbox_inches="tight")
        plt.close()

        df = per_cone_stats[["acc", "LP_nnet_mean", "n"]].dropna()
        if len(df) > 0:
            plt.figure()
            plt.scatter(df["LP_nnet_mean"], df["acc"], s=np.clip(df["n"] / 20.0, 5, 80), alpha=0.7)
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.xlabel(LABELS["s_nnet"]["lp"])
            plt.ylabel(r"Empirical $P(\mathrm{correct})$ (cone)")
            plt.title("Per-cone calibration summary: acc vs mean LP_nnet\n(point size ∝ n)")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{prefix}.per_cone.acc_vs_LP_nnet.png"), dpi=200)
            plt.close()

    # Histograms: per-cone ECE distributions (if present)
    if "ECE_dir_cone" in per_cone_stats.columns:
        e = per_cone_stats["ECE_dir_cone"].to_numpy(dtype=float)
        e = e[np.isfinite(e)]
        if len(e) > 0:
            plt.figure()
            plt.hist(e, bins=30, alpha=0.8)
            plt.xlabel("ECE per cone (LP_dir)")
            plt.ylabel("count")
            plt.title("Distribution of per-cone ECE (LP_dir)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{prefix}.per_cone.ECE_dir_hist.png"), dpi=200)
            plt.close()

    if "ECE_nnet_cone" in per_cone_stats.columns:
        e = per_cone_stats["ECE_nnet_cone"].to_numpy(dtype=float)
        e = e[np.isfinite(e)]
        if len(e) > 0:
            plt.figure()
            plt.hist(e, bins=30, alpha=0.8)
            plt.xlabel("ECE per cone (LP_nnet)")
            plt.ylabel("count")
            plt.title("Distribution of per-cone ECE (LP_nnet)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{prefix}.per_cone.ECE_nnet_hist.png"), dpi=200)
            plt.close()


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--gt", type=str, default=None, help="GT STAR path (RELION angles columns)")
    ap.add_argument("--pred", type=str, default=None, help="Pred STAR path (angles + scores)")
    ap.add_argument("--single", type=str, default=None, help="Single STAR containing predicted + *_ori GT angles")

    ap.add_argument("--merge_key", type=str, default="rlnImageName",
                    help="Column used for merging when using --gt/--pred")

    # Pred angle columns
    ap.add_argument("--pred_rot", type=str, default="rlnAngleRot")
    ap.add_argument("--pred_tilt", type=str, default="rlnAngleTilt")
    ap.add_argument("--pred_psi", type=str, default="rlnAnglePsi")

    # GT angle columns for --single
    ap.add_argument("--gt_rot", type=str, default="rlnAngleRot_ori")
    ap.add_argument("--gt_tilt", type=str, default="rlnAngleTilt_ori")
    ap.add_argument("--gt_psi", type=str, default="rlnAnglePsi_ori")

    # STAR columns for raw scores
    ap.add_argument("--fom_col", type=str, default="rlnParticleFigureOfMerit",
                    help="STAR column used for s_nnet (raw network score)")
    ap.add_argument("--dirz_col", type=str, default="rlnDirectionalZscore",
                    help="STAR column used for Z_dir (robust directional score)")

    # Plot-only lower cutoff for Z_dir (ignore super-negative tail in acc-vs-score plots)
    ap.add_argument("--zdir_plot_min", type=float, default=-5.0,
                    help="For accuracy-vs-Z_dir plot only: include only particles with Z_dir > this threshold.")
    ap.add_argument("--zdir_plot_min_disable", action="store_true",
                    help="Disable plot-only Z_dir lower cutoff.")

    # Winsorization for Z_dir (outlier-robustness)
    ap.add_argument("--zdir_winsor_q", type=float, default=0.005,
                    help="Winsorize Z_dir by clipping to [q, 1-q] quantiles (0 disables).")
    ap.add_argument("--zdir_winsor_for", type=str, default="fit,plots",
                    help="Comma-separated subset of {fit,plots,metrics}. Default: fit,plots.")
    ap.add_argument("--print_winsor_info", action="store_true")

    # cryoPARES cone settings
    ap.add_argument("--symmetry", type=str, required=True)
    ap.add_argument("--hp_order", type=int, required=True)
    ap.add_argument("--device", type=str, default="cpu")

    # evaluation settings
    ap.add_argument("--correct_deg", type=float, default=5.0)
    ap.add_argument("--n_bins", type=int, default=15)
    ap.add_argument("--reliability_bins", type=int, default=10)
    ap.add_argument("--no_fit", action="store_true")
    ap.add_argument("--seed", type=int, default=0)

    # outputs
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--out_prefix", type=str, default="calibration")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--compare_plots", action="store_true")

    # NEW: per-cone diagnostics
    ap.add_argument("--per_cone_stats", action="store_true",
                    help="Compute additional per-cone calibration diagnostics using LP_nnet/LP_dir.")
    ap.add_argument("--min_particles_per_cone", type=int, default=200,
                    help="Minimum particles required per cone to compute per-cone diagnostics.")
    ap.add_argument("--per_cone_ece_bins", type=int, default=15,
                    help="Number of probability bins for per-cone ECE computation.")

    args = ap.parse_args()

    # Validate input mode
    if args.single is None:
        if args.gt is None or args.pred is None:
            raise ValueError("Provide either --single OR both --gt and --pred.")
    else:
        if args.gt is not None or args.pred is not None:
            raise ValueError("Use either --single OR --gt/--pred, not both.")

    os.makedirs(args.out_dir, exist_ok=True)

    # Load data + extract angles
    if args.single:
        df = read_star_as_df(args.single)
        require_cols(df, [args.pred_rot, args.pred_tilt, args.pred_psi], "--single predicted angles")
        require_cols(df, [args.gt_rot, args.gt_tilt, args.gt_psi], "--single GT *_ori angles")

        pred_angles = df[[args.pred_rot, args.pred_tilt, args.pred_psi]].to_numpy(dtype=float)
        gt_angles = df[[args.gt_rot, args.gt_tilt, args.gt_psi]].to_numpy(dtype=float)
    else:
        gt_df = read_star_as_df(args.gt)
        pred_df = read_star_as_df(args.pred)
        require_cols(gt_df, [args.pred_rot, args.pred_tilt, args.pred_psi], "--gt angles")
        require_cols(pred_df, [args.pred_rot, args.pred_tilt, args.pred_psi], "--pred angles")

        merged = merge_gt_pred(gt_df, pred_df, args.merge_key)
        df = merged

        pred_angles = merged[[args.pred_rot, args.pred_tilt, args.pred_psi]].to_numpy(dtype=float)
        require_cols(merged, [args.pred_rot + "_gt", args.pred_tilt + "_gt", args.pred_psi + "_gt"], "merged GT angles")
        gt_angles = merged[[args.pred_rot + "_gt", args.pred_tilt + "_gt", args.pred_psi + "_gt"]].to_numpy(dtype=float)

    # Rotmats and symmetry-aware errors (MATCH compare_poses.py)
    sym_mats = get_symmetry_matrices(args.symmetry)
    R_pred = eulers_to_rotmats_relion(pred_angles)
    R_gt = eulers_to_rotmats_relion(gt_angles)

    err = angular_errors_with_sym_all(R_gt, R_pred, sym_mats)
    correct = (err <= args.correct_deg).astype(np.int32)

    print(f"[INFO] RELION_EULER_CONVENTION = {RELION_EULER_CONVENTION}")
    print(f"[INFO] symmetry group size = {sym_mats.shape[0]}")
    print(f"[INFO] correct definition: err_deg <= {args.correct_deg:.2f} deg")
    print(f"[INFO] Global % < {args.correct_deg:g}° = {100.0 * float(np.mean(err < args.correct_deg)):.2f}%")

    # Cone assignment (always computed; required for per_cone.csv and optional per_cone_stats)
    assigner = CryoPARESConeAssigner(symmetry=args.symmetry, hp_order=args.hp_order, device=args.device)
    cone_id = assigner.rotmats_to_cone_id(R_pred)
    print(f"[INFO] cones: hp_order={args.hp_order} => n_cones={assigner.n_cones}, n_psi={assigner.n_psi}")

    # Per-particle table
    per_particle = pd.DataFrame({
        "err_deg": err,
        "correct": correct,
        "cone_id": cone_id.astype(np.int64),
    })
    # Raw scores
    if args.fom_col in df.columns:
        per_particle["s_nnet"] = pd.to_numeric(df[args.fom_col], errors="coerce")
    else:
        breakpoint()
        print(f"[WARN] s_nnet source column not found: {args.fom_col}")

    if args.dirz_col in df.columns:
        per_particle["Z_dir"] = pd.to_numeric(df[args.dirz_col], errors="coerce")
    else:
        print(f"[WARN] Z_dir source column not found: {args.dirz_col}")

    # Basic per-cone summary (always)
    gcone = per_particle.groupby("cone_id", observed=True)
    per_cone = gcone.agg(
        n=("correct", "size"),
        acc=("correct", "mean"),
        err_median=("err_deg", "median"),
        err_mean=("err_deg", "mean"),
    ).reset_index()

    if "s_nnet" in per_particle.columns:
        per_cone["s_nnet_mean"] = gcone["s_nnet"].mean().to_numpy()
    if "Z_dir" in per_particle.columns:
        per_cone["Z_dir_mean"] = gcone["Z_dir"].mean().to_numpy()

    per_cone = per_cone.sort_values("n", ascending=False)
    per_cone_path = os.path.join(args.out_dir, f"{args.out_prefix}.per_cone.csv")
    per_cone.to_csv(per_cone_path, index=False)
    print(f"[OK] per-cone CSV: {per_cone_path}")

    # Winsorization settings for Z_dir
    apply = set([t.strip() for t in args.zdir_winsor_for.split(",") if t.strip()])
    valid = {"fit", "plots", "metrics"}
    bad = sorted(list(apply - valid))
    if bad:
        raise ValueError(f"--zdir_winsor_for contains invalid entries: {bad}. Valid: fit,plots,metrics")

    logistic_cache: Dict[str, Dict[str, Any]] = {}

    def eval_score(key: str):
        """
        key in {'s_nnet','Z_dir'}:
          - Spearman(score, err_deg)
          - binned accuracy-vs-score (raw score)
          - logistic mapping -> LP_* (if enabled)
          - reliability plot uses LP_* (not raw score)
        """
        if key not in per_particle.columns:
            return

        raw = pd.to_numeric(per_particle[key], errors="coerce").to_numpy(dtype=float)
        m_all = np.isfinite(raw) & np.isfinite(per_particle["err_deg"].to_numpy(dtype=float))
        s_all = raw[m_all]
        e_all = per_particle["err_deg"].to_numpy(dtype=float)[m_all]
        y_all = per_particle["correct"].to_numpy(dtype=int)[m_all]

        s_metrics = s_all
        s_fit = s_all
        s_plot = s_all

        # Winsorization ONLY for Z_dir
        if key == "Z_dir" and args.zdir_winsor_q > 0:
            if "metrics" in apply:
                s_metrics, lo_m, hi_m = winsorize(s_all, args.zdir_winsor_q)
            if "fit" in apply:
                s_fit, lo_f, hi_f = winsorize(s_all, args.zdir_winsor_q)
            if "plots" in apply:
                s_plot, lo_p, hi_p = winsorize(s_all, args.zdir_winsor_q)

            if args.print_winsor_info:
                if "metrics" in apply:
                    print(f"[Z_dir winsor metrics] q={args.zdir_winsor_q:g} -> clip [{lo_m:.4g}, {hi_m:.4g}]")
                if "fit" in apply:
                    print(f"[Z_dir winsor fit]     q={args.zdir_winsor_q:g} -> clip [{lo_f:.4g}, {hi_f:.4g}]")
                if "plots" in apply:
                    print(f"[Z_dir winsor plots]   q={args.zdir_winsor_q:g} -> clip [{lo_p:.4g}, {hi_p:.4g}]")

        sp = spearman_corr(s_metrics, e_all)

        # Plot-only filter for Z_dir acc-vs-score
        if key == "Z_dir" and not args.zdir_plot_min_disable:
            zmin = args.zdir_plot_min
            m_plot = (s_plot > zmin)
            if np.sum(m_plot) < 10:
                m_plot = np.ones_like(s_plot, dtype=bool)
            s_plot2 = s_plot[m_plot]
            e_plot2 = e_all[m_plot]
            y_plot2 = y_all[m_plot]
            print(f"[INFO] For accuracy-vs-{key} plot only: using {len(s_plot2)} samples with {key} > {zmin:g}")
        else:
            s_plot2, e_plot2, y_plot2 = s_plot, e_all, y_all

        binned = bin_stats_quantile(s_plot2, e_plot2, y_plot2, n_bins=args.n_bins)
        binned_path = os.path.join(args.out_dir, f"{args.out_prefix}.{key}.binned.csv")
        binned.to_csv(binned_path, index=False)

        auc = None
        ece = None
        p_test = None
        y_test = None

        if not args.no_fit:
            cal = logistic_calibrate(s_fit, y_all, seed=args.seed)
            if cal is not None:
                auc = cal["auc"]
                ece = cal["ece"]
                p_test = cal["p"]
                y_test = cal["y"]
                a, b = cal["a"], cal["b"]
                logistic_cache[key] = {"p": p_test, "y": y_test, "a": a, "b": b}

                # Add LP_* for ALL particles where score is finite
                lp_all = np.full_like(raw, np.nan, dtype=float)
                lp_all[m_all] = logistic_predict(s_all, a, b)
                if key == "s_nnet":
                    per_particle["LP_nnet"] = lp_all
                else:
                    per_particle["LP_dir"] = lp_all

        print(f"\n=== {key} ===")
        print(f"n={len(s_all)}")
        if sp is not None:
            print(f"Spearman(score, error_deg) = {sp:.4f}  (more negative is better)")
        print(f"Binned curve saved: {binned_path}")
        if auc is not None:
            print(f"ROC-AUC (after logistic mapping) = {auc:.4f}")
        if ece is not None:
            print(f"ECE (after logistic mapping) = {ece:.4f}")

        if args.plots and len(binned) > 0:
            save_score_plots(args.out_dir, args.out_prefix, key, binned, p_test, y_test, args.reliability_bins)

    # Evaluate scores
    if "s_nnet" in per_particle.columns:
        eval_score("s_nnet")
    if "Z_dir" in per_particle.columns:
        eval_score("Z_dir")

    # Save per-particle CSV (now including LP_* if fit succeeded)
    per_particle_path = os.path.join(args.out_dir, f"{args.out_prefix}.per_particle.csv")
    per_particle.to_csv(per_particle_path, index=False)
    print(f"[OK] per-particle CSV: {per_particle_path}")

    # Compare plots (global)
    if args.compare_plots and args.plots:
        zmin = None if args.zdir_plot_min_disable else args.zdir_plot_min
        zq = args.zdir_winsor_q if ("plots" in apply) else 0.0
        save_compare_plots(args.out_dir, args.out_prefix, per_particle,
                           args.n_bins, args.reliability_bins,
                           zmin, zq,
                           logistic_cache if ("s_nnet" in logistic_cache and "Z_dir" in logistic_cache) else None)
        print(f"[OK] comparison plots written: {args.out_dir}/{args.out_prefix}.compare.*.png")

    # NEW: per-cone calibration diagnostics (actually uses cones meaningfully)
    if args.per_cone_stats:
        if ("LP_nnet" not in per_particle.columns) and ("LP_dir" not in per_particle.columns):
            print("[WARN] --per_cone_stats requested but no LP_* columns found. "
                  "Enable logistic fit (remove --no_fit) to compute per-cone calibration metrics.")
        per_cone_stats = compute_per_cone_stats(per_particle,
                                                min_particles_per_cone=args.min_particles_per_cone,
                                                ece_bins=args.per_cone_ece_bins)
        per_cone_stats_path = os.path.join(args.out_dir, f"{args.out_prefix}.per_cone_stats.csv")
        per_cone_stats.to_csv(per_cone_stats_path, index=False)
        print(f"[OK] per-cone stats CSV: {per_cone_stats_path}")
        print(f"[INFO] per-cone stats: kept {len(per_cone_stats)} cones with n >= {args.min_particles_per_cone}")

        if args.plots and len(per_cone_stats) > 0:
            plot_per_cone_stats(args.out_dir, args.out_prefix, args.hp_order, per_cone_stats)
            print(f"[OK] per-cone plots written: {args.out_dir}/{args.out_prefix}.cone_map.*.png and {args.out_prefix}.per_cone.*.png")

    if args.plots:
        print(f"[OK] plots written under {args.out_dir} with prefix {args.out_prefix}")


if __name__ == "__main__":
    main()
