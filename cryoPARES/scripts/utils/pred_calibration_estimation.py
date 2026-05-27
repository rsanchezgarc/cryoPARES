#!/usr/bin/env python3
"""
pred_calibration_estimation_pr_auc.py

Calibration / reliability analysis for CryoPARES confidence proxies using STAR files.

What this script does
---------------------
1. Reads predicted orientations and ground-truth orientations from STAR files.
2. Computes symmetry-aware angular errors.
3. Evaluates raw confidence scores as ranking signals using:
     - ROC-AUC
     - PR-AUC (Average Precision)
     - Spearman correlation with angular error
4. Fits a post-hoc logistic calibrator on a train split and evaluates on a held-out test split using:
     - Brier score
     - Log loss
     - Expected Calibration Error (ECE)
5. Saves per-particle results, binned raw-score curves, and optional plots.

Important interpretation rule
------------------------------
- ROC-AUC and PR-AUC are ranking metrics. They should be computed on the raw score,
  and they do not change under any monotonic calibration transform such as a logistic map.
- Brier score, log loss, and ECE measure probability calibration and are the metrics
  that should improve after post-hoc calibration.

This script keeps the calibration logic simple and defensible for rebuttal use.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
import healpy as hp
import starfile

from scipy.spatial.transform import Rotation
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from cryoPARES.constants import RELION_EULER_CONVENTION

try:
    from cryoPARES.models.image2sphere.so3Components import SO3OutputGrid
except Exception as e:
    raise ImportError(
        "Could not import cryoPARES SO3OutputGrid. Ensure cryoPARES is installed and importable.\n"
        f"Original error: {e}"
    )

# -----------------------------------------------------------------------------
# Labels
# -----------------------------------------------------------------------------
LABELS = {
    "s_nnet": {
        "raw": r"$s_{\mathrm{nnet}}$",
        "raw_long": r"$s_{\mathrm{nnet}}$ (raw network score)",
        "lp": r"$LP_{\mathrm{nnet}}$",
        "lp_long": r"$LP_{\mathrm{nnet}}$ (post-hoc calibrated probability)",
    },
    "Z_dir": {
        "raw": r"$\tilde{Z}_{\mathrm{dir}}$",
        "raw_long": r"$\tilde{Z}_{\mathrm{dir}}$ (direction-normalized score)",
        "lp": r"$LP_{\mathrm{dir}}$",
        "lp_long": r"$LP_{\mathrm{dir}}$ (post-hoc calibrated probability)",
    },
}

COLOR_SNNET = "tab:blue"
COLOR_ZDIR = "tab:orange"
MARKER_SNNET = "o"
MARKER_ZDIR = "s"


# -----------------------------------------------------------------------------
# STAR helpers
# -----------------------------------------------------------------------------
def read_star_as_df(path: str) -> pd.DataFrame:
    obj = starfile.read(path)
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, dict):
        if "particles" in obj:
            return obj["particles"]
        # fall back to the first table if needed
        return next(iter(obj.values()))
    raise TypeError(f"Unsupported starfile.read return type: {type(obj)}")


def require_cols(df: pd.DataFrame, cols: List[str], where: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {where}: {missing}\nAvailable: {list(df.columns)[:80]}...")


_COORD_COLS = ["rlnCoordinateX", "rlnCoordinateY"]
_MGRAPH_COL = "rlnMicrographName"
_COORD_KEY  = "__coord_merge_key__"


def _mic_basename(s: str) -> str:
    """Strip directory prefix from a micrograph path, keeping only the basename."""
    return os.path.basename(str(s))


def _make_coord_key(df: pd.DataFrame, round_px: int = 2) -> Optional[pd.Series]:
    """
    Build a composite key: basename(rlnMicrographName) + rounded rlnCoordinateX/Y.
    Returns None if the required columns are absent.
    """
    if not all(c in df.columns for c in [_MGRAPH_COL] + _COORD_COLS):
        return None
    mic = df[_MGRAPH_COL].astype(str).map(_mic_basename)
    cx  = df[_COORD_COLS[0]].round(round_px).astype(str)
    cy  = df[_COORD_COLS[1]].round(round_px).astype(str)
    return mic + "|" + cx + "|" + cy


def merge_gt_pred(gt: pd.DataFrame, pred: pd.DataFrame, merge_key: str, gt_angle_cols: List[str]) -> pd.DataFrame:
    if merge_key in gt.columns and merge_key in pred.columns:
        merged = pred.merge(gt, on=merge_key, suffixes=("", "_gt"), how="inner")
        if len(merged) > 0:
            return merged

        # rlnImageName join failed (different path prefixes or extraction schemes).
        # Fall back to coordinate-based matching: basename(micrograph) + (X, Y).
        print(f"[WARN] Inner join on '{merge_key}' returned 0 rows. "
              f"Falling back to coordinate-based matching "
              f"({_MGRAPH_COL} basename + {_COORD_COLS}).")
        pred_key = _make_coord_key(pred)
        gt_key   = _make_coord_key(gt)
        if pred_key is None or gt_key is None:
            missing_pred = [c for c in [_MGRAPH_COL] + _COORD_COLS if c not in pred.columns]
            missing_gt   = [c for c in [_MGRAPH_COL] + _COORD_COLS if c not in gt.columns]
            sample_pred  = pred[merge_key].astype(str).iloc[:3].tolist()
            sample_gt    = gt[merge_key].astype(str).iloc[:3].tolist()
            raise ValueError(
                f"Join on '{merge_key}' returned 0 rows and coordinate fallback is not possible "
                f"(missing cols — pred: {missing_pred}, gt: {missing_gt}).\n"
                f"  pred sample: {sample_pred}\n"
                f"  gt   sample: {sample_gt}"
            )

        pred2 = pred.copy()
        gt2   = gt.copy()
        pred2[_COORD_KEY] = pred_key
        gt2[_COORD_KEY]   = gt_key

        # Warn if coordinate keys are not unique within either set (duplicate coords in same mic)
        for label, df2, key2 in [("pred", pred2, pred_key), ("gt", gt2, gt_key)]:
            n_dup = int(key2.duplicated().sum())
            if n_dup > 0:
                print(f"[WARN] {n_dup} duplicate coordinate keys in {label} — "
                      f"these particles may cross-match incorrectly.")

        merged = pred2.merge(gt2, on=_COORD_KEY, suffixes=("", "_gt"), how="inner")
        merged = merged.drop(columns=[_COORD_KEY])

        if len(merged) == 0:
            sample_pred = pred_key.iloc[:3].tolist()
            sample_gt   = gt_key.iloc[:3].tolist()
            raise ValueError(
                f"Coordinate-based merge also returned 0 rows.\n"
                f"  pred coord key sample: {sample_pred}\n"
                f"  gt   coord key sample: {sample_gt}\n"
                f"These STARs may not share the same particle set."
            )

        print(f"[INFO] Coordinate-based merge succeeded: {len(merged)} matched particles "
              f"(pred={len(pred)}, gt={len(gt)}).")
        return merged

    if len(gt) != len(pred):
        raise ValueError(
            f"GT and pred have different lengths and merge key '{merge_key}' not found in both. "
            f"GT={len(gt)}, pred={len(pred)}"
        )

    df = pred.copy()
    for c in gt_angle_cols:
        if c not in gt.columns:
            raise KeyError(f"GT file missing '{c}' and merge by row order was required.")
        df[c + "_gt"] = gt[c].to_numpy()
    return df


# -----------------------------------------------------------------------------
# Pose / symmetry helpers
# -----------------------------------------------------------------------------
def eulers_to_rotmats_relion(angles_deg: np.ndarray) -> np.ndarray:
    return Rotation.from_euler(RELION_EULER_CONVENTION, angles_deg, degrees=True).as_matrix().astype(np.float32)


def get_symmetry_matrices(sym: str) -> np.ndarray:
    sym = sym.upper()
    if sym == "C1":
        return np.eye(3, dtype=np.float32)[None, ...]
    try:
        rg = Rotation.create_group(sym)
        return rg.as_matrix().astype(np.float32)
    except ValueError as e:
        raise ValueError(f"Invalid symmetry group: {sym}. Error: {str(e)}")


def angular_errors_with_sym_all(R_gt: np.ndarray, R_pred: np.ndarray, sym_mats: np.ndarray) -> np.ndarray:
    # Compute R_rel[n] = R_pred[n] @ R_gt[n]^T  (N, 3, 3)
    # Reference scalar form: trace(sym[k] @ R_pred @ R_gt^T)
    # Vectorized: traces[n, k] = einsum('kij,nji->nk', sym_mats, R_rel)
    R_rel = np.einsum("nij,nkj->nik", R_pred, R_gt)  # R_pred @ R_gt^T
    traces = np.einsum("kij,nji->nk", sym_mats, R_rel)
    angles = np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0))
    return np.degrees(np.min(angles, axis=1)).astype(np.float32)


# -----------------------------------------------------------------------------
# CryoPARES cone assignment (optional; retained for compatibility)
# -----------------------------------------------------------------------------
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
                f"n_so3_pixels ({self.n_so3_pixels}) not divisible by n_cones ({self.n_cones})."
            )
        self.n_psi = int(self.n_so3_pixels // self.n_cones)

    def rotmats_to_cone_id(self, rotmats_np: np.ndarray) -> np.ndarray:
        rotmats = torch.from_numpy(rotmats_np).to(self.device)
        with torch.no_grad():
            _, so3_indices = self.so3_grid.nearest_rotmat_idx(rotmats, reduce_sym=True)
        so3_indices = so3_indices.view(-1).cpu().numpy().astype(np.int64)
        return so3_indices // self.n_psi


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
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


def pearson_corr(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if len(x) < 3:
        return None
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m] - x[m].mean()
    y = y[m] - y[m].mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom == 0.0:
        return None
    return float((x @ y) / denom)


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(p) & np.isfinite(y)
    p = p[m]
    y = y[m]
    if len(p) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(p)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        sel = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        if not np.any(sel):
            continue
        conf = float(np.mean(p[sel]))
        acc = float(np.mean(y[sel]))
        w = float(np.sum(sel)) / n
        ece += w * abs(acc - conf)
    return float(ece)


def safe_roc_auc(y: np.ndarray, s: np.ndarray) -> Optional[float]:
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    m = np.isfinite(y) & np.isfinite(s)
    y = y[m]
    s = s[m]
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, s))


def safe_pr_auc(y: np.ndarray, s: np.ndarray) -> Optional[float]:
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    m = np.isfinite(y) & np.isfinite(s)
    y = y[m]
    s = s[m]
    if len(np.unique(y)) < 2:
        return None
    return float(average_precision_score(y, s))


def safe_brier(p: np.ndarray, y: np.ndarray) -> Optional[float]:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    m = np.isfinite(p) & np.isfinite(y)
    p = p[m]
    y = y[m]
    if len(p) == 0:
        return None
    return float(brier_score_loss(y, p))


def safe_logloss(p: np.ndarray, y: np.ndarray) -> Optional[float]:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    m = np.isfinite(p) & np.isfinite(y)
    p = p[m]
    y = y[m]
    if len(p) == 0 or len(np.unique(y)) < 2:
        return None
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return float(log_loss(y, np.c_[1.0 - p, p], labels=[0, 1]))


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


# -----------------------------------------------------------------------------
# Calibration fit
# -----------------------------------------------------------------------------
def fit_logistic_calibrator(scores: np.ndarray, y: np.ndarray, seed: int = 0, test_size: float = 0.3):
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=int)
    m = np.isfinite(scores) & np.isfinite(y)
    scores = scores[m]
    y = y[m]

    if len(scores) < 10 or len(np.unique(y)) < 2:
        return None

    X = scores.reshape(-1, 1)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    clf = LogisticRegression(solver="lbfgs")
    clf.fit(Xtr, ytr)
    p_cal = clf.predict_proba(Xte)[:, 1]

    smin = float(np.min(Xtr))
    smax = float(np.max(Xtr))
    p_raw = (Xte[:, 0] - smin) / (smax - smin + 1e-12)
    p_raw = np.clip(p_raw, 1e-6, 1.0 - 1e-6)

    out = {
        "a": float(clf.coef_.ravel()[0]),
        "b": float(clf.intercept_.ravel()[0]),
        "score_test": Xte[:, 0].copy(),
        "y_test": yte.copy(),
        "p_cal_test": p_cal.copy(),
        "p_raw_test": p_raw.copy(),
        "roc_auc_raw": safe_roc_auc(yte, Xte[:, 0]),
        "pr_auc_raw": safe_pr_auc(yte, Xte[:, 0]),
        "roc_auc_cal": safe_roc_auc(yte, p_cal),
        "pr_auc_cal": safe_pr_auc(yte, p_cal),
        "brier_raw": safe_brier(p_raw, yte),
        "brier_cal": safe_brier(p_cal, yte),
        "logloss_raw": safe_logloss(p_raw, yte),
        "logloss_cal": safe_logloss(p_cal, yte),
        "ece_raw": expected_calibration_error(p_raw, yte, n_bins=15),
        "ece_cal": expected_calibration_error(p_cal, yte, n_bins=15),
    }
    return out


def logistic_predict(scores: np.ndarray, a: float, b: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    z = a * scores + b
    return sigmoid(z)


# -----------------------------------------------------------------------------
# Binned plots / summaries
# -----------------------------------------------------------------------------
def quantile_binned_stats(scores: np.ndarray, y: np.ndarray, err_deg: np.ndarray, n_bins: int = 15) -> pd.DataFrame:
    df = pd.DataFrame({"score": scores, "correct": y.astype(int), "err_deg": err_deg})
    df = df[np.isfinite(df["score"]) & np.isfinite(df["err_deg"]) & np.isfinite(df["correct"])]
    if len(df) == 0:
        return pd.DataFrame(columns=["n", "score_mean", "acc", "err_mean", "err_median"])

    q = min(n_bins, max(2, len(df) // 200))
    df["bin"] = pd.qcut(df["score"], q=q, duplicates="drop")
    g = df.groupby("bin", observed=True)
    return g.agg(
        n=("score", "size"),
        score_mean=("score", "mean"),
        acc=("correct", "mean"),
        err_mean=("err_deg", "mean"),
        err_median=("err_deg", "median"),
    ).reset_index(drop=True)


def reliability_points_quantile(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.DataFrame({"p": p, "y": y})
    df = df[np.isfinite(df["p"]) & np.isfinite(df["y"])]
    if len(df) == 0:
        return np.array([]), np.array([])
    q = min(n_bins, max(2, len(df) // 200))
    df["bin"] = pd.qcut(df["p"], q=q, duplicates="drop")
    g = df.groupby("bin", observed=True)
    return g["p"].mean().to_numpy(), g["y"].mean().to_numpy()


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def save_score_plots(out_dir: str, prefix: str, key: str, binned: pd.DataFrame,
                     p_test: Optional[np.ndarray], y_test: Optional[np.ndarray],
                     reliability_bins: int = 10):
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    if key == "s_nnet":
        color, marker = COLOR_SNNET, MARKER_SNNET
    else:
        color, marker = COLOR_ZDIR, MARKER_ZDIR

    plt.figure()
    plt.plot(binned["score_mean"], binned["acc"], marker=marker, color=color)
    plt.xlabel(f"Mean {LABELS[key]['raw']} (bin)")
    plt.ylabel(r"Empirical $P(\mathrm{correct})$")
    plt.title(f"Accuracy vs {LABELS[key]['raw']}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}.{key}.acc_vs_score.png"), dpi=200)
    plt.close()

    if p_test is not None and y_test is not None:
        confs, accs = reliability_points_quantile(p_test, y_test, n_bins=reliability_bins)
        plt.figure()
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="ideal")
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


# -----------------------------------------------------------------------------
# Main evaluation
# -----------------------------------------------------------------------------
def evaluate_one_score(key: str,
                       per_particle: pd.DataFrame,
                       out_dir: str,
                       out_prefix: str,
                       n_bins: int,
                       reliability_bins: int,
                       seed: int,
                       test_size: float,
                       do_plots: bool,
                       zdir_plot_min: Optional[float] = -5.0,
                       zdir_plot_min_disable: bool = False,
                       zdir_winsor_q: float = 0.005,
                       zdir_winsor_for: str = "fit,plots") -> Optional[Dict[str, Any]]:
    if key not in per_particle.columns:
        return None

    raw = pd.to_numeric(per_particle[key], errors="coerce").to_numpy(dtype=float)
    err = pd.to_numeric(per_particle["err_deg"], errors="coerce").to_numpy(dtype=float)
    correct = per_particle["correct"].to_numpy(dtype=int)

    mask = np.isfinite(raw) & np.isfinite(err)
    scores = raw[mask]
    err = err[mask]
    correct = correct[mask]

    if len(scores) == 0:
        return None

    s_for_fit = scores.copy()
    s_for_plot = scores.copy()
    apply = {x.strip() for x in zdir_winsor_for.split(",") if x.strip()}
    if key == "Z_dir" and zdir_winsor_q > 0:
        if "fit" in apply:
            lo = float(np.quantile(s_for_fit, zdir_winsor_q))
            hi = float(np.quantile(s_for_fit, 1.0 - zdir_winsor_q))
            s_for_fit = np.clip(s_for_fit, lo, hi)
        if "plots" in apply:
            lo = float(np.quantile(s_for_plot, zdir_winsor_q))
            hi = float(np.quantile(s_for_plot, 1.0 - zdir_winsor_q))
            s_for_plot = np.clip(s_for_plot, lo, hi)

    fit = fit_logistic_calibrator(s_for_fit, correct, seed=seed, test_size=test_size)
    if fit is None:
        return None

    lp_all = np.full_like(raw, np.nan, dtype=float)
    lp_all[mask] = logistic_predict(s_for_fit if len(s_for_fit) == len(scores) else scores, fit["a"], fit["b"])
    if key == "s_nnet":
        per_particle["LP_nnet"] = lp_all
    else:
        per_particle["LP_dir"] = lp_all

    spearman = spearman_corr(scores, err)
    roc_auc = safe_roc_auc(correct, scores)
    pr_auc = safe_pr_auc(correct, scores)

    plot_scores = s_for_plot.copy()
    plot_err = err.copy()
    plot_correct = correct.copy()
    if key == "Z_dir" and not zdir_plot_min_disable and zdir_plot_min is not None:
        m_plot = plot_scores > zdir_plot_min
        if np.sum(m_plot) >= 10:
            plot_scores = plot_scores[m_plot]
            plot_err = plot_err[m_plot]
            plot_correct = plot_correct[m_plot]
            print(f"[INFO] For accuracy-vs-{key} plot only: using {len(plot_scores)} samples with {key} > {zdir_plot_min:g}")

    binned = quantile_binned_stats(plot_scores, plot_correct, plot_err, n_bins=n_bins)
    binned_path = os.path.join(out_dir, f"{out_prefix}.{key}.binned.csv")
    binned.to_csv(binned_path, index=False)

    print(f"\n=== {key} ===")
    print(f"n={len(scores)}")
    if spearman is not None:
        print(f"Spearman(score, error_deg) = {spearman:.4f}  (more negative is better)")
    if roc_auc is not None:
        print(f"ROC-AUC (raw score) = {roc_auc:.4f}")
    if pr_auc is not None:
        print(f"PR-AUC (raw score) = {pr_auc:.4f}")
    print(f"Binned curve saved: {binned_path}")

    print(f"Held-out calibration split (test_size={test_size:.2f})")
    print(f"  ROC-AUC (raw test) = {fit['roc_auc_raw']:.4f}")
    print(f"  PR-AUC  (raw test) = {fit['pr_auc_raw']:.4f}")
    print(f"  ROC-AUC (cal test) = {fit['roc_auc_cal']:.4f}")
    print(f"  PR-AUC  (cal test) = {fit['pr_auc_cal']:.4f}")
    print(f"  Brier   raw={fit['brier_raw']:.4f}  cal={fit['brier_cal']:.4f}")
    print(f"  LogLoss raw={fit['logloss_raw']:.4f}  cal={fit['logloss_cal']:.4f}")
    print(f"  ECE     raw={fit['ece_raw']:.4f}  cal={fit['ece_cal']:.4f}")

    if do_plots:
        save_score_plots(out_dir, out_prefix, key, binned, fit["p_cal_test"], fit["y_test"], reliability_bins=reliability_bins)

    return {
        "key": key,
        "spearman": spearman,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "test_roc_auc_raw": fit["roc_auc_raw"],
        "test_pr_auc_raw": fit["pr_auc_raw"],
        "test_roc_auc_cal": fit["roc_auc_cal"],
        "test_pr_auc_cal": fit["pr_auc_cal"],
        "test_brier_raw": fit["brier_raw"],
        "test_brier_cal": fit["brier_cal"],
        "test_logloss_raw": fit["logloss_raw"],
        "test_logloss_cal": fit["logloss_cal"],
        "test_ece_raw": fit["ece_raw"],
        "test_ece_cal": fit["ece_cal"],
    }


# -----------------------------------------------------------------------------
# RELION baseline (rlnMaxValueProbDistribution) evaluation
# -----------------------------------------------------------------------------
def evaluate_relion_baseline(per_particle: pd.DataFrame,
                             out_dir: str,
                             out_prefix: str,
                             n_bins: int,
                             reliability_bins: int,
                             seed: int,
                             test_size: float,
                             do_plots: bool) -> Optional[Dict[str, Any]]:
    """
    Evaluate rlnMaxValueProbDistribution (MVPD) as a confidence proxy, using the
    same metrics as evaluate_one_score, and additionally report its Spearman and
    Pearson correlations with s_nnet and Z_dir where available.

    MVPD is RELION's per-particle maximum value of the probability distribution
    from the E-step of the gold-standard refinement. It is the natural RELION
    baseline for any learned confidence score.
    """
    key = "mvpd"
    col = "rlnMaxValueProbDistribution"

    if key not in per_particle.columns:
        return None

    raw     = pd.to_numeric(per_particle[key], errors="coerce").to_numpy(dtype=float)
    err     = pd.to_numeric(per_particle["err_deg"], errors="coerce").to_numpy(dtype=float)
    correct = per_particle["correct"].to_numpy(dtype=int)

    mask    = np.isfinite(raw) & np.isfinite(err)
    scores  = raw[mask]
    err_m   = err[mask]
    correct_m = correct[mask]

    if len(scores) == 0:
        return None

    spearman_err = spearman_corr(scores, err_m)
    roc_auc      = safe_roc_auc(correct_m, scores)
    pr_auc       = safe_pr_auc(correct_m, scores)

    fit = fit_logistic_calibrator(scores, correct_m, seed=seed, test_size=test_size)

    binned      = quantile_binned_stats(scores, correct_m, err_m, n_bins=n_bins)
    binned_path = os.path.join(out_dir, f"{out_prefix}.{key}.binned.csv")
    binned.to_csv(binned_path, index=False)

    print(f"\n=== {col} (RELION baseline) ===")
    print(f"n={len(scores)}")
    if spearman_err is not None:
        print(f"Spearman({col}, error_deg)  = {spearman_err:.4f}  (more negative is better)")
    if roc_auc is not None:
        print(f"ROC-AUC (raw) = {roc_auc:.4f}")
    if pr_auc is not None:
        print(f"PR-AUC  (raw) = {pr_auc:.4f}")

    # Cross-correlations with CryoPARES scores
    cross: Dict[str, Any] = {}
    for other_key, other_col in [("s_nnet", "s_nnet"), ("Z_dir", "Z_dir")]:
        if other_key not in per_particle.columns:
            continue
        other_raw = pd.to_numeric(per_particle[other_key], errors="coerce").to_numpy(dtype=float)
        both_mask = mask & np.isfinite(other_raw)
        if both_mask.sum() < 3:
            continue
        a = raw[both_mask]
        b = other_raw[both_mask]
        sp = spearman_corr(a, b)
        pe = pearson_corr(a, b)
        cross[f"spearman_vs_{other_key}"] = sp
        cross[f"pearson_vs_{other_key}"]  = pe
        print(f"Correlation with {other_key}:")
        print(f"  Spearman = {sp:.4f}" if sp is not None else "  Spearman = n/a")
        print(f"  Pearson  = {pe:.4f}" if pe is not None else "  Pearson  = n/a")

    out: Dict[str, Any] = {
        "key":      key,
        "spearman": spearman_err,
        "roc_auc":  roc_auc,
        "pr_auc":   pr_auc,
        **cross,
    }

    if fit is not None:
        print(f"Held-out calibration split (test_size={test_size:.2f})")
        print(f"  ROC-AUC (raw test) = {fit['roc_auc_raw']:.4f}")
        print(f"  PR-AUC  (raw test) = {fit['pr_auc_raw']:.4f}")
        print(f"  ROC-AUC (cal test) = {fit['roc_auc_cal']:.4f}")
        print(f"  PR-AUC  (cal test) = {fit['pr_auc_cal']:.4f}")
        print(f"  Brier   raw={fit['brier_raw']:.4f}  cal={fit['brier_cal']:.4f}")
        print(f"  LogLoss raw={fit['logloss_raw']:.4f}  cal={fit['logloss_cal']:.4f}")
        print(f"  ECE     raw={fit['ece_raw']:.4f}  cal={fit['ece_cal']:.4f}")
        out.update({
            "test_roc_auc_raw":  fit["roc_auc_raw"],
            "test_pr_auc_raw":   fit["pr_auc_raw"],
            "test_roc_auc_cal":  fit["roc_auc_cal"],
            "test_pr_auc_cal":   fit["pr_auc_cal"],
            "test_brier_raw":    fit["brier_raw"],
            "test_brier_cal":    fit["brier_cal"],
            "test_logloss_raw":  fit["logloss_raw"],
            "test_logloss_cal":  fit["logloss_cal"],
            "test_ece_raw":      fit["ece_raw"],
            "test_ece_cal":      fit["ece_cal"],
        })

    if do_plots and fit is not None:
        save_score_plots(out_dir, out_prefix, "s_nnet", binned,
                         fit["p_cal_test"], fit["y_test"],
                         reliability_bins=reliability_bins)

    print(f"Binned curve saved: {binned_path}")
    return out
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--gt", type=str, default=None, help="GT STAR path (RELION angles columns)")
    ap.add_argument("--pred", type=str, default=None, help="Pred STAR path (angles + scores)")
    ap.add_argument("--single", type=str, default=None, help="Single STAR containing both predicted angles and GT angles (specify GT cols with --gt_rot/tilt/psi)")

    ap.add_argument("--merge_key", type=str, default="rlnImageName", help="Column used for merging when using --gt/--pred")

    ap.add_argument("--pred_rot", type=str, default="rlnAngleRot")
    ap.add_argument("--pred_tilt", type=str, default="rlnAngleTilt")
    ap.add_argument("--pred_psi", type=str, default="rlnAnglePsi")

    ap.add_argument("--gt_rot", type=str, default="rlnAngleRot")
    ap.add_argument("--gt_tilt", type=str, default="rlnAngleTilt")
    ap.add_argument("--gt_psi", type=str, default="rlnAnglePsi")

    ap.add_argument("--fom_col", type=str, default="rlnParticleFigureOfMerit", help="STAR column used for s_nnet")
    ap.add_argument("--dirz_col", type=str, default="rlnDirectionalZscore", help="STAR column used for Z_dir")
    ap.add_argument("--mvpd_col", type=str, default="rlnMaxValueProbDistribution",
                    help="STAR column for RELION baseline score (rlnMaxValueProbDistribution)")

    ap.add_argument("--symmetry", type=str, required=True)
    ap.add_argument("--hp_order", type=int, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--correct_deg", type=float, default=5.0)

    ap.add_argument("--test_size", type=float, default=0.3, help="Fraction reserved for held-out calibration evaluation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_bins", type=int, default=15)
    ap.add_argument("--reliability_bins", type=int, default=10)

    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--out_prefix", type=str, default="calibration")
    ap.add_argument("--plots", action="store_true")

    ap.add_argument("--zdir_plot_min", type=float, default=-5.0)
    ap.add_argument("--zdir_plot_min_disable", action="store_true")
    ap.add_argument("--zdir_winsor_q", type=float, default=0.005)
    ap.add_argument("--zdir_winsor_for", type=str, default="fit,plots")

    args = ap.parse_args()

    if args.single is None:
        if args.gt is None or args.pred is None:
            raise ValueError("Provide either --single OR both --gt and --pred.")
    else:
        if args.gt is not None or args.pred is not None:
            raise ValueError("Use either --single OR --gt/--pred, not both.")

    os.makedirs(args.out_dir, exist_ok=True)

    # Load data and angles
    if args.single:
        df = read_star_as_df(args.single)
        require_cols(df, [args.pred_rot, args.pred_tilt, args.pred_psi], "--single predicted angles")
        require_cols(df, [args.gt_rot, args.gt_tilt, args.gt_psi], "--single GT *_ori angles")
        pred_angles = df[[args.pred_rot, args.pred_tilt, args.pred_psi]].to_numpy(dtype=float)
        gt_angles = df[[args.gt_rot, args.gt_tilt, args.gt_psi]].to_numpy(dtype=float)
    else:
        gt_df = read_star_as_df(args.gt)
        pred_df = read_star_as_df(args.pred)
        # BUG FIX: check GT file for GT angle columns, not pred angle columns
        require_cols(gt_df, [args.gt_rot, args.gt_tilt, args.gt_psi], "--gt angles")
        require_cols(pred_df, [args.pred_rot, args.pred_tilt, args.pred_psi], "--pred angles")
        # BUG FIX: pass GT angle column names so merge_gt_pred row-order fallback uses them
        merged = merge_gt_pred(gt_df, pred_df, args.merge_key,
                               [args.gt_rot, args.gt_tilt, args.gt_psi])
        print(f"[INFO] Matched particles after merge: {len(merged)}"
              f" (pred={len(pred_df)}, gt={len(gt_df)})")
        if len(merged) == 0:
            raise ValueError("Merge produced 0 particles. Cannot proceed.")
        df = merged
        pred_angles = merged[[args.pred_rot, args.pred_tilt, args.pred_psi]].to_numpy(dtype=float)
        # Resolve GT angle column names in the merged dataframe.
        # merge() uses suffixes=("", "_gt") on the pred (left) and gt (right) sides.
        # A GT column keeps its original name unless it collides with a pred column,
        # in which case pandas appends "_gt". The row-order fallback in merge_gt_pred
        # always stores GT columns as gt_col+"_gt".
        use_merge_key = args.merge_key in gt_df.columns and args.merge_key in pred_df.columns
        gt_cols_raw = [args.gt_rot, args.gt_tilt, args.gt_psi]
        pred_cols_raw = [args.pred_rot, args.pred_tilt, args.pred_psi]
        if use_merge_key:
            # A GT col gets _gt suffix only if it collides with a pred col name
            gt_angle_cols_in_merged = [
                (c + "_gt" if c in pred_cols_raw else c) for c in gt_cols_raw
            ]
        else:
            # Row-order fallback always appends _gt
            gt_angle_cols_in_merged = [c + "_gt" for c in gt_cols_raw]
        require_cols(merged, gt_angle_cols_in_merged, "merged GT angles")
        gt_angles = merged[gt_angle_cols_in_merged].to_numpy(dtype=float)

    sym_mats = get_symmetry_matrices(args.symmetry)
    R_pred = eulers_to_rotmats_relion(pred_angles)
    R_gt = eulers_to_rotmats_relion(gt_angles)
    err = angular_errors_with_sym_all(R_gt, R_pred, sym_mats)
    correct = (err <= args.correct_deg).astype(np.int32)

    print(f"[INFO] RELION_EULER_CONVENTION = {RELION_EULER_CONVENTION}")
    print(f"[INFO] symmetry group size = {sym_mats.shape[0]}")
    print(f"[INFO] correctness threshold = {args.correct_deg:.2f} deg")
    print(f"[INFO] Global % < {args.correct_deg:g} deg = {100.0 * float(np.mean(err < args.correct_deg)):.2f}%")

    assigner = CryoPARESConeAssigner(symmetry=args.symmetry, hp_order=args.hp_order, device=args.device)
    cone_id = assigner.rotmats_to_cone_id(R_pred)
    print(f"[INFO] cones: hp_order={args.hp_order} => n_cones={assigner.n_cones}, n_psi={assigner.n_psi}")

    per_particle = pd.DataFrame({
        "err_deg": err,
        "correct": correct,
        "cone_id": cone_id.astype(np.int64),
    })

    if args.fom_col in df.columns:
        per_particle["s_nnet"] = pd.to_numeric(df[args.fom_col], errors="coerce")
    else:
        print(f"[WARN] s_nnet source column not found: {args.fom_col}")

    if args.dirz_col in df.columns:
        per_particle["Z_dir"] = pd.to_numeric(df[args.dirz_col], errors="coerce")
    else:
        print(f"[WARN] Z_dir source column not found: {args.dirz_col}")

    # rlnMaxValueProbDistribution comes from the GT/refinement STAR.
    # After merge it may be suffixed with _gt if it collided with a pred column
    # (unlikely for this column, but we check both names defensively).
    _mvpd_col_in_df = None
    for candidate in [args.mvpd_col, args.mvpd_col + "_gt"]:
        if candidate in df.columns:
            _mvpd_col_in_df = candidate
            break
    if _mvpd_col_in_df is not None:
        per_particle["mvpd"] = pd.to_numeric(df[_mvpd_col_in_df], errors="coerce")
        print(f"[INFO] RELION baseline loaded from column '{_mvpd_col_in_df}'")
    else:
        print(f"[WARN] RELION baseline column not found: {args.mvpd_col} "
              f"(neither plain nor _gt suffix). Skipping baseline comparison.")

    metrics_rows = []
    if "s_nnet" in per_particle.columns:
        m = evaluate_one_score(
            "s_nnet", per_particle, args.out_dir, args.out_prefix,
            args.n_bins, args.reliability_bins, args.seed, args.test_size,
            args.plots, args.zdir_plot_min, args.zdir_plot_min_disable,
            args.zdir_winsor_q, args.zdir_winsor_for,
        )
        if m is not None:
            metrics_rows.append(m)
    if "Z_dir" in per_particle.columns:
        m = evaluate_one_score(
            "Z_dir", per_particle, args.out_dir, args.out_prefix,
            args.n_bins, args.reliability_bins, args.seed, args.test_size,
            args.plots, args.zdir_plot_min, args.zdir_plot_min_disable,
            args.zdir_winsor_q, args.zdir_winsor_for,
        )
        if m is not None:
            metrics_rows.append(m)
    if "mvpd" in per_particle.columns:
        m = evaluate_relion_baseline(
            per_particle, args.out_dir, args.out_prefix,
            args.n_bins, args.reliability_bins, args.seed, args.test_size,
            args.plots,
        )
        if m is not None:
            metrics_rows.append(m)

    per_particle_path = os.path.join(args.out_dir, f"{args.out_prefix}.per_particle.csv")
    per_particle.to_csv(per_particle_path, index=False)
    print(f"[OK] per-particle CSV: {per_particle_path}")

    metrics_path = os.path.join(args.out_dir, f"{args.out_prefix}.metrics.csv")
    if len(metrics_rows) > 0:
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df.to_csv(metrics_path, index=False)
        print(f"[OK] metrics CSV: {metrics_path}")

    print("\n=== compact summary ===")
    print(f"mean error = {float(np.mean(err)):.3f} deg")
    print(f"median error = {float(np.median(err)):.3f} deg")
    print(f"% < {args.correct_deg:g} deg = {100.0 * float(np.mean(err <= args.correct_deg)):.2f}%")
    print(f"% < 10 deg = {100.0 * float(np.mean(err <= 10.0)):.2f}%")

    if args.plots:
        print(f"[OK] plots written under {args.out_dir} with prefix {args.out_prefix}")


if __name__ == "__main__":
    main()