import starfile
import numpy as np
from cryoPARES.constants import RELION_EULER_CONVENTION
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
import os
import warnings
import matplotlib.pyplot as plt
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MICROGRAPH_COL_CANDIDATES = ["rlnMicrographName", "rlnImageName"]
COORD_COL_CANDIDATES = [
    ("rlnCoordinateX", "rlnCoordinateY"),
    ("rlnMicrographCoordinatesX", "rlnMicrographCoordinatesY"),
]


# ---------------------------------------------------------------------------
# Starfile I/O
# ---------------------------------------------------------------------------

def load_starfile(filename, particles_key="particles"):
    """Load a RELION starfile and return the particles dataframe."""
    try:
        data = starfile.read(filename)
        if particles_key is None:
            particles_key = list(data.keys())[-1]
        return data[particles_key]
    except Exception as e:
        raise Exception(f"Error reading starfile {filename}: {str(e)}")


# ---------------------------------------------------------------------------
# Primary match strategy: rlnImageName
# ---------------------------------------------------------------------------

def extract_particle_id(image_name):
    """Extract particle number from RELION image name (e.g., '0001@/path/stack.mrcs')."""
    num, stackname = image_name.split('@')
    return str(int(num)) + "_" + os.path.basename(stackname)


def match_particles_by_image_name(df1, df2, allow_partial=True):
    """
    Match particles between two dataframes via rlnImageName and sort them.
    Returns (df1_sorted, df2_sorted, n_matched) or raises if no match found.
    """
    df1 = df1.copy()
    df2 = df2.copy()

    df1['particle_id'] = df1['rlnImageName'].apply(extract_particle_id)
    df2['particle_id'] = df2['rlnImageName'].apply(extract_particle_id)

    particles1 = set(df1['particle_id'])
    particles2 = set(df2['particle_id'])

    if allow_partial:
        common = particles1 & particles2
        if not common:
            raise ValueError("No matching particles found via rlnImageName.")

        df1_f = df1[df1['particle_id'].isin(common)].sort_values('particle_id').reset_index(drop=True)
        df2_f = df2[df2['particle_id'].isin(common)].sort_values('particle_id').reset_index(drop=True)

        n1, n2 = len(particles1), len(particles2)
        if len(common) < n1 or len(common) < n2:
            warnings.warn(
                f"Partial match via rlnImageName: {len(common)} common out of "
                f"{n1} / {n2}. Statistics computed on the matching subset."
            )
    else:
        if particles1 != particles2:
            raise ValueError(
                f"Starfiles contain different particles.\n"
                f"Only in first: {len(particles1 - particles2)}\n"
                f"Only in second: {len(particles2 - particles1)}"
            )
        df1_f = df1.sort_values('particle_id').reset_index(drop=True)
        df2_f = df2.sort_values('particle_id').reset_index(drop=True)

    if not (df1_f['particle_id'] == df2_f['particle_id']).all():
        raise ValueError("Failed to align particles after sorting by rlnImageName.")

    return df1_f, df2_f, len(df1_f)


# ---------------------------------------------------------------------------
# Fallback match strategy: micrograph basename + coordinates
# ---------------------------------------------------------------------------

def _extract_micrograph_basename(series: pd.Series) -> pd.Series:
    """
    Normalise micrograph identifiers to a bare filename, handling:
      - full paths, relative paths, bare names
      - stack references:  000006@raw_data/.../Foo.mrcs  ->  Foo.mrcs
    """
    def _norm(val: str) -> str:
        s = str(val)
        if "@" in s:
            s = s.split("@", 1)[1]
        return os.path.basename(s)

    return series.map(_norm)


def _detect_coord_columns(df: pd.DataFrame):
    """Return (micrograph_col, x_col, y_col) or raise."""
    mg_col = next((c for c in MICROGRAPH_COL_CANDIDATES if c in df.columns), None)
    if mg_col is None:
        raise ValueError(f"No micrograph column found. Tried: {MICROGRAPH_COL_CANDIDATES}")

    coord_pair = next(
        ((x, y) for x, y in COORD_COL_CANDIDATES if x in df.columns and y in df.columns),
        None,
    )
    if coord_pair is None:
        raise ValueError(f"No coordinate columns found. Tried: {COORD_COL_CANDIDATES}")

    return mg_col, coord_pair[0], coord_pair[1]


def _prepare_coord_frame(df: pd.DataFrame, mg_col, x_col, y_col,
                         scale: float, bin_size: float) -> pd.DataFrame:
    out = df[[mg_col, x_col, y_col]].copy()
    out = out.rename(columns={mg_col: "__mg", x_col: "__x_raw", y_col: "__y_raw"})
    out["__mg_key"] = _extract_micrograph_basename(out["__mg"])
    out["__x"] = pd.to_numeric(out["__x_raw"], errors="coerce") * scale
    out["__y"] = pd.to_numeric(out["__y_raw"], errors="coerce") * scale
    if out["__x"].isna().any() or out["__y"].isna().any():
        raise ValueError("Non-numeric coordinates encountered.")
    out["__row_id"] = np.arange(len(out), dtype=np.int64)
    out["__bin_x"] = np.floor(out["__x"] / bin_size).astype(np.int64)
    out["__bin_y"] = np.floor(out["__y"] / bin_size).astype(np.int64)
    return out


def match_particles_by_coordinates(df1, df2,
                                   margin: float = 5.0,
                                   scale1: float = 1.0,
                                   scale2: float = 1.0):
    """
    Match particles by micrograph basename + coordinate proximity.

    Returns (df1_matched, df2_matched, n_matched) where rows are aligned
    1-to-1 by the nearest neighbour within `margin` (in the scaled space).

    This is intentionally a many-to-one-safe approach: each row in df1 is
    matched to at most one row in df2 (the first hit within the margin).
    """
    mg1, x1, y1 = _detect_coord_columns(df1)
    mg2, x2, y2 = _detect_coord_columns(df2)

    bin_size = max(1.0, float(margin))

    a = _prepare_coord_frame(df1, mg1, x1, y1, scale1, bin_size)
    b = _prepare_coord_frame(df2, mg2, x2, y2, scale2, bin_size)

    # Sanity-check basename overlap.
    common_mgs = set(a["__mg_key"].unique()) & set(b["__mg_key"].unique())
    if not common_mgs:
        raise ValueError(
            "Coordinate fallback: no shared micrograph basenames found. "
            "Check that both files refer to the same dataset and consider "
            "adjusting --scale1 / --scale2."
        )
    warnings.warn(
        f"Coordinate fallback: matching on {len(common_mgs)} shared micrographs "
        f"with margin={margin} px (scale1={scale1}, scale2={scale2})."
    )

    # Expand b into the 3x3 neighbourhood of each bin.
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
        raise ValueError("Coordinate fallback: no candidate pairs found after binning.")

    within = merged[
            ((merged["__x_a"] - merged["__x_b"]).abs() <= margin)
            & ((merged["__y_a"] - merged["__y_b"]).abs() <= margin)
    ].copy()

    if within.empty:
        raise ValueError(
            f"Coordinate fallback: no pairs within margin={margin} px. "
            "Try increasing --coord_margin."
        )

    # Keep the closest match in b for each row in a.
    within["__dist"] = np.hypot(
        within["__x_a"] - within["__x_b"],
        within["__y_a"] - within["__y_b"],
    )
    best = (
        within.sort_values("__dist")
        .drop_duplicates(subset="__row_id_a", keep="first")
    )

    idx1 = best["__row_id_a"].to_numpy()
    idx2 = best["__row_id_b"].to_numpy()

    df1_matched = df1.iloc[idx1].reset_index(drop=True)
    df2_matched = df2.iloc[idx2].reset_index(drop=True)

    warnings.warn(
        f"Coordinate fallback matched {len(idx1)} particles "
        f"(df1 had {len(df1)}, df2 had {len(df2)})."
    )

    return df1_matched, df2_matched, len(idx1)


# ---------------------------------------------------------------------------
# Unified match dispatcher
# ---------------------------------------------------------------------------

def match_particles(df1, df2, allow_partial=True,
                    coord_margin: float = 5.0,
                    scale1: float = 1.0,
                    scale2: float = 1.0):
    """
    Match particles between two dataframes.

    Strategy 1 (primary): rlnImageName-based exact match.
    Strategy 2 (fallback): micrograph basename + coordinate proximity,
        used when rlnImageName is absent in either dataframe or yields
        zero matches.

    coord_margin, scale1, scale2 are forwarded to the coordinate fallback.
    """
    has_image_name = (
        'rlnImageName' in df1.columns and 'rlnImageName' in df2.columns
    )

    if has_image_name:
        try:
            df1_m, df2_m, n = match_particles_by_image_name(df1, df2, allow_partial)
            print(f"Matched {n} particles via rlnImageName.")
            return df1_m, df2_m
        except ValueError as e:
            warnings.warn(
                f"rlnImageName matching failed ({e}); "
                "falling back to coordinate-based matching."
            )
    else:
        warnings.warn(
            "rlnImageName not present in both dataframes; "
            "using coordinate-based matching."
        )

    df1_m, df2_m, n = match_particles_by_coordinates(
        df1, df2,
        margin=coord_margin,
        scale1=scale1,
        scale2=scale2,
    )
    print(f"Matched {n} particles via coordinates.")
    return df1_m, df2_m


# ---------------------------------------------------------------------------
# Rotation utilities
# ---------------------------------------------------------------------------

def euler_to_matrix(angles):
    return Rotation.from_euler(RELION_EULER_CONVENTION, angles, degrees=True).as_matrix()


def matrix_to_euler(matrix):
    return Rotation.from_matrix(matrix).as_euler(RELION_EULER_CONVENTION, degrees=True)


def get_symmetry_matrices(sym_group):
    if sym_group.upper() == 'C1':
        return [np.eye(3)]
    group = f"{sym_group[0].upper()}{sym_group[1:]}" if sym_group[0].upper() in ['C', 'D'] else sym_group.upper()
    try:
        return Rotation.create_group(group).as_matrix()
    except ValueError as e:
        raise ValueError(f"Invalid symmetry group: {sym_group}. Error: {str(e)}")


def calculate_angular_difference(R1, R2, sym_matrices):
    R_diffs = np.einsum('ijk,kl->ijl', sym_matrices, np.dot(R2, R1.T))
    traces = np.einsum('ijj->i', R_diffs)
    angles = np.arccos(np.clip((traces - 1) / 2, -1.0, 1.0))
    return np.degrees(np.min(angles))


def apply_transformation(angles, transform_matrix):
    R = euler_to_matrix(angles)
    return matrix_to_euler(np.dot(transform_matrix, R))


def optimize_transformation(angles1, angles2, sym_matrices):
    """Vectorised Nelder-Mead: precomputes all rotation matrices to avoid per-call overhead."""
    R1 = Rotation.from_euler(RELION_EULER_CONVENTION, angles1, degrees=True).as_matrix()
    R2 = Rotation.from_euler(RELION_EULER_CONVENTION, angles2, degrees=True).as_matrix()
    sym = np.asarray(sym_matrices)

    def objective(params):
        R_t = Rotation.from_euler(RELION_EULER_CONVENTION, params, degrees=True).as_matrix()
        R2t = np.einsum('ij,kjl->kil', R_t, R2)           # (N,3,3): R_t @ R2[k]
        R_prod = np.einsum('kij,klj->kil', R2t, R1)        # (N,3,3): R2t[k] @ R1[k].T
        R_sym = sym[:, None] @ R_prod[None]                 # (S,N,3,3)
        traces = np.einsum('snii->sn', R_sym)               # (S,N)
        # min angle over symmetry = arccos of max trace (arccos is decreasing)
        min_angles = np.degrees(np.arccos(np.clip((np.max(traces, axis=0) - 1) / 2, -1.0, 1.0)))
        return float(np.mean(min_angles))

    result = minimize(objective, [0, 0, 0], method='Nelder-Mead')
    return Rotation.from_euler(RELION_EULER_CONVENTION, result.x, degrees=True).as_matrix(), result.fun


def optimize_transformation_svd(angles1, angles2, sym_matrices, n_iter=5):
    """Find global frame alignment via Procrustes SVD + EM symmetry assignment.

    O(n_iter * N) — no iterative optimiser, much faster than Nelder-Mead for large N.
    Minimises a Frobenius-distance proxy that is equivalent to the angular metric in practice.
    """
    R1 = Rotation.from_euler(RELION_EULER_CONVENTION, angles1, degrees=True).as_matrix()
    R2 = Rotation.from_euler(RELION_EULER_CONVENTION, angles2, degrees=True).as_matrix()
    sym = np.asarray(sym_matrices)
    sym_T = sym.transpose(0, 2, 1)  # (S,3,3) transposed
    R_t = np.eye(3)

    for _ in range(n_iter):
        R2t = np.einsum('ij,kjl->kil', R_t, R2)            # (N,3,3)
        R_prod = np.einsum('kij,klj->kil', R2t, R1)         # (N,3,3): R2t[k] @ R1[k].T
        # Tr(sym[s] @ R_prod[k]) — max trace = min angle
        traces = np.einsum('sij,kji->sk', sym, R_prod)       # (S,N)
        best_s = np.argmax(traces, axis=0)                   # (N,)
        # Procrustes target: T[k] = sym[best_s[k]].T @ R1[k]
        T = np.einsum('kij,kjl->kil', sym_T[best_s], R1)    # (N,3,3)
        M = np.einsum('kij,klj->il', T, R2)                 # (3,3): sum_k T[k] @ R2[k].T
        U, _, Vh = np.linalg.svd(M)
        R_t = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vh)]) @ Vh

    # Final mean error
    R2t = np.einsum('ij,kjl->kil', R_t, R2)
    R_prod = np.einsum('kij,klj->kil', R2t, R1)
    R_sym = sym[:, None] @ R_prod[None]
    traces_f = np.einsum('snii->sn', R_sym)
    min_angles = np.degrees(np.arccos(np.clip((np.max(traces_f, axis=0) - 1) / 2, -1.0, 1.0)))
    return R_t, float(np.mean(min_angles))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_angular_errors(angular_errors, output_prefix=None):
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(angular_errors)), angular_errors, alpha=0.6, s=20)
    plt.xlabel('Particle Index')
    plt.ylabel('Angular Error (degrees)')
    plt.title('Angular Errors per Particle')
    plt.grid(True, alpha=0.3)
    mean_e = np.mean(angular_errors)
    med_e = np.median(angular_errors)
    plt.axhline(y=mean_e, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_e:.2f}°')
    plt.axhline(y=med_e, color='orange', linestyle='--', alpha=0.7, label=f'Median: {med_e:.2f}°')
    plt.legend()
    if output_prefix:
        plt.savefig(f'{output_prefix}_angular_errors.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_shift_errors(shift_errors, shift_x_errors, shift_y_errors, output_prefix=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].scatter(range(len(shift_errors)), shift_errors, alpha=0.6, s=20, color='blue')
    axes[0].set_xlabel('Particle Index')
    axes[0].set_ylabel('Shift Error Magnitude (Å)')
    axes[0].set_title('Shift Error Magnitude per Particle')
    axes[0].grid(True, alpha=0.3)
    mean_s = np.mean(shift_errors)
    med_s = np.median(shift_errors)
    axes[0].axhline(y=mean_s, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_s:.2f}Å')
    axes[0].axhline(y=med_s, color='orange', linestyle='--', alpha=0.7, label=f'Median: {med_s:.2f}Å')
    axes[0].legend()

    axes[1].scatter(shift_x_errors, shift_y_errors, alpha=0.6, s=20, color='green')
    axes[1].set_xlabel('X Shift Error (Å)')
    axes[1].set_ylabel('Y Shift Error (Å)')
    axes[1].set_title('X vs Y Shift Errors')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1].axvline(x=0, color='black', linestyle='-', alpha=0.3)

    n_arrows = min(100, len(shift_x_errors))
    step = max(1, len(shift_x_errors) // n_arrows)
    idxs = range(0, len(shift_x_errors), step)
    axes[2].quiver(
        list(idxs), [0] * len(list(idxs)),
        [shift_x_errors[i] for i in idxs],
        [shift_y_errors[i] for i in idxs],
        angles='xy', scale_units='xy', scale=1, alpha=0.7, width=0.003,
    )
    axes[2].set_xlabel('Particle Index')
    axes[2].set_ylabel('Shift Error (Å)')
    axes[2].set_title('Shift Error Vectors (sampled)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if output_prefix:
        plt.savefig(f'{output_prefix}_shift_errors.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_combined_errors(angular_errors, shift_errors, output_prefix=None):
    plt.figure(figsize=(8, 6))
    plt.scatter(angular_errors, shift_errors, alpha=0.6, s=20, color='purple')
    plt.xlabel('Angular Error (degrees)')
    plt.ylabel('Shift Error Magnitude (Å)')
    plt.title('Angular vs Shift Errors')
    plt.grid(True, alpha=0.3)
    corr = np.corrcoef(angular_errors, shift_errors)[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {corr:.3f}',
             transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    if output_prefix:
        plt.savefig(f'{output_prefix}_combined_errors.png', dpi=300, bbox_inches='tight')
    plt.show()


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def analyze_angular_differences(starfile1: str,
                                starfile2: str,
                                sym: str,
                                align_frames: bool = False,
                                align_frames_n_particles: Optional[int] = 2000,
                                save_transformation: Optional[str] = None,
                                read_transformation: Optional[str] = None,
                                plot_angular: bool = False,
                                plot_shifts: bool = False,
                                plot_combined: bool = False,
                                plot_all: bool = False,
                                save_plots: Optional[str] = None,
                                coord_margin: float = 5.0,
                                scale1: float = 1.0,
                                scale2: float = 1.0):
    """
    Compare angular assignments between two RELION starfiles.

    :param starfile1: First starfile
    :param starfile2: Second starfile
    :param sym: Symmetry group (C1, C2, C3, ..., D2, D3, ..., T, O, I)
    :param align_frames: Try to find optimal alignment between reference frames
    :param align_frames_n_particles: Number of particles used to estimate the frame alignment (None = use all). Subsampling speeds up the optimisation with negligible accuracy loss.
    :param save_transformation: Save the frame alignment matrix to this .npy file (only used with --align_frames)
    :param read_transformation: Load a previously saved frame alignment matrix from this .npy file instead of re-running the optimisation
    :param plot_angular: Show scatter plot of angular errors vs particle index
    :param plot_shifts: Show scatter plots of shift errors
    :param plot_combined: Show scatter plot of angular vs shift errors
    :param plot_all: Show all scatter plots
    :param save_plots: Save plots with this prefix
    :param coord_margin: Coordinate match margin in pixels (fallback strategy)
    :param scale1: Multiply starfile1 coordinates by this factor (fallback strategy)
    :param scale2: Multiply starfile2 coordinates by this factor (fallback strategy)
    """
    symmetry = sym.upper()

    df1 = load_starfile(starfile1)
    df2 = load_starfile(starfile2)

    df1, df2 = match_particles(
        df1, df2,
        allow_partial=True,
        coord_margin=coord_margin,
        scale1=scale1,
        scale2=scale2,
    )

    sym_matrices = get_symmetry_matrices(symmetry)

    angles1 = df1[['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']].values
    angles2 = df2[['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']].values

    shifts_available = all(
        col in df1.columns and col in df2.columns
        for col in ['rlnOriginXAngst', 'rlnOriginYAngst']
    )

    shift_x_errors = shift_y_errors = shift_errors = []

    if shifts_available:
        shifts1 = df1[['rlnOriginXAngst', 'rlnOriginYAngst']].values
        shifts2 = df2[['rlnOriginXAngst', 'rlnOriginYAngst']].values
        shift_diffs = shifts1 - shifts2
        shift_x_errors = shift_diffs[:, 0]
        shift_y_errors = shift_diffs[:, 1]
        shift_errors = np.linalg.norm(shift_diffs, axis=1)

    transform_matrix = np.eye(3)
    if read_transformation is not None:
        transform_matrix = np.load(read_transformation)
        print(f"Loaded transformation matrix from {read_transformation}")
    elif align_frames:
        print("Optimizing reference frame alignment...")
        n_total = len(angles1)
        if align_frames_n_particles is not None and n_total > align_frames_n_particles:
            idx = np.random.default_rng().choice(n_total, align_frames_n_particles, replace=False)
            a1_opt, a2_opt = angles1[idx], angles2[idx]
            print(f"  Using {align_frames_n_particles} of {n_total} particles for alignment.")
        else:
            a1_opt, a2_opt = angles1, angles2
        transform_matrix, min_error = optimize_transformation_svd(a1_opt, a2_opt, sym_matrices)
        print(f"Optimal alignment found with mean error: {min_error:.2f}°")
        if save_transformation is not None:
            np.save(save_transformation, transform_matrix)
            print(f"Saved transformation matrix to {save_transformation}")

    if not np.allclose(transform_matrix, np.eye(3)):
        # Vectorised application of the transform to all angles2
        R2_all = Rotation.from_euler(RELION_EULER_CONVENTION, angles2, degrees=True).as_matrix()
        R2_transformed = np.einsum('ij,kjl->kil', transform_matrix, R2_all)
        angles2 = Rotation.from_matrix(R2_transformed).as_euler(RELION_EULER_CONVENTION, degrees=True)

    # Vectorised angular difference computation (avoids Python loops over particles)
    _sym = np.asarray(sym_matrices)
    R1_mats = Rotation.from_euler(RELION_EULER_CONVENTION, angles1, degrees=True).as_matrix()
    R2_mats = Rotation.from_euler(RELION_EULER_CONVENTION, angles2, degrees=True).as_matrix()
    R_prod = np.einsum('kij,klj->kil', R2_mats, R1_mats)   # R2[k] @ R1[k].T, (N,3,3)
    R_sym_all = _sym[:, None] @ R_prod[None]                 # (S,N,3,3)
    traces_all = np.einsum('snii->sn', R_sym_all)            # (S,N)
    angular_diffs = np.degrees(np.arccos(np.clip((np.max(traces_all, axis=0) - 1) / 2, -1.0, 1.0)))

    stats = {
        'mean': np.mean(angular_diffs),
        'std': np.std(angular_diffs),
        'median': np.median(angular_diffs),
        'iqr': np.percentile(angular_diffs, 75) - np.percentile(angular_diffs, 25),
        'percent_below_5': np.mean(angular_diffs < 5) * 100,
        'percent_below_10': np.mean(angular_diffs < 10) * 100,
        'transform_matrix': transform_matrix if not np.allclose(transform_matrix, np.eye(3)) else None,
        'n_particles': len(angular_diffs),
        'shifts_available': shifts_available,
    }

    if shifts_available:
        stats.update({
            'shift_mean': np.mean(shift_errors),
            'shift_std': np.std(shift_errors),
            'shift_median': np.median(shift_errors),
            'shift_iqr': np.percentile(shift_errors, 75) - np.percentile(shift_errors, 25),
        })

    if plot_all or plot_angular:
        plot_angular_errors(angular_diffs, save_plots)
    if shifts_available and (plot_all or plot_shifts):
        plot_shift_errors(shift_errors, shift_x_errors, shift_y_errors, save_plots)
    if shifts_available and (plot_all or plot_combined):
        plot_combined_errors(angular_diffs, shift_errors, save_plots)

    print(f"Found {stats['n_particles']} matching particles.")
    print(f"\nAnalyzing angular differences with {symmetry} symmetry:")
    print(f"Mean: {stats['mean']:.2f}°")
    print(f"Standard Deviation: {stats['std']:.2f}°")
    print(f"Median: {stats['median']:.2f}°")
    print(f"IQR: {stats['iqr']:.2f}°")
    print(f"\nPercentage of particles with angular error:")
    print(f"< 5°: {stats['percent_below_5']:.1f}%")
    print(f"< 10°: {stats['percent_below_10']:.1f}%")

    if stats['shifts_available']:
        print(f"\nShift errors (Å):")
        print(f"Mean: {stats['shift_mean']:.2f}")
        print(f"Standard Deviation: {stats['shift_std']:.2f}")
        print(f"Median: {stats['shift_median']:.2f}")
        print(f"IQR: {stats['shift_iqr']:.2f}")
    else:
        print("\nShift information not available in both starfiles.")

    if stats['transform_matrix'] is not None:
        print("\nTransformation matrix applied:")
        print(stats['transform_matrix'])

    return stats


def main():
    from argParseFromDoc import parse_function_and_call
    parse_function_and_call(analyze_angular_differences)


if __name__ == "__main__":
    main()