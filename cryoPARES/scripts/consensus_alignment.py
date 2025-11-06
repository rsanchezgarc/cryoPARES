"""
Consensus Alignment from Multiple RELION Star Files (with Optics Groups)
========================================================================

Changes vs. your original:
- Reads RELION optics groups (data_optics table) when present.
- Verifies optics compatibility: for every matched particle, the optics row
  referenced in each input STAR must match (all shared columns except
  rlnOpticsGroup). If not, the script raises with a concise diff.
- Writes an optics table in the result (preferring the first file's optics
  definitions) and subsets it to only the optics groups used in output.
- Continues to support STARs without optics; in that case, optics handling is skipped.
"""

import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import starfile
import torch
from scipy.spatial.transform import Rotation

from cryoPARES.constants import RELION_EULER_CONVENTION, RELION_ANGLES_NAMES
from cryoPARES.geometry.metrics_angles import rotation_error_with_sym, mean_rot_matrix
from cryoPARES.geometry.convert_angles import euler_angles_to_matrix, matrix_to_euler_angles


# Primary keys for matching particles across star files
PRIMARY_KEYS = ["rlnMicrographName", "rlnCoordinateX", "rlnCoordinateY"]

# Coordinate tolerance for matching (pixels)
COORDINATE_TOLERANCE_PX = 0.5


@dataclass
class StarTables:
    """Container for one STAR file's relevant tables."""
    particles: pd.DataFrame
    optics: Optional[pd.DataFrame]  # may be None if file has no optics table
    path: str


def _normalize_micrograph_paths(df: pd.DataFrame, root_dir: Optional[str]) -> pd.DataFrame:
    """
    Normalize rlnMicrographName to improve matching across star files.

    If root_dir is provided, convert to relative paths. Otherwise, use basename only.
    """
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


def _read_star_file(fpath: str, root_dir: Optional[str]) -> StarTables:
    """Read RELION star file and return particles (& optics if present)."""
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"Star file not found: {fpath}")

    data = starfile.read(fpath)

    particles = None
    optics = None

    if isinstance(data, dict):
        # Standard RELION naming
        if "particles" in data:
            particles = data["particles"]
        else:
            # Fallback: use last table
            particles = list(data.values())[-1]
        if "optics" in data:
            optics = data["optics"]
    else:
        particles = data

    # Normalize micrograph paths
    particles = _normalize_micrograph_paths(particles, root_dir)

    # Verify required columns in particles
    for key in PRIMARY_KEYS:
        if key not in particles.columns:
            raise KeyError(f"Missing required column '{key}' in {fpath}")

    for angle_col in RELION_ANGLES_NAMES:
        if angle_col not in particles.columns:
            raise KeyError(f"Missing required angle column '{angle_col}' in {fpath}")

    # If optics present, verify rlnOpticsGroup exists in particles
    if optics is not None:
        if "rlnOpticsGroup" not in particles.columns:
            raise KeyError(
                f"Found an optics table in '{fpath}' but particles lack 'rlnOpticsGroup'"
            )
        if "rlnOpticsGroup" not in optics.columns:
            raise KeyError(
                f"Optics table in '{fpath}' lacks 'rlnOpticsGroup'"
            )

        # Ensure optics group IDs are integers for robust merging
        optics = optics.copy()
        optics["rlnOpticsGroup"] = optics["rlnOpticsGroup"].astype(int)
        particles = particles.copy()
        particles["rlnOpticsGroup"] = particles["rlnOpticsGroup"].astype(int)

    return StarTables(particles=particles, optics=optics, path=fpath)


def _match_particles_across_files(
    stars: List[StarTables],
    coordinate_tolerance: float = COORDINATE_TOLERANCE_PX
) -> Tuple[List[pd.DataFrame], pd.DataFrame, pd.Index]:
    """
    Match particles across multiple star files using PRIMARY_KEYS.

    Returns only particles present in ALL input files (intersection).
    Uses fuzzy matching for coordinates within tolerance.

    Returns:
        matched_dfs: List of dataframes with matched particles (same order as input)
        match_info: DataFrame with matching statistics
        original_order_indices: Indices to restore original order of first file
    """
    if len(stars) < 2:
        raise ValueError("Need at least 2 star files for consensus alignment")

    dfs = [s.particles.copy() for s in stars]

    # Round coordinates for fuzzy matching
    for i, df in enumerate(dfs):
        df = df.copy()
        df['_rlnCoordinateX_rounded'] = np.round(df['rlnCoordinateX'] / coordinate_tolerance) * coordinate_tolerance
        df['_rlnCoordinateY_rounded'] = np.round(df['rlnCoordinateY'] / coordinate_tolerance) * coordinate_tolerance
        df['_match_key'] = (
            df['rlnMicrographName'].astype(str) + '||' +
            df['_rlnCoordinateX_rounded'].astype(str) + '||' +
            df['_rlnCoordinateY_rounded'].astype(str)
        )
        dfs[i] = df

    # Find intersection of all match keys
    common_keys = set(dfs[0]['_match_key'])
    for df in dfs[1:]:
        common_keys &= set(df['_match_key'])

    if len(common_keys) == 0:
        raise ValueError("No matching particles found across all input files")

    # Filter and sort each dataframe by common keys
    matched_dfs = []
    for i, df in enumerate(dfs):
        df_matched = df[df['_match_key'].isin(common_keys)].copy()
        # Remember original indices before sorting (only for first file)
        if i == 0:
            original_indices = df_matched.index.copy()
        df_matched = df_matched.sort_values('_match_key').reset_index(drop=True)

        # Remove temporary columns
        df_matched = df_matched.drop(columns=['_rlnCoordinateX_rounded', '_rlnCoordinateY_rounded', '_match_key'])
        matched_dfs.append(df_matched)

    # Create mapping to restore original order of first file
    # Build a dataframe with both original indices and match keys for first file
    df0_for_mapping = dfs[0][dfs[0]['_match_key'].isin(common_keys)].copy()
    df0_for_mapping['_original_idx'] = original_indices
    df0_for_mapping = df0_for_mapping.sort_values('_match_key').reset_index(drop=True)
    df0_for_mapping['_sorted_idx'] = df0_for_mapping.index

    # Create a series that maps from sorted position to original position
    # Then create the inverse mapping
    df0_for_mapping = df0_for_mapping.sort_values('_original_idx').reset_index(drop=True)
    sorted_to_original_map = df0_for_mapping['_sorted_idx'].values

    # Verify all dataframes have same number of particles in same order
    n_matched = len(matched_dfs[0])
    for df in matched_dfs[1:]:
        if len(df) != n_matched:
            raise ValueError("Dataframe sizes don't match after filtering")

    # Create match info
    match_info = pd.DataFrame({
        'file_idx': range(len(dfs)),
        'file_path': [s.path for s in stars],
        'n_original': [len(s.particles) for s in stars],
        'n_matched': [n_matched] * len(dfs),
        'fraction_retained': [n_matched / len(s.particles) for s in stars]
    })

    return matched_dfs, match_info, sorted_to_original_map


def _build_optics_signature_maps(stars: List[StarTables]) -> List[Optional[Dict[int, Tuple[Tuple[str, ...], Tuple]]]]:
    """
    For each file, build a map from rlnOpticsGroup -> (columns, values) signature
    (excluding rlnOpticsGroup itself). Used to compare optics rows across files.

    Returns: list of dicts (or None if no optics in that file).
    """
    sig_maps: List[Optional[Dict[int, Tuple[Tuple[str, ...], Tuple]]]] = []
    for s in stars:
        if s.optics is None:
            sig_maps.append(None)
            continue
        opt = s.optics.copy()
        cols = [c for c in opt.columns if c != "rlnOpticsGroup"]
        cols_sorted = tuple(sorted(cols))
        # Normalize dtypes for robust equality
        opt_norm = opt.copy()
        for c in cols:
            # Cast to string representation to avoid tiny float diffs causing false mismatches
            opt_norm[c] = opt_norm[c].map(lambda v: None if pd.isna(v) else str(v))
        d: Dict[int, Tuple[Tuple[str, ...], Tuple]] = {}
        for _, row in opt_norm.iterrows():
            grp = int(row["rlnOpticsGroup"])
            sig = tuple(row[c] for c in cols_sorted)
            d[grp] = (cols_sorted, sig)
        sig_maps.append(d)
    return sig_maps


def _verify_optics_compatibility(
    stars: List[StarTables],
    matched_dfs: List[pd.DataFrame],
    max_report: int = 10
):
    """
    Ensure that for every matched particle, the referenced optics rows are identical
    across files (comparing on the intersection of optics columns, excluding rlnOpticsGroup).

    If any mismatch is found, raise ValueError describing the first few mismatches.
    """
    # Quick exit if none (or only some) files have optics
    num_with_optics = sum(1 for s in stars if s.optics is not None)
    if num_with_optics == 0:
        print("No optics tables detected in any input STAR — skipping optics compatibility checks.")
        return

    if num_with_optics != len(stars):
        warnings.warn(
            "Some inputs have optics tables and others do not. "
            "Optics compatibility cannot be fully verified; proceeding using the first file's optics (if present)."
        )
        return

    # Build per-file signature maps
    sig_maps = _build_optics_signature_maps(stars)

    # Determine shared optics columns (excluding rlnOpticsGroup)
    shared_cols = None
    for s in stars:
        cols = set(s.optics.columns) - {"rlnOpticsGroup"}
        shared_cols = cols if shared_cols is None else (shared_cols & cols)
    shared_cols = tuple(sorted(shared_cols)) if shared_cols else tuple()

    # If no shared columns, nothing to compare (unlikely)
    if not shared_cols:
        warnings.warn("Optics tables have disjoint columns across files; skipping compatibility check.")
        return

    # Precompute normalized lookup tables for each file
    # Map optics group -> dict(col -> str(value)) only for shared_cols
    def norm_val(v):
        return None if pd.isna(v) else str(v)

    lookups: List[Dict[int, Dict[str, Optional[str]]]] = []
    for s in stars:
        d: Dict[int, Dict[str, Optional[str]]] = {}
        if s.optics is None:
            lookups.append(d)
            continue
        opt = s.optics.copy()
        for _, row in opt.iterrows():
            gid = int(row["rlnOpticsGroup"])
            d[gid] = {c: norm_val(row[c]) for c in shared_cols}
        lookups.append(d)

    # Compare, particle-by-particle, the referenced optics rows
    mismatches = []
    n = len(matched_dfs[0])
    for idx in range(n):
        # collect the shared_cols dict for each file reference
        refs = []
        for fi, (s, df) in enumerate(zip(stars, matched_dfs)):
            if s.optics is None:
                refs.append((fi, None))
                continue
            gid = int(df.iloc[idx]["rlnOpticsGroup"])
            refs.append((fi, lookups[fi].get(gid, None)))

        # Skip if any file lacks optics
        if any(r[1] is None for r in refs):
            continue

        # Compare all to the first
        base = refs[0][1]
        for fi, ref in refs[1:]:
            if ref != base:
                mismatches.append(
                    (idx, [r[1] for r in refs])
                )
                break

        if len(mismatches) >= max_report:
            break

    if mismatches:
        # Prepare a concise message
        lines = [
            "Optics compatibility check FAILED: referenced optics rows differ across files "
            f"for {len(mismatches)}+ matched particle(s). Showing up to {max_report}:"
        ]
        for idx, ref_list in mismatches:
            row_desc = []
            for fi, ref in enumerate(ref_list):
                row_desc.append(f"[file {fi+1}] {ref}")
            lines.append(f"  - particle idx {idx}: " + " | ".join(row_desc))
        lines.append(
            "Ensure that the input STARs reference identical optics (pixel size, voltage, Cs, etc.) "
            "for the same particles, or re-export with consistent optics groups."
        )
        raise ValueError("\n".join(lines))
    else:
        print("Optics compatibility check PASSED across all matched particles.")


def _compute_angular_errors(
    rot_matrices_list: List[torch.Tensor],
    symmetry: str
) -> torch.Tensor:
    """
    Compute pairwise angular errors between rotation matrices.

    Args:
        rot_matrices_list: List of [N, 3, 3] rotation matrices (one per star file)
        symmetry: Point group symmetry

    Returns:
        errors: [N, n_files, n_files] pairwise angular errors in radians
    """
    n_files = len(rot_matrices_list)
    n_particles = rot_matrices_list[0].shape[0]

    # Stack all rotation matrices: [N, n_files, 3, 3]
    rot_stack = torch.stack(rot_matrices_list, dim=1)

    # Compute all pairwise errors
    errors = torch.zeros(n_particles, n_files, n_files)

    for i in range(n_files):
        for j in range(i + 1, n_files):
            err = rotation_error_with_sym(
                rot_stack[:, i, :, :],
                rot_stack[:, j, :, :],
                symmetry=symmetry
            )
            errors[:, i, j] = err
            errors[:, j, i] = err  # Symmetric

    return errors


def _compute_geodesic_average(
    rot_matrices_list: List[torch.Tensor],
    weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute geodesic average of rotation matrices on SO(3).

    Args:
        rot_matrices_list: List of [N, 3, 3] rotation matrices
        weights: Optional [N, n_files] weights for each rotation

    Returns:
        avg_rot: [N, 3, 3] averaged rotation matrices
    """
    # Stack all rotations: [N, n_files, 3, 3]
    rot_stack = torch.stack(rot_matrices_list, dim=1)

    n_particles = rot_stack.shape[0]
    n_files = rot_stack.shape[1]

    # If no weights provided, use uniform weights
    if weights is None:
        weights = torch.ones(n_particles, n_files) / n_files

    # Compute mean over file dimension (dim=1)
    avg_rot, dispersion = mean_rot_matrix(rot_stack, dim=1, weights=weights, compute_dispersion=True)

    return avg_rot

def consensus_alignment(
    input_stars: List[str],
    output_star: str,
    symmetry: str,
    consensus_mode: str = "combined",
    angular_threshold_degs: float = 5.0,
    coordinate_tolerance: float = COORDINATE_TOLERANCE_PX,
    root_dir: Optional[str] = None,
    merge_columns: Optional[List[str]] = None,
    weights_per_file: Optional[List[float]] = None
):
    """
    Compute consensus alignment from multiple RELION star files.

    :param input_stars: List of input star file paths (minimum 2 required)
    :param output_star: Output star file path for consensus alignment
    :param symmetry: Point group symmetry (C1, C2, D2, T, O, I, etc.)
    :param consensus_mode: Consensus operation mode:
        - "drop": Keep only particles with max pairwise error < threshold
        - "average": Compute geodesic average of all poses
        - "combined": Average poses, then drop particles with high residual error
    :param angular_threshold_degs: Angular error threshold in degrees (default: 5.0)
    :param coordinate_tolerance: Tolerance for coordinate matching in pixels (default: 0.5)
    :param root_dir: Optional root directory for normalizing micrograph paths
    :param merge_columns: Optional list of metadata columns to merge from all files
        (by default, only first file's metadata is retained)
    :param weights_per_file: Optional weights for each input file when computing averages
        (must sum to 1.0, defaults to uniform weights)
    """
    # ==================================================================================
    # STEP 1: Validation
    # ==================================================================================
    if len(input_stars) < 2:
        raise ValueError("Need at least 2 input star files for consensus alignment")

    if consensus_mode not in ["drop", "average", "combined"]:
        raise ValueError(f"Invalid consensus_mode: {consensus_mode}. Must be 'drop', 'average', or 'combined'")

    if weights_per_file is not None:
        if len(weights_per_file) != len(input_stars):
            raise ValueError(f"Number of weights ({len(weights_per_file)}) must match number of input files ({len(input_stars)})")
        if not np.isclose(sum(weights_per_file), 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {sum(weights_per_file)}")
        weights_tensor = torch.tensor(weights_per_file, dtype=torch.float32)
    else:
        weights_tensor = None

    print(f"\n{'='*80}")
    print(f"Consensus Alignment")
    print(f"{'='*80}")
    print(f"Input files: {len(input_stars)}")
    for i, fpath in enumerate(input_stars):
        print(f"  [{i+1}] {fpath}")
    print(f"Symmetry: {symmetry}")
    print(f"Consensus mode: {consensus_mode}")
    print(f"Angular threshold: {angular_threshold_degs}°")
    print(f"{'='*80}\n")

    # ==================================================================================
    # STEP 2: Read and match particles
    # ==================================================================================
    print("Reading input star files...")
    stars = [_read_star_file(fpath, root_dir) for fpath in input_stars]

    print("\nMatching particles across files...")
    matched_dfs, match_info, sorted_to_original_map = _match_particles_across_files(stars, coordinate_tolerance)

    print("\nMatching statistics:")
    print(match_info.to_string(index=False))
    print(f"\nCommon particles: {len(matched_dfs[0])}")

    # ==================================================================================
    # STEP 2b: Optics compatibility check (if all files have optics)
    # ==================================================================================
    try:
        _verify_optics_compatibility(stars, matched_dfs)
    except ValueError as e:
        # Surface the error with a clear prefix to help users spot optics issues quickly.
        raise ValueError(f"Optics groups are not compatible across inputs:\n{e}") from None

    # ==================================================================================
    # STEP 3: Extract rotation matrices
    # ==================================================================================
    print("\nExtracting rotation matrices...")
    rot_matrices_list = []
    for i, df in enumerate(matched_dfs):
        # Extract Euler angles
        angles = df[list(RELION_ANGLES_NAMES)].values  # [N, 3]
        angles_rad = torch.tensor(np.deg2rad(angles), dtype=torch.float32)

        # Convert to rotation matrices using project's convention
        rot_mat = euler_angles_to_matrix(angles_rad, RELION_EULER_CONVENTION)
        rot_matrices_list.append(rot_mat)

    # ==================================================================================
    # STEP 4: Compute angular errors
    # ==================================================================================
    print(f"\nComputing pairwise angular errors (symmetry: {symmetry})...")
    errors_rad = _compute_angular_errors(rot_matrices_list, symmetry)
    errors_deg = torch.rad2deg(errors_rad)

    # Compute statistics
    max_errors = errors_deg.max(dim=1).values.max(dim=1).values.numpy()  # [N]
    mean_errors = errors_deg.sum(dim=(1, 2)).numpy() / (len(input_stars) * (len(input_stars) - 1))  # [N]

    print(f"\nAngular error statistics (degrees):")
    print(f"  Mean (across all pairs): {mean_errors.mean():.3f}°")
    print(f"  Std: {mean_errors.std():.3f}°")
    print(f"  Median: {np.median(mean_errors):.3f}°")
    print(f"  Min: {mean_errors.min():.3f}°")
    print(f"  Max: {mean_errors.max():.3f}°")
    print(f"  95th percentile: {np.percentile(mean_errors, 95):.3f}°")

    # ==================================================================================
    # STEP 5: Apply consensus operation
    # ==================================================================================
    print(f"\nApplying consensus operation: {consensus_mode}")

    # Start with the first dataframe as base (will retain its metadata, including rlnOpticsGroup)
    consensus_df = matched_dfs[0].copy()
    n_original = len(consensus_df)
    retained_indices = None  # Track which particles were retained

    if consensus_mode == "drop":
        # Keep only particles with max pairwise error below threshold
        mask = max_errors < angular_threshold_degs
        retained_indices = np.where(mask)[0]
        consensus_df = consensus_df[mask].reset_index(drop=True)

        # No pose averaging, use first file's poses
        retained_fraction = len(retained_indices) / n_original
        print(f"  Retained particles: {len(retained_indices)} / {n_original} ({retained_fraction*100:.1f}%)")
        print(f"  Dropped: {n_original - len(retained_indices)} particles with error > {angular_threshold_degs}°")

    elif consensus_mode == "average":
        # Compute geodesic average for all particles
        print("  Computing geodesic average of poses...")
        if weights_per_file is not None:
            # Expand weights to [N, n_files] for all particles
            weights_expanded = weights_tensor.unsqueeze(0).expand(len(consensus_df), -1)
            avg_rot = _compute_geodesic_average(rot_matrices_list, weights=weights_expanded)
        else:
            avg_rot = _compute_geodesic_average(rot_matrices_list, weights=None)

        # Convert back to Euler angles
        avg_angles_rad = matrix_to_euler_angles(avg_rot, RELION_EULER_CONVENTION)
        avg_angles_deg = torch.rad2deg(avg_angles_rad).numpy()

        # Update angles in consensus dataframe
        for i, col in enumerate(RELION_ANGLES_NAMES):
            consensus_df[col] = avg_angles_deg[:, i]

        print(f"  Updated poses for all {len(consensus_df)} particles")

    elif consensus_mode == "combined":
        # First compute average, then filter by residual error
        print("  Computing geodesic average of poses...")
        if weights_per_file is not None:
            weights_expanded = weights_tensor.unsqueeze(0).expand(len(consensus_df), -1)
            avg_rot = _compute_geodesic_average(rot_matrices_list, weights=weights_expanded)
        else:
            avg_rot = _compute_geodesic_average(rot_matrices_list, weights=None)

        # Compute residual errors (distance from average to each input)
        print("  Computing residual errors from average...")
        residual_errors = torch.zeros(len(consensus_df), len(input_stars))
        for i, rot_mat in enumerate(rot_matrices_list):
            residual_errors[:, i] = rotation_error_with_sym(avg_rot, rot_mat, symmetry=symmetry)

        # Maximum residual error per particle
        max_residual = torch.rad2deg(residual_errors.max(dim=1).values).numpy()

        # Filter particles
        mask = max_residual < angular_threshold_degs

        # Store indices for later filtering
        retained_indices = np.where(mask)[0]

        consensus_df = consensus_df[mask].reset_index(drop=True)

        # Update poses for retained particles
        avg_angles_rad = matrix_to_euler_angles(avg_rot[retained_indices], RELION_EULER_CONVENTION)
        avg_angles_deg = torch.rad2deg(avg_angles_rad).numpy()

        for i, col in enumerate(RELION_ANGLES_NAMES):
            consensus_df[col] = avg_angles_deg[:, i]

        retained_fraction = len(retained_indices) / n_original
        print(f"  Retained particles: {len(retained_indices)} / {n_original} ({retained_fraction*100:.1f}%)")
        print(f"  Dropped: {n_original - len(retained_indices)} particles with residual error > {angular_threshold_degs}°")

    # ==================================================================================
    # STEP 6: Merge additional metadata columns if requested
    # ==================================================================================
    if merge_columns:
        print(f"\nMerging additional metadata columns: {merge_columns}")
        for col in merge_columns:
            # Check which files have this column
            has_col = [col in s.particles.columns for s in stars]

            if not any(has_col):
                warnings.warn(f"Column '{col}' not found in any input file, skipping")
                continue

            # For each file that has the column, add it with suffix
            for i, (s, has) in enumerate(zip(stars, has_col)):
                if has and i > 0:  # Skip first file (already in consensus_df)
                    new_col_name = f"{col}_file{i+1}"

                    # Need to match indices after potential filtering
                    if len(consensus_df) < n_original:
                        # Filtering was applied, need to align by primary keys
                        merge_keys = PRIMARY_KEYS
                        merged = consensus_df.merge(
                            s.particles[merge_keys + [col]],
                            on=merge_keys,
                            how='left',
                            suffixes=('', f'_file{i+1}')
                        )
                        consensus_df[new_col_name] = merged[col + f'_file{i+1}']
                    else:
                        # No filtering, direct assignment
                        consensus_df[new_col_name] = s.particles[col].values

                    print(f"  Added column '{new_col_name}' from file {i+1}")

    # ==================================================================================
    # STEP 7: Prepare optics for output (subset + write)
    # ==================================================================================
    output_tables: Dict[str, pd.DataFrame] = {}

    # Always write particles
    output_tables["particles"] = consensus_df

    # Prefer optics from the first file if present, else from any file with optics
    first_optics = stars[0].optics
    if first_optics is not None and "rlnOpticsGroup" in consensus_df.columns:
        used_groups = sorted(set(consensus_df["rlnOpticsGroup"].astype(int).tolist()))
        optics_out = first_optics.copy()
        optics_out["rlnOpticsGroup"] = optics_out["rlnOpticsGroup"].astype(int)
        optics_out = optics_out[optics_out["rlnOpticsGroup"].isin(used_groups)].copy()

        # Sanity: ensure all used groups are present
        missing = set(used_groups) - set(optics_out["rlnOpticsGroup"].astype(int).tolist())
        if missing:
            raise ValueError(
                "Output references optics group(s) not present in the first input's optics table: "
                f"{sorted(missing)}. Ensure consistent optics groups or re-export inputs."
            )

        output_tables["optics"] = optics_out.reset_index(drop=True)
        print(f"\nOptics table written with {len(optics_out)} group(s) from first input.")
    else:
        # If no optics in first file, but some later file has optics AND everyone passed compatibility,
        # we could still include optics from that file. This is optional; we keep it simple and skip.
        if any(s.optics is not None for s in stars):
            warnings.warn(
                "Optics present in some inputs but not in the first; "
                "output will not include an optics table. (Particles still carry rlnOpticsGroup if present.)"
            )

    # ==================================================================================
    # STEP 8: Restore original order and write output
    # ==================================================================================
    # Restore the original order of the first input file
    # The sorted_to_original_map tells us: sorted_idx -> original_idx
    # We need to reorder so that particles appear in the same order as the first input file
    if len(consensus_df) == len(sorted_to_original_map):
        # No particles were dropped, can restore order
        reordered_consensus_df = consensus_df.iloc[sorted_to_original_map].reset_index(drop=True)
        output_tables["particles"] = reordered_consensus_df

        # Also need to reorder optics reference if present
        if "optics" in output_tables:
            # Optics groups are already correctly referenced, no need to change
            pass
    else:
        # Some particles were dropped (drop/combined mode), cannot restore order
        # Keep sorted order
        pass

    print(f"\nWriting consensus alignment to: {output_star}")

    # Create output directory if needed
    output_dir = os.path.dirname(output_star)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Write star file (write dict if we have optics, else just particles)
    if "optics" in output_tables:
        starfile.write(output_tables, output_star, overwrite=True)
    else:
        starfile.write(output_tables["particles"], output_star, overwrite=True)

    print(f"Successfully wrote {len(consensus_df)} particles to {output_star}")

    # ==================================================================================
    # STEP 9: Print final statistics
    # ==================================================================================
    print(f"\n{'='*80}")
    print(f"FINAL STATISTICS")
    print(f"{'='*80}")
    print(f"Total input particles (intersection): {n_original}")
    print(f"Output particles: {len(consensus_df)}")
    print(f"Retention rate: {len(consensus_df)/n_original*100:.1f}%")

    if consensus_mode in ["average", "combined"]:
        # Compute final angular errors for output poses vs input poses
        print(f"\nFinal pose quality:")

        # Extract output rotation matrices
        output_angles = consensus_df[list(RELION_ANGLES_NAMES)].values
        output_angles_rad = torch.tensor(np.deg2rad(output_angles), dtype=torch.float32)
        output_rot = euler_angles_to_matrix(output_angles_rad, RELION_EULER_CONVENTION)

        # Compute errors vs each input file
        for i, df_in in enumerate(matched_dfs):
            # Match indices if filtering was applied
            if retained_indices is not None:
                # Extract matching rows from input using stored indices
                angles_in = df_in.iloc[retained_indices][list(RELION_ANGLES_NAMES)].values
            else:
                angles_in = df_in[list(RELION_ANGLES_NAMES)].values

            angles_in_rad = torch.tensor(np.deg2rad(angles_in), dtype=torch.float32)
            rot_mat_matched = euler_angles_to_matrix(angles_in_rad, RELION_EULER_CONVENTION)

            errors = rotation_error_with_sym(output_rot, rot_mat_matched, symmetry=symmetry)
            errors_deg_out = torch.rad2deg(errors).numpy()

            import numpy as _np  # scoped to avoid polluting top-level
            print(f"  vs file {i+1}: mean={errors_deg_out.mean():.3f}°, "
                  f"median={_np.median(errors_deg_out):.3f}°, "
                  f"max={errors_deg_out.max():.3f}°")

    print(f"{'='*80}\n")


def main():
    from argParseFromDoc import parse_function_and_call
    parse_function_and_call(consensus_alignment)


if __name__ == "__main__":
    main()
