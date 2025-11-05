"""
Consensus Alignment from Multiple RELION Star Files
====================================================

This script computes consensus alignment from two or more RELION star files by matching
equivalent particles based on micrograph name and coordinates, computing angular errors
between poses, and applying consensus operations.

Algorithm
---------
1. **Particle Matching**: Match equivalent particles across star files using:
   - rlnMicrographName
   - rlnCoordinateX
   - rlnCoordinateY

2. **Angular Error Computation**: Calculate angular distance between equivalent particles
   considering molecular symmetry.

3. **Consensus Operations**: Apply one of three strategies:
   - **drop**: Discard particles with error > threshold (default: 5 degrees)
   - **average**: Compute geodesic average pose for all particles
   - **combined**: Average poses, then drop particles with high residual error

4. **Metadata Management**: Retain metadata from first star file by default, with options
   to incorporate specific columns from other star files.

Usage Examples
--------------
# Drop particles with high angular error
python -m cryoPARES.scripts.consensus_alignment \\
    --input_stars run1.star run2.star run3.star \\
    --output_star consensus.star \\
    --symmetry C1 \\
    --consensus_mode drop \\
    --angular_threshold_degs 5.0

# Compute geodesic average of all poses
python -m cryoPARES.scripts.consensus_alignment \\
    --input_stars run1.star run2.star \\
    --output_star consensus.star \\
    --symmetry C1 \\
    --consensus_mode average

# Combined: average then filter
python -m cryoPARES.scripts.consensus_alignment \\
    --input_stars run1.star run2.star run3.star \\
    --output_star consensus.star \\
    --symmetry C1 \\
    --consensus_mode combined \\
    --angular_threshold_degs 3.0

# Incorporate specific metadata columns from other star files
python -m cryoPARES.scripts.consensus_alignment \\
    --input_stars aligned1.star aligned2.star \\
    --output_star consensus.star \\
    --symmetry C1 \\
    --consensus_mode average \\
    --merge_columns rlnClassNumber rlnLogLikeliContribution

Key Features
------------
- Handles partial matches: only processes particles present in ALL input star files
- Supports all point group symmetries (C, D, T, O, I)
- Computes geodesic average on SO(3) for rotation matrices
- Comprehensive statistics: angular errors, retention rates, per-file contributions
- Validates coordinate matching with configurable tolerance (default: 0.5 pixels)
"""

import os
import warnings
from typing import List, Optional, Tuple

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
    df["rlnMicrographName"] = df["rlnMicrographName"].map(_to_rel_or_base)
    return df


def _read_star_file(fpath: str, root_dir: Optional[str]) -> pd.DataFrame:
    """Read RELION star file and return particles table."""
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"Star file not found: {fpath}")

    data = starfile.read(fpath)

    # Get particles table (usually named 'particles')
    if isinstance(data, dict):
        if 'particles' in data:
            df = data['particles']
        else:
            # Try to get the last table (common convention)
            df = list(data.values())[-1]
    else:
        df = data

    # Normalize micrograph paths
    df = _normalize_micrograph_paths(df, root_dir)

    # Verify required columns
    for key in PRIMARY_KEYS:
        if key not in df.columns:
            raise KeyError(f"Missing required column '{key}' in {fpath}")

    for angle_col in RELION_ANGLES_NAMES:
        if angle_col not in df.columns:
            raise KeyError(f"Missing required angle column '{angle_col}' in {fpath}")

    return df


def _match_particles_across_files(
    dfs: List[pd.DataFrame],
    coordinate_tolerance: float = COORDINATE_TOLERANCE_PX
) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    """
    Match particles across multiple star files using PRIMARY_KEYS.

    Returns only particles present in ALL input files (intersection).
    Uses fuzzy matching for coordinates within tolerance.

    Returns:
        matched_dfs: List of dataframes with matched particles (same order as input)
        match_info: DataFrame with matching statistics
    """
    if len(dfs) < 2:
        raise ValueError("Need at least 2 star files for consensus alignment")

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
        df_matched = df_matched.sort_values('_match_key').reset_index(drop=True)

        # Remove temporary columns
        df_matched = df_matched.drop(columns=['_rlnCoordinateX_rounded', '_rlnCoordinateY_rounded', '_match_key'])
        matched_dfs.append(df_matched)

    # Verify all dataframes have same number of particles in same order
    n_matched = len(matched_dfs[0])
    for df in matched_dfs[1:]:
        if len(df) != n_matched:
            raise ValueError("Dataframe sizes don't match after filtering")

    # Create match info
    match_info = pd.DataFrame({
        'file_idx': range(len(dfs)),
        'n_original': [len(df) for df in dfs],
        'n_matched': [n_matched] * len(dfs),
        'fraction_retained': [n_matched / len(df) for df in dfs]
    })

    return matched_dfs, match_info


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
    dfs = [_read_star_file(fpath, root_dir) for fpath in input_stars]

    print("\nMatching particles across files...")
    matched_dfs, match_info = _match_particles_across_files(dfs, coordinate_tolerance)

    print("\nMatching statistics:")
    print(match_info.to_string(index=False))
    print(f"\nCommon particles: {len(matched_dfs[0])}")

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

    # Start with the first dataframe as base (will retain its metadata)
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
            has_col = [col in df.columns for df in matched_dfs]

            if not any(has_col):
                warnings.warn(f"Column '{col}' not found in any input file, skipping")
                continue

            # For each file that has the column, add it with suffix
            for i, (df, has) in enumerate(zip(matched_dfs, has_col)):
                if has and i > 0:  # Skip first file (already in consensus_df)
                    new_col_name = f"{col}_file{i+1}"

                    # Need to match indices after potential filtering
                    if len(consensus_df) < n_original:
                        # Filtering was applied, need to align indices
                        # Match using primary keys
                        merge_keys = PRIMARY_KEYS
                        merged = consensus_df.merge(
                            df[merge_keys + [col]],
                            on=merge_keys,
                            how='left',
                            suffixes=('', f'_file{i+1}')
                        )
                        consensus_df[new_col_name] = merged[col + f'_file{i+1}']
                    else:
                        # No filtering, direct assignment
                        consensus_df[new_col_name] = df[col].values

                    print(f"  Added column '{new_col_name}' from file {i+1}")

    # ==================================================================================
    # STEP 7: Write output
    # ==================================================================================
    print(f"\nWriting consensus alignment to: {output_star}")

    # Create output directory if needed
    output_dir = os.path.dirname(output_star)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Write star file
    starfile.write(consensus_df, output_star, overwrite=True)

    print(f"Successfully wrote {len(consensus_df)} particles to {output_star}")

    # ==================================================================================
    # STEP 8: Print final statistics
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
        for i, rot_mat in enumerate(rot_matrices_list):
            # Match indices if filtering was applied
            if retained_indices is not None:
                # Extract matching rows from input using stored indices
                rot_mat_matched = rot_mat[retained_indices]
            else:
                rot_mat_matched = rot_mat

            errors = rotation_error_with_sym(output_rot, rot_mat_matched, symmetry=symmetry)
            errors_deg = torch.rad2deg(errors).numpy()

            print(f"  vs file {i+1}: mean={errors_deg.mean():.3f}°, "
                  f"median={np.median(errors_deg):.3f}°, "
                  f"max={errors_deg.max():.3f}°")

    print(f"{'='*80}\n")


def main():
    from argParseFromDoc import parse_function_and_call
    parse_function_and_call(consensus_alignment)


if __name__ == "__main__":
    main()