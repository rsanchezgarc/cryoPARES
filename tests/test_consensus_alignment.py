"""
Test script for consensus_alignment.py using synthetic data.

Creates fake star files with known poses and tests all consensus modes.
"""

import os
import tempfile
import shutil
import numpy as np
import pandas as pd
import starfile
from scipy.spatial.transform import Rotation
import torch

from cryoPARES.constants import RELION_EULER_CONVENTION, RELION_ANGLES_NAMES
from cryoPARES.scripts.consensus_alignment import consensus_alignment
from cryoPARES.geometry.convert_angles import euler_angles_to_matrix
from cryoPARES.geometry.metrics_angles import rotation_error_with_sym


def read_star_particles(fpath: str) -> pd.DataFrame:
    """Helper to read particles from star file, handling both DataFrame and dict formats."""
    data = starfile.read(fpath)
    return data if isinstance(data, pd.DataFrame) else data['particles']


def create_fake_star_file(
    output_path: str,
    n_particles: int = 100,
    base_rotations: np.ndarray = None,
    noise_degrees: float = 0.0,
    seed: int = None
) -> pd.DataFrame:
    """
    Create a fake RELION star file with synthetic particle data.

    Args:
        output_path: Where to save the star file
        n_particles: Number of particles to generate
        base_rotations: Optional [N, 3] base Euler angles (degrees). If None, random.
        noise_degrees: Amount of Gaussian noise to add to angles (degrees)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with the generated particle data
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate or use base rotations
    if base_rotations is None:
        # Generate random rotations
        rots = Rotation.random(n_particles, random_state=seed)
        angles = rots.as_euler(RELION_EULER_CONVENTION, degrees=True)
    else:
        angles = base_rotations.copy()

    # Add noise if specified
    if noise_degrees > 0:
        noise = np.random.randn(n_particles, 3) * noise_degrees
        angles += noise

    # Create particle dataframe
    particles = {
        'rlnImageName': [f'{i+1:06d}@fake_stack.mrcs' for i in range(n_particles)],
        'rlnMicrographName': [f'mic_{i // 10:03d}.mrc' for i in range(n_particles)],
        'rlnCoordinateX': np.random.uniform(100, 4000, n_particles),
        'rlnCoordinateY': np.random.uniform(100, 4000, n_particles),
        'rlnAngleRot': angles[:, 0],
        'rlnAngleTilt': angles[:, 1],
        'rlnAnglePsi': angles[:, 2],
        'rlnOriginXAngst': np.random.uniform(-5, 5, n_particles),
        'rlnOriginYAngst': np.random.uniform(-5, 5, n_particles),
        'rlnDefocusU': np.random.uniform(10000, 30000, n_particles),
        'rlnDefocusV': np.random.uniform(10000, 30000, n_particles),
        'rlnDefocusAngle': np.random.uniform(0, 180, n_particles),
        'rlnVoltage': [300.0] * n_particles,
        'rlnSphericalAberration': [2.7] * n_particles,
        'rlnAmplitudeContrast': [0.1] * n_particles,
    }

    df = pd.DataFrame(particles)

    # Write star file
    starfile.write(df, output_path, overwrite=True)

    return df


def test_consensus_alignment():
    """Run comprehensive tests on consensus_alignment script."""

    print("\n" + "="*80)
    print("CONSENSUS ALIGNMENT TEST")
    print("="*80)

    # Create temporary directory for test files
    test_dir = tempfile.mkdtemp(prefix='cryopares_consensus_test_')
    print(f"\nTest directory: {test_dir}")

    try:
        # ======================================================================
        # TEST 1: Generate synthetic data with known properties
        # ======================================================================
        print("\n" + "-"*80)
        print("TEST 1: Generating synthetic star files")
        print("-"*80)

        n_particles = 50
        seed = 42

        # Generate base rotations (ground truth)
        np.random.seed(seed)
        base_rots = Rotation.random(n_particles, random_state=seed)
        base_angles = base_rots.as_euler(RELION_EULER_CONVENTION, degrees=True)

        print(f"Generated {n_particles} particles with random poses")

        # Create first file to establish coordinates and micrograph names
        first_df = create_fake_star_file(
            os.path.join(test_dir, 'particles_noise0.0.star'),
            n_particles=n_particles,
            base_rotations=base_angles,
            noise_degrees=0.0,
            seed=seed
        )

        # Store the fixed coordinates and micrograph names
        fixed_coords_x = first_df['rlnCoordinateX'].values
        fixed_coords_y = first_df['rlnCoordinateY'].values
        fixed_micnames = first_df['rlnMicrographName'].values

        star_files = [os.path.join(test_dir, 'particles_noise0.0.star')]
        print(f"  Created {star_files[0]} (noise: 0.0°)")

        # Create additional files with same coordinates but noisy angles
        noise_levels = [2.0, 5.0]  # degrees

        for i, noise in enumerate(noise_levels):
            fpath = os.path.join(test_dir, f'particles_noise{noise:.1f}.star')

            # Generate noisy angles
            np.random.seed(seed + i + 1)
            noisy_angles = base_angles + np.random.randn(n_particles, 3) * noise

            # Create dataframe with SAME coordinates but noisy angles
            df = first_df.copy()
            df['rlnAngleRot'] = noisy_angles[:, 0]
            df['rlnAngleTilt'] = noisy_angles[:, 1]
            df['rlnAnglePsi'] = noisy_angles[:, 2]

            starfile.write(df, fpath, overwrite=True)
            star_files.append(fpath)
            print(f"  Created {fpath} (noise: {noise}°)")

        # ======================================================================
        # TEST 2: Mode "drop" - filter high error particles
        # ======================================================================
        print("\n" + "-"*80)
        print("TEST 2: Consensus mode 'drop'")
        print("-"*80)

        output_drop = os.path.join(test_dir, 'consensus_drop.star')

        consensus_alignment(
            input_stars=star_files,
            output_star=output_drop,
            symmetry='C1',
            consensus_mode='drop',
            angular_threshold_degs=10.0,
            coordinate_tolerance=0.5
        )

        # Verify output exists
        assert os.path.exists(output_drop), "Output file not created"

        # Read and check output
        df_drop = read_star_particles(output_drop)
        print(f"\n✓ Mode 'drop': Output has {len(df_drop)} particles (started with {n_particles})")

        # Verify it retained some particles (should drop particles with very high error)
        assert len(df_drop) <= n_particles, "Output has more particles than input!"
        assert len(df_drop) > 0, "All particles were dropped!"

        # ======================================================================
        # TEST 3: Mode "average" - geodesic averaging
        # ======================================================================
        print("\n" + "-"*80)
        print("TEST 3: Consensus mode 'average'")
        print("-"*80)

        output_avg = os.path.join(test_dir, 'consensus_average.star')

        consensus_alignment(
            input_stars=star_files,
            output_star=output_avg,
            symmetry='C1',
            consensus_mode='average'
        )

        # Verify output
        assert os.path.exists(output_avg), "Output file not created"
        df_avg = read_star_particles(output_avg)
        print(f"\n✓ Mode 'average': Output has {len(df_avg)} particles (all retained)")

        # Should keep all particles in average mode
        assert len(df_avg) == n_particles, "Average mode should retain all particles"

        # Verify averaged poses are reasonable by comparing to the zero-noise reference
        avg_angles = df_avg[list(RELION_ANGLES_NAMES)].values
        avg_rots = Rotation.from_euler(RELION_EULER_CONVENTION, avg_angles, degrees=True)

        # Compare with zero-noise file (our reference)
        df_ref = read_star_particles(star_files[0])
        ref_angles = df_ref[list(RELION_ANGLES_NAMES)].values
        ref_rots = Rotation.from_euler(RELION_EULER_CONVENTION, ref_angles, degrees=True)

        # Compare with highest noise file
        df_noisy = read_star_particles(star_files[2])
        noisy_angles = df_noisy[list(RELION_ANGLES_NAMES)].values
        noisy_rots = Rotation.from_euler(RELION_EULER_CONVENTION, noisy_angles, degrees=True)

        # Compute errors relative to zero-noise reference
        errors_avg = []
        errors_noisy = []
        for i in range(n_particles):
            # Error of averaged pose to reference
            err_avg = (avg_rots[i].inv() * ref_rots[i]).magnitude()
            errors_avg.append(np.degrees(err_avg))

            # Error of noisy pose to reference
            err_noisy = (noisy_rots[i].inv() * ref_rots[i]).magnitude()
            errors_noisy.append(np.degrees(err_noisy))

        mean_err_avg = np.mean(errors_avg)
        mean_err_noisy = np.mean(errors_noisy)

        print(f"  Mean error relative to zero-noise reference:")
        print(f"    Averaged poses: {mean_err_avg:.3f}°")
        print(f"    Noisy poses (5°): {mean_err_noisy:.3f}°")
        print(f"  ✓ Averaging reduced error by {mean_err_noisy - mean_err_avg:.3f}°")

        # Averaged should be better than the noisiest input
        assert mean_err_avg < mean_err_noisy, "Averaging did not improve pose accuracy!"

        # ======================================================================
        # TEST 4: Mode "combined" - average + filter
        # ======================================================================
        print("\n" + "-"*80)
        print("TEST 4: Consensus mode 'combined'")
        print("-"*80)

        output_combined = os.path.join(test_dir, 'consensus_combined.star')

        consensus_alignment(
            input_stars=star_files,
            output_star=output_combined,
            symmetry='C1',
            consensus_mode='combined',
            angular_threshold_degs=8.0
        )

        # Verify output
        assert os.path.exists(output_combined), "Output file not created"
        df_combined = read_star_particles(output_combined)
        print(f"\n✓ Mode 'combined': Output has {len(df_combined)} particles")

        assert len(df_combined) <= n_particles, "Output has more particles than input!"
        assert len(df_combined) > 0, "All particles were dropped!"

        # ======================================================================
        # TEST 5: Test with partial matching (different particles in files)
        # ======================================================================
        print("\n" + "-"*80)
        print("TEST 5: Partial matching (different particle sets)")
        print("-"*80)

        # Create two files with overlapping but different particles
        df1 = create_fake_star_file(
            os.path.join(test_dir, 'partial1.star'),
            n_particles=40,
            seed=100
        )

        df2 = create_fake_star_file(
            os.path.join(test_dir, 'partial2.star'),
            n_particles=40,
            seed=200
        )

        # Modify coordinates so only ~30 particles overlap
        df2['rlnCoordinateX'] = df1['rlnCoordinateX'].iloc[:30].tolist() + df2['rlnCoordinateX'].iloc[30:].tolist()
        df2['rlnCoordinateY'] = df1['rlnCoordinateY'].iloc[:30].tolist() + df2['rlnCoordinateY'].iloc[30:].tolist()
        df2['rlnMicrographName'] = df1['rlnMicrographName'].iloc[:30].tolist() + df2['rlnMicrographName'].iloc[30:].tolist()

        # Save modified df2
        starfile.write(df2, os.path.join(test_dir, 'partial2.star'), overwrite=True)

        output_partial = os.path.join(test_dir, 'consensus_partial.star')

        consensus_alignment(
            input_stars=[
                os.path.join(test_dir, 'partial1.star'),
                os.path.join(test_dir, 'partial2.star')
            ],
            output_star=output_partial,
            symmetry='C1',
            consensus_mode='average'
        )

        df_partial = read_star_particles(output_partial)
        print(f"\n✓ Partial match: {len(df_partial)} common particles found (expected ~30)")
        assert 25 <= len(df_partial) <= 35, "Unexpected number of matched particles"

        # ======================================================================
        # TEST 6: Test with symmetry
        # ======================================================================
        print("\n" + "-"*80)
        print("TEST 6: Symmetry handling (C2)")
        print("-"*80)

        # Create files where second file has 180° rotation (C2 symmetry)
        df_sym1 = create_fake_star_file(
            os.path.join(test_dir, 'sym1.star'),
            n_particles=20,
            seed=300
        )

        # Apply C2 rotation to second file
        angles1 = df_sym1[list(RELION_ANGLES_NAMES)].values
        rots1 = Rotation.from_euler(RELION_EULER_CONVENTION, angles1, degrees=True)

        # C2 symmetry: 180° rotation around Z
        c2_rot = Rotation.from_euler('Z', 180, degrees=True)
        rots2 = rots1 * c2_rot
        angles2 = rots2.as_euler(RELION_EULER_CONVENTION, degrees=True)

        df_sym2 = df_sym1.copy()
        df_sym2['rlnAngleRot'] = angles2[:, 0]
        df_sym2['rlnAngleTilt'] = angles2[:, 1]
        df_sym2['rlnAnglePsi'] = angles2[:, 2]

        starfile.write(df_sym2, os.path.join(test_dir, 'sym2.star'), overwrite=True)

        output_sym = os.path.join(test_dir, 'consensus_sym.star')

        consensus_alignment(
            input_stars=[
                os.path.join(test_dir, 'sym1.star'),
                os.path.join(test_dir, 'sym2.star')
            ],
            output_star=output_sym,
            symmetry='C2',
            consensus_mode='drop',
            angular_threshold_degs=5.0
        )

        df_sym = read_star_particles(output_sym)
        print(f"\n✓ Symmetry C2: All {len(df_sym)} particles retained (expected all due to symmetry)")

        # With C2 symmetry, the poses should be considered equivalent
        # So we should keep all particles
        assert len(df_sym) == 20, "C2 symmetry not properly handled"

        # ======================================================================
        # TEST 7: Merge columns
        # ======================================================================
        print("\n" + "-"*80)
        print("TEST 7: Merging metadata columns")
        print("-"*80)

        # Add extra columns to test files
        df_meta1 = create_fake_star_file(
            os.path.join(test_dir, 'meta1.star'),
            n_particles=15,
            seed=400
        )
        df_meta1['rlnClassNumber'] = np.random.randint(1, 4, 15)
        starfile.write(df_meta1, os.path.join(test_dir, 'meta1.star'), overwrite=True)

        df_meta2 = df_meta1.copy()
        df_meta2['rlnClassNumber'] = np.random.randint(1, 4, 15)
        df_meta2['rlnLogLikeliContribution'] = np.random.uniform(100, 200, 15)
        # Add small noise to angles
        for col in RELION_ANGLES_NAMES:
            df_meta2[col] += np.random.randn(15) * 1.0
        starfile.write(df_meta2, os.path.join(test_dir, 'meta2.star'), overwrite=True)

        output_meta = os.path.join(test_dir, 'consensus_meta.star')

        consensus_alignment(
            input_stars=[
                os.path.join(test_dir, 'meta1.star'),
                os.path.join(test_dir, 'meta2.star')
            ],
            output_star=output_meta,
            symmetry='C1',
            consensus_mode='average',
            merge_columns=['rlnClassNumber', 'rlnLogLikeliContribution']
        )

        df_meta_out = read_star_particles(output_meta)
        print(f"\n✓ Merged columns:")
        print(f"  - Base columns from file 1: {len([c for c in df_meta_out.columns if not c.endswith('_file2')])}")
        print(f"  - Merged columns from file 2: {len([c for c in df_meta_out.columns if c.endswith('_file2')])}")

        # Check that merged columns exist
        assert 'rlnClassNumber' in df_meta_out.columns, "Base rlnClassNumber missing"
        assert 'rlnClassNumber_file2' in df_meta_out.columns, "Merged rlnClassNumber_file2 missing"
        assert 'rlnLogLikeliContribution_file2' in df_meta_out.columns, "Merged rlnLogLikeliContribution_file2 missing"

        print("  ✓ All merged columns present")

        # ======================================================================
        # FINAL SUMMARY
        # ======================================================================
        print("\n" + "="*80)
        print("ALL TESTS PASSED! ✓")
        print("="*80)
        print("\nTest summary:")
        print("  1. Synthetic data generation: PASSED")
        print("  2. Consensus mode 'drop': PASSED")
        print("  3. Consensus mode 'average': PASSED")
        print("  4. Consensus mode 'combined': PASSED")
        print("  5. Partial matching: PASSED")
        print("  6. Symmetry handling (C2): PASSED")
        print("  7. Metadata column merging: PASSED")
        print("\n" + "="*80 + "\n")

        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        if os.path.exists(test_dir):
            print(f"\nCleaning up test directory: {test_dir}")
            shutil.rmtree(test_dir)


if __name__ == "__main__":
    success = test_consensus_alignment()
    exit(0 if success else 1)