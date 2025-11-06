"""
Diagnostic test for consensus_alignment.py using synthetic data.

Key improvements:
- Computes ALL pose errors in matrix space using the project's converters:
  euler_angles_to_matrix(...) + rotation_error_with_sym(...)
- Cross-checks the averaged rotations in THREE ways:
  (1) output file vs zero-noise reference (what you had)
  (2) a recomputed average done INSIDE the test (same kernel as the script)
  (3) output file vs the recomputed average (to test I/O conversion)

If something goes wrong, this test prints targeted diagnostics so you
can see whether the issue is in averaging, angle conversion, or matching.
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
from cryoPARES.geometry.metrics_angles import rotation_error_with_sym, mean_rot_matrix


# -------------------------- helpers --------------------------

def read_star_particles(fpath: str) -> pd.DataFrame:
    data = starfile.read(fpath)
    return data if isinstance(data, pd.DataFrame) else data['particles']


def create_fake_star_file(
    output_path: str,
    n_particles: int,
    base_rotations: np.ndarray,
    noise_degrees: float,
    seed: int
) -> pd.DataFrame:
    if seed is not None:
        np.random.seed(seed)

    angles = base_rotations.copy()
    if noise_degrees > 0:
        angles = angles + np.random.randn(n_particles, 3) * noise_degrees

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
    starfile.write(df, output_path, overwrite=True)
    return df


def _angles_deg_to_rot_torch(angles_deg: np.ndarray) -> torch.Tensor:
    ang_rad = torch.tensor(np.deg2rad(angles_deg), dtype=torch.float32)
    return euler_angles_to_matrix(ang_rad, RELION_EULER_CONVENTION)


def _mean_err_deg(A: torch.Tensor, B: torch.Tensor, symmetry: str = 'C1') -> float:
    errs = rotation_error_with_sym(A, B, symmetry=symmetry)
    return float(torch.rad2deg(errs).mean().cpu().numpy())


def _stack_inputs_to_rotmats(dfs):
    """Return [N,K,3,3] torch rotation matrices from a list of particle DataFrames."""
    mats = []
    for df in dfs:
        mats.append(_angles_deg_to_rot_torch(df[list(RELION_ANGLES_NAMES)].values))
    return torch.stack(mats, dim=1)  # [N,K,3,3] (K = len(dfs))


# -------------------------- the test --------------------------

def test_consensus_alignment():
    print("\n" + "="*80)
    print("CONSENSUS ALIGNMENT TEST (diagnostic)")
    print("="*80)

    test_dir = tempfile.mkdtemp(prefix='cryopares_consensus_test_')
    print(f"\nTest directory: {test_dir}")

    try:
        # ------------------------------------------------------------------
        # Synthetic data
        # ------------------------------------------------------------------
        n_particles = 50
        seed = 42

        np.random.seed(seed)
        base_rots = Rotation.random(n_particles, random_state=seed)
        base_angles = base_rots.as_euler(RELION_EULER_CONVENTION, degrees=True)

        # create base file (0° noise)
        df0 = create_fake_star_file(
            os.path.join(test_dir, 'particles_noise0.0.star'),
            n_particles=n_particles,
            base_rotations=base_angles,
            noise_degrees=0.0,
            seed=seed
        )

        star_files = [os.path.join(test_dir, 'particles_noise0.0.star')]
        print(f"  Created {star_files[0]} (noise: 0.0°)")

        # make noisy clones with identical coordinates/micrographs
        for i, noise in enumerate([2.0, 5.0]):
            fpath = os.path.join(test_dir, f'particles_noise{noise:.1f}.star')
            np.random.seed(seed + i + 1)
            noisy_angles = base_angles + np.random.randn(n_particles, 3) * noise
            df = df0.copy()
            for j, col in enumerate(RELION_ANGLES_NAMES):
                df[col] = noisy_angles[:, j]
            starfile.write(df, fpath, overwrite=True)
            star_files.append(fpath)
            print(f"  Created {fpath} (noise: {noise}°)")

        # ------------------------------------------------------------------
        # Mode "average"
        # ------------------------------------------------------------------
        print("\n" + "-"*80)
        print("TEST: Consensus mode 'average' (with diagnostics)")
        print("-"*80)

        out_avg = os.path.join(test_dir, 'consensus_average.star')
        consensus_alignment(
            input_stars=star_files,
            output_star=out_avg,
            symmetry='C1',
            consensus_mode='average'
        )
        assert os.path.exists(out_avg), "Output file not created"
        df_avg = read_star_particles(out_avg)
        assert len(df_avg) == n_particles, "Average mode should retain all particles"

        # Build all matrices (project converters!)
        avg_R  = _angles_deg_to_rot_torch(df_avg[list(RELION_ANGLES_NAMES)].values)     # [N,3,3]
        ref_R  = _angles_deg_to_rot_torch(read_star_particles(star_files[0])[list(RELION_ANGLES_NAMES)].values)
        noisy_R = _angles_deg_to_rot_torch(read_star_particles(star_files[2])[list(RELION_ANGLES_NAMES)].values)

        # 1) Output vs reference
        mean_err_avg = _mean_err_deg(avg_R, ref_R, 'C1')
        mean_err_noisy = _mean_err_deg(noisy_R, ref_R, 'C1')
        print(f"\n[Eval #1] vs reference:")
        print(f"  Averaged poses: {mean_err_avg:.3f}°")
        print(f"  Noisy (5°):     {mean_err_noisy:.3f}°")

        # 2) Recompute the average inside the test (same kernel the script uses).
        #    The script stacks per-file rotmats [N,K,3,3] and calls mean_rot_matrix(dim=1).
        inputs_R = _stack_inputs_to_rotmats([read_star_particles(f) for f in star_files])  # [N,K,3,3]
        avg_R_recomp, _ = mean_rot_matrix(inputs_R, dim=1, weights=None, compute_dispersion=False)
        mean_err_recomp = _mean_err_deg(avg_R_recomp, ref_R, 'C1')
        print(f"\n[Eval #2] recomputed mean_rot_matrix vs reference: {mean_err_recomp:.3f}°")

        # 3) Output vs recomputed (checks round-trip/write path)
        mean_err_out_vs_recomp = _mean_err_deg(avg_R, avg_R_recomp, 'C1')
        print(f"[Eval #3] output vs recomputed mean: {mean_err_out_vs_recomp:.3f}°")

        # Hard assertions:
        # (A) The recomputed mean should beat the 5° noisy input (if averaging is correct)
        assert mean_err_recomp < mean_err_noisy + 1e-6, (
            f"Recomputed mean (matrix-space) is not better than noisy: "
            f"{mean_err_recomp:.3f}° vs {mean_err_noisy:.3f}°"
        )

        # (B) The file we wrote should match the recomputed mean closely (I/O correctness)
        assert mean_err_out_vs_recomp < 1.0, (
            f"Output poses differ from recomputed mean by {mean_err_out_vs_recomp:.3f}° "
            f"(suspect angle conversion / write path)."
        )

        # (C) And therefore output should beat the noisy input too
        assert mean_err_avg < mean_err_noisy, "Averaging did not improve pose accuracy!"

        print("\n✓ Average mode: all diagnostics OK")

        # ------------------------------------------------------------------
        # The rest of your original tests (drop/combined/partial/symmetry/merge)
        # ------------------------------------------------------------------

        print("\n" + "-"*80)
        print("TEST: Consensus mode 'drop'")
        print("-"*80)
        out_drop = os.path.join(test_dir, 'consensus_drop.star')
        consensus_alignment(
            input_stars=star_files,
            output_star=out_drop,
            symmetry='C1',
            consensus_mode='drop',
            angular_threshold_degs=10.0,
            coordinate_tolerance=0.5
        )
        assert os.path.exists(out_drop)
        df_drop = read_star_particles(out_drop)
        print(f"✓ drop: Output has {len(df_drop)} particles (<= {n_particles})")
        assert 0 < len(df_drop) <= n_particles

        print("\n" + "-"*80)
        print("TEST: Consensus mode 'combined'")
        print("-"*80)
        out_comb = os.path.join(test_dir, 'consensus_combined.star')
        consensus_alignment(
            input_stars=star_files,
            output_star=out_comb,
            symmetry='C1',
            consensus_mode='combined',
            angular_threshold_degs=8.0
        )
        assert os.path.exists(out_comb)
        df_comb = read_star_particles(out_comb)
        print(f"✓ combined: Output has {len(df_comb)} particles")
        assert 0 < len(df_comb) <= n_particles

        print("\n" + "-"*80)
        print("TEST: Partial matching")
        print("-"*80)
        df1 = create_fake_star_file(os.path.join(test_dir, 'partial1.star'), 40,
                                    Rotation.random(40, random_state=100).as_euler(RELION_EULER_CONVENTION, True),
                                    0.0, 100)
        df2 = create_fake_star_file(os.path.join(test_dir, 'partial2.star'), 40,
                                    Rotation.random(40, random_state=200).as_euler(RELION_EULER_CONVENTION, True),
                                    0.0, 200)
        df2['rlnCoordinateX'] = df1['rlnCoordinateX'].iloc[:30].tolist() + df2['rlnCoordinateX'].iloc[30:].tolist()
        df2['rlnCoordinateY'] = df1['rlnCoordinateY'].iloc[:30].tolist() + df2['rlnCoordinateY'].iloc[30:].tolist()
        df2['rlnMicrographName'] = df1['rlnMicrographName'].iloc[:30].tolist() + df2['rlnMicrographName'].iloc[30:].tolist()
        starfile.write(df2, os.path.join(test_dir, 'partial2.star'), overwrite=True)

        out_partial = os.path.join(test_dir, 'consensus_partial.star')
        consensus_alignment(
            input_stars=[os.path.join(test_dir, 'partial1.star'),
                         os.path.join(test_dir, 'partial2.star')],
            output_star=out_partial,
            symmetry='C1',
            consensus_mode='average'
        )
        df_partial = read_star_particles(out_partial)
        print(f"✓ partial: {len(df_partial)} common particles (expected ~30)")
        assert 25 <= len(df_partial) <= 35

        print("\n" + "-"*80)
        print("TEST: Symmetry C2")
        print("-"*80)
        df_sym1 = create_fake_star_file(os.path.join(test_dir, 'sym1.star'), 20,
                                        Rotation.random(20, random_state=300).as_euler(RELION_EULER_CONVENTION, True),
                                        0.0, 300)
        angles1 = df_sym1[list(RELION_ANGLES_NAMES)].values
        rots1 = Rotation.from_euler(RELION_EULER_CONVENTION, angles1, degrees=True)
        c2 = Rotation.from_euler('Z', 180, degrees=True)
        angles2 = (c2 * rots1).as_euler(RELION_EULER_CONVENTION, degrees=True)
        df_sym2 = df_sym1.copy()
        df_sym2[RELION_ANGLES_NAMES[0]] = angles2[:, 0]
        df_sym2[RELION_ANGLES_NAMES[1]] = angles2[:, 1]
        df_sym2[RELION_ANGLES_NAMES[2]] = angles2[:, 2]
        starfile.write(df_sym2, os.path.join(test_dir, 'sym2.star'), overwrite=True)

        out_sym = os.path.join(test_dir, 'consensus_sym.star')
        consensus_alignment(
            input_stars=[os.path.join(test_dir, 'sym1.star'),
                         os.path.join(test_dir, 'sym2.star')],
            output_star=out_sym,
            symmetry='C2',
            consensus_mode='drop',
            angular_threshold_degs=5.0
        )
        df_sym = read_star_particles(out_sym)
        print(f"✓ symmetry C2: {len(df_sym)} particles (expected 20)")
        assert len(df_sym) == 20

        print("\n" + "-"*80)
        print("TEST: Merge columns")
        print("-"*80)
        df_meta1 = create_fake_star_file(os.path.join(test_dir, 'meta1.star'), 15,
                                         Rotation.random(15, random_state=400).as_euler(RELION_EULER_CONVENTION, True),
                                         0.0, 400)
        df_meta1['rlnClassNumber'] = np.random.randint(1, 4, 15)
        starfile.write(df_meta1, os.path.join(test_dir, 'meta1.star'), overwrite=True)

        df_meta2 = df_meta1.copy()
        df_meta2['rlnClassNumber'] = np.random.randint(1, 4, 15)
        df_meta2['rlnLogLikeliContribution'] = np.random.uniform(100, 200, 15)
        for col in RELION_ANGLES_NAMES:
            df_meta2[col] += np.random.randn(15) * 1.0
        starfile.write(df_meta2, os.path.join(test_dir, 'meta2.star'), overwrite=True)

        out_meta = os.path.join(test_dir, 'consensus_meta.star')
        consensus_alignment(
            input_stars=[os.path.join(test_dir, 'meta1.star'),
                         os.path.join(test_dir, 'meta2.star')],
            output_star=out_meta,
            symmetry='C1',
            consensus_mode='average',
            merge_columns=['rlnClassNumber', 'rlnLogLikeliContribution']
        )
        df_meta_out = read_star_particles(out_meta)
        assert 'rlnClassNumber' in df_meta_out.columns
        assert 'rlnClassNumber_file2' in df_meta_out.columns
        assert 'rlnLogLikeliContribution_file2' in df_meta_out.columns
        print("✓ merge columns OK")

        print("\n" + "="*80)
        print("ALL TESTS PASSED! ✓")
        print("="*80)
        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        if os.path.exists(test_dir):
            print(f"\nCleaning up test directory: {test_dir}")
            shutil.rmtree(test_dir)


if __name__ == "__main__":
    ok = test_consensus_alignment()
    exit(0 if ok else 1)
