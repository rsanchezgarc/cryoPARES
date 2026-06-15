"""
Tests for half-model consensus pruning (allCombinations inference).

These tests exercise the file-based consensus phase
(:func:`cryoPARES.inference.consensus.run_consensus_phase`) and the underlying reuse of
:func:`cryoPARES.scripts.consensus_alignment.consensus_alignment`. They run on CPU and do
not perform reconstruction (``reconstruct=False``), so they are fast and dependency-light.
"""
import os

import numpy as np
import pandas as pd
import pytest
import starfile
import torch
from scipy.spatial.transform import Rotation

from cryoPARES.constants import (RELION_ANGLES_NAMES, RELION_EULER_CONVENTION,
                                  CONSENSUS_ANGULAR_ERROR_NAME)
from cryoPARES.geometry.convert_angles import matrix_to_euler_angles
from cryoPARES.geometry.symmetry import getSymmetryGroup
from cryoPARES.inference.consensus import run_consensus_phase, allcombinations_star_suffix


def _mats_to_euler_degs(rotmats: torch.Tensor) -> np.ndarray:
    """Convert (N, 3, 3) rotation matrices to RELION ZYZ Euler angles in degrees."""
    return torch.rad2deg(matrix_to_euler_angles(rotmats, RELION_EULER_CONVENTION)).numpy()


def _write_pred_star(path: str, image_names, rotmats: torch.Tensor):
    """Write a minimal cryoPARES-style prediction star (particles + optics)."""
    eulers = _mats_to_euler_degs(rotmats)
    particles = pd.DataFrame({"rlnImageName": list(image_names)})
    for i, col in enumerate(RELION_ANGLES_NAMES):
        particles[col] = eulers[:, i]
    particles["rlnOpticsGroup"] = 1
    optics = pd.DataFrame({"rlnOpticsGroup": [1], "rlnImagePixelSize": [1.0], "rlnImageSize": [64]})
    starfile.write({"optics": optics, "particles": particles}, path, overwrite=True)


def _names(prefix, n):
    return [f"{i+1:06d}@{prefix}.mrcs" for i in range(n)]


def _read_particles(path):
    data = starfile.read(path)
    return data["particles"] if isinstance(data, dict) else data


def _setup_allcombinations(results_dir, *, half1_matching, half1_cross, half2_matching, half2_cross,
                           names1, names2, basename="parts"):
    """Write the 4 per-combination prediction star files for an allCombinations run."""
    _write_pred_star(os.path.join(results_dir, basename + allcombinations_star_suffix("half1", "half1")),
                     names1, half1_matching)
    _write_pred_star(os.path.join(results_dir, basename + allcombinations_star_suffix("half1", "half2")),
                     names1, half1_cross)
    _write_pred_star(os.path.join(results_dir, basename + allcombinations_star_suffix("half2", "half2")),
                     names2, half2_matching)
    _write_pred_star(os.path.join(results_dir, basename + allcombinations_star_suffix("half2", "half1")),
                     names2, half2_cross)
    return basename


def test_consensus_drops_disagreeing_and_keeps_cross_pose(tmp_path):
    rng = np.random.default_rng(0)
    n = 8
    names1, names2 = _names("h1", n), _names("h2", n)

    base1 = torch.tensor(Rotation.random(n, random_state=0).as_matrix(), dtype=torch.float32)
    base2 = torch.tensor(Rotation.random(n, random_state=1).as_matrix(), dtype=torch.float32)

    # Cross predictions agree with matching except for two deliberately-perturbed particles.
    perturb = Rotation.from_euler("Z", 30, degrees=True).as_matrix()
    cross1 = base1.clone()
    cross1[0] = torch.tensor(perturb, dtype=torch.float32) @ base1[0]
    cross1[3] = torch.tensor(perturb, dtype=torch.float32) @ base1[3]

    results_dir = str(tmp_path)
    _setup_allcombinations(results_dir,
                           half1_matching=base1, half1_cross=cross1,
                           half2_matching=base2, half2_cross=base2.clone(),
                           names1=names1, names2=names2)

    run_consensus_phase(results_dir, symmetry="C1", thr_degs=10.0, reconstruct=False)

    # half1: particles 0 and 3 disagree by 30deg > 10deg threshold -> dropped
    h1 = _read_particles(os.path.join(results_dir, "parts_consensus_half1.star"))
    kept_names = set(h1["rlnImageName"])
    assert names1[0] not in kept_names and names1[3] not in kept_names
    assert len(h1) == n - 2
    # The error column is present
    assert CONSENSUS_ANGULAR_ERROR_NAME in h1.columns
    assert (h1[CONSENSUS_ANGULAR_ERROR_NAME] < 10.0).all()

    # Survivors keep the CROSS-half pose (== cross1 angles, which equal base1 for survivors)
    expected = pd.DataFrame(_mats_to_euler_degs(cross1), columns=list(RELION_ANGLES_NAMES))
    expected["rlnImageName"] = names1
    merged = h1.merge(expected, on="rlnImageName", suffixes=("", "_exp"))
    for col in RELION_ANGLES_NAMES:
        np.testing.assert_allclose(merged[col].values, merged[col + "_exp"].values, atol=1e-3)

    # half2: all agree -> all kept
    h2 = _read_particles(os.path.join(results_dir, "parts_consensus_half2.star"))
    assert len(h2) == n

    # Merged file exists and tags rlnRandomSubset
    merged_all = _read_particles(os.path.join(results_dir, "parts_consensus.star"))
    assert set(merged_all["rlnRandomSubset"].unique()) == {1, 2}
    assert len(merged_all) == (n - 2) + n


def test_consensus_is_symmetry_aware(tmp_path):
    n = 5
    names1, names2 = _names("h1", n), _names("h2", n)
    base1 = torch.tensor(Rotation.random(n, random_state=2).as_matrix(), dtype=torch.float32)
    base2 = torch.tensor(Rotation.random(n, random_state=3).as_matrix(), dtype=torch.float32)

    # Make every half1 cross prediction differ from matching by a non-identity C4 operation.
    sym_mats = getSymmetryGroup("C4", as_matrix=True)  # (4, 3, 3)
    c4_op = sym_mats[1].to(torch.float32)  # 90deg about Z
    assert not torch.allclose(c4_op, torch.eye(3))
    cross1 = torch.einsum("ij,njk->nik", c4_op, base1)

    def run(sym):
        results_dir = str(tmp_path / sym)
        os.makedirs(results_dir, exist_ok=True)
        _setup_allcombinations(results_dir,
                               half1_matching=base1, half1_cross=cross1,
                               half2_matching=base2, half2_cross=base2.clone(),
                               names1=names1, names2=names2)
        run_consensus_phase(results_dir, symmetry=sym, thr_degs=10.0, reconstruct=False)
        return _read_particles(os.path.join(results_dir, "parts_consensus_half1.star"))

    # Under C4 the C4-related poses agree -> all kept
    assert len(run("C4")) == n
    # Under C1 they differ by ~90deg -> all dropped
    assert len(run("C1")) == 0
