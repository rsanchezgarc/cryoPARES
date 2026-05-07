# tests/test_simulate_particles.py
"""
Regression tests for single-GPU/CPU particle simulation.

Tests verify:
- Correct number of output particles
- Output MRC stacks can be opened and have the right shape
- Output STAR file correctly maps every particle to a valid image index in its stack
- Multiple output files (images_per_file < n_particles) are handled correctly
- SNR noise addition is directionally correct (noisier with lower SNR)
"""
import os
import tempfile
import unittest

import mrcfile
import numpy as np
import pandas as pd
import starfile

from cryoPARES.simulation.simulateParticlesHelper import run_simulation


def _make_volume(path: str, box: int = 32, px_A: float = 1.0) -> None:
    """Write a simple sphere volume to an MRC file."""
    rng = np.random.default_rng(0)
    vol = rng.standard_normal((box, box, box)).astype(np.float32)
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(vol)
        mrc.voxel_size = (px_A, px_A, px_A)


def _make_star(path: str, n: int, box: int = 32, px_A: float = 1.0) -> None:
    """Write a minimal RELION STAR file with random poses and CTF params."""
    rng = np.random.default_rng(1)
    optics_df = pd.DataFrame({
        "rlnOpticsGroup":            [1],
        "rlnImageSize":              [box],
        "rlnImagePixelSize":         [px_A],
        "rlnCtfDataArePhaseFlipped": [0],
        "rlnVoltage":                [300.0],
        "rlnSphericalAberration":    [2.7],
        "rlnAmplitudeContrast":      [0.1],
    })
    particles_df = pd.DataFrame({
        "rlnImageName":    [f"{i+1}@dummy.mrcs" for i in range(n)],
        "rlnOpticsGroup":  [1] * n,
        "rlnAngleRot":     rng.uniform(0, 360, n).tolist(),
        "rlnAngleTilt":    rng.uniform(0, 180, n).tolist(),
        "rlnAnglePsi":     rng.uniform(0, 360, n).tolist(),
        "rlnOriginXAngst": [0.0] * n,
        "rlnOriginYAngst": [0.0] * n,
        "rlnDefocusU":     rng.uniform(5000, 25000, n).tolist(),
        "rlnDefocusV":     rng.uniform(5000, 25000, n).tolist(),
        "rlnDefocusAngle": rng.uniform(0, 180, n).tolist(),
        "rlnCtfBfactor":   [0.0] * n,
        "rlnCoordinateX":  [16.0] * n,
        "rlnCoordinateY":  [16.0] * n,
    })
    starfile.write({"optics": optics_df, "particles": particles_df}, path, overwrite=True)


def _check_star_consistency(star_path: str, output_dir: str) -> None:
    """
    Assert that every rlnImageName entry in the STAR file points to a valid
    (in-range) image inside its referenced MRC stack.
    """
    data = starfile.read(star_path)
    parts_df = data["particles"] if isinstance(data, dict) else data
    for img_name in parts_df["rlnImageName"]:
        idx_str, fname = img_name.split("@")
        idx = int(idx_str)
        stack_path = os.path.join(output_dir, fname)
        assert os.path.exists(stack_path), f"Stack not found: {stack_path}"
        with mrcfile.open(stack_path, permissive=True) as mrc:
            n_imgs = mrc.data.shape[0] if mrc.data.ndim == 3 else 1
        assert 1 <= idx <= n_imgs, (
            f"Image index {idx} out of range [1, {n_imgs}] in {fname}"
        )


class TestSimulateParticlesSingleGPU(unittest.TestCase):

    BOX = 32
    PX_A = 1.0

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.vol_path = os.path.join(self.tmp, "volume.mrc")
        self.star_path = os.path.join(self.tmp, "particles.star")
        _make_volume(self.vol_path, box=self.BOX, px_A=self.PX_A)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run(self, n_particles: int, images_per_file: int, snr=None,
             apply_ctf: bool = True, n_first_particles=None) -> list:
        _make_star(self.star_path, n_particles, box=self.BOX, px_A=self.PX_A)
        out_dir = os.path.join(self.tmp, "out")
        paths = run_simulation(
            volume=self.vol_path,
            in_star=self.star_path,
            output_dir=out_dir,
            basename="stack",
            images_per_file=images_per_file,
            batch_size=4,
            simulation_mode="central_slice",
            apply_ctf=apply_ctf,
            snr=snr,
            num_workers=0,
            px_A=self.PX_A,
            device="cpu",
            disable_tqdm=True,
            n_first_particles=n_first_particles,
        )
        return paths, out_dir

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_output_files_created(self):
        """run_simulation returns non-empty list of existing MRC paths."""
        paths, _ = self._run(n_particles=10, images_per_file=100)
        self.assertTrue(len(paths) > 0)
        for p in paths:
            self.assertTrue(os.path.exists(p), f"Missing: {p}")

    def test_total_particle_count_single_file(self):
        """When n_particles <= images_per_file, exactly one file is produced."""
        n = 7
        paths, _ = self._run(n_particles=n, images_per_file=100)
        self.assertEqual(len(paths), 1)
        with mrcfile.open(paths[0], permissive=True) as mrc:
            self.assertEqual(mrc.data.shape[0], n)

    def test_total_particle_count_multiple_files(self):
        """When n_particles > images_per_file, correct files and total count."""
        n = 11
        ipf = 4
        paths, _ = self._run(n_particles=n, images_per_file=ipf)
        expected_files = (n + ipf - 1) // ipf  # ceil division
        self.assertEqual(len(paths), expected_files)
        total = sum(
            mrcfile.open(p, permissive=True).data.shape[0] for p in paths
        )
        self.assertEqual(total, n)

    def test_image_shape_matches_volume(self):
        """Each output image has the same spatial dims as the input volume."""
        paths, _ = self._run(n_particles=5, images_per_file=100)
        with mrcfile.open(paths[0], permissive=True) as mrc:
            _, h, w = mrc.data.shape
        self.assertEqual(h, self.BOX)
        self.assertEqual(w, self.BOX)

    def test_n_first_particles(self):
        """n_first_particles limits output to requested subset."""
        n_first = 3
        paths, _ = self._run(n_particles=10, images_per_file=100, n_first_particles=n_first)
        total = sum(
            mrcfile.open(p, permissive=True).data.shape[0] for p in paths
        )
        self.assertEqual(total, n_first)

    def test_no_ctf(self):
        """apply_ctf=False runs without error."""
        paths, _ = self._run(n_particles=5, images_per_file=100, apply_ctf=False)
        self.assertTrue(len(paths) > 0)

    def test_snr_lower_means_noisier(self):
        """Smaller SNR must produce higher pixel std (more noise)."""
        _make_star(self.star_path, 20, box=self.BOX, px_A=self.PX_A)

        def mean_std(snr_val):
            out_dir = os.path.join(self.tmp, f"snr_{snr_val}")
            paths = run_simulation(
                volume=self.vol_path,
                in_star=self.star_path,
                output_dir=out_dir,
                basename="stack",
                images_per_file=100,
                batch_size=4,
                simulation_mode="central_slice",
                apply_ctf=False,
                snr=snr_val,
                num_workers=0,
                px_A=self.PX_A,
                device="cpu",
                disable_tqdm=True,
            )
            with mrcfile.open(paths[0], permissive=True) as mrc:
                return float(mrc.data.std())

        std_low_snr = mean_std(0.01)
        std_high_snr = mean_std(10.0)
        self.assertGreater(
            std_low_snr, std_high_snr,
            f"Expected lower SNR to produce higher std, got low={std_low_snr:.4f} high={std_high_snr:.4f}"
        )


class TestSimulateParticlesStarConsistency(unittest.TestCase):
    """
    Tests that verify the output STAR file correctly maps all particles to
    valid image indices in the output MRC stacks.
    These are the regression tests guarding against the multi-GPU STAR-writing bug.
    """

    BOX = 32
    PX_A = 1.0

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.vol_path = os.path.join(self.tmp, "volume.mrc")
        self.star_path = os.path.join(self.tmp, "particles.star")
        _make_volume(self.vol_path, box=self.BOX, px_A=self.PX_A)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run_and_write_star(self, n_particles: int, images_per_file: int,
                            n_first_particles=None) -> str:
        """Run simulation and produce the output STAR via simulate_particles_cli logic."""
        from cryoPARES.simulation.simulateParticles import _write_output_star
        _make_star(self.star_path, n_particles, box=self.BOX, px_A=self.PX_A)
        out_dir = os.path.join(self.tmp, "out")
        paths = run_simulation(
            volume=self.vol_path,
            in_star=self.star_path,
            output_dir=out_dir,
            basename="stack",
            images_per_file=images_per_file,
            batch_size=4,
            simulation_mode="central_slice",
            apply_ctf=False,
            snr=None,
            num_workers=0,
            px_A=self.PX_A,
            device="cpu",
            disable_tqdm=True,
            n_first_particles=n_first_particles,
        )
        out_star = _write_output_star(
            stack_paths=paths,
            output_dir=out_dir,
            basename="stack",
            in_star=self.star_path,
            images_per_file=images_per_file,
            n_first_particles=n_first_particles,
        )
        return out_star, out_dir

    def test_star_consistency_single_file(self):
        """STAR indices are valid when all particles fit in one file."""
        out_star, out_dir = self._run_and_write_star(n_particles=8, images_per_file=100)
        _check_star_consistency(out_star, out_dir)

    def test_star_consistency_multiple_files(self):
        """STAR indices are valid when particles span multiple files."""
        out_star, out_dir = self._run_and_write_star(n_particles=11, images_per_file=4)
        _check_star_consistency(out_star, out_dir)

    def test_star_consistency_exact_multiple(self):
        """STAR indices are valid when n_particles is exact multiple of images_per_file."""
        out_star, out_dir = self._run_and_write_star(n_particles=12, images_per_file=4)
        _check_star_consistency(out_star, out_dir)

    def test_star_consistency_n_first_particles(self):
        """STAR indices are valid when n_first_particles limits the run."""
        out_star, out_dir = self._run_and_write_star(
            n_particles=20, images_per_file=4, n_first_particles=7
        )
        _check_star_consistency(out_star, out_dir)

    def test_star_particle_count_matches(self):
        """STAR file has exactly n_particles rows."""
        n = 9
        out_star, _ = self._run_and_write_star(n_particles=n, images_per_file=4)
        data = starfile.read(out_star)
        parts_df = data["particles"] if isinstance(data, dict) else data
        self.assertEqual(len(parts_df), n)


if __name__ == "__main__":
    unittest.main()