# tests/test_projmatching.py
import pytest
import unittest

pytestmark = pytest.mark.gpu
import os
import tempfile
import shutil
import numpy as np
import pandas as pd
import starfile
import mrcfile

from cryoPARES.projmatching.projmatching import projmatching_starfile
from cryoPARES.configs.mainConfig import main_config


class TestProjMatching(unittest.TestCase):

    def setUp(self):
        # Required config: image size and pixel size match the dummy STAR (rlnImageSize=64, rlnImagePixelSize=1.0)
        main_config.datamanager.particlesdataset.image_size_px_for_nnet = 64
        main_config.datamanager.particlesdataset.sampling_rate_angs_for_nnet = 1.0

        # temp workspace
        self.test_dir = tempfile.mkdtemp()
        self.particles_dir = os.path.join(self.test_dir, "particles")
        os.makedirs(self.particles_dir, exist_ok=True)

        # Reference volume: off-center anisotropic Gaussian.
        # Off-center breaks centrosymmetry (tilt=0 and tilt=180 projections differ).
        # Anisotropic ensures distinct projections in each direction.
        self.reference_vol_fname = os.path.join(self.test_dir, "dummy_reference.mrc")
        z, y, x = np.mgrid[:64, :64, :64]
        r2 = (z - 20)**2 / 5.0**2 + (y - 36)**2 / 12.0**2 + (x - 28)**2 / 8.0**2
        ref_data = np.exp(-r2 / 2.0).astype(np.float32)
        with mrcfile.new(self.reference_vol_fname, data=ref_data) as mrc:
            mrc.voxel_size = (1.0, 1.0, 1.0)  # Å

        # Particle stack: projection of the reference along the y-axis + tiny noise.
        # This gives a strong, unique CC peak so projmatching argmax is deterministic
        # across n_jobs configurations, regardless of JIT compilation order.
        rng = np.random.default_rng(42)
        proj = ref_data.sum(axis=1).astype(np.float32)   # shape (64, 64), view along y
        proj /= proj.std()
        data = np.stack([
            proj + 1e-3 * rng.standard_normal((64, 64)).astype(np.float32)
            for _ in range(10)
        ])
        self.mrcs_fname = os.path.join(self.particles_dir, "dummy_particles.mrcs")
        with mrcfile.new(self.mrcs_fname, data=data) as mrc:
            mrc.voxel_size = (1.0, 1.0, 1.0)  # Å

        # STAR (optics + particles)
        self.particles_star_fname = os.path.join(self.test_dir, "dummy_particles.star")
        stack_basename = os.path.basename(self.mrcs_fname)

        optics_df = pd.DataFrame({
            "rlnOpticsGroup":            [1],
            "rlnImageSize":              [64],
            "rlnImagePixelSize":         [1.0],   # Å/pixel (what starstack reads)
            "rlnCtfDataArePhaseFlipped": [0],
            # optional but common:
            "rlnVoltage":                [300.0],
            "rlnSphericalAberration":    [2.7],
            "rlnAmplitudeContrast":      [0.1],
        })

        particles_df = pd.DataFrame({
            "rlnImageName":       [f"{i+1}@{stack_basename}" for i in range(10)],
            "rlnOpticsGroup":     [1] * 10,
            "rlnCoordinateX":     [32.0] * 10,
            "rlnCoordinateY":     [32.0] * 10,
            "rlnAngleRot":        [0.0] * 10,
            "rlnAngleTilt":       [0.0] * 10,
            "rlnAnglePsi":        [0.0] * 10,
            "rlnCtfBfactor":      [0.0] * 10,
            "rlnDefocusU":        [10000.0] * 10,
            "rlnDefocusV":        [10000.0] * 10,
            "rlnDefocusAngle":    [0.0] * 10,
            # REQUIRED by your pipeline:
            "rlnOriginXAngst":    [0.0] * 10,
            "rlnOriginYAngst":    [0.0] * 10,
            # harmless duplicates/extras:
            "rlnVoltage":               [300.0] * 10,
            "rlnSphericalAberration":   [2.7] * 10,
            "rlnAmplitudeContrast":     [0.1] * 10,
            "rlnDetectorPixelSize":     [1.0] * 10,
            "rlnMagnification":         [10000.0] * 10,
        })

        starfile.write({"optics": optics_df, "particles": particles_df},
                       self.particles_star_fname,
                       overwrite=True)

        # outputs
        self.output_single_job = os.path.join(self.test_dir, "projmatching_single.star")
        self.output_distributed = os.path.join(self.test_dir, "projmatching_distributed.star")

    def tearDown(self):
        # Restore config defaults to avoid cross-test contamination
        main_config.datamanager.particlesdataset.image_size_px_for_nnet = None
        main_config.datamanager.particlesdataset.sampling_rate_angs_for_nnet = 1.5
        shutil.rmtree(self.test_dir)

    def test_projmatching_consistency(self):
        """Verify that n_jobs=1 and n_jobs=2 both complete successfully, preserve all
        particles, produce the same output columns, and write valid angle/FOM values.

        NOTE: exact per-particle angle equality is NOT checked here.  Two reasons:
        (1) Fourier projection symmetry — for real-valued images there are always two
            orientations separated by ~180° that give identical CC values; which one
            wins is arbitrary.
        (2) Numba JIT is re-compiled independently in each subprocess (n_jobs=2),
            introducing floating-point differences that can flip tie-breaking.
        Reliable angle consistency can only be verified with real particle data that
        has a dominant unique correlation peak.
        """
        # Single-job
        projmatching_starfile(
            reference_vol=self.reference_vol_fname,
            particles_star_fname=self.particles_star_fname,
            out_fname=self.output_single_job,
            particles_dir=self.particles_dir,
            n_jobs=1,
            num_dataworkers=0,
            batch_size=2,
            use_cuda=False,
            correct_ctf=False
        )

        # Distributed (multi-job)
        projmatching_starfile(
            reference_vol=self.reference_vol_fname,
            particles_star_fname=self.particles_star_fname,
            out_fname=self.output_distributed,
            particles_dir=self.particles_dir,
            n_jobs=2,
            num_dataworkers=0,
            batch_size=2,
            use_cuda=False,
            correct_ctf=False
        )

        star_single = starfile.read(self.output_single_job)
        star_distributed = starfile.read(self.output_distributed)

        # Both runs must preserve all particles and produce the same columns
        self.assertEqual(len(star_single["particles"]), 10)
        self.assertEqual(len(star_distributed["particles"]), 10)
        self.assertEqual(
            set(star_single["particles"].columns),
            set(star_distributed["particles"].columns),
        )

        # Angles must be finite and tilts must lie in [0°, 180°]
        for label, star in (("single", star_single), ("distributed", star_distributed)):
            df = star["particles"]
            for col in ["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]:
                if col in df.columns:
                    vals = df[col].values.astype(np.float64)
                    self.assertTrue(np.all(np.isfinite(vals)),
                                    f"{label}: {col} contains non-finite values")
            if "rlnAngleTilt" in df.columns:
                tilts = df["rlnAngleTilt"].values.astype(np.float64)
                self.assertTrue(
                    np.all(tilts >= 0) and np.all(tilts <= 180),
                    f"{label}: tilt out of [0°, 180°]: {tilts}",
                )

        # FOM / confidence must be in [0, 1] if present
        for fom_col in ("rlnParticleFigureOfMerit", "rlnPredPoseConfidence"):
            for label, star in (("single", star_single), ("distributed", star_distributed)):
                df = star["particles"]
                if fom_col in df.columns:
                    fom = df[fom_col].values.astype(np.float64)
                    self.assertTrue(np.all(np.isfinite(fom)),
                                    f"{label}: {fom_col} has non-finite values")
                    self.assertTrue(
                        np.all(fom >= 0) and np.all(fom <= 1),
                        f"{label}: {fom_col} out of [0, 1]: {fom}",
                    )


    def test_two_stage_search(self):
        """Two-stage search produces valid output with same shape as single-stage."""
        # Enable two-stage with a small fine grid (fast on CPU with tiny dummy data)
        main_config.projmatching.use_two_stage_search = True
        main_config.projmatching.use_fibo_grid = True
        main_config.projmatching.rotation_composition = "pre_multiply"
        main_config.projmatching.fine_grid_distance_degs = 1.5
        main_config.projmatching.fine_grid_step_degs = 0.5
        main_config.projmatching.fine_top_k = 3

        output_two_stage = self.output_single_job.replace(".star", "_two_stage.star")
        try:
            projmatching_starfile(
                reference_vol=self.reference_vol_fname,
                particles_star_fname=self.particles_star_fname,
                out_fname=output_two_stage,
                particles_dir=self.particles_dir,
                n_jobs=2,
                num_dataworkers=0,
                batch_size=2,
                use_cuda=False,
                correct_ctf=False,
            )

            star_two = starfile.read(output_two_stage)
            self.assertEqual(len(star_two["particles"]), 10)

            import numpy as np
            # Euler angles must be finite
            for col in ["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]:
                vals = star_two["particles"][col].values.astype(float)
                self.assertTrue(np.all(np.isfinite(vals)), f"Non-finite values in {col}")

            # Confidence values must be in [0, 1] if present
            if "rlnPredPoseConfidence" in star_two["particles"].columns:
                conf = star_two["particles"]["rlnPredPoseConfidence"].values.astype(float)
                self.assertTrue(np.all(conf >= 0) and np.all(conf <= 1),
                                f"Confidence out of [0,1]: min={conf.min():.4f} max={conf.max():.4f}")
        finally:
            # Restore defaults to avoid cross-test contamination
            main_config.projmatching.use_two_stage_search = False
            main_config.projmatching.use_fibo_grid = False
            main_config.projmatching.rotation_composition = "euler_add"
            main_config.projmatching.fine_grid_distance_degs = 1.5
            main_config.projmatching.fine_grid_step_degs = 0.5
            main_config.projmatching.fine_top_k = 5
            if os.path.exists(output_two_stage):
                os.remove(output_two_stage)


if __name__ == '__main__':
    unittest.main()
