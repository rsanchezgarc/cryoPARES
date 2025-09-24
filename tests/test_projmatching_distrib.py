# tests/test_projmatching.py
import unittest
import os
import tempfile
import shutil
import numpy as np
import pandas as pd
import starfile
import mrcfile

from cryoPARES.projmatching.projmatching import projmatching_starfile


class TestProjMatching(unittest.TestCase):

    def setUp(self):
        # temp workspace
        self.test_dir = tempfile.mkdtemp()
        self.particles_dir = os.path.join(self.test_dir, "particles")
        os.makedirs(self.particles_dir, exist_ok=True)

        # dummy MRCS stack
        self.mrcs_fname = os.path.join(self.particles_dir, "dummy_particles.mrcs")
        data = np.zeros((10, 64, 64), dtype=np.float32)
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

        # dummy reference volume
        self.reference_vol_fname = os.path.join(self.test_dir, "dummy_reference.mrc")
        with mrcfile.new(self.reference_vol_fname,
                         data=np.zeros((64, 64, 64), dtype=np.float32)) as mrc:
            mrc.voxel_size = (1.0, 1.0, 1.0)  # Å

        # outputs
        self.output_single_job = os.path.join(self.test_dir, "projmatching_single.star")
        self.output_distributed = os.path.join(self.test_dir, "projmatching_distributed.star")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_projmatching_consistency(self):
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

        # "Distributed" (multi-job)
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

        # Compare outputs
        star_single = starfile.read(self.output_single_job)
        star_distributed = starfile.read(self.output_distributed)

        # same shape
        self.assertEqual(star_single["particles"].shape, star_distributed["particles"].shape)

        # numeric columns likely modified by projmatching
        candidate_cols = [
            "rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi",
            "rlnOriginX", "rlnOriginY",               # if your tool writes pixel shifts
            "rlnOriginXAngst", "rlnOriginYAngst",     # or writes Å shifts
            "rlnPredPoseConfidence"                   # if produced
        ]

        for col in candidate_cols:
            if col in star_single["particles"].columns and col in star_distributed["particles"].columns:
                a = np.asarray(star_single["particles"][col].values, dtype=np.float64)
                b = np.asarray(star_distributed["particles"][col].values, dtype=np.float64)
                self.assertTrue(
                    np.allclose(a, b, atol=1e-5),
                    f"Mismatch in column {col}"
                )


if __name__ == '__main__':
    unittest.main()
