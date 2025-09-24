# tests/test_reconstruct.py
import unittest
import os
import tempfile
import shutil
import numpy as np
import pandas as pd
import starfile
import mrcfile

from cryoPARES.reconstruction.reconstruct import reconstruct_starfile
# "distributed" just means running with n_jobs > 1
from cryoPARES.reconstruction.reconstruct import reconstruct_starfile as distributed_reconstruct


class TestReconstruct(unittest.TestCase):

    def setUp(self):
        # temp workspace
        self.test_dir = tempfile.mkdtemp()
        self.particles_dir = os.path.join(self.test_dir, "particles")
        os.makedirs(self.particles_dir, exist_ok=True)

        # create a dummy MRCS stack
        self.mrcs_fname = os.path.join(self.particles_dir, "dummy_particles.mrcs")
        data = np.zeros((10, 64, 64), dtype=np.float32)
        with mrcfile.new(self.mrcs_fname, data=data) as mrc:
            mrc.voxel_size = (1.0, 1.0, 1.0)

        # create a minimal RELION-style STAR with data_optics and data_particles
        self.particles_star_fname = os.path.join(self.test_dir, "dummy_particles.star")
        stack_basename = os.path.basename(self.mrcs_fname)

        # optics table (starstack expects rlnImagePixelSize here)
        optics_df = pd.DataFrame({
            "rlnOpticsGroup":            [1],
            "rlnImageSize":              [64],
            "rlnImagePixelSize":         [1.0],  # Å/pixel
            "rlnCtfDataArePhaseFlipped": [0],
            "rlnVoltage":                [300.0],
            "rlnSphericalAberration":    [2.7],
            "rlnAmplitudeContrast":      [0.1],
        })

        # particles table
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
            # origin shifts (required!)
            "rlnOriginXAngst":    [0.0] * 10,
            "rlnOriginYAngst":    [0.0] * 10,
            # duplicate/extra fields (harmless)
            "rlnVoltage":               [300.0] * 10,
            "rlnSphericalAberration":   [2.7] * 10,
            "rlnAmplitudeContrast":     [0.1] * 10,
            "rlnDetectorPixelSize":     [1.0] * 10,
            "rlnMagnification":         [10000.0] * 10,
        })

        # write STAR file with optics and particles blocks
        starfile.write({"optics": optics_df, "particles": particles_df},
                       self.particles_star_fname,
                       overwrite=True)

        # output paths
        self.output_single_job = os.path.join(self.test_dir, "reconstruct_single.mrc")
        self.output_distributed = os.path.join(self.test_dir, "reconstruct_distributed.mrc")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_reconstruct_consistency(self):
        # Single-job reconstruction
        reconstruct_starfile(
            particles_star_fname=self.particles_star_fname,
            symmetry="C1",
            output_fname=self.output_single_job,
            particles_dir=self.particles_dir,
            n_jobs=1,
            num_dataworkers=0,
            batch_size=2,
            use_cuda=False,
            correct_ctf=True
        )

        # Distributed (multi-job) reconstruction
        distributed_reconstruct(
            particles_star_fname=self.particles_star_fname,
            symmetry="C1",
            output_fname=self.output_distributed,
            particles_dir=self.particles_dir,
            n_jobs=2,
            num_dataworkers=1,
            batch_size=2,
            use_cuda=False,
            correct_ctf=True
        )

        # Compare volumes
        with mrcfile.open(self.output_single_job, permissive=True) as mrc_single, \
             mrcfile.open(self.output_distributed, permissive=True) as mrc_distributed:

            self.assertEqual(mrc_single.data.shape, mrc_distributed.data.shape)
            self.assertTrue(np.allclose(mrc_single.data, mrc_distributed.data, atol=1e-5))


if __name__ == '__main__':
    unittest.main()
