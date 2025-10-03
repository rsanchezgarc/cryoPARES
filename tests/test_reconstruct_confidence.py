"""
Tests for reconstruction with confidence weighting.

These tests verify that:
1. Reconstruction without confidence works correctly (baseline)
2. Reconstruction with confidence weighting produces valid results
3. Multiple poses per particle are handled correctly
4. Bug fixes maintain or improve reconstruction quality
"""
import unittest
import os
import tempfile
import shutil
import numpy as np
import pandas as pd
import starfile
import mrcfile
import torch

from cryoPARES.reconstruction.reconstructor import Reconstructor, ReconstructionParticlesDataset
from cryoPARES.constants import RELION_PRED_POSE_CONFIDENCE_NAME


class TestReconstructConfidence(unittest.TestCase):
    """Test reconstruction with and without confidence weighting"""

    def setUp(self):
        """Create test data with known geometry"""
        self.test_dir = tempfile.mkdtemp()
        self.particles_dir = os.path.join(self.test_dir, "particles")
        os.makedirs(self.particles_dir, exist_ok=True)

        # Create test parameters
        self.box_size = 64
        self.n_particles = 20
        self.sampling_rate = 1.5  # Angstroms/pixel

        # Create a simple test volume (sphere with gradient)
        self.reference_vol = self._create_test_volume()

        # Save reference volume
        self.ref_vol_path = os.path.join(self.test_dir, "reference.mrc")
        with mrcfile.new(self.ref_vol_path, data=self.reference_vol.astype(np.float32), overwrite=True) as mrc:
            mrc.voxel_size = (self.sampling_rate,) * 3

        # Create particle stack with known projections
        self.mrcs_fname = os.path.join(self.particles_dir, "test_particles.mrcs")
        self.particles_data, self.angles, self.confidences = self._create_test_particles()

        with mrcfile.new(self.mrcs_fname, data=self.particles_data, overwrite=True) as mrc:
            mrc.voxel_size = (self.sampling_rate,) * 3

        # Create STAR files (with and without confidence)
        self.star_no_conf = os.path.join(self.test_dir, "particles_no_conf.star")
        self.star_with_conf = os.path.join(self.test_dir, "particles_with_conf.star")
        self._create_star_files()

        # Output paths
        self.output_no_conf = os.path.join(self.test_dir, "recon_no_conf.mrc")
        self.output_with_conf = os.path.join(self.test_dir, "recon_with_conf.mrc")
        self.output_uniform_conf = os.path.join(self.test_dir, "recon_uniform_conf.mrc")

    def tearDown(self):
        """Clean up test directory"""
        shutil.rmtree(self.test_dir)

    def _create_test_volume(self):
        """Create a simple test volume: sphere with radial gradient"""
        vol = np.zeros((self.box_size, self.box_size, self.box_size), dtype=np.float32)
        center = self.box_size // 2
        radius = self.box_size // 4

        for z in range(self.box_size):
            for y in range(self.box_size):
                for x in range(self.box_size):
                    r = np.sqrt((x - center)**2 + (y - center)**2 + (z - center)**2)
                    if r < radius:
                        vol[z, y, x] = (1.0 - r / radius) * 100.0

        return vol

    def _create_test_particles(self):
        """Create particle projections with random orientations and varying confidence"""
        particles = np.zeros((self.n_particles, self.box_size, self.box_size), dtype=np.float32)
        angles = np.zeros((self.n_particles, 3), dtype=np.float32)  # rot, tilt, psi
        confidences = np.zeros(self.n_particles, dtype=np.float32)

        np.random.seed(42)

        for i in range(self.n_particles):
            # Random Euler angles
            rot = np.random.uniform(0, 360)
            tilt = np.random.uniform(0, 180)
            psi = np.random.uniform(0, 360)
            angles[i] = [rot, tilt, psi]

            # Create projection (simplified: just take a central slice for this test)
            # In reality, would rotate volume and extract slice
            particles[i] = self.reference_vol[self.box_size // 2, :, :]

            # Add some noise
            noise_level = np.random.uniform(0, 5.0)
            particles[i] += np.random.randn(self.box_size, self.box_size) * noise_level

            # Assign confidence inversely proportional to noise
            # First half: high confidence (0.8-1.0)
            # Second half: low confidence (0.2-0.5)
            if i < self.n_particles // 2:
                confidences[i] = np.random.uniform(0.8, 1.0)
            else:
                confidences[i] = np.random.uniform(0.2, 0.5)

        return particles, angles, confidences

    def _create_star_files(self):
        """Create STAR files with and without confidence column"""
        stack_basename = os.path.basename(self.mrcs_fname)

        # Optics table
        optics_df = pd.DataFrame({
            "rlnOpticsGroup": [1],
            "rlnImageSize": [self.box_size],
            "rlnImagePixelSize": [self.sampling_rate],
            "rlnCtfDataArePhaseFlipped": [0],
            "rlnVoltage": [300.0],
            "rlnSphericalAberration": [2.7],
            "rlnAmplitudeContrast": [0.1],
        })

        # Particles table (base)
        particles_df = pd.DataFrame({
            "rlnImageName": [f"{i+1}@{stack_basename}" for i in range(self.n_particles)],
            "rlnOpticsGroup": [1] * self.n_particles,
            "rlnAngleRot": self.angles[:, 0],
            "rlnAngleTilt": self.angles[:, 1],
            "rlnAnglePsi": self.angles[:, 2],
            "rlnOriginXAngst": [0.0] * self.n_particles,
            "rlnOriginYAngst": [0.0] * self.n_particles,
            "rlnDefocusU": [10000.0] * self.n_particles,
            "rlnDefocusV": [10000.0] * self.n_particles,
            "rlnDefocusAngle": [0.0] * self.n_particles,
        })

        # STAR without confidence
        starfile.write(
            {"optics": optics_df, "particles": particles_df},
            self.star_no_conf,
            overwrite=True
        )

        # STAR with confidence
        particles_df_conf = particles_df.copy()
        particles_df_conf[RELION_PRED_POSE_CONFIDENCE_NAME] = self.confidences
        starfile.write(
            {"optics": optics_df, "particles": particles_df_conf},
            self.star_with_conf,
            overwrite=True
        )

    def test_reconstruction_no_confidence_single_pose(self):
        """Test baseline: reconstruction without confidence weighting works"""
        reconstructor = Reconstructor(
            symmetry="C1",
            correct_ctf=False,
            eps=1e-3,
            weight_with_confidence=False
        )

        reconstructor.backproject_particles(
            particles_star_fname=self.star_no_conf,
            particles_dir=self.particles_dir,
            batch_size=4,
            num_dataworkers=0
        )

        vol = reconstructor.generate_volume(self.output_no_conf)

        # Basic sanity checks
        self.assertEqual(vol.shape, (self.box_size, self.box_size, self.box_size))
        self.assertFalse(torch.isnan(vol).any(), "Reconstructed volume contains NaN")
        self.assertFalse(torch.isinf(vol).any(), "Reconstructed volume contains Inf")

        # Volume should have non-zero content
        self.assertGreater(vol.abs().max().item(), 0, "Volume is all zeros")

        print(f"✓ No confidence reconstruction: max={vol.max():.3f}, min={vol.min():.3f}, mean={vol.mean():.3f}")

    def test_reconstruction_with_confidence_single_pose(self):
        """Test reconstruction with confidence weighting"""
        reconstructor = Reconstructor(
            symmetry="C1",
            correct_ctf=False,
            eps=1e-3,
            weight_with_confidence=True
        )

        reconstructor.backproject_particles(
            particles_star_fname=self.star_with_conf,
            particles_dir=self.particles_dir,
            batch_size=4,
            num_dataworkers=0
        )

        vol = reconstructor.generate_volume(self.output_with_conf)

        # Basic sanity checks
        self.assertEqual(vol.shape, (self.box_size, self.box_size, self.box_size))
        self.assertFalse(torch.isnan(vol).any(), "Reconstructed volume contains NaN")
        self.assertFalse(torch.isinf(vol).any(), "Reconstructed volume contains Inf")
        self.assertGreater(vol.abs().max().item(), 0, "Volume is all zeros")

        print(f"✓ With confidence reconstruction: max={vol.max():.3f}, min={vol.min():.3f}, mean={vol.mean():.3f}")

    def test_confidence_weighting_effect(self):
        """Test that confidence weighting produces different (hopefully better) results"""
        # Reconstruct without confidence
        reconstructor_no_conf = Reconstructor(
            symmetry="C1",
            correct_ctf=False,
            eps=1e-3,
            weight_with_confidence=False
        )
        reconstructor_no_conf.backproject_particles(
            particles_star_fname=self.star_no_conf,
            particles_dir=self.particles_dir,
            batch_size=4,
            num_dataworkers=0
        )
        vol_no_conf = reconstructor_no_conf.generate_volume()

        # Reconstruct with confidence
        reconstructor_with_conf = Reconstructor(
            symmetry="C1",
            correct_ctf=False,
            eps=1e-3,
            weight_with_confidence=True
        )
        reconstructor_with_conf.backproject_particles(
            particles_star_fname=self.star_with_conf,
            particles_dir=self.particles_dir,
            batch_size=4,
            num_dataworkers=0
        )
        vol_with_conf = reconstructor_with_conf.generate_volume()

        # Volumes should be different (confidence should have an effect)
        diff = (vol_no_conf - vol_with_conf).abs().mean()
        print(f"✓ Mean difference between weighted/unweighted: {diff:.6f}")

        # They should be different but not wildly different
        self.assertGreater(diff.item(), 1e-6, "Confidence weighting had no effect")

        # Standard deviations should be similar order of magnitude
        ratio = vol_with_conf.std() / vol_no_conf.std()
        print(f"✓ Std ratio (with_conf/no_conf): {ratio:.3f}")
        self.assertGreater(ratio, 0.1, "Confidence weighting collapsed the volume")
        self.assertLess(ratio, 10.0, "Confidence weighting exploded the volume")

    def test_uniform_confidence_equals_no_confidence(self):
        """Test that uniform confidence=1.0 gives same result as no confidence"""
        # Create STAR with uniform confidence = 1.0
        stack_basename = os.path.basename(self.mrcs_fname)
        optics_df = pd.DataFrame({
            "rlnOpticsGroup": [1],
            "rlnImageSize": [self.box_size],
            "rlnImagePixelSize": [self.sampling_rate],
            "rlnCtfDataArePhaseFlipped": [0],
            "rlnVoltage": [300.0],
            "rlnSphericalAberration": [2.7],
            "rlnAmplitudeContrast": [0.1],
        })

        particles_df = pd.DataFrame({
            "rlnImageName": [f"{i+1}@{stack_basename}" for i in range(self.n_particles)],
            "rlnOpticsGroup": [1] * self.n_particles,
            "rlnAngleRot": self.angles[:, 0],
            "rlnAngleTilt": self.angles[:, 1],
            "rlnAnglePsi": self.angles[:, 2],
            "rlnOriginXAngst": [0.0] * self.n_particles,
            "rlnOriginYAngst": [0.0] * self.n_particles,
            "rlnDefocusU": [10000.0] * self.n_particles,
            "rlnDefocusV": [10000.0] * self.n_particles,
            "rlnDefocusAngle": [0.0] * self.n_particles,
            RELION_PRED_POSE_CONFIDENCE_NAME: [1.0] * self.n_particles,
        })

        star_uniform = os.path.join(self.test_dir, "uniform_conf.star")
        starfile.write({"optics": optics_df, "particles": particles_df}, star_uniform, overwrite=True)

        # Reconstruct without confidence
        reconstructor_no_conf = Reconstructor(
            symmetry="C1",
            correct_ctf=False,
            eps=1e-3,
            weight_with_confidence=False
        )
        reconstructor_no_conf.backproject_particles(
            particles_star_fname=self.star_no_conf,
            particles_dir=self.particles_dir,
            batch_size=4,
            num_dataworkers=0
        )
        vol_no_conf = reconstructor_no_conf.generate_volume()

        # Reconstruct with uniform confidence = 1.0
        reconstructor_uniform = Reconstructor(
            symmetry="C1",
            correct_ctf=False,
            eps=1e-3,
            weight_with_confidence=True
        )
        reconstructor_uniform.backproject_particles(
            particles_star_fname=star_uniform,
            particles_dir=self.particles_dir,
            batch_size=4,
            num_dataworkers=0
        )
        vol_uniform = reconstructor_uniform.generate_volume()

        # Should be very close (numerical precision differences allowed)
        diff = (vol_no_conf - vol_uniform).abs().max()
        print(f"✓ Max difference (no_conf vs uniform_conf=1.0): {diff:.6e}")
        self.assertLess(diff.item(), 1e-4, "Uniform confidence=1.0 differs from no confidence")

    def test_dataset_confidence_loading(self):
        """Test that ReconstructionParticlesDataset loads confidence correctly"""
        # Without confidence
        dataset_no_conf = ReconstructionParticlesDataset(
            particles_star_fname=self.star_no_conf,
            particles_dir=self.particles_dir,
            correct_ctf=False,
            return_confidence=False
        )

        sample = dataset_no_conf[0]
        self.assertEqual(len(sample), 5, "Should return 5 items without confidence")

        # With confidence
        dataset_with_conf = ReconstructionParticlesDataset(
            particles_star_fname=self.star_with_conf,
            particles_dir=self.particles_dir,
            correct_ctf=False,
            return_confidence=True
        )

        sample = dataset_with_conf[0]
        self.assertEqual(len(sample), 6, "Should return 6 items with confidence")

        iid, img, ctf, rotMat, hwShiftAngs, confidence = sample

        # Check confidence shape and value
        self.assertIsInstance(confidence, torch.Tensor)
        print(f"✓ Confidence shape from dataset: {confidence.shape}")
        self.assertTrue(0.0 <= confidence.item() <= 1.5, f"Confidence {confidence.item()} out of range")

        # Verify it matches the stored value
        expected_conf = self.confidences[0]
        self.assertAlmostEqual(confidence.item(), expected_conf, places=5)

    def test_multiple_poses_per_particle(self):
        """Test reconstruction with multiple poses per particle (M > 1)"""
        # Create a simple test with known rotations
        reconstructor = Reconstructor(
            symmetry="C1",
            correct_ctf=False,
            eps=1e-3,
            weight_with_confidence=False
        )

        # Set metadata manually
        from cryoPARES.datamanager.particlesDataset import ParticlesDataset
        dataset = ReconstructionParticlesDataset(
            particles_star_fname=self.star_no_conf,
            particles_dir=self.particles_dir,
            correct_ctf=False,
            return_confidence=False
        )
        reconstructor.set_metadata_from_particles(dataset)

        # Create batch with multiple poses per particle
        B = 2  # batch size
        M = 3  # poses per particle

        imgs = torch.randn(B, self.box_size, self.box_size)
        ctf = torch.ones(B, self.box_size, self.box_size // 2 + 1)
        rotMats = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, M, -1, -1)  # [B, M, 3, 3]
        hwShiftAngs = torch.zeros(B, M, 2)  # [B, M, 2]

        # Should not raise an error
        try:
            reconstructor._backproject_batch(imgs, ctf, rotMats, hwShiftAngs, confidence=None)
            print("✓ Multiple poses per particle processed successfully")
        except Exception as e:
            self.fail(f"Failed to process multiple poses: {e}")

    def test_confidence_with_multiple_poses(self):
        """Test confidence weighting with multiple poses per particle"""
        reconstructor = Reconstructor(
            symmetry="C1",
            correct_ctf=False,
            eps=1e-3,
            weight_with_confidence=True
        )

        # Set metadata manually
        dataset = ReconstructionParticlesDataset(
            particles_star_fname=self.star_with_conf,
            particles_dir=self.particles_dir,
            correct_ctf=False,
            return_confidence=True
        )
        reconstructor.set_metadata_from_particles(dataset)

        # Create batch with multiple poses per particle
        B = 2
        M = 3

        imgs = torch.randn(B, self.box_size, self.box_size)
        ctf = torch.ones(B, self.box_size, self.box_size // 2 + 1)
        rotMats = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, M, -1, -1)  # [B, M, 3, 3]
        hwShiftAngs = torch.zeros(B, M, 2)
        confidence = torch.rand(B, M)  # [B, M] confidence per pose

        # Should not raise an error
        try:
            reconstructor._backproject_batch(imgs, ctf, rotMats, hwShiftAngs, confidence=confidence)
            print("✓ Confidence with multiple poses processed successfully")
        except Exception as e:
            self.fail(f"Failed to process confidence with multiple poses: {e}")


class TestReconstructBufferConsistency(unittest.TestCase):
    """Test that buffers accumulate correctly"""

    def test_buffer_accumulation(self):
        """Test that running backprojection twice gives 2x the values"""
        # Create minimal test setup
        test_dir = tempfile.mkdtemp()
        try:
            particles_dir = os.path.join(test_dir, "particles")
            os.makedirs(particles_dir)

            box_size = 32
            n_particles = 5

            # Create particle stack
            mrcs_fname = os.path.join(particles_dir, "particles.mrcs")
            particles_data = np.random.randn(n_particles, box_size, box_size).astype(np.float32)
            with mrcfile.new(mrcs_fname, data=particles_data, overwrite=True) as mrc:
                mrc.voxel_size = (1.0, 1.0, 1.0)

            # Create STAR file
            star_fname = os.path.join(test_dir, "particles.star")
            stack_basename = os.path.basename(mrcs_fname)

            optics_df = pd.DataFrame({
                "rlnOpticsGroup": [1],
                "rlnImageSize": [box_size],
                "rlnImagePixelSize": [1.0],
                "rlnCtfDataArePhaseFlipped": [0],
                "rlnVoltage": [300.0],
                "rlnSphericalAberration": [2.7],
                "rlnAmplitudeContrast": [0.1],
            })

            particles_df = pd.DataFrame({
                "rlnImageName": [f"{i+1}@{stack_basename}" for i in range(n_particles)],
                "rlnOpticsGroup": [1] * n_particles,
                "rlnAngleRot": [0.0] * n_particles,
                "rlnAngleTilt": [0.0] * n_particles,
                "rlnAnglePsi": [0.0] * n_particles,
                "rlnOriginXAngst": [0.0] * n_particles,
                "rlnOriginYAngst": [0.0] * n_particles,
                "rlnDefocusU": [10000.0] * n_particles,
                "rlnDefocusV": [10000.0] * n_particles,
                "rlnDefocusAngle": [0.0] * n_particles,
            })

            starfile.write({"optics": optics_df, "particles": particles_df}, star_fname, overwrite=True)

            # Backproject once
            reconstructor = Reconstructor(symmetry="C1", correct_ctf=False, eps=1e-3, weight_with_confidence=False)
            reconstructor.backproject_particles(
                particles_star_fname=star_fname,
                particles_dir=particles_dir,
                batch_size=2,
                num_dataworkers=0
            )

            numerator_1x = reconstructor.numerator.clone()
            weights_1x = reconstructor.weights.clone()

            # Backproject again (should double the values)
            reconstructor.backproject_particles(
                particles_star_fname=star_fname,
                particles_dir=particles_dir,
                batch_size=2,
                num_dataworkers=0
            )

            numerator_2x = reconstructor.numerator.clone()
            weights_2x = reconstructor.weights.clone()

            # Check that values doubled
            np.testing.assert_allclose(
                numerator_2x.numpy(), 2 * numerator_1x.numpy(), rtol=1e-5,
                err_msg="Numerator did not double after second backprojection"
            )
            np.testing.assert_allclose(
                weights_2x.numpy(), 2 * weights_1x.numpy(), rtol=1e-5,
                err_msg="Weights did not double after second backprojection"
            )

            print("✓ Buffer accumulation is consistent")

        finally:
            shutil.rmtree(test_dir)


if __name__ == '__main__':
    unittest.main(verbosity=2)
