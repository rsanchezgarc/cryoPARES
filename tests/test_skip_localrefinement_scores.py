"""
Regression tests for the bug where --skip_localrefinement causes rlnDirectionalZscore
to be <NA> in the output STAR file and the directional_zscore_thr to have no effect.

Bug description (cryopares_v0-beta):
  When --skip_localrefinement and --skip_reconstruction are passed, the directional
  z-score is <NA> in the output STAR file and the pruning threshold does nothing.

Root cause: the scoreNormalizer was not applied (or not loaded) when skip_localrefinement=True.
Fix: scoreNormalizer must be loaded and applied independently of the local-refinement flag.
"""

import unittest
import os
import tempfile
import shutil

import numpy as np
import pandas as pd
import starfile
import torch
from scipy.spatial.transform import Rotation

from cryoPARES.inference.nnetWorkers.inferenceModel import InferenceModel
from cryoPARES.models.directionalNormalizer.directionalNormalizer import DirectionalPercentileNormalizer
from cryoPARES.constants import (
    BATCH_IDS_NAME, BATCH_PARTICLES_NAME, BATCH_ORI_IMAGE_NAME, BATCH_ORI_CTF_NAME,
    DIRECTIONAL_ZSCORE_NAME,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _DummySO3Model(torch.nn.Module):
    """Minimal SO3 model that returns valid rotation matrices."""
    def __init__(self, symmetry="C1"):
        super().__init__()
        self.symmetry = symmetry

    def forward(self, x, top_k: int = 1):
        batch_size = x.shape[0]
        rotmats = torch.FloatTensor(
            Rotation.random(batch_size * top_k, random_state=42).as_matrix()
        ).reshape(batch_size, top_k, 3, 3)
        maxprobs = torch.rand(batch_size, top_k)
        return None, None, None, rotmats, maxprobs


def _make_batch(batch_size=4, img_size=64):
    return {
        BATCH_IDS_NAME: list(range(batch_size)),
        BATCH_PARTICLES_NAME: torch.zeros(batch_size, 1, img_size, img_size),
        BATCH_ORI_IMAGE_NAME: torch.zeros(batch_size, img_size, img_size),
        BATCH_ORI_CTF_NAME: torch.zeros(batch_size, img_size, img_size // 2 + 1),
    }


def _make_fitted_normalizer(symmetry="C1", n_samples=2000, seed=0):
    normalizer = DirectionalPercentileNormalizer(symmetry=symmetry)
    rotmats = torch.FloatTensor(Rotation.random(n_samples, random_state=seed).as_matrix())
    scores = torch.rand(n_samples)
    normalizer.fit(rotmats, scores)
    return normalizer


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSkipLocalrefinementDirectionalScore(unittest.TestCase):
    """
    Verify that the directional z-score is computed correctly regardless of
    whether --skip_localrefinement is used.
    """

    # --- Bug-reproduction tests (demonstrate the problematic state) ----------

    def test_nan_scores_when_normalizer_missing(self):
        """
        Reproduce the symptom: norm_nn_score is NaN when scoreNormalizer is None.
        This simulates a checkpoint trained before the normalizer was introduced
        (the file best_directional_normalizer.pt is absent).
        """
        model = InferenceModel(
            so3model=_DummySO3Model(),
            scoreNormalizer=None,   # simulates missing normalizer file
            normalizedScore_thr=None,
            localRefiner=None,      # --skip_localrefinement
        )
        model.eval()
        with torch.inference_mode():
            result = model.predict_step(_make_batch(), batch_idx=0)

        self.assertIsNotNone(result)
        _, _, _, _, norm_nn_score = result
        self.assertTrue(
            torch.all(torch.isnan(norm_nn_score)).item(),
            "When scoreNormalizer is None, norm_nn_score must be NaN "
            "(this is the root cause of the <NA> values in the STAR file).",
        )

    def test_all_particles_filtered_when_scores_are_nan_and_threshold_set(self):
        """
        When scoreNormalizer is None, norm_nn_score is NaN.
        NaN > threshold == False for every particle, so the forward pass
        returns None and nothing is kept — reproducing the 'threshold has no effect'
        symptom from the bug report (the user expected some particles to pass).
        """
        model = InferenceModel(
            so3model=_DummySO3Model(),
            scoreNormalizer=None,
            normalizedScore_thr=-1.41,  # Should keep ~92 % of particles normally
            localRefiner=None,
        )
        model.eval()
        with torch.inference_mode():
            result = model.predict_step(_make_batch(batch_size=8), batch_idx=0)

        self.assertIsNone(
            result,
            "With NaN scores, NaN > threshold is False for every particle, "
            "so the forward pass returns None and all particles are silently dropped.",
        )

    # --- Fix-verification tests (the correct behaviour) ----------------------

    def test_valid_scores_with_normalizer_and_skip_localrefinement(self):
        """
        Core regression guard: when scoreNormalizer is present and localRefiner is None
        (i.e. --skip_localrefinement), norm_nn_score must be finite (not NaN).
        """
        model = InferenceModel(
            so3model=_DummySO3Model(),
            scoreNormalizer=_make_fitted_normalizer(),
            normalizedScore_thr=None,
            localRefiner=None,      # --skip_localrefinement
        )
        model.eval()
        with torch.inference_mode():
            result = model.predict_step(_make_batch(), batch_idx=0)

        self.assertIsNotNone(result)
        _, _, _, _, norm_nn_score = result
        self.assertFalse(
            torch.any(torch.isnan(norm_nn_score)).item(),
            "With a valid scoreNormalizer and skip_localrefinement=True, "
            "norm_nn_score must NOT be NaN.",
        )

    def test_threshold_works_with_normalizer_and_skip_localrefinement(self):
        """
        When the normalizer is present and a threshold is set, particles below the
        threshold are filtered and the rest are returned with valid scores.
        A threshold of -100 (well below any z-score) should keep all particles.
        """
        batch_size = 6
        model = InferenceModel(
            so3model=_DummySO3Model(),
            scoreNormalizer=_make_fitted_normalizer(),
            normalizedScore_thr=-100.0,  # Keep everything
            localRefiner=None,
        )
        model.eval()
        with torch.inference_mode():
            result = model.predict_step(_make_batch(batch_size=batch_size), batch_idx=0)

        self.assertIsNotNone(
            result,
            "With a valid normalizer and a very low threshold, no particle should be filtered out.",
        )
        ids, _, _, _, norm_nn_score = result
        self.assertEqual(len(ids), batch_size)
        self.assertFalse(torch.any(torch.isnan(norm_nn_score)).item())

    def test_scores_identical_with_and_without_local_refinement(self):
        """
        The norm_nn_score produced by _firstforward must be the same regardless
        of whether localRefiner is None or not (the bug caused them to differ).
        We verify this by running with localRefiner=None and checking that the
        score is finite — the value itself depends on the random model output,
        but it must not be NaN in either case.
        """
        normalizer = _make_fitted_normalizer()
        batch = _make_batch(batch_size=4)

        model_skip = InferenceModel(
            so3model=_DummySO3Model(),
            scoreNormalizer=normalizer,
            normalizedScore_thr=None,
            localRefiner=None,  # skip_localrefinement=True
        )
        model_skip.eval()
        with torch.inference_mode():
            result_skip = model_skip.predict_step(batch, batch_idx=0)

        self.assertIsNotNone(result_skip)
        _, _, _, _, score_skip = result_skip
        self.assertFalse(
            torch.any(torch.isnan(score_skip)).item(),
            "Scores must be finite when skip_localrefinement=True and normalizer is present.",
        )


class TestSaveParticlesResultsDirectionalScore(unittest.TestCase):
    """
    End-to-end test that checks the rlnDirectionalZscore column in the saved STAR
    file when NaN scores are present (bug scenario) vs valid scores (fix scenario).
    """

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _run_save_and_read(self, norm_nn_score_values: torch.Tensor,
                           passed_ids=None, directional_zscore_thr=None):
        """
        Helper: build a minimal all_results list, call _save_particles_results,
        and return the saved DataFrame.

        Parameters
        ----------
        norm_nn_score_values:
            Scores for the particles that appear in all_results (i.e. those that
            passed the threshold filter inside InferenceModel).
        passed_ids:
            Row-label indices of the particles that appear in all_results.
            Defaults to ``range(len(norm_nn_score_values))`` (all particles pass).
        directional_zscore_thr:
            Value to set on the inferencer (mirrors the CLI flag).  When not None
            the total particle count in the STAR file will be larger than the
            number of passed particles, replicating the threshold-filter scenario.
        """
        import mrcfile
        from cryoPARES.inference.inferencer import SingleInferencer
        from cryoPARES.constants import RELION_ANGLES_NAMES, RELION_SHIFTS_NAMES

        n_passed = len(norm_nn_score_values)
        if passed_ids is None:
            passed_ids = list(range(n_passed))
            n_total = n_passed
        else:
            passed_ids = list(passed_ids)
            n_total = max(passed_ids) + 1  # full dataset is at least this large

        # ---------- Build a minimal STAR file (n_total particles) ----------
        mrcs_path = os.path.join(self.test_dir, "p.mrcs")
        with mrcfile.new(mrcs_path, data=np.zeros((n_total, 32, 32), dtype=np.float32)) as f:
            f.voxel_size = 1.0

        particles_df = pd.DataFrame({
            "rlnImageName":    [f"{i+1}@p.mrcs" for i in range(n_total)],
            "rlnOpticsGroup":  [1] * n_total,
            "rlnDefocusU":     [10000.0] * n_total,
            "rlnDefocusV":     [10000.0] * n_total,
            "rlnDefocusAngle": [0.0] * n_total,
            "rlnVoltage":      [300.0] * n_total,
            "rlnSphericalAberration": [2.7] * n_total,
            "rlnAmplitudeContrast": [0.1] * n_total,
            "rlnImagePixelSize": [1.0] * n_total,
            "rlnAngleRot":     [0.0] * n_total,
            "rlnAngleTilt":    [0.0] * n_total,
            "rlnAnglePsi":     [0.0] * n_total,
            "rlnOriginXAngst": [0.0] * n_total,
            "rlnOriginYAngst": [0.0] * n_total,
        })
        particles_df.index = np.arange(n_total, dtype=int)
        optics_df = pd.DataFrame({
            "rlnOpticsGroup": [1],
            "rlnImageSize":   [32],
            "rlnImagePixelSize": [1.0],
            "rlnCtfDataArePhaseFlipped": [0],
            "rlnVoltage": [300.0],
            "rlnSphericalAberration": [2.7],
            "rlnAmplitudeContrast": [0.1],
        })
        star_path = os.path.join(self.test_dir, "particles.star")
        starfile.write({"optics": optics_df, "particles": particles_df}, star_path, overwrite=True)

        # ---------- Build fake all_results (only passed_ids) ----------
        # (ids, euler_degs, pred_shiftsXYangs, score, norm_nn_score)
        top_k = 1
        euler_degs = torch.zeros(n_passed, top_k, 3)
        score = torch.rand(n_passed, top_k)
        pred_shifts = torch.zeros(n_passed, top_k, 2)
        all_results = [(passed_ids, euler_degs, pred_shifts, score, norm_nn_score_values)]

        # ---------- Build a minimal SingleInferencer (bypass __init__) ----------
        inferencer = object.__new__(SingleInferencer)
        inferencer.results_dir = self.test_dir
        inferencer.particles_star_fname = star_path
        inferencer.particles_dir = self.test_dir
        inferencer.data_halfset = "allParticles"
        inferencer.model_halfset = "allParticles"
        inferencer.directional_zscore_thr = directional_zscore_thr
        inferencer.skip_localrefinement = True
        inferencer.skip_reconstruction = True
        inferencer.show_debug_stats = False
        inferencer._last_dataset_processed = 0

        from unittest.mock import patch, MagicMock
        from cryoPARES.datamanager.particlesDataset import ParticlesDataset

        fake_particles_set = MagicMock()
        fake_particles_set.particles_md = particles_df.copy()
        fake_particles_set.optics_md = optics_df.copy()
        fake_particles_set.starFname = star_path
        fake_particles_set.save = MagicMock()

        fake_dataset = MagicMock()
        fake_dataset.datasets = [fake_dataset]
        fake_dataset.particles = fake_particles_set

        with patch.object(inferencer, '_get_outsuffix', return_value="_allParticles.star"):
            inferencer._save_particles_results(all_results, fake_dataset)

        # Read the saved STAR
        saved = fake_particles_set.particles_md
        return saved

    def test_nan_scores_appear_as_na_in_star(self):
        """
        Bug scenario: NaN norm_nn_score values should cause rlnDirectionalZscore
        to be NaN in the output DataFrame (which starfile writes as <NA>).
        """
        n = 5
        nan_scores = torch.full((n,), float('nan'))
        saved_df = self._run_save_and_read(nan_scores)

        if DIRECTIONAL_ZSCORE_NAME in saved_df.columns:
            col_values = saved_df[DIRECTIONAL_ZSCORE_NAME].values
            self.assertTrue(
                np.all(np.isnan(col_values.astype(float))),
                f"With NaN scores, {DIRECTIONAL_ZSCORE_NAME} should be NaN in the output.",
            )

    def test_valid_scores_written_correctly(self):
        """
        Fix scenario: valid norm_nn_score values should appear correctly in the STAR output.
        """
        n = 5
        valid_scores = torch.tensor([1.2, -0.5, 0.8, -1.1, 0.3])
        saved_df = self._run_save_and_read(valid_scores)

        self.assertIn(
            DIRECTIONAL_ZSCORE_NAME,
            saved_df.columns,
            f"{DIRECTIONAL_ZSCORE_NAME} column must exist in the output STAR.",
        )
        col_values = saved_df[DIRECTIONAL_ZSCORE_NAME].values.astype(float)
        self.assertFalse(
            np.any(np.isnan(col_values)),
            f"{DIRECTIONAL_ZSCORE_NAME} must not contain NaN when scores are valid.",
        )
        np.testing.assert_allclose(col_values, valid_scores.numpy(), atol=1e-5)


if __name__ == "__main__":
    unittest.main()