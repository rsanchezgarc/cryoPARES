# tests/test_inference_distrib.py
import unittest
import os
import tempfile
import shutil
import numpy as np
import pandas as pd
import starfile
import mrcfile

from unittest.mock import patch
import cryoPARES.inference.infer as infer  # we test infer.distributed_inference


class TestDistributedInference(unittest.TestCase):

    def setUp(self):
        # temp workspace
        self.test_dir = tempfile.mkdtemp()
        self.particles_dir = os.path.join(self.test_dir, "particles")
        os.makedirs(self.particles_dir, exist_ok=True)

        # dummy MRCS stack
        self.mrcs_fname = os.path.join(self.particles_dir, "dummy_particles.mrcs")
        with mrcfile.new(self.mrcs_fname, data=np.zeros((10, 64, 64), dtype=np.float32)) as mrc:
            mrc.voxel_size = (1.0, 1.0, 1.0)  # Å

        # STAR (optics + particles) with halfset labels 1/2 alternating
        self.particles_star_fname = os.path.join(self.test_dir, "dummy_particles.star")
        stack_basename = os.path.basename(self.mrcs_fname)

        optics_df = pd.DataFrame({
            "rlnOpticsGroup":            [1],
            "rlnImageSize":              [64],
            "rlnImagePixelSize":         [1.0],   # Å/pixel
            "rlnCtfDataArePhaseFlipped": [0],
            "rlnVoltage":                [300.0],
            "rlnSphericalAberration":    [2.7],
            "rlnAmplitudeContrast":      [0.1],
        })

        n = 10
        particles_df = pd.DataFrame({
            "rlnImageName":       [f"{i+1}@{stack_basename}" for i in range(n)],
            "rlnOpticsGroup":     [1] * n,
            "rlnCoordinateX":     [32.0] * n,
            "rlnCoordinateY":     [32.0] * n,
            "rlnAngleRot":        [0.0] * n,
            "rlnAngleTilt":       [0.0] * n,
            "rlnAnglePsi":        [0.0] * n,
            "rlnCtfBfactor":      [0.0] * n,
            "rlnDefocusU":        [10000.0] * n,
            "rlnDefocusV":        [10000.0] * n,
            "rlnDefocusAngle":    [0.0] * n,
            "rlnOriginXAngst":    [0.0] * n,
            "rlnOriginYAngst":    [0.0] * n,
            "rlnRandomSubset":    [1 if i % 2 == 0 else 2 for i in range(n)],
        })
        # Use original row numbers as labels (0..n-1)
        particles_df.index = np.arange(n, dtype=int)

        starfile.write({"optics": optics_df, "particles": particles_df},
                       self.particles_star_fname, overwrite=True)

        # results + fake checkpoints
        self.results_dir = os.path.join(self.test_dir, "results")
        os.makedirs(self.results_dir, exist_ok=True)

        # We’ll mock SingleInferencer so checkpoints are irrelevant, but provide dirs anyway
        self.checkpoint_dir = os.path.join(self.test_dir, "ckpts")
        os.makedirs(os.path.join(self.checkpoint_dir, "half1"), exist_ok=True)
        os.makedirs(os.path.join(self.checkpoint_dir, "half2"), exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _make_fake_inferencer(self):
        """Stub SingleInferencer to avoid loading real models/checkpoints and GPU work."""
        star_path = self.particles_star_fname

        class FakeSingleInferencer:
            def __init__(self, *,
                         particles_star_fname,
                         checkpoint_dir,
                         results_dir,
                         data_halfset,
                         model_halfset,
                         particles_dir=None,
                         batch_size=1,
                         num_data_workers=0,
                         use_cuda=False,
                         n_cpus_if_no_cuda=1,
                         compile_model=False,
                         top_k_poses_nnet=1,
                         reference_map=None,
                         reference_mask=None,
                         directional_zscore_thr=None,
                         skip_localrefinement=True,
                         skip_reconstruction=True,
                         subset_idxs=None,
                         show_debug_stats=False,
                         float32_matmul_precision="high"):
                self.particles_star_fname = particles_star_fname
                self.data_halfset = data_halfset
                self.model_halfset = model_halfset
                self.subset_idxs = subset_idxs

            def run(self):
                # Emulate structure expected by infer._flatten_inference_output:
                # list_over_model_halfset [ list_over_datasets [ (particles_md_list, vol) ] ]
                star = starfile.read(star_path)
                df = star["particles"] if isinstance(star, dict) else star

                # filter by halfset like _load_particles_indices_with_halfset
                if "rlnRandomSubset" in df.columns and self.data_halfset in ("half1", "half2"):
                    target = 1 if self.data_halfset == "half1" else 2
                    df = df[df["rlnRandomSubset"].astype(int) == target].copy()

                # IMPORTANT: subset_idxs are **row labels** (original df.index), not positions
                if self.subset_idxs is not None:
                    subset = [int(i) for i in self.subset_idxs]
                    subset = [i for i in subset if i in df.index]
                    # preserve the given order
                    df = df.loc[subset].copy()

                # add deterministic mock outputs
                df = df.copy()
                df["rlnAngleRot"] += 0.0
                df["rlnAngleTilt"] += 0.0
                df["rlnAnglePsi"] += 0.0
                df["rlnPredPoseConfidence"] = 1.0  # pretend perfect confidence
                df["__data_half__"] = self.data_halfset
                df["__model_half__"] = self.model_halfset

                # Return ([[([df], None)]]) — matches infer._flatten_inference_output logic
                return [[([df], None)]]

        return FakeSingleInferencer

    def _make_fake_ctx(self):
        """Multiprocessing context that runs worker target inline (no real procs)."""
        import queue

        class FakeProc:
            def __init__(self, target, args):
                self._target = target
                self._args = args
                self.exitcode = 0
                self._alive = False
                self.pid = 0  # dummy

            def start(self):
                self._alive = True
                self._target(*self._args)
                self._alive = False

            def is_alive(self):
                return self._alive

            def join(self, timeout=None):
                pass

            def terminate(self):
                self.exitcode = -1
                self._alive = False

        class FakeManager:
            def __enter__(self): return self
            def __exit__(self, exc_type, exc, tb): return False

            class _Q:
                def __init__(self): self.q = queue.Queue()
                def put(self, x): self.q.put(x)
                def get_nowait(self): return self.q.get_nowait()
                def empty(self): return self.q.empty()

            def Queue(self): return self._Q()

        class FakeCtx:
            def Process(self, target, args): return FakeProc(target=target, args=args)
            def Manager(self): return FakeManager()

        return FakeCtx()

    @staticmethod
    def _align_for_compare(df: pd.DataFrame) -> pd.DataFrame:
        """Align rows deterministically by particle id and (if present) half labels."""
        # Extract particle id before '@' as stable key
        pid = df["rlnImageName"].str.extract(r"^(\d+)", expand=False).astype(int)
        df = df.assign(__pid__=pid)

        sort_cols = ["__pid__"]
        for c in ["__data_half__", "__model_half__"]:
            if c in df.columns:
                sort_cols.append(c)

        return df.sort_values(sort_cols).drop(columns=["__pid__"]).reset_index(drop=True)

    def test_distributed_inference_consistency(self):
        FakeSingleInferencer = self._make_fake_inferencer()

        # Patch the SingleInferencer used *inside* infer.distributed_inference and workers
        with patch("cryoPARES.inference.infer.SingleInferencer", FakeSingleInferencer):
            # -------- Single-process path (n_jobs=1) --------
            out_single = infer.distributed_inference(
                particles_star_fname=self.particles_star_fname,
                checkpoint_dir=self.checkpoint_dir,
                results_dir=self.results_dir,
                data_halfset="allParticles",
                model_halfset="matchingHalf",
                particles_dir=self.particles_dir,
                batch_size=3,
                n_jobs=1,
                num_data_workers=0,
                use_cuda=False,
                n_cpus_if_no_cuda=1,
                compile_model=False,
                top_k_poses_nnet=1,
                reference_map=None,
                reference_mask=None,
                directional_zscore_thr=None,
                skip_localrefinement=True,
                skip_reconstruction=True,
                subset_idxs=None,
                float32_matmul_precision="high",
            )

            # Flatten SingleInferencer output using infer helper
            dfs_single = infer._flatten_inference_output(out_single)
            self.assertTrue(len(dfs_single) >= 1)
            df_single = pd.concat(dfs_single, axis=0)

            # -------- “Multiprocess” path (n_jobs=2), but run inline via fake ctx --------
            fake_ctx = self._make_fake_ctx()
            with patch("cryoPARES.inference.infer.multiprocessing.get_context", return_value=fake_ctx):
                out_distrib = infer.distributed_inference(
                    particles_star_fname=self.particles_star_fname,
                    checkpoint_dir=self.checkpoint_dir,
                    results_dir=self.results_dir,
                    data_halfset="allParticles",
                    model_halfset="matchingHalf",
                    particles_dir=self.particles_dir,
                    batch_size=3,
                    n_jobs=2,                     # triggers multiprocess branch
                    num_data_workers=0,
                    use_cuda=False,
                    n_cpus_if_no_cuda=1,
                    compile_model=False,
                    top_k_poses_nnet=1,
                    reference_map=None,
                    reference_mask=None,
                    directional_zscore_thr=None,
                    skip_localrefinement=True,
                    skip_reconstruction=True,    # avoid shared recon buffers
                    subset_idxs=None,
                    float32_matmul_precision="high",
                )

            # out_distrib is a dict mapping "model_<h>__data_<h>" -> aggregated DataFrame
            self.assertIsInstance(out_distrib, dict)
            dfs_multi = [v for v in out_distrib.values() if v is not None]
            self.assertTrue(len(dfs_multi) >= 1)
            df_multi = pd.concat(dfs_multi, axis=0)

            # --- Align rows deterministically before comparing ---
            df_single_aligned = self._align_for_compare(df_single)
            df_multi_aligned = self._align_for_compare(df_multi)

            # Compare numeric columns only
            compare_cols = [
                c for c in df_single_aligned.columns
                if c in df_multi_aligned.columns and pd.api.types.is_numeric_dtype(df_single_aligned[c])
            ]
            self.assertTrue(
                np.allclose(
                    df_single_aligned[compare_cols].to_numpy(),
                    df_multi_aligned[compare_cols].to_numpy(),
                    atol=1e-8
                )
            )

            # Optionally enforce halfset labels match exactly after alignment
            if "rlnRandomSubset" in df_single_aligned.columns and "rlnRandomSubset" in df_multi_aligned.columns:
                self.assertTrue(
                    (df_single_aligned["rlnRandomSubset"].to_numpy() ==
                     df_multi_aligned["rlnRandomSubset"].to_numpy()).all()
                )


if __name__ == "__main__":
    unittest.main()
