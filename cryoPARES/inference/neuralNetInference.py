import copy
import json
import os
import os.path as osp
import subprocess
import sys
import tempfile
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import starfile
import torch
from joblib import Parallel, delayed
from numba import Literal
from pytorch_lightning import LightningModule
from tqdm import tqdm

from cryoPARES import constants
from cryoPARES.configManager.inject_defaults import inject_defaults_from_config, CONFIG_PARAM
from cryoPARES.configs.mainConfig import main_config
from cryoPARES.datamanager.particlesDataset import ParticlesDataset
from cryoPARES.datamanager.relionStarDataset import ParticlesRelionStarDataset
from cryoPARES.utils.torchUtils import accelerator_selector


class NnetInferencer:
    @inject_defaults_from_config(main_config.inference.nnetinference, update_config_with_args=True)
    def __init__(self,
                 particles_star_fname: str,
                 model_fname: str,
                 star_out_fname: str,
                 particles_dir: Optional[str] = None,
                 angles_probs_fname: Optional[str] = None,
                 halfset: Literal["half1", "half2", "allParticles"] = "allParticles",
                 combining_mode: str = "copy", #TODO: Check this and move to config
                 keep_original_sphere: bool = False,
                 keep_original_inplane: bool = False,
                 keep_original_shifts: bool = False,
                 rm_particles_lt_score: Optional[float] = None,
                 n_dataworkers: int = CONFIG_PARAM(),
                 batch_size: int = CONFIG_PARAM(),
                 use_cuda: bool = CONFIG_PARAM(config=main_config.inference)):
        """Initialize neural network inference.

        Args:
            particles_star_fname: Path to starfile containing particles
            model_fname: Path to model checkpoint
            star_out_fname: Output starfile path
            particles_dir: Directory containing particles (defaults to starfile dir)
            angles_probs_fname: Path to save angle probabilities
            halfset: which halfset of particles to use
            combining_mode: How to combine orientations ("copy" or "average")
            keep_original_sphere: Keep original sphere parameters
            keep_original_inplane: Keep original in-plane parameters
            keep_original_shifts: Keep original shift parameters
            rm_particles_lt_score: Remove particles below score threshold
            n_dataworkers: Number of data workers
            batch_size: Batch size for inference
            use_cuda: Whether to use GPU acceleration
        """
        self.particles_star_fname = osp.expanduser(particles_star_fname)
        self.model_fname = osp.expanduser(model_fname)
        self.star_out_fname = star_out_fname
        self.particles_dir = particles_dir or osp.dirname(self.particles_star_fname)
        self.angles_probs_fname = angles_probs_fname
        self.halfset = halfset
        self.combining_mode = combining_mode
        self.keep_original_sphere = keep_original_sphere
        self.keep_original_inplane = keep_original_inplane
        self.keep_original_shifts = keep_original_shifts
        self.rm_particles_lt_score = rm_particles_lt_score
        self.n_dataworkers = n_dataworkers
        self.batch_size = batch_size
        self.use_cuda = use_cuda

        # Setup device and model
        self.accel, self.n_devices = accelerator_selector(
            use_cuda=self.use_cuda,
            n_cpus_torch=self.n_dataworkers
        )
        self.n_splits = self.n_devices if self.accel != "cpu" else 1

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run neural network inference pipeline.

        Returns:
            Tuple containing particles DataFrame and optics DataFrame
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Split data if using multiple devices
            input_star_fnames = self._split_input_data(tmp_dir)

            # Setup output filenames for chunks
            chunk_star_fnames = []
            chunk_angles_probs_fnames = []
            for i in range(len(input_star_fnames)):
                chunk_star_fnames.append(osp.join(tmp_dir, f"chunk_{i}.star"))
                chunk_angles_probs_fnames.append(osp.join(tmp_dir, f"chunk_{i}.probs.npy"))

            # Load model
            pl_model = self._load_model()

            # Process chunks in parallel
            info_list = self._process_chunks(
                input_star_fnames,
                chunk_star_fnames,
                chunk_angles_probs_fnames,
                pl_model
            )

            # Combine results
            symmetry = info_list[0]["symmetry"]
            out_data = self._combine_results(
                chunk_star_fnames,
                chunk_angles_probs_fnames,
                symmetry
            )

            return out_data["particles"], out_data["optics"]


    def _load_model(self) -> LightningModule:
        """Load neural network model."""
        from cryoPARES.models import get_model_class

        kwargs = {}


        kwargs["map_location"] = torch.device("cpu")
        model_class = get_model_class()
        pl_model, _ = model_class.load_from_checkpoint(self.model_fname, **kwargs)
        return pl_model

    def _process_chunks(self,
                        input_star_fnames: List[str],
                        chunk_star_fnames: List[str],
                        chunk_angles_probs_fnames: List[str],
                        pl_model: LightningModule) -> List[Dict[str, Any]]:
        """Process data chunks in parallel."""

        def worker_fn(i: int) -> Dict[str, Any]:
            return self._inference_worker(
                particles_star_fname=input_star_fnames[i],
                pl_model=pl_model,
                star_out_fname=chunk_star_fnames[i],
                angles_probs_fname=chunk_angles_probs_fnames[i],
                gpu_id=i if self.use_gpu else -1
            )

        return Parallel(n_jobs=self.n_splits, backend='loky')(
            delayed(worker_fn)(i) for i in range(self.n_splits)
        )

    def _inference_worker(self,
                          particles_star_fname: str,
                          pl_model: LightningModule,
                          star_out_fname: str,
                          angles_probs_fname: str,
                          gpu_id: int) -> Dict[str, Any]:
        """Worker function for inference on a data chunk."""
        # Set device
        if gpu_id >= 0:
            pl_model.to(f"cuda:{gpu_id}")
        device = pl_model.device

        # Get symmetry
        symmetry = pl_model.symmetry

        # Create dataset
        dataset = self._create_dataset(particles_star_fname, halfset=halfset)

        # Create dataloader
        dataloader = dataset.create_dataloader(
            batch_size=self.batch_size,
            num_workers=self.n_dataworkers,
            shuffle=False,
            pin_memory=gpu_id >= 0
        )

        # Process batches
        results = self._process_batches(
            dataloader,
            pl_model,
            device,
            dataset.desired_particle_size_angstroms,
            gpu_id
        )

        # Save results
        self._save_worker_results(
            results,
            dataset,
            star_out_fname,
            angles_probs_fname,
            symmetry
        )

        return {"symmetry": symmetry}

    def _create_dataset(self, particles_star_fname: str, symmetry: str, halfset: Optional[int]):
        """Create dataset for inference."""

        return ParticlesRelionStarDataset(
            particles_star_fname=particles_star_fname,
            particles_dir=self.particles_dir,
            symmetry=symmetry,
            halfset=halfset
        )

    def _process_batches(self,
                         dataloader,
                         pl_model: LightningModule,
                         device: torch.device,
                         particle_size_angstroms: float,
                         gpu_id: int) -> Dict[str, Any]:
        """Process batches through the model."""
        results = {
            'output_dfs': [],
            'geodesic_errs': [],
            'cone_errs': [],
            'inplane_errs': [],
            'angles_probs': []
        }

        n_items = 0
        for batch_idx, batch in tqdm(enumerate(dataloader),
                                     disable=(gpu_id != 0),
                                     total=len(dataloader)):
            # Get predictions
            preds = pl_model.predict_eulers_degrees_and_shift_fraction(
                pl_model.transfer_batch_to_device(batch, device, 0),
                report_metrics='rlnAngleRot' in batch.metadata
            )

            # Process predictions
            batch_results = self._process_batch_predictions(
                preds,
                batch,
                n_items,
                particle_size_angstroms
            )

            # Update results
            for k, v in batch_results.items():
                results[k].extend(v)

            n_items += len(batch_results['output_dfs'])

        return results

    def _save_worker_results(self,
                             results: Dict[str, Any],
                             dataset,
                             star_out_fname: str,
                             angles_probs_fname: str,
                             symmetry: str) -> None:
        """Save results from a worker."""
        # Combine DataFrames
        out_df = pd.concat(results['output_dfs'])

        # Save starfile
        starfile.write(
            {'optics': dataset.optics, 'particles': out_df},
            star_out_fname,
            overwrite=True
        )

        # Save statistics
        self._save_statistics(
            results,
            star_out_fname
        )

        # Save angle probabilities if requested
        if angles_probs_fname and results['angles_probs']:
            np.save(
                angles_probs_fname,
                np.concatenate(results['angles_probs'])
            )

    def _combine_results(self,
                         chunk_star_fnames: List[str],
                         chunk_angles_probs_fnames: List[str],
                         symmetry: str) -> Dict[str, Any]:
        """Combine results from all workers."""
        # Join starfiles
        out_data = join_star_files(chunk_star_fnames, self.star_out_fname)

        # Post-process if needed
        if self.rm_particles_lt_score is not None:
            out_data = self._apply_score_filtering(out_data, symmetry)

        # Save final results
        self._save_final_results(out_data)

        return out_data

    def _save_final_results(self, out_data: Dict[str, Any]) -> None:
        """Save final combined results."""
        if self.star_out_fname:
            # Clean up any NaN values
            particles = out_data["particles"]
            particles['rlnOriginXAngst'] = particles['rlnOriginXAngst'].fillna(0)
            particles['rlnOriginYAngst'] = particles['rlnOriginYAngst'].fillna(0)

            starfile.write(out_data, self.star_out_fname, overwrite=True)

    def _save_statistics(self,
                         results: Dict[str, Any],
                         output_fname: str) -> None:
        """Save statistical results from inference."""
        stats = {
            'geodesic_degs_err': self._calculate_stats(results['geodesic_errs']),
            'cone_degs_err': self._calculate_stats(results['cone_errs']),
            'inplane_degs_err': self._calculate_stats(results['inplane_errs'])
        }

        stats_fname = os.path.splitext(output_fname)[0] + '_stats.json'
        with open(stats_fname, 'w') as f:
            json.dump(stats, f)

    @staticmethod
    def _calculate_stats(data: List[np.ndarray]) -> Dict[str, Any]:
        """Calculate statistics for a metric."""
        if not data:
            return {'stats': (0.0, 0.0), 'n_items': 0}

        combined = np.concatenate(data)
        return {
            'stats': (float(combined.mean()), float(combined.std())),
            'n_items': combined.shape[0]
        }

def execute_neuralNetInference(**kwargs):
    from argParseFromDoc import generate_command_for_argparseFromDoc
    cmd = generate_command_for_argparseFromDoc(
        "cryoPARES.train.runTrainOnePartition",
        fun=NnetInferencer.__init__,
        use_module=True,
        python_executable=sys.executable,
        **kwargs
    )
    print(cmd)  # TODO: Use loggers
    subprocess.run(
        cmd.split(),
        cwd=os.path.abspath(os.path.join(__file__, "..", "..", ".."))
    )

def main():
    """Command-line entry point for neural network inference."""
    from argParseFromDoc import AutoArgumentParser

    parser = AutoArgumentParser(prog="neural network inference")
    parser.add_args_from_function(NnetInferencer.__init__)
    args = parser.parse_args()

    inferencer = NnetInferencer(**vars(args))
    inferencer.run()


if __name__ == "__main__":
    main()