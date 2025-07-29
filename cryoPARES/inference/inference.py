import glob
import os
import sys

import numpy as np
import torch
import yaml
from lightning import seed_everything
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from typing import Optional, List, Literal, Dict, Any, Tuple

from cryoPARES.constants import RELION_ANGLES_NAMES, RELION_SHIFTS_NAMES, RELION_PRED_POSE_CONFIDENCE_NAME, \
    RELION_EULER_CONVENTION, DIRECTIONAL_ZSCORE_NAME

from cryoPARES import constants
from cryoPARES.configManager.configParser import ConfigOverrideSystem
from cryoPARES.configManager.inject_defaults import inject_defaults_from_config, CONFIG_PARAM
from cryoPARES.configs.mainConfig import main_config
from cryoPARES.constants import BEST_DIRECTIONAL_NORMALIZER, BEST_CHECKPOINT_BASENAME, BEST_MODEL_SCRIPT_BASENAME
from cryoPARES.geometry.convert_angles import matrix_to_euler_angles
from cryoPARES.inference.nnetWorkers.inferenceModel import InferenceModel
from cryoPARES.models.model import PlModel
from cryoPARES.datamanager.datamanager import DataManager
from cryoPARES.projmatching.projMatching import ProjectionMatcher
from cryoPARES.reconstruction.reconstruction import Reconstructor
from cryoPARES.utils.paths import get_most_recent_file


class SingleInferencer:

    @inject_defaults_from_config(main_config.inference, update_config_with_args=True)
    def __init__(self,
                 particles_star_fname: str,
                 checkpoint_dir: str,
                 results_dir: str,
                 data_halfset: Literal["half1", "half2", "allParticles"] = "allParticles",
                 model_halfset: Literal["half1", "half2", "allCombinations", "matchingHalf"] = "matchingHalf",
                 particles_dir: Optional[str] = None,
                 batch_size: int = CONFIG_PARAM(),
                 num_data_workers: int = CONFIG_PARAM(config=main_config.datamanager),
                 use_cuda: bool = CONFIG_PARAM(),
                 n_cpus_if_no_cuda: int = CONFIG_PARAM(),
                 compile_model: bool = False,
                 top_k: int = CONFIG_PARAM(),
                 reference_map: Optional[str] = None,
                 directional_zscore_thr: Optional[float] = CONFIG_PARAM(),
                 perform_localrefinement: bool = CONFIG_PARAM(),
                 perform_reconstruction: bool = CONFIG_PARAM(),
                 update_progessbar_n_batches: int = CONFIG_PARAM(),
                 subset_idxs: Optional[List[int]] = None,
                 n_first_particles: Optional[int]  = None
                 ):
        """

        :param particles_star_fname:
        :param checkpoint_dir:
        :param results_dir:
        :param data_halfset:
        :param model_halfset:
        :param particles_dir:
        :param batch_size:
        :param num_data_workers:
        :param use_cuda:
        :param n_cpus_if_no_cuda:
        :param compile_model:
        :param top_k:
        :param reference_map: If not provided, it will be tried to load from the checkpointç
        :param directional_zscore_thr:
        :param perform_localrefinement:
        :param perform_reconstruction:
        :param update_progessbar_n_batches:
        :param subset_idxs:
        :param n_first_particles:
        """

        self.particles_star_fname = particles_star_fname
        self.particles_dir = particles_dir

        self.checkpoint_dir = checkpoint_dir
        if n_first_particles is not None:
            assert subset_idxs is None, "Error, only n_first_particles or subset_idxs can be provided"
            subset_idxs = range(n_first_particles)
        assert os.path.isdir(checkpoint_dir), f"checkpoint_dir {checkpoint_dir} needs to be a directory"
        main_config.datamanager.num_augmented_copies_per_batch = 1  # We have not implemented test-time augmentation

        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.n_cpus_if_no_cuda = n_cpus_if_no_cuda
        self.results_dir = results_dir
        self.data_halfset = data_halfset
        self.model_halfset = model_halfset

        self.compile_model = compile_model
        self.top_k = top_k
        self.reference_map = reference_map
        self.directional_zscore_thr = directional_zscore_thr
        self.perform_localrefinement = perform_localrefinement
        self.perform_reconstruction = perform_reconstruction
        self.update_progessbar_n_batches = update_progessbar_n_batches

        if results_dir is not None:
            os.makedirs(results_dir, exist_ok=True)

        self._setup_accelerator()
        self._pbar = None
        self._model = None
        self._reconstructor = None
        self._datamanager = None
        self._subset_idxs = subset_idxs
        self._last_dataset_processed = 0

    def _setup_accelerator(self):
        """Setup the computation device and count."""
        if self.use_cuda and torch.cuda.is_available():
            accelerator = "cuda"
            device_count = torch.cuda.device_count()
        else:
            accelerator = "cpu"
            device_count = self.n_cpus_if_no_cuda

        print(f'devices={device_count} accelerator={accelerator}', flush=True)
        torch.set_num_threads(max(1, torch.get_num_threads() // device_count))

        self.device, self.device_count = accelerator, device_count

    def _setup_reconstructor(self, symmetry: Optional[str] = None):
        if symmetry is None:
            symmetry = self._get_symmetry()
        reconstructor = Reconstructor(symmetry=symmetry, correct_ctf=True)
        reconstructor._get_reconstructionParticlesDataset(self.particles_star_fname, self.particles_dir)
        self._reconstructor = self._get_reconstructor(self.particles_star_fname, self.particles_dir, symmetry)
        return reconstructor

    @staticmethod
    def _get_reconstructor(particles_star_fname, particles_dir, symmetry: str):
        reconstructor = Reconstructor(symmetry=symmetry, correct_ctf=True)
        reconstructor._get_reconstructionParticlesDataset(particles_star_fname, particles_dir)
        return reconstructor

    def _setup_model(self, rank: Optional[int] = None):
        """Setup the model for inference."""

        # so3Model_fname = os.path.join(self.checkpoint_dir, self.model_halfset, "checkpoints", BEST_MODEL_SCRIPT_BASENAME)
        # so3Model = torch.jit.load(so3Model_fname)

        so3Model_fname = os.path.join(self.checkpoint_dir, self.model_halfset, "checkpoints", BEST_CHECKPOINT_BASENAME)
        so3Model = PlModel.load_from_checkpoint(so3Model_fname)

        percentilemodel_fname = os.path.join(self.checkpoint_dir, self.model_halfset, "checkpoints",
                                             BEST_DIRECTIONAL_NORMALIZER)
        percentilemodel = torch.load(percentilemodel_fname, weights_only=False)

        if self.reference_map is None:
            reference_map = os.path.join(self.checkpoint_dir, self.model_halfset, "reconstructions", "0.mrc")
        else:
            reference_map = self.reference_map

        if self.perform_localrefinement:
            localRefiner = ProjectionMatcher(reference_vol=reference_map)
        else:
            localRefiner = None

        if self.perform_reconstruction:
            reconstructor = self._setup_reconstructor(so3Model.symmetry)
        else:
            reconstructor = None

        model = InferenceModel(so3Model, percentilemodel, self.directional_zscore_thr, localRefiner,
                               reconstructor=reconstructor,
                               return_top_k=self.top_k)

        # Handle missing symmetry attribute
        if not hasattr(model, 'symmetry'):
            # Try to get from hyperparameters or set default
            if hasattr(model, 'hparams') and hasattr(model.hparams, 'symmetry'):
                model.symmetry = model.hparams.symmetry
            else:
                raise RuntimeError("Symmetry not found in model")

        device = f'cuda:{rank}' if rank is not None else 'cuda:0'
        if self.device == "cpu":
            device = "cpu"

        model = model.to(device)

        if self.compile_model:
            print(f"Compiling model on device {device}")
            model = torch.compile(model)

        model.eval()
        self._model = model
        return model

    def _get_symmetry(self):
        hparams = os.path.join(self.checkpoint_dir, self.model_halfset, "hparams.yaml")
        try:
            with open(hparams) as f:
                symmetry = yaml.safe_load(f)["symmetry"]
            return symmetry
        except (FileNotFoundError, yaml.YAMLError, KeyError) as e:
            raise RuntimeError(f"Failed to load symmetry from {hparams}: {e}")

    def _get_datamanager(self, rank: Optional[int] = None):
        """Setup the dataloader for inference."""
        data_halfset = None
        if self.data_halfset == "half1":
            data_halfset = 1
        elif self.data_halfset == "half2":
            data_halfset = 2
        elif self.data_halfset == "allParticles":
            data_halfset = None
        else:
            raise ValueError(f"Error, not valid self.data_halfset {self.data_halfset}")

        symmetry = self._get_symmetry()
        datamanager = DataManager(
            star_fnames=self.particles_star_fname,
            symmetry=symmetry,
            particles_dir=self.particles_dir,
            batch_size=self.batch_size,
            augment_train=False,  # No augmentation during inference
            halfset=data_halfset,
            is_global_zero=rank == 0 if rank is not None else True,
            save_train_val_partition_dir=None,  # Not needed for inference
            return_ori_imagen=True,  # Needed for inference,
            subset_idxs=self._subset_idxs
        )
        return datamanager

    def _setup_dataloader(self, rank: Optional[int] = None):
        self._datamanager = self._get_datamanager(rank)
        return self._datamanager.predict_dataloader()

    def _process_batch(self, model: PlModel, batch: Dict[str, Any] | None, batch_idx: int):
        """Process a single batch of data using predict_step."""
        device = self.device
        if batch is not None:
            if hasattr(model, 'transfer_batch_to_device'):
                batch = model.transfer_batch_to_device(batch, self.device, dataloader_idx=0)
            else:
                # Fallback: manually transfer tensors
                non_blocking = True
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(device, non_blocking=non_blocking)
                    elif isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                        batch[key] = [v.to(device, non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v for v in
                                      value]

            # Use predict_step as requested
            result = model.predict_step(batch, batch_idx=batch_idx, dataloader_idx=0)
        else:
            result = model.flush()
        if result is None:
            return None
        ids, pred_rotmats, pred_shiftsXYangs, score, norm_nn_score = result
        # Convert rotation matrices to euler angles
        euler_degs = torch.rad2deg(matrix_to_euler_angles(pred_rotmats, RELION_EULER_CONVENTION))

        return ids, euler_degs, pred_shiftsXYangs, score, norm_nn_score

    def run(self):
        if self.model_halfset == "allCombinations":
            model_halfset_list = ["half1", "half2"]
        elif self.model_halfset == "matchingHalf":
            model_halfset_list = [None]
        else:
            model_halfset_list = [self.model_halfset]

        if self.data_halfset == "allParticles":
            data_halfset_list = ["half1", "half2"]
        else:
            data_halfset_list = [self.data_halfset]

        out_list = []
        for model_halfset in model_halfset_list:
            for data_halfset in data_halfset_list:
                if model_halfset == None:
                    self.model_halfset = data_halfset
                else:
                    self.model_halfset = model_halfset
                self.data_halfset = data_halfset
                print(f"Running inference for data {self.data_halfset} with model {self.model_halfset}")
                out = self._run()
                out_list.append(out)
            self._model = None
        return out_list

    def _get_pbar(self, total):
        if self._pbar is None:
            return tqdm(total=total, desc="Processing batches")
        else:
            if hasattr(self._pbar, "set_total_steps"):
                self._pbar.set_total_steps(total)
            return self._pbar

    def _get_outsuffix(self, extension):
        if self.model_halfset == "allParticles":
            return f"_data_{self.data_halfset}_model_{self.model_halfset}.{extension}"
        else:
            return f"_{self.data_halfset}.{extension}"

    def _run(self):
        """Main entry point for running async inference."""

        self._last_dataset_processed = 0
        # Get total dataset info
        dataloader = self._setup_dataloader()
        dataset = dataloader.dataset
        total_batches = len(dataloader)

        print(f"Processing {len(dataset)} particles in {total_batches} batches")

        if self._model is None:
            model = self._setup_model()
        else:
            model = self._model

        pbar = self._get_pbar(total_batches)
        try:
            all_results = self._process_all_batches(model, dataloader, pbar=pbar)
            print("Aggregating results and saving to STAR files...")
            particles_md = self._save_particles_results(all_results, dataset)
            if self.perform_reconstruction: print("Materializing reconstruction...")
            vol = self._save_reconstruction(self.perform_reconstruction)
            return particles_md, vol
        finally:
            pbar.close()

    def _process_all_batches(self, model, dataloader, pbar=None):
        all_results = []
        with torch.inference_mode():
            for batch_idx, batch in enumerate(dataloader):
                result = self._process_batch(model, batch, batch_idx)
                if result:
                    all_results.append(result)
                if pbar is not None: pbar.update(1)
            result = self._process_batch(model, batch=None, batch_idx=-1)  # batch=None flushes the model buffer
            if result is not None:
                all_results.append(result)
        return all_results

    def _save_reconstruction(self, save_to_file: bool = True, materialize: bool = True):
        vol = None
        if save_to_file and self.results_dir is not None:
            if save_to_file:
                if materialize:
                    out_fname = os.path.join(self.results_dir, f"reconstruction" + self._get_outsuffix("mrc"))
                    vol = self._model.reconstructor.generate_volume(out_fname)
                else:
                    out_fname = os.path.join(self.results_dir, f"mapcomponents" + self._get_outsuffix("npz"))
                    components = self._model.reconstructor.get_buffers()
                    components = {k: v.detach().cpu() for k, v in components.items()}
                    components["sampling_rate"] = np.array([self._model.reconstructor.sampling_rate])
                    np.savez(out_fname, **components)
                print(f"{out_fname} saved!", flush=True)
        return vol

    def _save_particles_results(self, all_results: List[Any], dataset, save_to_file: bool = True):

        # Create result tensors to hold data from all batches
        n_particles = sum(len(v[0]) for v in all_results)
        if n_particles == 0:
            print("Warning: No particle results to save.")
            return []

        result_arrays = {
            'eulerdegs': torch.zeros((n_particles, self.top_k, 3), dtype=torch.float32),
            'score': torch.zeros((n_particles, self.top_k), dtype=torch.float32),
            'shiftsXYangs': torch.zeros((n_particles, self.top_k, 2), dtype=torch.float32),
            'top1_directional_zscore': torch.zeros((n_particles), dtype=torch.float32),
            'ids': [None] * n_particles
        }

        # Fill result arrays by concatenating all batch results
        current_idx = 0
        for batch_idx, elem in enumerate(all_results):
            idd, euler_degs, pred_shiftsXYangs, maxprobs, top1_directional_zscore = elem

            batch_size = len(idd)
            end_idx = current_idx + batch_size

            result_arrays['eulerdegs'][current_idx:end_idx] = euler_degs.cpu()
            result_arrays['score'][current_idx:end_idx] = maxprobs.cpu()
            if pred_shiftsXYangs is None:
                pred_shiftsXYangs = torch.zeros(*euler_degs.shape[:-1], 2)
            result_arrays['shiftsXYangs'][current_idx:end_idx] = pred_shiftsXYangs.cpu()
            result_arrays['top1_directional_zscore'][current_idx:end_idx] = top1_directional_zscore.cpu()
            result_arrays['ids'][current_idx:end_idx] = idd
            current_idx = end_idx

        all_processed_ids_np = np.array(result_arrays["ids"])
        assert len(all_processed_ids_np) == len(np.unique(all_processed_ids_np)), \
            "Duplicate particle IDs found in the inference results."

        # Create a mapping from particle ID to its index in the results array
        particles_md_list = []
        for datIdx, _dataset in enumerate(dataset.datasets):
            particlesSet = _dataset.particles
            particles_md = particlesSet.particles_md

            # Find which of the processed IDs are in the current dataset's index
            ids_to_update_in_df = []
            result_indices = []
            for i, id_val in enumerate(all_processed_ids_np):
                if id_val in particles_md.index:
                    ids_to_update_in_df.append(id_val)
                    result_indices.append(i)

            if len(ids_to_update_in_df) == 0:
                continue

            # --- Assertion for shape consistency ---
            num_updates = len(ids_to_update_in_df)
            assert num_updates == len(result_indices), \
                "Shape mismatch between IDs to update and the filtered result data."

            # Update the DataFrame with the correctly ordered results
            particles_md.loc[ids_to_update_in_df, DIRECTIONAL_ZSCORE_NAME] = result_arrays["top1_directional_zscore"][
                result_indices].numpy()

            for k in range(self.top_k):
                suffix = "" if k == 0 else f"_top{k}"
                angles_names = [x + suffix for x in RELION_ANGLES_NAMES]
                shiftsXYangs_names = [x + suffix for x in RELION_SHIFTS_NAMES]
                confide_name = RELION_PRED_POSE_CONFIDENCE_NAME + suffix

                for col in angles_names + shiftsXYangs_names + [confide_name]:
                    if col not in particles_md.columns:
                        particles_md[col] = 0.0
                eulerdegs = result_arrays["eulerdegs"][result_indices, k, :].numpy()

                ######### Debug code
                r1 = Rotation.from_euler("ZYZ", eulerdegs, degrees=True)
                r2 = Rotation.from_euler("ZYZ", particles_md.loc[ids_to_update_in_df, angles_names], degrees=True)
                err = np.rad2deg((r1.inv() * r2).magnitude())
                # print("Error degs:", err)
                print("Median Error degs:", np.median(err))
                # breakpoint()
                ######## END of Debug code

                particles_md.loc[ids_to_update_in_df, angles_names] = eulerdegs

                particles_md.loc[ids_to_update_in_df, shiftsXYangs_names] = result_arrays["shiftsXYangs"][
                                                                            result_indices, k, :].numpy()
                particles_md.loc[ids_to_update_in_df, confide_name] = result_arrays["score"][result_indices, k].numpy()

            particles_md_list.append(particles_md)

            if self.results_dir is not None and save_to_file:
                if particlesSet.starFname is not None:
                    basename = os.path.basename(particlesSet.starFname).removesuffix(".star")
                else:
                    basename = "particles%d" % self._last_dataset_processed

                out_fname = os.path.join(self.results_dir, basename + self._get_outsuffix("star"))
                print(f"Results were saved at {out_fname}")
                particlesSet.save(out_fname)
            self._last_dataset_processed += 1
        return particles_md_list

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False  # Don't suppress exceptions


if __name__ == "__main__":
    seed_everything(111)
    os.environ[constants.SCRIPT_ENTRY_POINT] = "inference.py"
    print("---------------------------------------")
    print(" ".join(sys.argv))
    print("---------------------------------------")
    from cryoPARES.configManager.configParser import ConfigArgumentParser
    torch.set_float32_matmul_precision(constants.float32_matmul_precision)

    parser = ConfigArgumentParser(prog="infer_cryoPARES", description="Run inference with cryoPARES model",
                                  config_obj=main_config)
    parser.add_args_from_function(SingleInferencer.__init__)
    args, config_args = parser.parse_args()
    assert os.path.isdir(args.checkpoint_dir), f"Error, checkpoint_dir {args.checkpoint_dir} not found"
    config_fname = get_most_recent_file(args.checkpoint_dir, "configs_*.yml")
    ConfigOverrideSystem.update_config_from_file(main_config, config_fname, drop_paths=["inference", "projmatching"])

    main_config.models.image2sphere.so3components.i2sprojector.rand_fraction_points_to_project = 1
    #TODO: Remove the previous line, it is for debugging only

    with SingleInferencer(**vars(args)) as inferencer:
        inferencer.run()

    """

    """