import glob
import os
import sys
import warnings
from functools import cached_property

import numpy as np
import torch
import yaml
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
from cryoPARES.geometry.convert_angles import matrix_to_euler_angles, euler_angles_to_matrix
from cryoPARES.geometry.metrics_angles import rotation_error_with_sym
from cryoPARES.inference.nnetWorkers.inferenceModel import InferenceModel
from cryoPARES.models.model import PlModel
from cryoPARES.datamanager.datamanager import DataManager
from cryoPARES.projmatching.projMatcher import ProjectionMatcher
from cryoPARES.reconstruction.reconstructor import Reconstructor
from cryoPARES.utils.paths import get_most_recent_file
from cryoPARES.scripts.computeFsc import compute_fsc
from cryoPARES.utils.reconstructionUtils import get_vol


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
                 top_k_poses_nnet: int = CONFIG_PARAM(),
                 top_k_poses_localref: int = CONFIG_PARAM(config=main_config.projmatching),
                 reference_map: Optional[str] = None,
                 reference_mask: Optional[str] = None, #Only used for FSC estimation
                 directional_zscore_thr: Optional[float] = CONFIG_PARAM(),
                 skip_localrefinement: bool = CONFIG_PARAM(),
                 skip_reconstruction: bool = CONFIG_PARAM(),
                 subset_idxs: Optional[List[int]] = None,
                 n_first_particles: Optional[int] = None,
                 show_debug_stats: bool = False,
                 float32_matmul_precision: str = constants.float32_matmul_precision,
                 ):
        """
        Initializes the SingleInferencer for running inference on a set of particles.

        :param particles_star_fname: Path to the STAR file containing particle information.
        :param checkpoint_dir: Directory where the trained model checkpoints are stored.
        :param results_dir: Directory where the inference results will be saved.
        :param data_halfset: Specifies which half-set of the data to use ("half1", "half2", or "allParticles").
        :param model_halfset: Specifies which half-set of the model to use ("half1", "half2", "allCombinations", or "matchingHalf").
        :param particles_dir: Directory where the particle images are located. If None, paths in the STAR file are assumed to be absolute.
        :param batch_size: The number of particles to process in each batch.
        :param num_data_workers: The number of worker processes to use for data loading.
        :param use_cuda: Whether to use a CUDA-enabled GPU for inference.
        :param n_cpus_if_no_cuda: The number of CPU cores to use if CUDA is not available.
        :param compile_model: Whether to compile the model using `torch.compile` for potential speed-up.
        :param top_k_poses_nnet: The number of top predictions to predict with the nn for each particle.
        :param top_k_poses_localref: The number of top predictions to return after local refinement.
        :param reference_map: Path to the reference map for local refinement. If not provided, it will be loaded from the checkpoint.
        :param reference_mask: Path to the mask of the reference map. Used only for FSC calculation.
        :param directional_zscore_thr: The threshold for the directional Z-score to filter particles.
        :param skip_localrefinement: Whether to skip local refinement of the particle poses.
        :param skip_reconstruction: Whether to skip 3D reconstruction from the inferred poses.
        :param subset_idxs: A list of indices to process a subset of particles.
        :param n_first_particles: The number of first particles to process. Cannot be used with `subset_idxs`.
        :param show_debug_stats: Whether to print debug statistics, such as rotation errors if ground truth in the starfile.
        :param float32_matmul_precision: The precision used in multiplications. Speed/accuracy tradeoff
        """

        self.float32_matmul_precision = float32_matmul_precision
        self.particles_star_fname = particles_star_fname
        self.particles_dir = particles_dir
        assert top_k_poses_nnet >= top_k_poses_localref, "Error, top_k_poses_nnet >= top_k_poses_localref required"
        self.checkpoint_dir = checkpoint_dir
        if n_first_particles is not None:
            assert subset_idxs is None, "Error, only n_first_particles or subset_idxs can be provided"
            subset_idxs = range(n_first_particles)
        assert os.path.isdir(checkpoint_dir), f"checkpoint_dir {checkpoint_dir} needs to be a directory"
        main_config.datamanager.num_augmented_copies_per_batch = 1  # We have not implemented test-time augmentation
        main_config.datamanager.particlesdataset.store_data_in_memory = False

        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.n_cpus_if_no_cuda = n_cpus_if_no_cuda
        self.results_dir = results_dir
        self.data_halfset = data_halfset
        self.model_halfset = model_halfset

        self.compile_model = compile_model
        self.top_k_poses_nnet = top_k_poses_nnet
        self.top_k_poses_localref = top_k_poses_localref
        self.reference_map = reference_map
        self.reference_mask = reference_mask
        self.directional_zscore_thr = directional_zscore_thr
        self.skip_localrefinement = skip_localrefinement
        self.skip_reconstruction = skip_reconstruction
        self.update_progressbar_n_batches = main_config.inference.update_progressbar_n_batches
        self.show_debug_stats = show_debug_stats

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
        """
        Sets up the computational device (CPU or CUDA) and determines the number of available devices.
        """
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
        """
        Initializes the Reconstructor object, which is responsible for 3D reconstruction from 2D particle images.

        :param symmetry: The symmetry of the object being reconstructed. If not provided, it's retrieved from the class's `symmetry` attribute.
        """
        if symmetry is None:
            symmetry = self.symmetry
        self._reconstructor = self._get_reconstructor(self.particles_star_fname, self.particles_dir, symmetry)
        return self._reconstructor

    @staticmethod
    def _get_reconstructor(particles_star_fname, particles_dir, symmetry: str):
        """
        Creates and configures a Reconstructor instance.

        :param particles_star_fname: Path to the STAR file containing particle information.
        :param particles_dir: Directory where particle data is stored.
        :param symmetry: The symmetry of the object.
        :return: An initialized Reconstructor object.
        """
        reconstructor = Reconstructor(symmetry=symmetry, correct_ctf=True)
        reconstructor._get_reconstructionParticlesDataset(particles_star_fname, particles_dir)
        return reconstructor

    def _setup_model(self, rank: Optional[int] = None):
        """
        Loads and configures the machine learning model for inference. This includes loading the main model,
        the percentile model for score normalization, and setting up the local refiner and reconstructor if enabled.

        :param rank: The rank of the current process in a distributed setup.
        """

        try:
            so3Model_fname = os.path.join(self.checkpoint_dir, self.model_halfset, "checkpoints", BEST_MODEL_SCRIPT_BASENAME)
            so3Model = torch.jit.load(so3Model_fname)
        except (ValueError, IOError):
            try:
                so3Model_fname = os.path.join(self.checkpoint_dir, self.model_halfset, "checkpoints", BEST_CHECKPOINT_BASENAME)
                so3Model = PlModel.load_from_checkpoint(so3Model_fname)
            except FileNotFoundError:
                so3Model_fname = os.path.join(self.checkpoint_dir, self.model_halfset, "checkpoints", "last.ckpt")
                so3Model = PlModel.load_from_checkpoint(so3Model_fname)

        try:
            percentilemodel_fname = os.path.join(self.checkpoint_dir, self.model_halfset, "checkpoints",
                                                 BEST_DIRECTIONAL_NORMALIZER)
            percentilemodel = torch.load(percentilemodel_fname, weights_only=False)
        except FileNotFoundError:
            assert self.directional_zscore_thr is None, ("Error, if no percentilemodel available, you cannot set a "
                                                         "directional_zscore_thr")
            warnings.warn(f"No percentilemodel found at ({percentilemodel_fname}). Directional normalized z-scores"
                          f" won't be computed!!!")
            percentilemodel = None
        if self.reference_map is None:
            reference_map = os.path.join(self.checkpoint_dir, self.model_halfset, "reconstructions", "0.mrc")
        else:
            reference_map = self.reference_map

        if not self.skip_localrefinement:
            localRefiner = ProjectionMatcher(reference_vol=reference_map, top_k_poses_localref=self.top_k_poses_localref)
        else:
            localRefiner = None

        if not self.skip_reconstruction:
            reconstructor = self._setup_reconstructor(so3Model.symmetry)
        else:
            reconstructor = None

        model = InferenceModel(so3Model, percentilemodel, self.directional_zscore_thr, localRefiner,
                               reconstructor=reconstructor,
                               top_k_poses_nnet=self.top_k_poses_nnet)

        # Handle missing symmetry attribute
        if not hasattr(model, 'symmetry'):
            # Try to get from hyperparameters or set default
            if hasattr(model, 'hparams') and hasattr(model.hparams, 'symmetry'):
                model.symmetry = model.hparams.symmetry
            else:
                raise RuntimeError("Symmetry not found in model")

        device = f'cuda:{rank}' if rank is not None else 'cuda'
        if self.device == "cpu":
            device = "cpu"

        model = model.to(device)

        if self.compile_model:
            print(f"Compiling model on device {device}")
            model = torch.compile(model)

        model.eval()
        self._model = model
        return model

    @cached_property
    def symmetry(self):
        """
        The symmetry of the model, loaded from the hyperparameters file.
        """
        return self._get_symmetry(self.checkpoint_dir, self.model_halfset)

    @staticmethod
    def _get_symmetry(checkpoint_dir, model_halfset):
        """
        Retrieves the symmetry value from the `hparams.yaml` file in the checkpoint directory.

        :return: The symmetry value.
        :raises RuntimeError: If the symmetry cannot be loaded from the file.
        """
        hparams = os.path.join(checkpoint_dir, model_halfset, "hparams.yaml")
        try:
            with open(hparams) as f:
                symmetry = yaml.safe_load(f)["symmetry"]
            return symmetry
        except (FileNotFoundError, yaml.YAMLError, KeyError) as e:
            raise RuntimeError(f"Failed to load symmetry from {hparams}: {e}")

    def _get_datamanager(self, rank: Optional[int] = None):
        """
        Initializes the DataManager, which is responsible for loading and preparing the particle data.

        :param rank: The rank of the current process in a distributed setup.
        :return: An initialized DataManager object.
        """
        data_halfset = None
        if self.data_halfset == "half1":
            data_halfset = 1
        elif self.data_halfset == "half2":
            data_halfset = 2
        elif self.data_halfset == "allParticles":
            data_halfset = None
        else:
            raise ValueError(f"Error, not valid self.data_halfset {self.data_halfset}")

        datamanager = DataManager(
            star_fnames=self.particles_star_fname,
            symmetry=self.symmetry,
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
        """
        Creates the data loader for inference.

        :param rank: The rank of the current process in a distributed setup.
        :return: The PyTorch DataLoader for inference.
        """
        self._datamanager = self._get_datamanager(rank)
        return self._datamanager.predict_dataloader()

    def _process_batch(self, model: PlModel, batch: Dict[str, Any] | None, batch_idx: int,
                       gpu_offload=False):
        """
        Processes a single batch of data through the model and returns the predictions.

        :param model: The inference model.
        :param batch: The batch of data to process. If None, it flushes the model's buffer.
        :param batch_idx: The index of the current batch.
        :param gpu_offload: If True, the results will be moved from GPU to CPU
        :return: A tuple containing the particle IDs, predicted Euler angles, shifts, scores, and normalized scores. Returns None if there are no results.
        """
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
        if gpu_offload:
            euler_degs = euler_degs.cpu()
            pred_shiftsXYangs = pred_shiftsXYangs.cpu() if pred_shiftsXYangs is not None else None
            score = score.cpu()
            norm_nn_score = norm_nn_score.cpu()
        return ids, euler_degs, pred_shiftsXYangs, score, norm_nn_score

    def run(self):
        """
        Runs the inference process. It iterates through the specified data and model half-sets,
        performing inference for each combination.

        :return: A list of tuples, where each tuple contains the particle metadata and the reconstructed volume for each inference run.
        """

        torch.set_float32_matmul_precision(self.float32_matmul_precision)

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

        all_out_list = []
        for model_halfset in model_halfset_list:
            out_list = []
            sampling_rate = None
            for data_halfset in data_halfset_list:
                if model_halfset is None:
                    self.model_halfset = data_halfset
                else:
                    self.model_halfset = model_halfset
                self.data_halfset = data_halfset
                print(f"Running inference for data {self.data_halfset} with model {self.model_halfset}")
                out = self._run()
                if self._model and hasattr(self._model, "reconstructor"):
                    sampling_rate = self._model.reconstructor.sampling_rate
                    self._model.reconstructor.zero_buffers() # Clean the reconstructor after each half is processed
                out_list.append(out)
                all_out_list.append(out_list)

            if len(out_list) == 2 and not self.skip_reconstruction:
                vol1 = out_list[0][1]
                vol2 = out_list[1][1]

                if vol1 is not None and vol2 is not None and sampling_rate is not None:
                    print("Computing FSC...")
                    if self.reference_mask is not None:
                        reference_mask = get_vol(self.reference_mask, pixel_size=None)[0]
                    else:
                        reference_mask = None
                    fsc, spatial_freq, resolution_A, (res_05, res_0143) = compute_fsc(vol1.cpu().numpy(),
                                                                                      vol2.cpu().numpy(),
                                                                                      sampling_rate,
                                                                                      mask=reference_mask)
                    print(f"Resolution at FSC=0.143 ('gold-standard'): {res_0143:.3f} Å")
                    print(f"Resolution at FSC=0.5: {res_05:.3f} Å")
            self._model = None
        return all_out_list

    def _get_pbar(self, total):
        """
        Initializes or updates a progress bar for tracking the inference process.

        :param total: The total number of batches to process.
        :return: The progress bar object.
        """
        if self._pbar is None:
            return tqdm(total=total, desc="Processing batches")
        else:
            if hasattr(self._pbar, "set_total_steps"):
                self._pbar.set_total_steps(total)
            return self._pbar

    def _get_outsuffix(self, extension):
        """
        Generates a suffix for the output file names based on the current data and model half-sets.

        :param extension: The file extension.
        :return: The generated file suffix.
        """
        if self.model_halfset == "allParticles":
            return f"_data_{self.data_halfset}_model_{self.model_halfset}.{extension}"
        else:
            return f"_{self.data_halfset}.{extension}"

    def _run(self):
        """
        The main private method that executes a single inference run. It sets up the data loader and model,
        processes all batches, and saves the results.

        :return: A tuple containing the particle metadata and the reconstructed volume.
        """

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
            if not self.skip_reconstruction: print("Materializing reconstruction...")
            vol = self._save_reconstruction(not self.skip_reconstruction)
            return particles_md, vol
        finally:
            pbar.close()

    def _process_all_batches(self, model, dataloader, pbar=None, gpu_offload=True):
        """
        Iterates through all batches in the data loader, processes them, and collects the results.

        :param model: The inference model.
        :param dataloader: The data loader.
        :param pbar: The progress bar.
        :param gpu_offload: If true, results are moved from the GPU to the CPU
        :return: A list of results from processing each batch.
        """
        all_results = []
        with torch.inference_mode():
            for batch_idx, batch in enumerate(dataloader):
                result = self._process_batch(model, batch, batch_idx, gpu_offload=gpu_offload)
                if result:
                    all_results.append(result)
                if pbar is not None: pbar.update(1)
            result = self._process_batch(model, batch=None, batch_idx=-1, gpu_offload=gpu_offload)  # batch=None flushes the model buffer
            if result is not None:
                all_results.append(result)
        return all_results

    def _save_reconstruction(self, save_to_file: bool = True, materialize: bool = True):
        """
        Saves the reconstructed volume to a file.

        :param save_to_file: Whether to save the volume to a file.
        :param materialize: Whether to generate the full volume or save the components.
        :return: The reconstructed volume, or None if not generated.
        """
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
        """
        Aggregates the results from all batches and saves them to STAR files.

        :param all_results: A list of results from all processed batches.
        :param dataset: The dataset containing the particle information.
        :param save_to_file: Whether to save the results to STAR files.
        :return: A list of pandas DataFrames containing the particle metadata with the inference results.
        """

        # Create result tensors to hold data from all batches
        n_particles = sum(len(v[0]) for v in all_results)
        if n_particles == 0:
            print("Warning: No particle results to save.")
            return []

        assert all_results, "Error, no results were computed"
        n_poses = all_results[0][1].shape[1]
        result_arrays = {
            'eulerdegs': torch.zeros((n_particles, n_poses, 3), dtype=torch.float32),
            'score': torch.zeros((n_particles, n_poses), dtype=torch.float32),
            'shiftsXYangs': torch.zeros((n_particles, n_poses, 2), dtype=torch.float32),
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

            for k in range(n_poses):
                suffix = "" if k == 0 else f"_top{k+1}"
                angles_names = [x + suffix for x in RELION_ANGLES_NAMES]
                shiftsXYangs_names = [x + suffix for x in RELION_SHIFTS_NAMES]
                confide_name = RELION_PRED_POSE_CONFIDENCE_NAME + suffix

                for col in angles_names + shiftsXYangs_names + [confide_name]:
                    if col not in particles_md.columns:
                        particles_md[col] = 0.0

                eulerdegs = result_arrays["eulerdegs"][result_indices, k, :].numpy()
                shiftsXYangs = result_arrays["shiftsXYangs"][result_indices, k, :].numpy()
                if self.show_debug_stats:
                    ######## Debug code
                    r1 = torch.FloatTensor(Rotation.from_euler(RELION_EULER_CONVENTION,
                                                               eulerdegs,
                                                               degrees=True).as_matrix())
                    r2 = torch.FloatTensor(Rotation.from_euler(RELION_EULER_CONVENTION,
                                                               particles_md.loc[ids_to_update_in_df, angles_names],
                                                               degrees=True).as_matrix())
                    ang_err = torch.rad2deg(rotation_error_with_sym(r1, r2, symmetry=self.symmetry))

                    s2 = particles_md.loc[ids_to_update_in_df, shiftsXYangs_names].values
                    shift_error = np.sqrt(((shiftsXYangs - s2)**2).sum(-1))
                    print(f"Median Ang   Displacement (degs) (top-{k+1}):", np.median(ang_err))
                    print(f"Median Shift Displacement (Angs) (top-{k+1}):", np.median(shift_error))
                    ######## END of Debug code

                particles_md.loc[ids_to_update_in_df, angles_names] = eulerdegs
                particles_md.loc[ids_to_update_in_df, shiftsXYangs_names] = shiftsXYangs
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
    # from lightning import seed_everything
    # seed_everything(111)
    os.environ[constants.SCRIPT_ENTRY_POINT] = "inferencer.py"
    print("---------------------------------------")
    print(" ".join(sys.argv))
    print("---------------------------------------")
    from cryoPARES.configManager.configParser import ConfigArgumentParser

    parser = ConfigArgumentParser(prog="infer_cryoPARES", description="Run inference with cryoPARES model",
                                  config_obj=main_config)
    parser.add_args_from_function(SingleInferencer.__init__)
    args, config_args = parser.parse_args()
    assert os.path.isdir(args.checkpoint_dir), f"Error, checkpoint_dir {args.checkpoint_dir} not found"
    config_fname = get_most_recent_file(args.checkpoint_dir, "configs_*.yml")
    ConfigOverrideSystem.update_config_from_file(main_config, config_fname, drop_paths=["inference", "projmatching"])
    ConfigOverrideSystem.update_config_from_configstrings(main_config, config_args, verbose=True)
    with SingleInferencer(**vars(args)) as inferencer:
        inferencer.run()

    """
python -m cryoPARES.inference.inferencer \
--particles_star_fname /home/sanchezg/cryo/data/preAlignedParticles/EMPIAR-10166/data/1000particles.star  --results_dir /tmp/cryoPARES_train/cryoPARES_inference/ --particles_dir /home/sanchezg/cryo/data/preAlignedParticles/EMPIAR-10166/data --checkpoint_dir /tmp/cryoPARES_train/version_0/ --NOT_use_cuda --config inference.before_refiner_buffer_size=4 --batch_size 8     
    
    """
