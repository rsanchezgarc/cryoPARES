import glob
import os
import sys
import torch
import yaml
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
                 halfset: Literal["half1", "half2", "allParticles"] = "allParticles",
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
                 ):
        """

        :param particles_star_fname:
        :param checkpoint_dir:
        :param results_dir:
        :param halfset:
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
        """

        self.particles_star_fname = particles_star_fname
        self.checkpoint_dir = checkpoint_dir

        assert os.path.isdir(checkpoint_dir), f"checkpoint_dir {checkpoint_dir} needs to be a directory"
        main_config.datamanager.num_augmented_copies_per_batch = 1 # We have not implemented test-time augmentation

        self.particles_dir = particles_dir
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.n_cpus_if_no_cuda = n_cpus_if_no_cuda
        self.results_dir = results_dir
        self.halfset = halfset
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

    def _setup_reconstructor(self, symmetry: Optional[str]= None):
        if symmetry is None:
            symmetry = self._get_symmetry()
        reconstructor = Reconstructor(symmetry=symmetry)
        reconstructor._get_reconstructionParticlesDataset(self.particles_star_fname, self.particles_dir)
        self._reconstructor = reconstructor
        return reconstructor

    def _setup_model(self, rank: Optional[int] = None):
        """Setup the model for inference."""

        so3Model_fname = os.path.join(self.checkpoint_dir, self.halfset, "checkpoints", BEST_MODEL_SCRIPT_BASENAME)
        so3Model = torch.jit.load(so3Model_fname)

        percentilemodel_fname = os.path.join(self.checkpoint_dir, self.halfset, "checkpoints", BEST_DIRECTIONAL_NORMALIZER)
        percentilemodel = torch.load(percentilemodel_fname, weights_only=False)

        if self.reference_map is None:
            reference_map = os.path.join(self.checkpoint_dir, self.halfset, "reconstructions", "0.mrc")
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
        hparams = os.path.join(self.checkpoint_dir, self.halfset, "hparams.yaml")
        try:
            with open(hparams) as f:
                symmetry = yaml.safe_load(f)["symmetry"]
            return symmetry
        except (FileNotFoundError, yaml.YAMLError, KeyError) as e:
            raise RuntimeError(f"Failed to load symmetry from {hparams}: {e}")

    def _get_datamanager(self, rank: Optional[int] = None):
        """Setup the dataloader for inference."""
        halfset = None
        if self.halfset == "half1":
            halfset = 1
        elif self.halfset == "half2":
            halfset = 2

        symmetry = self._get_symmetry()
        datamanager = DataManager(
            star_fnames=self.particles_star_fname,
            symmetry=symmetry,
            particles_dir=self.particles_dir,
            batch_size=self.batch_size,
            augment_train=False,  # No augmentation during inference
            halfset=halfset,
            is_global_zero=rank == 0 if rank is not None else True,
            save_train_val_partition_dir=None,  # Not needed for inference
            return_ori_imagen=True, #Needed for inference,
            subset_idxs=self._subset_idxs
        )
        return datamanager

    def _setup_dataloader(self, rank: Optional[int] = None):
        self._datamanager = self._get_datamanager(rank)
        return  self._datamanager.predict_dataloader()

    def _process_batch(self, model: PlModel, batch: Dict[str, Any] | None, batch_idx: int):
        """Process a single batch of data using predict_step."""
        device = self.device
        if batch is not None:
            if hasattr(model, 'transfer_batch_to_device'):
                batch = model.transfer_batch_to_device(batch, self.device, dataloader_idx=0)
            else:
                # Fallback: manually transfer tensors
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(device, non_blocking=True)
                    elif isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                        batch[key] = [v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for v in value]

            # Use predict_step as requested
            result = model.predict_step(batch, batch_idx=batch_idx, dataloader_idx=0)
        else:
            result = model.flush()
        if result is None:
            return None
        ids, pred_rotmats, pred_shifts, score, norm_nn_score = result
        # Convert rotation matrices to euler angles
        euler_degs = torch.rad2deg(matrix_to_euler_angles(pred_rotmats, RELION_EULER_CONVENTION))

        return ids, euler_degs, pred_shifts, score, norm_nn_score

    def run(self):
        if self.halfset == "allParticles":
            out_list = []
            self.halfset = "half1"
            print(f"Running inference for {self.halfset}")
            out = self._run()
            out_list.append(out)
            self.halfset = "half2"
            print(f"Running inference for {self.halfset}")
            out = self._run()
            out_list.append(out)
            return out
        else:
            return self._run()

    def _get_pbar(self, total):
        if self._pbar is None:
            return tqdm(total=total, desc="Processing batches")
        else:
            return self._pbar

    def _run(self):
        """Main entry point for running async inference."""

        # Get total dataset info
        dataloader = self._setup_dataloader()
        dataset = dataloader.dataset
        total_batches = len(dataloader)

        print(f"Processing {len(dataset)} particles in {total_batches} batches")

        model = self._setup_model()
        all_results = {}
        pbar = self._get_pbar(total_batches)
        try:
            with torch.inference_mode():
                for batch_idx, batch in enumerate(dataloader):
                    result = self._process_batch(model, batch, batch_idx)
                    if result:
                        all_results[batch_idx] = result
                    pbar.update(1)
                result = self._process_batch(model, batch=None, batch_idx=-1) #This flushes the buffer
                if result:
                    all_results[-1] = result
            if self.perform_reconstruction and self.results_dir is not None:
                out_fname = os.path.join(self.results_dir, f"reconstruction_{self.halfset}_nnet.mrc")
                model.reconstructor.generate_volume(out_fname)

            # Aggregate results and save to STAR files
            particles_md = self._save_results(all_results, dataset)
            return particles_md
        finally:
            pbar.close()


    def _save_results(self, all_results: Dict[int, Any], dataset):
        """Save results to STAR files using original aggregation logic."""
        print("Aggregating results and saving to STAR files...")
        #TODO: This assumes that the order of the dataset is always fixed. The assumtion is safe if only one
        #starfile is provided

        # Create result tensors
        n_particles = sum(len(v[0])for v in all_results.values())
        result_arrays = {
            'eulerdegs': torch.zeros((n_particles, self.top_k, 3), dtype=torch.float32),
            'score': torch.zeros((n_particles, self.top_k), dtype=torch.float32),
            'shifts': torch.zeros((n_particles, self.top_k, 2), dtype=torch.float32),
            'top1_directional_zscore': torch.zeros((n_particles), dtype=torch.float32),
            'ids': [None]*n_particles
        }

        # Fill result arrays
        current_idx = 0
        for batch_idx in sorted(all_results.keys()):
            idd, euler_degs, pred_shifts, maxprobs, top1_directional_zscore = all_results[batch_idx]

            batch_size = len(idd)
            end_idx = current_idx + batch_size

            result_arrays['eulerdegs'][current_idx:end_idx] = euler_degs.cpu()
            result_arrays['score'][current_idx:end_idx] = maxprobs.cpu()
            result_arrays['shifts'][current_idx:end_idx] = pred_shifts.cpu()
            result_arrays['top1_directional_zscore'][current_idx:end_idx] = top1_directional_zscore.cpu()
            result_arrays['ids'][current_idx:end_idx] = idd
            current_idx = end_idx

        # Save to STAR files (same logic as original)
        for datIdx, _dataset in enumerate(dataset.datasets):
            particlesSet = _dataset.particles
            particles_md = particlesSet.particles_md
            ids = result_arrays["ids"]
            particles_md[DIRECTIONAL_ZSCORE_NAME] = 0.
            particles_md.loc[ids, DIRECTIONAL_ZSCORE_NAME] = result_arrays["top1_directional_zscore"].numpy()

            for k in range(self.top_k):
                suffix = "" if k == 0 else f"_top{k}"
                angles_names = [x + suffix for x in RELION_ANGLES_NAMES]
                shifts_names = [x + suffix for x in RELION_SHIFTS_NAMES]
                confide_name = RELION_PRED_POSE_CONFIDENCE_NAME + suffix

                particles_md[angles_names] = 0.
                particles_md[shifts_names] = 0.
                particles_md[confide_name] = 0.

                particles_md.loc[ids, angles_names] = result_arrays["eulerdegs"][..., k, :].numpy()
                particles_md.loc[ids, shifts_names] = result_arrays["shifts"][..., k, :].numpy()
                particles_md.loc[ids, confide_name] = result_arrays["score"][..., k].numpy()

            if particlesSet.starFname is not None:
                basename = os.path.basename(particlesSet.starFname).removesuffix(".star")
            else:
                basename = "particles%d"%datIdx
            if self.results_dir is not None:
                out_fname = os.path.join(self.results_dir, basename + f"_{self.halfset}_nnet.star")
                print(f"Results were saved at {out_fname}")
                particlesSet.save(out_fname)
            return particles_md

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False  # Don't suppress exceptions

if __name__ == "__main__":
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

    with SingleInferencer(**vars(args)) as inferencer:
        inferencer.run()

    """
    
    """