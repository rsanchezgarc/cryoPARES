import glob
import os
import sys
import torch
import threading
import queue
import traceback
import yaml
from tqdm import tqdm
from typing import Optional, List, Literal, Dict, Any, Tuple

from starstack.constants import RELION_ANGLES_NAMES, RELION_SHIFTS_NAMES, RELION_PRED_POSE_CONFIDENCE_NAME, \
    RELION_EULER_CONVENTION

from cryoPARES import constants
from cryoPARES.configManager.configParser import ConfigOverrideSystem
from cryoPARES.configManager.inject_defaults import inject_defaults_from_config, CONFIG_PARAM
from cryoPARES.configs.mainConfig import main_config
from cryoPARES.geometry.convert_angles import matrix_to_euler_angles
from cryoPARES.models.model import PlModel
from cryoPARES.datamanager.datamanager import DataManager


class AsyncPartitionInferencer:
    UPDATE_PROGRESS_BAR_N_BATCHES = 10

    @inject_defaults_from_config(main_config.inference, update_config_with_args=True)
    def __init__(self,
                 star_fnames: List[str],
                 checkpoint_dir: str,
                 results_dir: str,
                 halfset: Literal["half1", "half2", "allParticles"] = "allParticles",
                 particles_dir: Optional[List[str]] = None,
                 batch_size: int = CONFIG_PARAM(),
                 num_data_workers: int = CONFIG_PARAM(config=main_config.datamanager),
                 use_cuda: bool = CONFIG_PARAM(),
                 n_cpus_if_no_cuda: int = CONFIG_PARAM(),
                 compile_model: bool = False,
                 top_k: int = CONFIG_PARAM()
                 ):
        """

        :param star_fnames:
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
        """

        self.star_fnames = star_fnames
        self.checkpoint_dir = checkpoint_dir

        main_config.datamanager.num_augmented_copies_per_batch = 1 # We have not implemented test-time augmentation

        self.particles_dir = particles_dir
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.n_cpus_if_no_cuda = n_cpus_if_no_cuda
        self.results_dir = results_dir
        self.halfset = halfset
        self.compile_model = compile_model
        self.top_k = top_k
        os.makedirs(results_dir, exist_ok=True)

        self.accelerator, self.device_count = self._setup_accelerator()

        # For async processing
        self.models = []
        self.streams = []
        self.devices = []
        self.input_queues = []
        self.output_queue = queue.Queue()
        self.stop_processing = threading.Event()
        self.error_occurred = threading.Event()
        self.error_message = None
        self.processing_threads = []

    def _setup_accelerator(self):
        """Setup the computation device and count."""
        if self.use_cuda and torch.cuda.is_available():
            accelerator = "cuda"
            device_count = torch.cuda.device_count()
        else:
            accelerator = "cpu"
            device_count = self.n_cpus_if_no_cuda

        print(f'devices={device_count} accelerator={accelerator}', flush=True)
        torch.set_num_threads(min(1, torch.get_num_threads() // device_count))
        return accelerator, device_count

    def _setup_model(self, rank: Optional[int] = None):
        """Setup the model for inference."""

        scriptmodel_fname = os.path.join(self.checkpoint_dir, self.halfset, "checkpoints", "best_script.pt")
        if os.path.isfile(scriptmodel_fname):
            scriptmodel = torch.jit.load(scriptmodel_fname)
        else:
            scriptmodel = None
        checkpoint_fname = os.path.join(self.checkpoint_dir, self.halfset, "checkpoints", "best.ckpt")
        model = PlModel.load_from_checkpoint(
            checkpoint_fname,
            map_location="cpu",
            model=scriptmodel
        )

        # Handle missing symmetry attribute
        if not hasattr(model, 'symmetry'):
            # Try to get from hyperparameters or set default
            if hasattr(model, 'hparams') and hasattr(model.hparams, 'symmetry'):
                model.symmetry = model.hparams.symmetry
            else:
                raise RuntimeError("Symmetry not found in model")

        device = f'cuda:{rank}' if rank is not None else 'cuda:0'
        if self.accelerator == "cpu":
            device = "cpu"

        model = model.to(device)

        if self.compile_model:
            print(f"Compiling model on device {device}")
            model = torch.compile(model)

        model.eval()
        return model

    def _setup_dataloader(self, rank: Optional[int] = None, world_size: Optional[int] = None):
        """Setup the dataloader for inference."""
        halfset = None
        if self.halfset == "half1":
            halfset = 1
        elif self.halfset == "half2":
            halfset = 2

        hparams = os.path.join(self.checkpoint_dir, self.halfset, "hparams.yaml")
        with open(hparams) as f:
            symmetry = yaml.safe_load(f)["symmetry"]
        datamanager = DataManager(
            star_fnames=self.star_fnames,
            symmetry=symmetry,
            particles_dir=self.particles_dir,
            batch_size=self.batch_size,
            augment_train=False,  # No augmentation during inference
            halfset=halfset,
            is_global_zero=rank == 0 if rank is not None else True,
            save_train_val_partition_dir=None  # Not needed for inference
        )

        if rank is not None and world_size is not None:
            datamanager.setup_distributed(world_size, rank)

        return datamanager.predict_dataloader()

    def _process_batch(self, model: PlModel, batch: Dict[str, Any], batch_idx: int, device: torch.device):
        """Process a single batch of data using predict_step."""

        if hasattr(model, 'transfer_batch_to_device'):
            batch = model.transfer_batch_to_device(batch, device, dataloader_idx=0)
        else:
            # Fallback: manually transfer tensors
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device, non_blocking=True)
                elif isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                    batch[key] = [v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for v in value]

        # Use predict_step as requested
        result = model.predict_step(batch, batch_idx=batch_idx, dataloader_idx=0)
        (idd, (pred_rotmats, maxprobs, all_angles_probs),
         (pred_shifts, shifts_probs), errors, metadata) = result

        # Convert rotation matrices to euler angles
        euler_degs = torch.rad2deg(matrix_to_euler_angles(pred_rotmats, RELION_EULER_CONVENTION))

        # Set shifts to zero as in original code
        pred_shifts.fill_(0.)
        shifts_probs.fill_(1.)

        return idd, (euler_degs, maxprobs), (pred_shifts, shifts_probs), metadata

    def _gpu_worker(self, gpu_id: int, dataloader):
        """Async worker for a single GPU - processes subset of batches."""
        try:
            if self.accelerator == "cuda":
                torch.cuda.set_device(gpu_id)
                device = torch.device(f'cuda:{gpu_id}')
                stream = torch.cuda.Stream(device=device)
            else:
                device = torch.device("cpu")
                stream = None

            # Setup model for this GPU
            model = self._setup_model(gpu_id if self.accelerator == "cuda" else None)

            batch_results = []
            batch_count = 0

            # Process batches assigned to this GPU
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx % self.device_count != gpu_id:
                    continue  # Skip batches not assigned to this GPU

                if self.error_occurred.is_set():
                    break

                if stream is not None:
                    with torch.cuda.stream(stream):
                        with torch.inference_mode():
                            result = self._process_batch(model, batch, batch_idx, device)
                    stream.synchronize()
                else:
                    with torch.inference_mode():
                        result = self._process_batch(model, batch, batch_idx, device)

                batch_results.append((batch_idx, result))
                batch_count += 1

                # Report progress periodically
                if batch_count % self.UPDATE_PROGRESS_BAR_N_BATCHES == 0:
                    self.output_queue.put(("PROGRESS", gpu_id, self.UPDATE_PROGRESS_BAR_N_BATCHES))

            # Send remaining progress
            remaining = batch_count % self.UPDATE_PROGRESS_BAR_N_BATCHES
            if remaining > 0:
                self.output_queue.put(("PROGRESS", gpu_id, remaining))

            # Send results
            self.output_queue.put(("RESULTS", gpu_id, batch_results))

        except Exception as e:
            self.error_message = f"GPU {gpu_id} worker error: {str(e)}\n{traceback.format_exc()}"
            self.error_occurred.set()
            print(self.error_message)
        finally:
            self.output_queue.put(("WORKER_DONE", gpu_id, None))

    def _collect_results(self, total_batches: int) -> Tuple[Dict[int, Any], int]:
        """Collect results from all GPU workers."""
        all_results = {}
        workers_done = 0
        total_progress = 0

        # Setup progress bar
        pbar = tqdm(total=total_batches, desc="Processing batches")

        try:
            while workers_done < self.device_count:
                if self.error_occurred.is_set():
                    break

                try:
                    message_type, gpu_id, data = self.output_queue.get(timeout=1.0)

                    if message_type == "PROGRESS":
                        total_progress += data
                        pbar.update(data)

                    elif message_type == "RESULTS":
                        for batch_idx, result in data:
                            all_results[batch_idx] = result

                    elif message_type == "WORKER_DONE":
                        workers_done += 1

                except queue.Empty:
                    continue

            return all_results, total_progress

        finally:
            pbar.close()

    def run(self):
        """Main entry point for running async inference."""


        try:
            # Get total dataset info
            dataloader = self._setup_dataloader()
            dataset = dataloader.dataset
            total_batches = len(dataloader)

            print(f"Processing {len(dataset)} particles in {total_batches} batches")

            # Reset async state
            self.stop_processing.clear()
            self.error_occurred.clear()
            self.error_message = None

            if self.device_count > 1:
                print(f"Using async multi-GPU processing with {self.device_count} GPUs")

                # Start GPU worker threads
                self.processing_threads = []
                for gpu_id in range(self.device_count):
                    thread = threading.Thread(
                        target=self._gpu_worker,
                        args=(gpu_id, dataloader),
                        daemon=True
                    )
                    thread.start()
                    self.processing_threads.append(thread)

                # Collect results
                all_results, total_progress = self._collect_results(total_batches)

                # Wait for all workers to finish
                for thread in self.processing_threads:
                    thread.join(timeout=5.0)

            else:
                print("Using single GPU processing")
                # Single GPU processing
                model = self._setup_model()
                device = torch.device('cuda:0' if self.accelerator == "cuda" else "cpu")

                all_results = {}
                pbar = tqdm(total=total_batches, desc="Processing batches")

                try:
                    with torch.inference_mode():
                        for batch_idx, batch in enumerate(dataloader):
                            result = self._process_batch(model, batch, batch_idx, device)
                            all_results[batch_idx] = result
                            pbar.update(1)
                finally:
                    pbar.close()

            if self.error_occurred.is_set():
                raise RuntimeError(f"Error during processing: {self.error_message}")

            # Aggregate results and save to STAR files (same logic as original)
            self._save_results(all_results, dataset)

        except Exception as e:
            print(f"Inference failed: {e}")
            raise
        finally:
            self._cleanup()

    def _save_results(self, all_results: Dict[int, Any], dataset):
        """Save results to STAR files using original aggregation logic."""
        print("Aggregating results and saving to STAR files...")

        # Create result tensors
        n_particles = len(dataset)
        result_arrays = {
            'eulerdegs': torch.zeros((n_particles, self.top_k, 3), dtype=torch.float32),
            'rotprobs': torch.zeros((n_particles, self.top_k), dtype=torch.float32),
            'shifts': torch.zeros((n_particles, self.top_k, 2), dtype=torch.float32),
            'shiftprobs': torch.zeros((n_particles, self.top_k), dtype=torch.float32),
            'idxs': torch.arange(n_particles, dtype=torch.int64)
        }

        # Fill result arrays
        current_idx = 0
        for batch_idx in sorted(all_results.keys()):
            idd, (euler_degs, maxprobs), (pred_shifts, shifts_probs), metadata = all_results[batch_idx]

            batch_size = len(idd)
            end_idx = current_idx + batch_size

            result_arrays['eulerdegs'][current_idx:end_idx] = euler_degs.cpu()
            result_arrays['rotprobs'][current_idx:end_idx] = maxprobs.cpu()
            result_arrays['shifts'][current_idx:end_idx] = pred_shifts.cpu()
            result_arrays['shiftprobs'][current_idx:end_idx] = shifts_probs.cpu()

            current_idx = end_idx

        # Save to STAR files (same logic as original)
        for datIdx, _dataset in enumerate(dataset.datasets):
            particlesSet = _dataset.particles
            particles_md = particlesSet.particles_md

            for k in range(self.top_k):
                suffix = "" if k == 0 else f"_top{k}"
                angles_names = [x + suffix for x in RELION_ANGLES_NAMES]
                shifts_names = [x + suffix for x in RELION_SHIFTS_NAMES]
                confide_name = RELION_PRED_POSE_CONFIDENCE_NAME + suffix

                particles_md[angles_names] = 0.
                particles_md[shifts_names] = 0.
                particles_md[confide_name] = 0.

                # Get indices for this dataset
                idxs = result_arrays["idxs"]
                ids = particles_md.index[idxs]

                particles_md.loc[ids, angles_names] = result_arrays["eulerdegs"][..., k, :].numpy()
                particles_md.loc[ids, shifts_names] = result_arrays["shifts"][..., k, :].numpy()
                particles_md.loc[ids, confide_name] = (result_arrays["rotprobs"][..., k] *
                                                       result_arrays["shiftprobs"][..., k]).numpy()

            if particlesSet.starFname is not None:
                basename =  os.path.basename(particlesSet.starFname).removesuffix(".star")
            else:
                basename = "particles%d"%datIdx
            out_fname = os.path.join(self.results_dir, basename + "_nnet.star")
            print(f"Results were saved at {out_fname}")
            particlesSet.save(out_fname)

    def _cleanup(self):
        """Clean up resources."""
        self.stop_processing.set()

        # Clear queues
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                break


# Backwards compatibility - keep the same class name and interface
class PartitionInferencer(AsyncPartitionInferencer):
    """Backwards compatible class name."""
    pass


if __name__ == "__main__":
    os.environ[constants.PROJECT_NAME + "__ENTRY_POINT"] = "staticNnetInference.py"
    print("---------------------------------------")
    print(" ".join(sys.argv))
    print("---------------------------------------")
    from cryoPARES.configManager.configParser import ConfigArgumentParser


    parser = ConfigArgumentParser(prog="infer_cryoPARES", description="Run inference with cryoPARES model",
                                  config_obj=main_config)
    parser.add_args_from_function(PartitionInferencer.__init__)
    args, config_args = parser.parse_args()

    # Update config #TODO: This needs to happen before being added to config
    config_fname = max(glob.glob(os.path.join(args.checkpoint_dir, "configs_*.yml")), key=os.path.getmtime)
    ConfigOverrideSystem.update_config_from_file(main_config, config_fname, )

    PartitionInferencer(**vars(args)).run()