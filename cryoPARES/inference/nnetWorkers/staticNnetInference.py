import glob
import os
import sys
import torch
import threading
import queue
import traceback
import yaml
from sqlalchemy.orm import reconstructor
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


class PartitionInferencer:
    UPDATE_PROGRESS_BAR_N_BATCHES = 20 #TODO: Move this to config

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
                 top_k: int = CONFIG_PARAM(),
                 reference_map: Optional[str] = None,
                 directional_zscore_thr: Optional[float] = 0,
                 perform_localrefinement: bool = True,
                 perform_reconstruction: bool = True
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
        :param reference_map: If not provided, it will be tried to load from the checkpointç
        :param directional_zscore_thr:
        :param perform_localrefinement:
        :param perform_reconstruction:
        """
        assert halfset != "allParticles", "allParticlesMode has not been implemented yet"
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
        self.reference_map = reference_map
        self.directional_zscore_thr = directional_zscore_thr
        self.perform_localrefinement = perform_localrefinement
        self.perform_reconstruction = perform_reconstruction

        os.makedirs(results_dir, exist_ok=True)

        self._setup_accelerator()

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
        torch.set_num_threads(max(1, torch.get_num_threads() // device_count))

        self.device, self.device_count = accelerator, device_count

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
            reconstructor = Reconstructor(symmetry=so3Model.symmetry)
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
        return model

    def _setup_dataloader(self, rank: Optional[int] = None, world_size: Optional[int] = None):
        """Setup the dataloader for inference."""
        halfset = None
        if self.halfset == "half1":
            halfset = 1
        elif self.halfset == "half2":
            halfset = 2

        hparams = os.path.join(self.checkpoint_dir, self.halfset, "hparams.yaml")
        try:
            with open(hparams) as f:
                symmetry = yaml.safe_load(f)["symmetry"]
        except (FileNotFoundError, yaml.YAMLError, KeyError) as e:
            raise RuntimeError(f"Failed to load symmetry from {hparams}: {e}")

        datamanager = DataManager(
            star_fnames=self.star_fnames,
            symmetry=symmetry,
            particles_dir=self.particles_dir,
            batch_size=self.batch_size,
            augment_train=False,  # No augmentation during inference
            halfset=halfset,
            is_global_zero=rank == 0 if rank is not None else True,
            save_train_val_partition_dir=None,  # Not needed for inference
            return_ori_imagen=True #Needed for inference
        )

        if rank is not None and world_size is not None:
            datamanager.setup_distributed(world_size, rank)

        return datamanager.predict_dataloader()

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

    def _gpu_worker(self, gpu_id: int, dataloader):
        """Async worker for a single GPU - processes subset of batches."""
        raise NotImplementedError()
        try:
            if self.device == "cuda":
                torch.cuda.set_device(gpu_id)
                device = torch.device(f'cuda:{gpu_id}')
                stream = torch.cuda.Stream(device=device)
            else:
                device = torch.device("cpu")
                stream = None

            # Setup model for this GPU
            model = self._setup_model(gpu_id if self.device == "cuda" else None)

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
                            result = self._process_batch(model, batch, batch_idx)
                    stream.synchronize()
                else:
                    with torch.inference_mode():
                        result = self._process_batch(model, batch, batch_idx)
                if result is not None:
                    batch_results.append((batch_idx, result))
                    batch_count += 1

                # Report progress periodically
                if batch_count % self.UPDATE_PROGRESS_BAR_N_BATCHES == 0:
                    self.output_queue.put(("PROGRESS", gpu_id, self.UPDATE_PROGRESS_BAR_N_BATCHES))

            # Send remaining progress
            remaining = batch_count % self.UPDATE_PROGRESS_BAR_N_BATCHES
            if remaining > 0:
                self.output_queue.put(("PROGRESS", gpu_id, remaining))
            #TODO: We need to flush the predictor
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

    def run(self): #TODO: check if this is correct
        if self.halfset == "allParticles":
            self.halfset = "half1"
            self._run()

            self.halfset = "half2"
            self._run()
        else:
            return self._run()

    def _run(self):
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
                    thread.join(timeout=3.0)

            else:
                print("Using single GPU processing")
                # Single GPU processing
                model = self._setup_model()
                all_results = {}
                pbar = tqdm(total=total_batches, desc="Processing batches")
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
                finally:
                    pbar.close()
                    if self.perform_reconstruction:
                        out_fname = os.path.join(self.results_dir, f"reconstruction_{self.halfset}_nnet.mrc")
                        model.reconstructor.generate_volume(out_fname)

            if self.error_occurred.is_set():
                raise RuntimeError(f"Error during processing: {self.error_message}")

            # Aggregate results and save to STAR files
            self._save_results(all_results, dataset)

        except Exception as e:
            print(f"Inference failed: {e}")
            raise
        finally:
            self._cleanup()

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
            out_fname = os.path.join(self.results_dir, basename + f"_{self.halfset}_nnet.star")
            print(f"Results were saved at {out_fname}")
            particlesSet.save(out_fname)

    # def _cleanup(self):
    #     """Clean up resources."""
    #     self.stop_processing.set()
    #
    #     # Clear queues
    #     while not self.output_queue.empty():
    #         try:
    #             self.output_queue.get_nowait()
    #         except queue.Empty:
    #             break


    def _cleanup(self, timeout: float = 10.0):
        """
        Clean up resources including threads, queues, and CUDA resources.

        Args:
            timeout: Maximum time to wait for threads to finish (seconds)
        """
        print("Starting cleanup...")

        # 1. Signal all threads to stop
        self.stop_processing.set()

        # 2. Wait for processing threads to finish
        if hasattr(self, 'processing_threads') and self.processing_threads:
            print(f"Waiting for {len(self.processing_threads)} worker threads to finish...")

            for i, thread in enumerate(self.processing_threads):
                if thread.is_alive():
                    try:
                        thread.join(timeout=timeout / len(self.processing_threads))
                        if thread.is_alive():
                            print(f"Warning: Thread {i} did not finish within timeout")
                    except Exception as e:
                        print(f"Error waiting for thread {i}: {e}")

            # Clear the thread list
            self.processing_threads.clear()

        # 3. Drain the output queue safely
        if hasattr(self, 'output_queue'):
            drained_items = 0
            max_drain_attempts = 1000  # Prevent infinite loop

            try:
                while drained_items < max_drain_attempts:
                    try:
                        # Use a very short timeout to avoid blocking
                        self.output_queue.get(timeout=0.01)
                        drained_items += 1
                    except queue.Empty:
                        # Queue is actually empty now
                        break
                    except Exception as e:
                        print(f"Error draining queue: {e}")
                        break

                if drained_items >= max_drain_attempts:
                    print(f"Warning: Stopped draining queue after {max_drain_attempts} items")
                elif drained_items > 0:
                    print(f"Drained {drained_items} items from output queue")

            except Exception as e:
                print(f"Error during queue cleanup: {e}")

        # 4. Clear input queues if they exist
        if hasattr(self, 'input_queues'):
            for i, input_queue in enumerate(self.input_queues):
                if input_queue is not None:
                    try:
                        while True:
                            try:
                                input_queue.get_nowait()
                            except queue.Empty:
                                break
                    except Exception as e:
                        print(f"Error cleaning input queue {i}: {e}")
            self.input_queues.clear()

        # 5. Clean up CUDA resources
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                # Clear CUDA streams
                if hasattr(self, 'streams'):
                    for stream in self.streams:
                        if stream is not None:
                            try:
                                stream.synchronize()
                            except Exception as e:
                                print(f"Error synchronizing CUDA stream: {e}")
                    self.streams.clear()

                # Clear CUDA cache
                torch.cuda.empty_cache()

                # Synchronize all devices that were used
                for device_id in range(min(self.device_count, torch.cuda.device_count())):
                    try:
                        with torch.cuda.device(device_id):
                            torch.cuda.synchronize()
                    except Exception as e:
                        print(f"Error synchronizing CUDA device {device_id}: {e}")

            except Exception as e:
                print(f"Error during CUDA cleanup: {e}")

        # 6. Clear model references to help with memory cleanup
        if hasattr(self, 'models'):
            try:
                for model in self.models:
                    if model is not None:
                        # Move to CPU to free GPU memory
                        if hasattr(model, 'cpu'):
                            try:
                                model.cpu()
                            except Exception as e:
                                print(f"Error moving model to CPU: {e}")
                self.models.clear()
            except Exception as e:
                print(f"Error cleaning up models: {e}")

        # 7. Clear device references
        if hasattr(self, 'devices'):
            self.devices.clear()

        # 8. Reset state flags
        try:
            self.error_occurred.clear()
            self.error_message = None
        except Exception as e:
            print(f"Error resetting state: {e}")

        # 9. Force garbage collection (optional, but can help with memory)
        try:
            import gc
            gc.collect()
        except Exception as e:
            print(f"Error during garbage collection: {e}")

        print("Cleanup completed")


    def __del__(self):
        """Destructor to ensure cleanup happens even if not called explicitly."""
        try:
            self._cleanup(timeout=5.0)  # Shorter timeout for destructor
        except Exception as e:
            # Don't raise exceptions in destructor
            print(f"Error during destructor cleanup: {e}")
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup()
        return False  # Don't suppress exceptions

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

    with PartitionInferencer(**vars(args)) as inferencer:
        inferencer.run()