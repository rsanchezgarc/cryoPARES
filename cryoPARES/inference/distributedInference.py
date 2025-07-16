import os
import sys
import traceback
import time
from typing import Optional, Literal

import numpy as np
import pandas as pd
import torch

from torch import multiprocessing
from progressBarDistributed import SharedMemoryProgressBar, SharedMemoryProgressBarWorker
from torch.utils.data import ConcatDataset

from cryoPARES import constants
from cryoPARES.configManager.inject_defaults import CONFIG_PARAM, inject_defaults_from_config
from cryoPARES.configs.mainConfig import main_config
from cryoPARES.reconstruction.distributedReconstruct import create_shared_tensor
from cryoPARES.inference.inference import SingleInferencer
from cryoPARES.utils.paths import get_most_recent_file
from cryoPARES.configManager.configParser import ConfigArgumentParser, ConfigOverrideSystem


def worker(worker_id, output_q, *args, **kwargs):
    try:
        _worker(worker_id, output_q, *args, **kwargs)
    except Exception as e:
        print(f"[Worker {worker_id}] Exception occurred: {e}", file=sys.stderr)
        traceback.print_exc()
        raise e


def _worker(worker_id, output_q, pbar_fname, particles_idxs, inferencer_init_kwargs, main_config_updated,
            shared_numerator, shared_weights, shared_ctfsq, device=None):

    if device is not None:
        torch.cuda.set_device(device)
    ConfigOverrideSystem.update_config_from_dataclass(main_config, main_config_updated, verbose=False)
    inferencer_init_kwargs["subset_idxs"] = list(particles_idxs)
    inferencer = SingleInferencer(**inferencer_init_kwargs)
    with SharedMemoryProgressBarWorker(worker_id, pbar_fname) as pbar:
        inferencer._pbar = pbar
        out = inferencer.run()
        output_q.put((worker_id, out))

        reconstructor = inferencer._reconstructor
        if reconstructor is not None:
            with shared_numerator.get_lock():
                shared_numerator_tensor = torch.frombuffer(shared_numerator.get_obj(), dtype=torch.float32).reshape(
                    reconstructor.numerator.shape)
                shared_numerator_tensor += reconstructor.numerator.cpu()

            with shared_weights.get_lock():
                shared_weights_tensor = torch.frombuffer(shared_weights.get_obj(), dtype=torch.float32).reshape(
                    reconstructor.weights.shape)
                shared_weights_tensor += reconstructor.weights.cpu()

            with shared_ctfsq.get_lock():
                shared_ctfsq_tensor = torch.frombuffer(shared_ctfsq.get_obj(), dtype=torch.float32).reshape(
                    reconstructor.ctfsq.shape)
                shared_ctfsq_tensor += reconstructor.ctfsq.cpu()


@inject_defaults_from_config(main_config.inference, update_config_with_args=True)
def distributed_inference(
        particles_star_fname: str,
        checkpoint_dir: str,
        results_dir: str,
        halfset: Literal["half1", "half2", "allParticles"] = "allParticles",
        particles_dir: Optional[str] = None,
        batch_size: int = CONFIG_PARAM(),
        n_jobs: Optional[int] = None,
        num_data_workers: int = CONFIG_PARAM(config=main_config.datamanager),
        use_cuda: bool = CONFIG_PARAM(),
        compile_model: bool = False,
        top_k: int = CONFIG_PARAM(),
        reference_map: Optional[str] = None,
        directional_zscore_thr: Optional[float] = CONFIG_PARAM(),
        perform_localrefinement: bool = CONFIG_PARAM(),
        perform_reconstruction: bool = CONFIG_PARAM(),
        update_progessbar_n_batches: int = CONFIG_PARAM(),
        check_interval_secs: float = 2.  # seconds
):
    """

    :param particles_star_fname:
    :param checkpoint_dir:
    :param results_dir:
    :param halfset:
    :param particles_dir:
    :param batch_size:
    :param n_jobs:
    :param num_data_workers:
    :param use_cuda:
    :param compile_model:
    :param top_k:
    :param reference_map:
    :param directional_zscore_thr:
    :param perform_localrefinement:
    :param perform_reconstruction:
    :param update_progessbar_n_batches:
    :param check_interval_secs:
    :return:
    """
    ctx = multiprocessing.get_context('spawn')
    os.makedirs(results_dir, exist_ok=True)

    if halfset == "allParticles":
        partitions = ["half1", "half2"]
    else:
        partitions = [halfset]

    for partition in partitions:
        print(f"working on partition {partition}")
        inferencer_init_kwargs = dict(
            particles_star_fname=particles_star_fname,
            checkpoint_dir=checkpoint_dir,
            results_dir=None,
            halfset=partition,
            particles_dir=particles_dir,
            batch_size=batch_size,
            num_data_workers=num_data_workers,
            use_cuda=False,
            compile_model=compile_model,
            top_k=top_k,
            reference_map=reference_map,
            directional_zscore_thr=directional_zscore_thr,
            perform_localrefinement=perform_localrefinement,
            perform_reconstruction=perform_reconstruction,
            update_progessbar_n_batches=update_progessbar_n_batches
        )

        if use_cuda:
            n_cuda_devices = torch.cuda.device_count()
            print(f"Using {n_cuda_devices} CUDA devices")
            if n_jobs is None:
                n_jobs = n_cuda_devices
        else:
            n_cuda_devices = None
            if n_jobs is None:
                n_jobs = 1

        inferencer = SingleInferencer(**inferencer_init_kwargs)
        inferencer_init_kwargs["use_cuda"] = use_cuda
        dm = inferencer._get_datamanager()
        dataset = dm.create_dataset(None)
        n_particles = len(dataset)

        # Get shapes of result tensors
        reconstructor = inferencer._setup_reconstructor()
        if reconstructor is not None:
            numerator_shape = reconstructor.numerator.shape
            weights_shape = reconstructor.weights.shape
            ctfsq_shape = reconstructor.ctfsq.shape

            # Create shared memory tensors for results
            shared_numerator, _ = create_shared_tensor(numerator_shape, ctx=ctx)
            shared_weights, _ = create_shared_tensor(weights_shape, ctx=ctx)
            shared_ctfsq, _ = create_shared_tensor(ctfsq_shape, ctx=ctx)
        else:
            shared_numerator = None
            shared_weights = None
            shared_ctfsq = None

        with ctx.Manager() as manager,  SharedMemoryProgressBar(n_jobs) as pbar:
            output_q = manager.Queue()
            pbar_fname = pbar.shm_name

            processes = []
            for worker_id, particles_idxs in enumerate(np.array_split(range(n_particles), n_jobs)):
                # Determine device for this worker
                worker_device = None
                if n_cuda_devices is not None:
                    worker_device = f"cuda:{worker_id % n_cuda_devices}"

                p = ctx.Process(
                    target=worker,
                    args=(worker_id, output_q, pbar_fname, particles_idxs,
                          inferencer_init_kwargs, main_config,
                          shared_numerator, shared_weights, shared_ctfsq, worker_device)
                )
                p.start()
                processes.append(p)

            results = {}
            completed_workers = 0

            try:
                while completed_workers < n_jobs:
                    # Check for completed work in the queue
                    try:
                        while not output_q.empty():
                            worker_id, result = output_q.get_nowait()
                            results[worker_id] = result
                            print(f"Received results from worker {worker_id}")
                    except:
                        pass  # Queue was empty

                    # Check process health
                    all_done = True
                    for p in processes:
                        if not p.is_alive() and p.exitcode != 0:
                            print(f"Worker {p.pid} died with exit code {p.exitcode}!", file=sys.stderr)
                            # Terminate all processes
                            for q in processes:
                                if q.is_alive():
                                    print(f"Terminating worker {q.pid}...")
                                    q.terminate()
                            sys.exit(1)

                        if p.is_alive():
                            all_done = False

                    if all_done:
                        # All processes finished, collect any remaining results
                        while not output_q.empty():
                            worker_id, result = output_q.get_nowait()
                            results[worker_id] = result
                            print(f"Received final results from worker {worker_id}")
                        break

                    time.sleep(check_interval_secs)

                # Final join to clean up
                for p in processes:
                    p.join()

            except KeyboardInterrupt:
                print("Interrupted by user. Terminating workers...")
                for p in processes:
                    if p.is_alive():
                        p.terminate()
                sys.exit(1)

            # Process and save aggregated results
            if results:
                print(f"Aggregating results from {len(results)} workers...")
                aggregated_results = _aggregate_worker_results(results)

                # Save aggregated results
                if results_dir is not None:
                    if isinstance(dataset, ConcatDataset):
                        particlesSet = dataset.datasets[0].particles
                    else:
                        particlesSet = dataset.particles
                    particlesSet.particles_md = aggregated_results
                    results_file = os.path.join(results_dir, f"particles_{partition}_nnet.star")
                    particlesSet.save(results_file, overwrite=True)

        if reconstructor is not None:
            print("Backprojection done. Reconstructing")
            # Convert shared memory back to tensors
            final_numerator = torch.frombuffer(shared_numerator.get_obj(), dtype=torch.float32).reshape(numerator_shape)
            final_weights = torch.frombuffer(shared_weights.get_obj(), dtype=torch.float32).reshape(weights_shape)
            final_ctfsq = torch.frombuffer(shared_ctfsq.get_obj(), dtype=torch.float32).reshape(ctfsq_shape)

            # set results
            reconstructor.numerator = final_numerator
            reconstructor.weights = final_weights
            reconstructor.ctfsq = final_ctfsq
            output_fname  = os.path.join(results_dir, f"reconstruction_{partition}_nnet.mrc")
            reconstructor.generate_volume(output_fname)
            print(f"Reconstruction done for {partition}.")

def _aggregate_worker_results(results):
    """
    Aggregate results from multiple workers into a single coherent result.

    Args:
        results: Dict[worker_id, result] where result is the output from SingleInferencer.run()

    Returns:
        Aggregated results in the same format as SingleInferencer.run()
    """

    # Handle the case where results might be lists (if halfset was "allParticles")
    # or single DataFrames
    if not results:
        return None

    # Check if results are lists (allParticles case) or single results
    first_result = next(iter(results.values()))
    if isinstance(first_result, list):
        # This shouldn't happen in distributed mode since we handle partitions separately
        raise ValueError("Unexpected list result in distributed mode")

    # Aggregate single results
    all_dataframes = []
    for worker_id in sorted(results.keys()):
        result = results[worker_id]
        if result is not None:
            all_dataframes.append(result)

    if not all_dataframes:
        return None

    # Concatenate all results
    aggregated_result = pd.concat(all_dataframes, ignore_index=True)
    return aggregated_result

if __name__ == "__main__":
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    os.environ[constants.SCRIPT_ENTRY_POINT] = "distributedInference.py"
    print("---------------------------------------")
    print(" ".join(sys.argv))
    print("---------------------------------------")

    torch.set_float32_matmul_precision(constants.float32_matmul_precision)

    parser = ConfigArgumentParser(prog="distrib_infer_cryoPARES", description="Run inference with cryoPARES model",
                                  config_obj=main_config)
    parser.add_args_from_function(distributed_inference)
    args, config_args = parser.parse_args()
    assert os.path.isdir(args.checkpoint_dir), f"Error, checkpoint_dir {args.checkpoint_dir} not found"
    config_fname = get_most_recent_file(args.checkpoint_dir, "configs_*.yml") #max(glob.glob(os.path.join(args.checkpoint_dir, "configs_*.yml")), key=os.path.getmtime)
    ConfigOverrideSystem.update_config_from_file(main_config, config_fname, drop_paths=["inference", "projmatching"])

    distributed_inference(**vars(args))

    """
python -m cryoPARES.reconstruction.distributedReconstruct  --symmetry C1 --particles_star_fname /home/sanchezg/cryo/data/preAlignedParticles/EMPIAR-10166/data/allparticles.star --output_fname /tmp/reconstruction.mrc --use_only_n_first_batches 100    
    """
