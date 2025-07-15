import os
import sys
import traceback
import time
from typing import Optional, Literal

import numpy as np
import torch

from torch import multiprocessing
from progressBarDistributed import SharedMemoryProgressBar, SharedMemoryProgressBarWorker

from cryoPARES import constants
from cryoPARES.configManager.inject_defaults import CONFIG_PARAM, inject_defaults_from_config
from cryoPARES.configs.mainConfig import main_config
from cryoPARES.reconstruction.distributedReconstruct import create_shared_tensor
from cryoPARES.inference.inference import SingleInferencer
from cryoPARES.utils.paths import get_most_recent_file


def worker(worker_id, *args, **kwargs):
    try:
        _worker(worker_id, *args, **kwargs)
    except Exception as e:
        print(f"[Worker {worker_id}] Exception occurred: {e}", file=sys.stderr)
        traceback.print_exc()
        raise e


def _worker(worker_id, pbar_fname, particles_idxs, inferencer_init_kwargs,
            shared_numerator, shared_weights, shared_ctfsq, device=None):

    if device is not None:
        torch.cuda.set_device(device)


    inferencer_init_kwargs["subset_idxs"] = list(particles_idxs)
    inferencer = SingleInferencer(**inferencer_init_kwargs)
    batch_size = inferencer_init_kwargs["batch_size"]
    n_parts = len(particles_idxs)
    n_batches = n_parts // batch_size + int((n_parts%batch_size)!=0)
    print("n_batches", n_batches)
    with SharedMemoryProgressBarWorker(worker_id, pbar_fname) as pbar:
        pbar.set_total_steps(n_batches)
        inferencer._pbar = pbar
        out = inferencer.run()

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
        update_progessbar_n_batches: int = CONFIG_PARAM()
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
    :return:
    """

    ctx = multiprocessing.get_context('spawn')

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
        # reconstructor, particlesDs = inferencer._setup_reconstructor()
        dm = inferencer._get_datamanager()
        n_particles = len(particlesDs)

        # Get shapes of result tensors
        numerator_shape = reconstructor.numerator.shape
        weights_shape = reconstructor.weights.shape
        ctfsq_shape = reconstructor.ctfsq.shape

        # Create shared memory tensors for results
        shared_numerator, _ = create_shared_tensor(numerator_shape, ctx=ctx)
        shared_weights, _ = create_shared_tensor(weights_shape, ctx=ctx)
        shared_ctfsq, _ = create_shared_tensor(ctfsq_shape, ctx=ctx)


        with SharedMemoryProgressBar(n_jobs) as pbar:
            pbar_fname = pbar.shm_name

            processes = []
            for worker_id, particles_idxs in enumerate(np.array_split(range(n_particles), n_jobs)):
                # Determine device for this worker
                worker_device = None
                if n_cuda_devices is not None:
                    worker_device = f"cuda:{worker_id % n_cuda_devices}"

                p = ctx.Process(
                    target=worker,
                    args=(worker_id, pbar_fname, particles_idxs,
                          inferencer_init_kwargs,
                          shared_numerator, shared_weights, shared_ctfsq, worker_device)
                )
                p.start()
                processes.append(p)

            check_interval = 2  # seconds
            all_done = False
            try:
                while not all_done:
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

                    time.sleep(check_interval)

                # Final join to clean up any remaining zombie processes
                for p in processes:
                    p.join()

            except KeyboardInterrupt:
                print("Interrupted by user. Terminating workers...")
                for p in processes:
                    if p.is_alive():
                        p.terminate()
                sys.exit(1)

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
        print("Reconstruction done.")


if __name__ == "__main__":
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    os.environ[constants.SCRIPT_ENTRY_POINT] = "distributedInference.py"
    print("---------------------------------------")
    print(" ".join(sys.argv))
    print("---------------------------------------")
    from cryoPARES.configManager.configParser import ConfigArgumentParser, ConfigOverrideSystem

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
