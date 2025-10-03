import os
import sys
import traceback
import time
from typing import Optional, Literal

import numpy as np
from torch import multiprocessing
import torch
from progressBarDistributed import SharedMemoryProgressBar, SharedMemoryProgressBarWorker

from cryoPARES.configManager.inject_defaults import inject_defaults_from_config, inject_docs_from_config_params, CONFIG_PARAM
from cryoPARES.configs.mainConfig import main_config
from cryoPARES.reconstruction.reconstructor import Reconstructor

_RECONSTRUCTOR = None

def worker(worker_id, *args, **kwargs):
    try:
        _worker(worker_id, *args, **kwargs)
    except Exception as e:
        print(f"[Worker {worker_id}] Exception occurred: {e}", file=sys.stderr)
        traceback.print_exc()
        raise e


def _worker(worker_id, pbar_fname, particles_idxs, reconstructor_init_kwargs, reconstructor_run_kwargs,
            shared_numerator, shared_weights, shared_ctfsq, device=None):
    global _RECONSTRUCTOR
    if _RECONSTRUCTOR is None:
        # Initialize reconstructor in worker process
        _RECONSTRUCTOR = Reconstructor(**reconstructor_init_kwargs)

        # Move to appropriate CUDA device if specified
        if device is not None:
            _RECONSTRUCTOR.to(device)

    reconstructor_run_kwargs["subset_idxs"] = list(particles_idxs)

    with SharedMemoryProgressBarWorker(worker_id, pbar_fname) as pbar:
        pbar.set_total_steps(
            len(particles_idxs)
        )

        for n_parts in _RECONSTRUCTOR._backproject_particles(**reconstructor_run_kwargs, verbose=False):
            pbar.update(n_parts)

        # Write results directly to shared memory tensors
        # Use locks to prevent race conditions
        with shared_numerator.get_lock():
            shared_numerator_tensor = torch.frombuffer(shared_numerator.get_obj(), dtype=torch.float32).reshape(
                _RECONSTRUCTOR.numerator.shape)
            shared_numerator_tensor += _RECONSTRUCTOR.numerator.cpu()

        with shared_weights.get_lock():
            shared_weights_tensor = torch.frombuffer(shared_weights.get_obj(), dtype=torch.float32).reshape(
                _RECONSTRUCTOR.weights.shape)
            shared_weights_tensor += _RECONSTRUCTOR.weights.cpu()

        with shared_ctfsq.get_lock():
            shared_ctfsq_tensor = torch.frombuffer(shared_ctfsq.get_obj(), dtype=torch.float32).reshape(
                _RECONSTRUCTOR.ctfsq.shape)
            shared_ctfsq_tensor += _RECONSTRUCTOR.ctfsq.cpu()


def create_shared_tensor(shape, dtype=torch.float32, ctx=None):
    """Create a shared memory tensor"""
    size = int(np.prod(shape))  # Convert numpy.int64 to Python int
    if dtype == torch.float32:
        typecode = 'f'
    elif dtype == torch.float64:
        typecode = 'd'
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    if ctx is None:
        ctx = multiprocessing
    shared_array = ctx.Array(typecode, size)
    return shared_array, shape

@inject_docs_from_config_params
@inject_defaults_from_config(main_config.reconstruct, update_config_with_args=True)
def reconstruct_starfile(particles_star_fname: str,
                         symmetry: str,
                         output_fname: str,
                         particles_dir:Optional[str]=None,
                         n_jobs: int  = 1,
                         num_dataworkers: int = 1,
                         batch_size: int = 128,
                         use_cuda: bool = True,
                         correct_ctf: bool = CONFIG_PARAM(),
                         eps: float = CONFIG_PARAM(),
                         min_denominator_value: Optional[float] = None,
                         use_only_n_first_batches: Optional[int] = None,
                         float32_matmul_precision: Optional[str] = CONFIG_PARAM(),
                         weight_with_confidence: bool = CONFIG_PARAM(),
                         halfmap_subset: Optional[Literal["1", "2"]] = None
                         ):
    """
    Reconstruct a 3D volume from particle images with known poses.

    :param particles_star_fname: {particles_star_fname}
    :param symmetry: {symmetry}
    :param output_fname: {output_fname}
    :param particles_dir: {particles_dir}
    :param n_jobs: {n_jobs}
    :param num_dataworkers: {num_dataworkers}
    :param batch_size: {batch_size}
    :param use_cuda: {use_cuda}
    :param correct_ctf: {correct_ctf}
    :param eps: {eps}
    :param min_denominator_value: {min_denominator_value}
    :param use_only_n_first_batches: {use_only_n_first_batches}
    :param float32_matmul_precision: {float32_matmul_precision}
    :param weight_with_confidence: {weight_with_confidence}
    :param halfmap_subset: {halfmap_subset}
    """

    if n_jobs == 1:
        from .reconstructor import reconstruct_starfile as single_job_reconstruct_starfile
        single_job_reconstruct_starfile(particles_star_fname, symmetry, output_fname,
                                        particles_dir=particles_dir, num_dataworkers=num_dataworkers,
                                        batch_size=batch_size, use_cuda=use_cuda, correct_ctf=correct_ctf,
                                        eps=eps, min_denominator_value=min_denominator_value,
                                        use_only_n_first_batches=use_only_n_first_batches,
                                        float32_matmul_precision=float32_matmul_precision,
                                        weight_with_confidence=weight_with_confidence,
                                        halfmap_subset=halfmap_subset)
        return 0
    elif n_jobs <1:
        raise RuntimeError("Error, n_jobs>=1 required")

    ctx = multiprocessing.get_context('spawn')

    reconstructor_init_kwargs = dict(
        symmetry=symmetry, correct_ctf=correct_ctf, eps=eps, min_denominator_value=min_denominator_value,
    )
    reconstructor_run_kwargs = dict(
        particles_star_fname=particles_star_fname,
        particles_dir=particles_dir,
        batch_size=batch_size,
        num_dataworkers=num_dataworkers,
        use_only_n_first_batches=use_only_n_first_batches
    )

    if use_cuda:
        n_cuda_devices = torch.cuda.device_count()
        print(f"Using {n_cuda_devices} CUDA devices")
    else:
        n_cuda_devices = None

    # Initialize reconstructor to get shapes
    reconstructor = Reconstructor(**reconstructor_init_kwargs)
    particles = reconstructor._get_reconstructionParticlesDataset(
        reconstructor_run_kwargs["particles_star_fname"],
        reconstructor_run_kwargs["particles_dir"]
    ).particles

    # Get shapes of result tensors
    numerator_shape = reconstructor.numerator.shape
    weights_shape = reconstructor.weights.shape
    ctfsq_shape = reconstructor.ctfsq.shape

    # Create shared memory tensors for results
    shared_numerator, _ = create_shared_tensor(numerator_shape, ctx=ctx)
    shared_weights, _ = create_shared_tensor(weights_shape, ctx=ctx)
    shared_ctfsq, _ = create_shared_tensor(ctfsq_shape, ctx=ctx)

    n_particles = len(particles)

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
                      reconstructor_init_kwargs, reconstructor_run_kwargs,
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

    reconstructor.generate_volume(output_fname)
    print(f"Volume saved at {output_fname}")



def main():
    from argParseFromDoc import parse_function_and_call
    parse_function_and_call(reconstruct_starfile)


if __name__ == "__main__":
    main()
    """
python -m cryoPARES.reconstruction.reconstruct  --symmetry C1 --particles_star_fname /home/sanchezg/cryo/data/preAlignedParticles/EMPIAR-10166/data/allparticles.star --output_fname /tmp/reconstruction.mrc --use_only_n_first_batches 100    
    """