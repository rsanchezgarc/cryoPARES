import os
import numpy as np
from torch import multiprocessing
import torch
from more_itertools import batched
from progressBarDistributed import SharedMemoryProgressBar, SharedMemoryProgressBarWorker

from cryoPARES.reconstruction.reconstruction import Reconstructor

_RECONSTRUCTOR = None


def worker(pbar_fname, worker_id, n_particles, n_workers, particles_idxs,
           reconstructor_init_kwargs, reconstructor_run_kwargs,
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


def create_shared_tensor(shape, dtype=torch.float32):
    """Create a shared memory tensor"""
    size = int(np.prod(shape))  # Convert numpy.int64 to Python int
    if dtype == torch.float32:
        typecode = 'f'
    elif dtype == torch.float64:
        typecode = 'd'
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    shared_array = multiprocessing.Array(typecode, size)
    return shared_array, shape


def main():
    n_jobs = 2
    device = "cuda"
    outname = "/tmp/reconstructed_vol.mrc"

    reconstructor_init_kwargs = dict(
        symmetry="C1", correct_ctf=True, eps=1e-3, min_denominator_value=1e-4,
    )
    reconstructor_run_kwargs = dict(
        particles_star_fname="/home/sanchezg/cryo/data/preAlignedParticles/EMPIAR-10166/data/projections/1000proj_with_ctf.star",
        # particles_star_fname="/home/sanchezg/cryo/data/preAlignedParticles/EMPIAR-10166/data/allparticles.star",
        particles_dir="/home/sanchezg/cryo/data/preAlignedParticles/EMPIAR-10166/data/",
        batch_size=40,
        num_dataworkers=2,
        use_only_n_first_batches=None
    )

    if str(device).startswith("cuda"):
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
    shared_numerator, _ = create_shared_tensor(numerator_shape)
    shared_weights, _ = create_shared_tensor(weights_shape)
    shared_ctfsq, _ = create_shared_tensor(ctfsq_shape)

    n_particles = len(particles)

    with SharedMemoryProgressBar(n_jobs) as pbar:
        pbar_fname = pbar.shm_name

        processes = []
        for worker_id, particles_idxs in enumerate(np.array_split(range(n_particles), n_jobs)):
            # Determine device for this worker
            worker_device = None
            if n_cuda_devices is not None:
                worker_device = f"cuda:{worker_id % n_cuda_devices}"

            p = multiprocessing.Process(
                target=worker,
                args=(pbar_fname, worker_id, n_particles, n_jobs, particles_idxs,
                      reconstructor_init_kwargs, reconstructor_run_kwargs,
                      shared_numerator, shared_weights, shared_ctfsq, worker_device)
            )
            p.start()
            processes.append(p)

        # Wait for all processes to complete
        for p in processes:
            p.join()

    print("Backprojection done. Reconstructing")

    # Convert shared memory back to tensors
    final_numerator = torch.frombuffer(shared_numerator.get_obj(), dtype=torch.float32).reshape(numerator_shape)
    final_weights = torch.frombuffer(shared_weights.get_obj(), dtype=torch.float32).reshape(weights_shape)
    final_ctfsq = torch.frombuffer(shared_ctfsq.get_obj(), dtype=torch.float32).reshape(ctfsq_shape)

    # set results
    reconstructor.numerator = final_numerator
    reconstructor.weights = final_weights
    reconstructor.ctfsq = final_ctfsq

    reconstructor.generate_volume(outname)
    print("Reconstruction done.")


if __name__ == "__main__":
    # Required for multiprocessing with CUDA
    multiprocessing.set_start_method('spawn', force=True)
    main()