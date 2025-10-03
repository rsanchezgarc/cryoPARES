import os
import sys
import tempfile
import traceback
from typing import Optional, Literal

import numpy as np
import starfile
import torch
from progressBarDistributed import SharedMemoryProgressBar, SharedMemoryProgressBarWorker
from torch import multiprocessing

from cryoPARES.configManager.inject_defaults import inject_defaults_from_config
from cryoPARES.configs.mainConfig import main_config
from cryoPARES.constants import (
    BATCH_ORI_IMAGE_NAME,
    BATCH_POSE_NAME,
    BATCH_ORI_CTF_NAME,
    RELION_ANGLES_NAMES,
    RELION_SHIFTS_NAMES,
    RELION_PRED_POSE_CONFIDENCE_NAME,
)
from cryoPARES.projmatching.projMatcher import ProjectionMatcher, align_star as single_job_align_star
from cryoPARES.projmatching.projmatchingUtils.loggers import getWorkerLogger
from cryoPARES.projmatching.projMatcher import get_eulers
import gc

_MATCHER = None


# --------------------------
# Shared memory utilities
# --------------------------
def create_shared_tensor(shape, dtype=torch.float32, ctx=None, lock=False):
    """
    Create a shared multiprocessing.Array for storing tensor data.

    Args:
        shape (tuple[int]): Desired tensor shape.
        dtype (torch.dtype): PyTorch dtype (supports float32, float64, int64).
        ctx: Multiprocessing context (default: torch.multiprocessing).
        lock (bool): If False, disables lock for performance (safe when workers
                     write disjoint slices).

    Returns:
        (shared_array, shape): A multiprocessing.Array and the shape tuple.
    """
    size = int(np.prod(shape))
    if dtype == torch.float32:
        typecode = 'f'  # float32
    elif dtype == torch.float64:
        typecode = 'd'  # float64
    elif dtype == torch.int64:
        typecode = 'q'  # int64 (portable)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    if ctx is None:
        ctx = multiprocessing
    shared_array = ctx.Array(typecode, size, lock=lock)
    return shared_array, shape


def _ctypes_base(shared_array):
    """Return the underlying ctypes object from a multiprocessing.Array."""
    return shared_array.get_obj() if hasattr(shared_array, "get_obj") else shared_array


def _tensor_from_shared(shared_array, shape, dtype=torch.float32):
    """
    Create a zero-copy Torch Tensor view from a shared multiprocessing.Array.

    Args:
        shared_array: multiprocessing.Array allocated with create_shared_tensor().
        shape (tuple[int]): Desired tensor shape.
        dtype (torch.dtype): Expected dtype.

    Returns:
        torch.Tensor: A CPU tensor mapped to the shared memory.
    """
    base = _ctypes_base(shared_array)
    np_arr = np.ctypeslib.as_array(base)
    t = torch.from_numpy(np_arr).view(*shape)

    # Check dtype consistency
    expected = {torch.float32: torch.float32,
                torch.float64: torch.float64,
                torch.int64: torch.int64}[dtype]
    if t.dtype != expected:
        raise TypeError(f"Shared tensor dtype mismatch: got {t.dtype}, expected {dtype}")
    return t


# --------------------------
# Worker
# --------------------------
def worker(worker_id, *args, **kwargs):
    """Wrapper to catch exceptions in workers and print tracebacks."""
    try:
        _worker(worker_id, *args, **kwargs)
    except Exception as e:
        print(f"[Worker {worker_id}] Exception occurred: {e}", file=sys.stderr)
        traceback.print_exc()
        raise e


def _worker(worker_id, pbar_fname, particles_idxs, matcher_init_kwargs,
            run_kwargs, shared_results, device=None):
    """
    Worker process function for projection matching.

    Args:
        worker_id (int): Worker index.
        pbar_fname (str): Shared progress bar handle.
        particles_idxs (list[int]): Subset of particle indices assigned to this worker.
        matcher_init_kwargs (dict): Args to initialize ProjectionMatcher.
        run_kwargs (dict): Args for dataset loading.
        shared_results (dict): Shared memory arrays for results.
        device (str or None): Torch device string ("cuda:X" or None).
    """
    global _MATCHER
    if _MATCHER is None:
        _MATCHER = ProjectionMatcher(**matcher_init_kwargs)
        if device is not None:
            _MATCHER.to(device)

    verbose = matcher_init_kwargs.get('verbose', False)
    mainLogger = getWorkerLogger(verbose)
    mainLogger.info(f"Worker {worker_id} started")

    # Shared tensors
    results_corr_matrix = _tensor_from_shared(shared_results["corr"][0],
                                              shared_results["corr"][1], torch.float32)
    results_shiftsAngs_matrix = _tensor_from_shared(shared_results["shifts"][0],
                                                    shared_results["shifts"][1], torch.float32)
    results_eulerDegs_matrix = _tensor_from_shared(shared_results["eulers"][0],
                                                   shared_results["eulers"][1], torch.float32)
    stats_corr_matrix = _tensor_from_shared(shared_results["stats"][0],
                                            shared_results["stats"][1], torch.float32)
    n_cpus = run_kwargs["n_cpus"]
    dataset = _MATCHER.preprocess_particles(
        particles=run_kwargs["particles_star_fname"],
        data_rootdir=run_kwargs["data_rootdir"],
        batch_size=run_kwargs["batch_size"],
        n_cpus=run_kwargs["n_cpus"],
        halfset=run_kwargs["halfset"],
        subset_idxs=list(particles_idxs)
    )

    using_cuda = device is not None and str(device).startswith("cuda")
    dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=run_kwargs["batch_size"],
        num_workers=n_cpus,
        shuffle=False,
        pin_memory=using_cuda,
        multiprocessing_context='spawn' if n_cpus>0 else None
    )

    with SharedMemoryProgressBarWorker(worker_id, pbar_fname) as pbar:
        pbar.set_total_steps(len(dataset))
        _partIdx = 0
        non_blocking = bool(using_cuda)
        for batch in dl:
            n_items = len(batch[BATCH_ORI_IMAGE_NAME])
            partIdx = particles_idxs[_partIdx:_partIdx + n_items]
            _partIdx += n_items

            rotmats = batch[BATCH_POSE_NAME][0].to(device, non_blocking=non_blocking) \
                if using_cuda else batch[BATCH_POSE_NAME][0]
            parts = batch[BATCH_ORI_IMAGE_NAME].to(device, non_blocking=non_blocking) \
                if using_cuda else batch[BATCH_ORI_IMAGE_NAME]
            ctfs = batch[BATCH_ORI_CTF_NAME].to(device, non_blocking=non_blocking) \
                if using_cuda else batch[BATCH_ORI_CTF_NAME]

            rotmats = rotmats.unsqueeze(1) #We expect K poses per particle
            maxCorrs, predRotMats, predShiftsAngsXY, comparedWeight = _MATCHER.forward(parts, ctfs, rotmats)

            results_corr_matrix[partIdx, :] = maxCorrs.detach().cpu()
            results_shiftsAngs_matrix[partIdx, :] = predShiftsAngsXY.detach().cpu()
            results_eulerDegs_matrix[partIdx, :] = get_eulers(predRotMats).detach().cpu()
            stats_corr_matrix[partIdx, :] = comparedWeight.detach().cpu()

            pbar.update(n_items)
    mainLogger.info(f"Worker {worker_id} finished")


# --------------------------
# Main entry
# --------------------------
@inject_defaults_from_config(main_config.projmatching, update_config_with_args=True)
def projmatching_starfile(
        reference_vol: str,
        particles_star_fname: str,
        out_fname: str,
        particles_dir: Optional[str],
        mask_radius_angs: Optional[float] = None,
        grid_distance_degs: float = 8.0,
        grid_step_degs: float = 2.0,
        return_top_k_poses: int = 1,
        filter_resolution_angst: Optional[float] = None,
        n_jobs: int = 1,
        num_dataworkers: int = 1,
        batch_size: int = 1024,
        use_cuda: bool = True,
        verbose: bool = True,
        float32_matmul_precision: Literal["highest", "high", "medium"] = "high",
        gpu_id: Optional[int] = None,
        n_first_particles: Optional[int] = None,
        correct_ctf: bool = True,
        halfmap_subset: Optional[Literal["1", "2"]] = None,
):
    """
    Aligns particles from a STAR file to a reference volume using projection matching.

    This function can run in a single-process mode or be distributed across multiple
    jobs for parallel processing.

    Args:
        reference_vol: Path to the reference volume file (.mrc).
        particles_star_fname: Input STAR file with particle metadata.
        out_fname: Output STAR file with aligned particle poses.
        particles_dir: Root directory for particle image paths.
        mask_radius_angs: Mask radius in Angstroms.
        grid_distance_degs: Angular search range (degrees).
        grid_step_degs: Angular step size (degrees).
        return_top_k_poses: Number of top poses to save per particle.
        filter_resolution_angst: Low-pass filter the reference before matching.
        n_jobs: Number of parallel jobs.
        num_dataworkers: Number of CPU workers per DataLoader.
        batch_size: Batch size per job.
        use_cuda: If True, use GPU(s).
        verbose: If True, log progress.
        float32_matmul_precision: Precision mode for matmul.
        gpu_id: Specific GPU ID (if any).
        n_first_particles: Limit processing to first N particles.
        correct_ctf: Apply CTF correction if True.
        halfmap_subset: Select subset '1' or '2' for half-map validation.
    """

    if out_fname is not None:
        assert not os.path.isfile(
            out_fname
        ), f"Error, the starFnameOut {out_fname} already exists"

    ctx = multiprocessing.get_context('spawn')

    if n_jobs == 1:
        single_job_align_star(
            reference_vol=reference_vol,
            particles_star_fname=particles_star_fname,
            out_fname=out_fname,
            particles_dir=particles_dir,
            mask_radius_angs=mask_radius_angs,
            grid_distance_degs=grid_distance_degs,
            grid_step_degs=grid_step_degs,
            return_top_k_poses=return_top_k_poses,
            filter_resolution_angst=filter_resolution_angst,
            num_dataworkers=num_dataworkers,
            batch_size=batch_size,
            use_cuda=use_cuda,
            verbose=verbose,
            float32_matmul_precision=float32_matmul_precision,
            gpu_id=gpu_id,
            n_first_particles=n_first_particles,
            correct_ctf=correct_ctf,
            halfmap_subset=halfmap_subset
        )
        return

    torch.set_float32_matmul_precision(float32_matmul_precision)

    mainLogger = getWorkerLogger(verbose)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Particle subsampling
        if n_first_particles is not None:
            star_data = starfile.read(particles_star_fname)
            particles_df = star_data["particles"][:n_first_particles]
            optics_df = star_data["optics"]
            star_in_limited = os.path.join(tmpdir, f"input_particles_{os.path.basename(particles_star_fname)}")
            starfile.write({"optics": optics_df, "particles": particles_df}, star_in_limited)
            particles_star_fname = star_in_limited


        matcher_init_kwargs = dict(
            reference_vol=reference_vol,
            grid_distance_degs=grid_distance_degs,
            grid_step_degs=grid_step_degs,
            top_k_poses_localref=return_top_k_poses,
            max_resolution_A=filter_resolution_angst,
            verbose=verbose,
            correct_ctf=correct_ctf,
            mask_radius_angs=mask_radius_angs
        )
        run_kwargs = dict(
            particles_star_fname=particles_star_fname,
            data_rootdir=particles_dir,
            batch_size=batch_size,
            n_cpus=num_dataworkers,
            halfset=halfmap_subset
        )

        # Init matcher to infer dataset sizes
        matcher = ProjectionMatcher(**matcher_init_kwargs)
        dataset = matcher.preprocess_particles(
            particles=particles_star_fname,
            data_rootdir=particles_dir,
            batch_size=batch_size,
            n_cpus=num_dataworkers,
            halfset=halfmap_subset
        )
        n_particles = len(dataset)
        particlesStar = dataset.datasets[0].particles.copy()
        try:
            confidence = torch.tensor(
                particlesStar.particles_md.loc[:, RELION_PRED_POSE_CONFIDENCE_NAME].values,
                dtype=torch.float32
            )
        except KeyError:
            confidence = torch.ones(len(particlesStar.particles_md), dtype=torch.float32)

        # Allocate shared arrays
        corr_shape = (n_particles, return_top_k_poses)
        shifts_shape = (n_particles, return_top_k_poses, 2)
        eulers_shape = (n_particles, return_top_k_poses, 3)
        stats_shape = (n_particles, return_top_k_poses)

        shared_corr, _ = create_shared_tensor(corr_shape, dtype=torch.float32, ctx=ctx, lock=False)
        shared_shifts, _ = create_shared_tensor(shifts_shape, dtype=torch.float32, ctx=ctx, lock=False)
        shared_eulers, _ = create_shared_tensor(eulers_shape, dtype=torch.float32, ctx=ctx, lock=False)
        shared_stats, _ = create_shared_tensor(stats_shape, dtype=torch.float32, ctx=ctx, lock=False)

        shared_results = {"corr": (shared_corr, corr_shape),
                          "shifts": (shared_shifts, shifts_shape),
                          "eulers": (shared_eulers, eulers_shape),
                          "stats": (shared_stats, stats_shape)}

        # CUDA handling
        n_cuda_devices = torch.cuda.device_count() if use_cuda else 0
        if use_cuda:
            mainLogger.info(f"Using {n_cuda_devices} CUDA devices")

        # Launch workers
        with SharedMemoryProgressBar(n_jobs) as pbar:
            pbar_fname = pbar.shm_name
            processes = []
            particle_splits = np.array_split(np.arange(n_particles), n_jobs)
            for worker_id, particles_idxs in enumerate(particle_splits):
                worker_device = f"cuda:{worker_id % n_cuda_devices}" if n_cuda_devices > 0 else None
                p = ctx.Process(
                    target=worker,
                    args=(worker_id, pbar_fname, particles_idxs, matcher_init_kwargs, run_kwargs,
                          shared_results, worker_device)
                )
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
                if p.exitcode != 0:
                    mainLogger.error(f"Worker {p.pid} exited with code {p.exitcode}. Terminating.")
                    for q in processes:
                        if q.is_alive():
                            q.terminate()
                    sys.exit(1)

        mainLogger.info("All workers finished. Consolidating results.")

        # Read results
        predEulerDegs = _tensor_from_shared(shared_eulers, eulers_shape, torch.float32)
        predShiftsAngs = _tensor_from_shared(shared_shifts, shifts_shape, torch.float32)
        prob_x_y = _tensor_from_shared(shared_stats, stats_shape, torch.float32)

        # Finalize STAR file
        finalParticlesStar = particlesStar
        particles_md = finalParticlesStar.particles_md
        for k in range(return_top_k_poses):
            suffix = "" if k == 0 else f"_top{k}"
            angles_names = [x + suffix for x in RELION_ANGLES_NAMES]
            shifts_names = [x + suffix for x in RELION_SHIFTS_NAMES]
            confide_name = RELION_PRED_POSE_CONFIDENCE_NAME + suffix
            for col in angles_names + shifts_names + [confide_name]:
                if col not in particles_md.columns:
                    particles_md[col] = 0.0

            particles_md[angles_names] = predEulerDegs[:, k, :].numpy()
            particles_md[shifts_names] = predShiftsAngs[:, k, :].numpy()
            particles_md[confide_name] = (confidence * prob_x_y[:, k]).numpy()

        if out_fname is not None:
            finalParticlesStar.save(starFname=out_fname)
            mainLogger.info(f"Particles were saved at {out_fname}")

        gc.collect()
        if use_cuda:
            torch.cuda.empty_cache()


def main():
    """Main entry point: parse CLI args from docstrings and run projmatching_starfile()."""
    from argParseFromDoc import parse_function_and_call
    parse_function_and_call(projmatching_starfile)


if __name__ == "__main__":
    main()
