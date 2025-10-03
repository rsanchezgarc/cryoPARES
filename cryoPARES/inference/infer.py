import os
import sys
import traceback
import time
from typing import Optional, Literal, List, Dict, Any

import numpy as np
import pandas as pd
import torch
import starfile  # fast STAR I/O

from torch import multiprocessing
from progressBarDistributed import SharedMemoryProgressBar, SharedMemoryProgressBarWorker

from cryoPARES import constants
from cryoPARES.configManager.inject_defaults import CONFIG_PARAM, inject_defaults_from_config, inject_docs_from_config_params
from cryoPARES.configs.mainConfig import main_config
from cryoPARES.reconstruction.reconstruct import create_shared_tensor
from cryoPARES.inference.inferencer import SingleInferencer
from cryoPARES.utils.paths import get_most_recent_file
from cryoPARES.scripts.computeFsc import compute_fsc
from cryoPARES.utils.reconstructionUtils import get_vol
from cryoPARES.configManager.configParser import ConfigArgumentParser, ConfigOverrideSystem
from cryoPARES.utils.checkpointReader import CheckpointReader


# -----------------------------
# Orchestrator
# -----------------------------

@inject_docs_from_config_params
@inject_defaults_from_config(main_config.inference, update_config_with_args=True)
def distributed_inference(
        particles_star_fname: str,
        checkpoint_dir: str,
        results_dir: str,
        data_halfset: Literal["half1", "half2", "allParticles"] = "allParticles",
        model_halfset: Literal["half1", "half2", "allCombinations", "matchingHalf"] = "matchingHalf",
        particles_dir: Optional[str] = None,
        batch_size: int = CONFIG_PARAM(),
        n_jobs: Optional[int] = None,
        num_dataworkers: int = CONFIG_PARAM(config=main_config.datamanager),
        use_cuda: bool = CONFIG_PARAM(),
        n_cpus_if_no_cuda: int = CONFIG_PARAM(),
        compile_model: bool = False,
        top_k_poses_nnet: int = CONFIG_PARAM(),
        top_k_poses_localref: int = CONFIG_PARAM(config=main_config.projmatching),
        grid_distance_degs: float = CONFIG_PARAM(config=main_config.projmatching),
        reference_map: Optional[str] = None,
        reference_mask: Optional[str] = None,
        directional_zscore_thr: Optional[float] = CONFIG_PARAM(),
        skip_localrefinement: bool = CONFIG_PARAM(),
        skip_reconstruction: bool = CONFIG_PARAM(),
        subset_idxs: Optional[List[int]] = None,
        n_first_particles: Optional[int] = None,
        check_interval_secs: float = 2.0
):
    """
    Distributed inference across particles and devices, mirroring the **halfset selection logic**
    used by :class:`cryoPARES.inference.inference.SingleInferencer`.

    Parameters
    ----------
    particles_star_fname : str
        {particles_star_fname}
    checkpoint_dir : str
        {checkpoint_dir}
    results_dir : str
        {results_dir}
    data_halfset : {{"half1", "half2", "allParticles"}}, default "allParticles"
        {data_halfset}
    model_halfset : {{"half1", "half2", "allCombinations", "matchingHalf"}}, default "matchingHalf"
        {model_halfset}
    particles_dir : str, optional
        {particles_dir}
    batch_size : int
        {batch_size}
    n_jobs : int, optional
        {n_jobs}
    num_dataworkers : int
        {num_dataworkers}
    use_cuda : bool
        {use_cuda}
    n_cpus_if_no_cuda : int
        {n_cpus_if_no_cuda}
    compile_model : bool
        {compile_model}
    top_k_poses_nnet : int
        {top_k_poses_nnet}
    top_k_poses_localref : int
        {top_k_poses_localref}
    grid_distance_degs : float
        {grid_distance_degs}
    reference_map : str, optional
        {reference_map}
    reference_mask : str, optional
        {reference_mask}
    directional_zscore_thr : float, optional
        {directional_zscore_thr}
    skip_localrefinement : bool
        {skip_localrefinement}
    skip_reconstruction : bool
        {skip_reconstruction}
    subset_idxs : list[int], optional
        {subset_idxs}
    n_first_particles: int, optional
        {n_first_particles}
    check_interval_secs : float
        {check_interval_secs}

    Notes
    -----
    - For **n_jobs == 1**, this function delegates completely to `SingleInferencer`, which internally
      handles `data_halfset` and `model_halfset` (including "matchingHalf" and "allCombinations").
    - For **n_jobs > 1**, particles are split by half-set and distributed across workers; model half
      selection follows the resolved policy per half to avoid cross-half coupling.
    """

    torch.set_float32_matmul_precision(main_config.inference.float32_matmul_precision)

    os.makedirs(results_dir, exist_ok=True)

    # Determine default parallelism
    if use_cuda and torch.cuda.is_available():
        n_cuda_devices = torch.cuda.device_count()
        if n_jobs is None or n_jobs <= 0:
            n_jobs = n_cuda_devices if n_cuda_devices > 0 else 1
    else:
        n_cuda_devices = None
        if n_jobs is None or n_jobs <= 0:
            n_jobs = 1

    # -------------------------
    # FAST PATH: single process
    # -------------------------
    if n_jobs == 1:
        print("Single-process mode")
        inferencer = SingleInferencer(
            particles_star_fname=particles_star_fname,
            checkpoint_dir=checkpoint_dir,
            results_dir=results_dir,  # let SingleInferencer write outputs
            data_halfset=data_halfset,  # pass through as provided
            model_halfset=model_halfset,  # pass through as provided
            particles_dir=particles_dir,
            batch_size=batch_size,
            num_dataworkers=num_dataworkers,
            use_cuda=use_cuda,
            n_cpus_if_no_cuda=n_cpus_if_no_cuda,
            compile_model=compile_model,
            top_k_poses_nnet=top_k_poses_nnet,
            top_k_poses_localref=top_k_poses_localref,
            grid_distance_degs=grid_distance_degs,
            reference_map=reference_map,
            reference_mask=reference_mask,
            directional_zscore_thr=directional_zscore_thr,
            skip_localrefinement=skip_localrefinement,
            skip_reconstruction=skip_reconstruction,
            subset_idxs=subset_idxs,  # interpreted inside per half
            n_first_particles=n_first_particles,
            show_debug_stats=False,
        )
        out = inferencer.run()
        # Return whatever SingleInferencer returned; files are already written to results_dir
        return out

    # -------------------------
    # MULTI-PROCESS PATH
    # -------------------------
    print(f"Multiprocess mode: spawning {n_jobs} worker(s).")
    ctx = multiprocessing.get_context('spawn')
    aggregated_results: Dict[str, Optional[pd.DataFrame]] = {}

    # Track reconstructions per model-half to compute FSC
    fsc_by_model: Dict[str, dict] = {}

    # Build lists mirroring SingleInferencer.run() logic
    if model_halfset == "allCombinations":
        model_halfset_list: List[Optional[str]] = ["half1", "half2"]
    elif model_halfset == "matchingHalf":
        model_halfset_list = [None]
    else:
        model_halfset_list = [model_halfset]

    if data_halfset == "allParticles":
        data_halfset_list = ["half1", "half2"]
    else:
        data_halfset_list = [data_halfset]

    checkpoint_reader = CheckpointReader(checkpoint_dir)
    for m_half in model_halfset_list:
        fsc_by_model = {m_half:{"half1": None, "half2": None, "sampling_rate": None}}
        for d_half in data_halfset_list:
            resolved_model_halfset = d_half if m_half is None else m_half
            print(f"\n=== Running data {d_half} with model {resolved_model_halfset} ===")

            # Determine particle rows for this half via starfile
            all_indices = _load_particles_indices_with_halfset(
                particles_star_fname, d_half, subset_idxs=subset_idxs
            )
            n_particles_total = len(all_indices)
            print(f"Total particles to process for data {d_half}: {n_particles_total}")

            # Shared reconstructor buffers (optional)
            shared_numerator = shared_weights = shared_ctfsq = None
            numerator_shape = weights_shape = ctfsq_shape = None
            reconstructor_parent = None

            if not skip_reconstruction:
                symmetry = SingleInferencer._get_symmetry(checkpoint_reader, resolved_model_halfset)
                reconstructor_parent = SingleInferencer._get_reconstructor(particles_star_fname,
                                                                           particles_dir,
                                                                           symmetry=symmetry)
                numerator_shape = reconstructor_parent.numerator.shape
                weights_shape = reconstructor_parent.weights.shape
                ctfsq_shape = reconstructor_parent.ctfsq.shape

                shared_numerator, _ = create_shared_tensor(numerator_shape, ctx=ctx)
                shared_weights, _ = create_shared_tensor(weights_shape, ctx=ctx)
                shared_ctfsq, _ = create_shared_tensor(ctfsq_shape, ctx=ctx)

            with ctx.Manager() as manager, SharedMemoryProgressBar(n_jobs) as pbar:
                output_q = manager.Queue()
                pbar_fname = pbar.shm_name

                split_indices = np.array_split(all_indices, n_jobs)

                processes = []
                for worker_id, part_idxs in enumerate(split_indices):
                    worker_device = None
                    if torch.cuda.is_available():
                        n_cuda = torch.cuda.device_count()
                        if n_cuda > 0:
                            worker_device = f"cuda:{worker_id % n_cuda}"
                    if n_first_particles:
                        part_idxs = part_idxs[:n_first_particles]
                    inferencer_init_kwargs = dict(
                        particles_star_fname=particles_star_fname,
                        checkpoint_dir=checkpoint_dir,
                        results_dir=None,
                        data_halfset=d_half,
                        model_halfset=resolved_model_halfset,
                        particles_dir=particles_dir,
                        batch_size=batch_size,
                        num_dataworkers=num_dataworkers,
                        use_cuda=use_cuda,
                        n_cpus_if_no_cuda=n_cpus_if_no_cuda if not use_cuda else n_cpus_if_no_cuda,
                        compile_model=compile_model,
                        top_k_poses_nnet=top_k_poses_nnet,
                        top_k_poses_localref=top_k_poses_localref,
                        reference_map=reference_map,
                        reference_mask=reference_mask,
                        directional_zscore_thr=directional_zscore_thr,
                        skip_localrefinement=skip_localrefinement,
                        skip_reconstruction=skip_reconstruction,
                        subset_idxs=list(map(int, part_idxs)),
                        show_debug_stats=False,
                    )

                    p = ctx.Process(
                        target=worker,
                        args=(worker_id, output_q, pbar_fname,
                              inferencer_init_kwargs, main_config,
                              shared_numerator, shared_weights, shared_ctfsq, worker_device)
                    )
                    p.start()
                    processes.append(p)

                results: Dict[int, Any] = {}
                try:
                    while True:
                        while not output_q.empty():
                            worker_id, result = output_q.get_nowait()
                            results[worker_id] = result
                            print(f"Received results from worker {worker_id}")

                        all_done = True
                        for p in processes:
                            if not p.is_alive() and p.exitcode not in (0, None):
                                print(f"Worker {p.pid} died with exit code {p.exitcode}!", file=sys.stderr)
                                for q in processes:
                                    if q.is_alive():
                                        print(f"Terminating worker {q.pid}...")
                                        q.terminate()
                                sys.exit(1)

                            if p.is_alive():
                                all_done = False

                        if all_done:
                            while not output_q.empty():
                                worker_id, result = output_q.get_nowait()
                                results[worker_id] = result
                                print(f"Received final results from worker {worker_id}")
                            break

                        time.sleep(check_interval_secs)

                    for p in processes:
                        p.join()

                except KeyboardInterrupt:
                    print("Interrupted by user. Terminating workers...")
                    for p in processes:
                        if p.is_alive():
                            p.terminate()
                    sys.exit(1)
                finally:
                    for p in processes:
                        p.join()
            key = f"model_{resolved_model_halfset}__data_{d_half}"
            if results:
                print(f"Aggregating results from {len(results)} workers for {key}...")
                aggregated_particles, aggregated_optics = _aggregate_worker_results(results)
                aggregated_results[key] = aggregated_particles
                if aggregated_particles is not None and results_dir is not None:
                    basename = os.path.basename(particles_star_fname).removesuffix(".star")
                    out_star = os.path.join(results_dir, basename + f"_{d_half}.star")
                    star_payload = {"particles": aggregated_particles}
                    if aggregated_optics is not None:
                        star_payload["optics"] = aggregated_optics
                    starfile.write(star_payload, out_star, overwrite=True)
                    print(f"Saved aggregated STAR: {out_star}")
            else:
                aggregated_results[key] = None
            if not skip_reconstruction and reconstructor_parent is not None:
                print("Backprojection done. Reconstructing...")
                final_numerator = torch.frombuffer(shared_numerator.get_obj(), dtype=torch.float32).reshape(numerator_shape)
                final_weights = torch.frombuffer(shared_weights.get_obj(), dtype=torch.float32).reshape(weights_shape)
                final_ctfsq = torch.frombuffer(shared_ctfsq.get_obj(), dtype=torch.float32).reshape(ctfsq_shape)

                reconstructor_parent.numerator = final_numerator
                reconstructor_parent.weights = final_weights
                reconstructor_parent.ctfsq = final_ctfsq

                out_mrc = os.path.join(results_dir, f"reconstruction_{d_half}.mrc")
                reconstructor_parent.generate_volume(out_mrc)
                print(f"Reconstruction saved for {d_half}: {out_mrc}")
                fsc_by_model[m_half][d_half] = out_mrc
                try:
                    fsc_by_model[m_half]["sampling_rate"] = getattr(reconstructor_parent, "sampling_rate",
                                                                    fsc_by_model[m_half]["sampling_rate"])
                except Exception:
                    pass

        # Compute FSC when reconstructions for both halves exist
        try:
            for model_key, rec in fsc_by_model.items():
                if rec.get("half1") and rec.get("half2") and rec.get("sampling_rate"):
                    print(f"Computing FSC, " "" if model_key is None else f" for {model_key}...")
                    vol1 = get_vol(rec["half1"], pixel_size=None)[0]
                    vol2 = get_vol(rec["half2"], pixel_size=None)[0]
                    mask_arr = None
                    if reference_mask is not None:
                        mask_arr = get_vol(reference_mask, pixel_size=None)[0]
                    fsc, spatial_freq, resolution_A, (res_05, res_0143) = compute_fsc(
                        vol1, vol2, rec["sampling_rate"], mask=mask_arr
                    )
                    print(f"[FSC] Resolution at 0.143: {res_0143:.3f} Å\n"
                          f"                   at 0.5: {res_05:.3f} Å")
        except Exception as e:
            print(f"FSC computation failed: {e}")
        return aggregated_results


# -----------------------------
# Helpers
# -----------------------------

def _device_index_from_str(device: Optional[str]) -> Optional[int]:
    if device is None:
        return None
    if isinstance(device, int):
        return device
    try:
        return int(str(device).split(":")[-1])
    except Exception:
        return None


def _flatten_inference_output(out: Any) -> List[tuple[pd.DataFrame, pd.DataFrame | None]]:
    """ Flatten SingleInferencer.run() output into a list of (particles_df, optics_df) pairs.

    Expected nested structure from SingleInferencer.run():
      list_over_model_halfset [
        list_over_data_half [
          (particles_md_list, optics_md_list, vol)
        ]
      ]

    We pair the items from particles_md_list and optics_md_list by index.
    """
    pairs: List[tuple[pd.DataFrame, pd.DataFrame | None]] = []
    if out is None:
        return pairs

    def _add_pairs(p_list, o_list):
        if isinstance(p_list, list):
            # optics list may be None or list-like of same length
            if isinstance(o_list, list):
                for p_df, o_df in zip(p_list, o_list):
                    if isinstance(p_df, pd.DataFrame):
                        pairs.append((p_df, o_df if isinstance(o_df, pd.DataFrame) else None))
            else:
                for p_df in p_list:
                    if isinstance(p_df, pd.DataFrame):
                        pairs.append((p_df, None))

    if not isinstance(out, list):
        maybe = out
        if isinstance(maybe, tuple) and len(maybe) >= 2:
            _add_pairs(maybe[0], maybe[1])
        return pairs

    for out_list in out:
        if isinstance(out_list, list):
            for elem in out_list:
                if isinstance(elem, tuple) and len(elem) >= 2:
                    _add_pairs(elem[0], elem[1])
    return pairs

    if not isinstance(out, list):
        maybe = out
        if isinstance(maybe, tuple) and len(maybe) >= 1 and isinstance(maybe[0], list):
            for df in maybe[0]:
                if isinstance(df, pd.DataFrame):
                    dfs.append(df)
        return dfs

    for out_list in out:
        if isinstance(out_list, list):
            for elem in out_list:
                if isinstance(elem, tuple) and len(elem) >= 1:
                    particles_md_list = elem[0]
                    if isinstance(particles_md_list, list):
                        for df in particles_md_list:
                            if isinstance(df, pd.DataFrame):
                                dfs.append(df)
    return dfs


def _aggregate_worker_results(results: Dict[int, Any]) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """ Aggregate results from multiple workers into (particles_df, optics_df)."""
    if not results:
        return None, None
    part_dfs: List[pd.DataFrame] = []
    opt_dfs: List[pd.DataFrame] = []
    for worker_id in sorted(results.keys()):
        out = results[worker_id]
        for p_df, o_df in _flatten_inference_output(out):
            if isinstance(p_df, pd.DataFrame):
                part_dfs.append(p_df)
            if isinstance(o_df, pd.DataFrame):
                opt_dfs.append(o_df)

    agg_particles = pd.concat(part_dfs, axis=0) if part_dfs else None
    # Optics tables are typically identical; we de-duplicate rows if multiple are present.
    if opt_dfs:
        try:
            agg_optics = pd.concat(opt_dfs, axis=0).drop_duplicates().reset_index(drop=True)
        except Exception:
            agg_optics = opt_dfs[0]
    else:
        agg_optics = None
    return agg_particles, agg_optics


def _load_particles_indices_with_halfset(
        star_path: str,
        data_half: Literal["half1", "half2"],
        subset_idxs: Optional[List[int]] = None
) -> np.ndarray:
    """Read STAR and return the *row indices* to process for the requested data half.

    If rlnRandomSubset exists (1/2), filter accordingly; otherwise process all.
    subset_idxs, if provided, is applied *after* filtering by half.
    """
    star = starfile.read(star_path)
    if isinstance(star, dict):
        if "particles" in star:
            df = star["particles"]
        else:
            df = next(iter(star.values()))
    else:
        df = star

    idx = np.arange(len(df))
    if "rlnRandomSubset" in df.columns:
        mask = (df["rlnRandomSubset"].astype(int) == (1 if data_half == "half1" else 2))
        idx = df.index[mask].to_numpy()

    if subset_idxs is not None:
        subset_idxs = np.array(list(map(int, subset_idxs)), dtype=np.int64)
        valid = (subset_idxs >= 0) & (subset_idxs < len(idx))
        idx = idx[subset_idxs[valid]]

    return idx


# -----------------------------
# Worker
# -----------------------------

def worker(worker_id, output_q, *args, **kwargs):
    try:
        _worker(worker_id, output_q, *args, **kwargs)
    except Exception as e:
        print(f"[Worker {worker_id}] Exception occurred: {e}", file=sys.stderr)
        traceback.print_exc()
        raise e

_INFERENCER = None
def _worker(worker_id,
            output_q,
            pbar_fname,
            inferencer_init_kwargs,
            main_config_updated,
            shared_numerator,
            shared_weights,
            shared_ctfsq,
            device=None):
    global _INFERENCER
    if device is not None and torch.cuda.is_available():
        dev_index = _device_index_from_str(device)
        if dev_index is not None:
            torch.cuda.set_device(dev_index)

    ConfigOverrideSystem.update_config_from_dataclass(main_config, main_config_updated, verbose=False)

    if _INFERENCER is None:
        inferencer = SingleInferencer(**inferencer_init_kwargs)
        _INFERENCER = inferencer
    else:
        inferencer = _INFERENCER

    with SharedMemoryProgressBarWorker(worker_id, pbar_fname) as pbar:
        inferencer._pbar = pbar
        particles_md, optics_md = inferencer._run(materialize_reconstruction=False)[:2]  #TODO: We don't need to reinitialize the model to run severl  _run(), use a global var
        output_q.put((worker_id, (particles_md, optics_md)))

        # Accumulate reconstructor buffers into shared memory if present
        reconstructor = getattr(inferencer, "_reconstructor", None)
        if reconstructor is None:
            reconstructor = getattr(getattr(inferencer, "_model", None), "reconstructor", None)

        if reconstructor is not None and all(x is not None for x in (shared_numerator, shared_weights, shared_ctfsq)):
            with shared_numerator.get_lock():
                shared_num_t = torch.frombuffer(shared_numerator.get_obj(), dtype=torch.float32).reshape(
                    reconstructor.numerator.shape)
                shared_num_t += reconstructor.numerator.detach().cpu()

            with shared_weights.get_lock():
                shared_w_t = torch.frombuffer(shared_weights.get_obj(), dtype=torch.float32).reshape(
                    reconstructor.weights.shape)
                shared_w_t += reconstructor.weights.detach().cpu()

            with shared_ctfsq.get_lock():
                shared_c_t = torch.frombuffer(shared_ctfsq.get_obj(), dtype=torch.float32).reshape(
                    reconstructor.ctfsq.shape)
                shared_c_t += reconstructor.ctfsq.detach().cpu()
            inferencer.clean_reconstructer_buffers()


def main():
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    os.environ[constants.SCRIPT_ENTRY_POINT] = 'infer.py'
    print('---------------------------------------')
    print(' '.join(sys.argv))
    print('---------------------------------------')

    parser = ConfigArgumentParser(prog='distrib_infer_cryoPARES',
                                  description='Run inference with cryoPARES model (distributed)',
                                  config_obj=main_config)
    parser.add_args_from_function(distributed_inference)
    args, config_args = parser.parse_args()

    # Support both directory and ZIP checkpoints
    if args.checkpoint_dir.endswith('.zip'):
        assert os.path.isfile(args.checkpoint_dir), f"Error, checkpoint_dir {args.checkpoint_dir} not found"
    else:
        assert os.path.isdir(args.checkpoint_dir), f"Error, checkpoint_dir {args.checkpoint_dir} not found"

    # Load config from checkpoint
    with CheckpointReader(args.checkpoint_dir) as reader:
        config_files = reader.glob('configs_*.yml')
        if not config_files:
            raise FileNotFoundError(f"No configs_*.yml found in {args.checkpoint_dir}")
        # Get most recent config file
        config_fname_rel = sorted(config_files)[-1]
        config_text = reader.read_text(config_fname_rel)

    # Update config from file content
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        f.write(config_text)
        temp_config_path = f.name

    try:
        ConfigOverrideSystem.update_config_from_file(main_config, temp_config_path, drop_paths=['inference', 'projmatching'])
    finally:
        os.unlink(temp_config_path)

    ConfigOverrideSystem.update_config_from_configstrings(main_config, config_args, verbose=True)

    distributed_inference(**vars(args))


if __name__ == '__main__':
    main()

    """

python -m cryoPARES.inference.infer \
--particles_star_fname /home/sanchezg/cryo/data/preAlignedParticles/EMPIAR-10166/data/1000particles.star  --results_dir /tmp/cryoPARES_train/cryoPARES_inference/ --particles_dir /home/sanchezg/cryo/data/preAlignedParticles/EMPIAR-10166/data --checkpoint_dir /tmp/cryoPARES_train/version_0/ --NOT_use_cuda --config inference.before_refiner_buffer_size=4 --batch_size 8 

    """