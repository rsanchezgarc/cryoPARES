"""
System utility functions for cryoPARES.

This module provides utility functions for system-level operations like
managing resource limits.
"""

import os
import resource
import warnings


def increase_file_descriptor_limit():
    """
    Attempt to maximize the file descriptor limit to avoid 'too many open files' errors.

    This sets the soft limit to match the hard limit (maximum allowed by the system).
    This is better than using a fixed value like 65536 since different systems have
    different hard limits.

    CryoPARES opens file handlers for each .mrcs file in RELION .star files, which can
    quickly exceed default system limits. This function automatically increases the limit
    to the maximum allowed, eliminating the need for users to manually run 'ulimit -n'
    before training or inference.

    Raises a warning if the final limit is very small (<=1024), which may cause issues
    when working with many .mrcs files.

    Returns:
        int: The actual limit that was set

    Example:
        >>> from cryoPARES.utils.systemUtils import increase_file_descriptor_limit
        >>> limit = increase_file_descriptor_limit()
        File descriptor limit increased from 1024 to 65536
    """
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)

        # If already at maximum, no need to change
        if soft >= hard:
            final_limit = soft
        else:
            # Set soft limit to hard limit (maximum allowed)
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
            print(f"File descriptor limit increased from {soft} to {hard}")
            final_limit = hard

    except (ValueError, OSError) as e:
        # Permission denied or other error - continue with current limit
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        warnings.warn(
            f"Could not increase file descriptor limit to maximum ({hard}): {e}. "
            f"Current limit: {soft}. You may need to increase system limits or run "
            f"'ulimit -n {hard}' manually if you encounter 'too many open files' errors."
        )
        final_limit = soft

    # Warn if the final limit is very small
    if final_limit <= 1024:
        warnings.warn(
            f"File descriptor limit is very low ({final_limit}). This may cause "
            f"'too many open files' errors when working with many .mrcs files. "
            f"Consider increasing system limits (e.g., edit /etc/security/limits.conf "
            f"or /etc/sysctl.conf) and restarting your session."
        )

    return final_limit


def setup_torch_env(matmul_precision: str = "high") -> None:
    """Set compile-cache dir and matmul precision; call early in every main()."""
    import tempfile
    import torch
    from pathlib import Path
    from cryoPARES.configs.mainConfig import main_config

    preferred = Path(main_config.cachedir) / "torch_inductor"
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        test = preferred / ".write_test"
        test.touch()
        test.unlink()
        inductor_cache = preferred
    except OSError as e:
        tmp = tempfile.mkdtemp(prefix="cryoPARES_inductor_")
        warnings.warn(
            f"Cannot write to inductor cache {preferred} ({e}). "
            f"Using temporary directory {tmp} instead."
        )
        inductor_cache = Path(tmp)

    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(inductor_cache))

    try:
        import torch._inductor.config as _inductor_cfg
        _inductor_cfg.fx_graph_cache = True
    except Exception:
        pass

    torch.set_float32_matmul_precision(matmul_precision)
    _auto_configure_nccl()


def _auto_configure_nccl() -> None:
    """Probe whether NCCL P2P works and disable it if it hangs.

    Runs only when: 2+ GPUs are present, NCCL_P2P_DISABLE is not already set,
    and CUDA reports peer access as possible between at least one GPU pair.
    The probe spawns a minimal 2-process NCCL all_reduce with a timeout; if it
    hangs, NCCL_P2P_DISABLE=1 is set before Lightning spawns DDP workers.
    The result is cached in the cryoPARES cache dir so the test only runs once
    per unique GPU+NCCL configuration.
    """
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return
    if 'NCCL_P2P_DISABLE' in os.environ:
        return

    # Only probe when CUDA itself reports P2P as possible between any pair;
    # if it's definitively unavailable NCCL will use a different transport anyway.
    n = torch.cuda.device_count()
    p2p_possible = any(torch.cuda.can_device_access_peer(i, j)
                       for i in range(n) for j in range(n) if i != j)
    if not p2p_possible:
        return

    # Cache key: GPU names + NCCL version (re-probes after driver/NCCL upgrades)
    nccl_ver = ".".join(str(x) for x in torch.cuda.nccl.version())
    gpu_names = "_".join(torch.cuda.get_device_name(i).replace(" ", "-")
                         for i in range(n))
    cache_key = f"nccl_p2p_{gpu_names}_{nccl_ver}"

    from cryoPARES.configs.mainConfig import main_config
    cache_dir = os.path.join(main_config.cachedir, "nccl_probe")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, cache_key + ".txt")

    if os.path.exists(cache_file):
        if open(cache_file).read().strip() == "disabled":
            warnings.warn(
                "GPU direct (NCCL P2P) communication is not supported on this machine — "
                "multi-GPU training will use a slower inter-GPU transport. "
                f"Delete {cache_file} and unset NCCL_P2P_DISABLE to re-probe after a "
                "driver/NCCL upgrade."
            )
            os.environ['NCCL_P2P_DISABLE'] = '1'
        return

    print("Probing NCCL GPU direct (P2P) communication (runs once, result cached)...",
          flush=True)
    disable = _probe_nccl_p2p(timeout_secs=10)

    with open(cache_file, "w") as f:
        f.write("disabled" if disable else "ok")

    if disable:
        os.environ['NCCL_P2P_DISABLE'] = '1'
        warnings.warn(
            "GPU direct (NCCL P2P) communication is not supported on this machine — "
            "the probe timed out. Multi-GPU training will use a slower inter-GPU transport. "
            f"Result cached in {cache_file}; delete it and unset NCCL_P2P_DISABLE to "
            "re-probe after a driver/NCCL upgrade."
        )
    else:
        print("GPU direct (NCCL P2P) communication probe passed — P2P enabled.",
              flush=True)


def _probe_nccl_p2p(timeout_secs: int = 10) -> bool:
    """Spawn a minimal 2-rank NCCL all_reduce. Returns True if it hangs/fails."""
    import sys
    import subprocess
    import tempfile

    script = """
import torch, torch.distributed as dist, torch.multiprocessing as mp

def _worker(rank, world_size):
    dist.init_process_group('nccl', rank=rank, world_size=world_size,
                             init_method='tcp://127.0.0.1:29573')
    dist.all_reduce(torch.zeros(1).cuda(rank))
    dist.destroy_process_group()

if __name__ == '__main__':
    mp.spawn(_worker, args=(2,), nprocs=2, join=True)
    print('ok')
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        tmpfile = f.name

    try:
        proc = subprocess.run(
            [sys.executable, tmpfile],
            capture_output=True, text=True, timeout=timeout_secs,
        )
        return proc.returncode != 0 or 'ok' not in proc.stdout
    except subprocess.TimeoutExpired:
        return True  # hung → disable P2P
    finally:
        try:
            os.unlink(tmpfile)
        except OSError:
            pass