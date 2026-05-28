"""
Measure peak GPU memory vs (batch_size × n_rotations) for projmatching forward().

Runs the actual ProjectionMatcher forward pass with synthetic particles and records
torch.cuda.max_memory_allocated() for each (n_rotations, batch_size) combination.
Fits peak_memory = baseline + slope × B×N and extrapolates safe B×N limits for
8 / 16 / 32 / 48 / 80 GB GPUs.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_batch_memory.py [--box 336] [--gpu_id 0]
"""
import argparse
import gc
import math
import sys
import tempfile

import mrcfile
import numpy as np
import torch


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_projmatcher(box: int, pixel_size: float, device: str, n_rotations_hint: int = 209):
    """Return a ProjectionMatcher loaded onto *device* with a random reference volume."""
    from cryoPARES.configs.mainConfig import main_config
    from cryoPARES.projmatching.projMatcher import ProjectionMatcher

    # Patch config so box/pixel-size match
    main_config.datamanager.particlesdataset.image_size_px_for_nnet = box
    main_config.datamanager.particlesdataset.sampling_rate_angs_for_nnet = pixel_size
    main_config.projmatching.use_fibo_grid = True
    main_config.projmatching.rotation_composition = "pre_multiply"
    main_config.projmatching.use_two_stage_search = False

    # Write a temporary MRC volume
    vol = np.random.randn(box, box, box).astype(np.float32) * 0.01
    with tempfile.NamedTemporaryFile(suffix=".mrc", delete=False) as f:
        vol_fname = f.name
    with mrcfile.new(vol_fname, overwrite=True) as mrc:
        mrc.set_data(vol)
        mrc.voxel_size = pixel_size

    pm = ProjectionMatcher(
        reference_vol=vol_fname,
        pixel_size=pixel_size,
        grid_distance_degs=6,
        grid_step_degs=2,      # doesn't matter — we'll override rotmats directly
        correct_ctf=True,
        verbose=False,
    )
    pm.eval()
    pm.to(device)
    import os; os.unlink(vol_fname)
    return pm


def make_inputs(pm, batch_size: int, device: str):
    """Create synthetic (imgs, ctfs, rotmats) for one forward() call."""
    H, W = pm.image_shape[-2], pm.image_shape[-1]
    imgs = torch.randn(batch_size, H, W, device=device, dtype=torch.float32)
    # CTF: real-valued (B, H, W//2+1) — flat unit CTF (no modulation)
    ctfs = torch.ones(batch_size, H, W // 2 + 1, device=device, dtype=torch.float32)
    rotmats = torch.eye(3, device=device, dtype=torch.float32) \
                   .unsqueeze(0).unsqueeze(0) \
                   .expand(batch_size, 1, -1, -1).contiguous()
    return imgs, ctfs, rotmats


def run_forward(pm, batch_size: int, device: str):
    """
    Run one forward() pass and return peak GPU memory in bytes.
    Returns None on OOM or Triton XBLOCK error.
    """
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    imgs, ctfs, rotmats = make_inputs(pm, batch_size, device)
    try:
        with torch.inference_mode():
            _ = pm(imgs, ctfs, rotmats)
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)
        del imgs, ctfs, rotmats, _
        torch.cuda.empty_cache()
        return peak
    except (torch.cuda.OutOfMemoryError, RuntimeError, AssertionError) as e:
        msg = str(e)
        if any(k in msg for k in ("out of memory", "XBLOCK", "CUDA", "memory")):
            del imgs, ctfs, rotmats
            torch.cuda.empty_cache()
            gc.collect()
            return None
        raise


def warmup(pm, device: str):
    """One small forward pass to trigger torch.compile and warm up the kernel cache."""
    torch.cuda.empty_cache()
    imgs, ctfs, rotmats = make_inputs(pm, 1, device)
    with torch.inference_mode():
        try:
            _ = pm(imgs, ctfs, rotmats)
        except Exception:
            pass
    del imgs, ctfs, rotmats
    torch.cuda.empty_cache()
    gc.collect()


# ---------------------------------------------------------------------------
# main sweep
# ---------------------------------------------------------------------------

def sweep(box: int, pixel_size: float, device: str):
    # Grid configs: (distance_deg, step_deg) → realistic n_rotations
    # We test a range of batch_sizes for each config to build memory curve
    grid_configs = [
        (4, 2),    # ~63 pts  (fibo 4/2)
        (6, 2),    # ~209 pts (fibo 6/2)
        (4, 1),    # ~488 pts (fibo 4/1)
        (6, 1),    # ~1638 pts (fibo 6/1)
    ]

    # batch_sizes to probe
    BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]

    all_rows = []   # (n_rots, bs, total, peak_gb, status)

    for dist, step in grid_configs:
        from cryoPARES.configs.mainConfig import main_config
        main_config.projmatching.use_fibo_grid = True
        main_config.projmatching.rotation_composition = "pre_multiply"
        main_config.projmatching.use_two_stage_search = False
        main_config.projmatching.grid_distance_degs = dist
        main_config.projmatching.grid_step_degs = step

        pm = make_projmatcher(box, pixel_size, device)
        n_rots = ProjectionMatcher._count_rotations(True, dist, step)
        print(f"\n=== fibo {dist}/{step} ({n_rots} pts) ===")

        warmup(pm, device)

        for bs in BATCH_SIZES:
            total = bs * n_rots
            peak = run_forward(pm, bs, device)
            if peak is None:
                status = "OOM"
                peak_gb = float("nan")
                print(f"  bs={bs:4d}  B×N={total:6d}  PEAK=   OOM")
            else:
                peak_gb = peak / 1e9
                status = "OK"
                print(f"  bs={bs:4d}  B×N={total:6d}  PEAK={peak_gb:.2f} GB")
            all_rows.append((n_rots, bs, total, peak_gb, status))

        del pm
        torch.cuda.empty_cache()
        gc.collect()

    return all_rows


def fit_and_extrapolate(rows, box: int):
    """
    Fit peak_mem = baseline + slope × (B×N) from OK rows.
    Extrapolate max_BN for target VRAM sizes.
    """
    import numpy as np

    ok = [(r[2], r[3]) for r in rows if r[4] == "OK" and not math.isnan(r[3])]
    if len(ok) < 3:
        print("Not enough OK data points for fitting.")
        return

    x = np.array([r[0] for r in ok], dtype=float)   # B×N
    y = np.array([r[1] for r in ok], dtype=float)   # peak GB

    # Linear fit: y = a + b*x
    A = np.column_stack([np.ones_like(x), x])
    b_vec, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    baseline_gb, slope_gb_per_bn = b_vec

    print(f"\n=== Linear fit: peak_GB = {baseline_gb:.2f} + {slope_gb_per_bn*1e6:.4f}e-6 × B×N ===")
    print(f"    (baseline: {baseline_gb:.2f} GB for volume + overhead)")
    print(f"    (slope:    {slope_gb_per_bn*1e9:.2f} bytes per (B×N))")
    print()

    # Per-projection cost in MB (each projection is H×W//2+1×2 float32 + correlation output)
    H = box
    bytes_per_proj = H * (H // 2 + 1) * 2 * 4   # rfft, two float32
    print(f"    (raw bytes per proj at box={H}: {bytes_per_proj/1e3:.1f} KB)")
    print(f"    (empirical overhead factor: {slope_gb_per_bn * 1e9 / bytes_per_proj:.1f}×)")
    print()

    target_vrams = [8, 16, 24, 32, 48, 64, 80]
    print(f"{'VRAM (GB)':>10} | {'max B×N':>10} | {'@ n=64 bs':>10} | {'@ n=209 bs':>11} | {'@ n=488 bs':>11} | {'@ n=1638 bs':>12}")
    print("-" * 75)
    for vram in target_vrams:
        avail = vram - baseline_gb
        if avail <= 0:
            max_bn = 0
        else:
            max_bn = int(avail / slope_gb_per_bn)
        bs_64   = max(0, max_bn // 64)
        bs_209  = max(0, max_bn // 209)
        bs_488  = max(0, max_bn // 488)
        bs_1638 = max(0, max_bn // 1638)
        print(f"{vram:>10} | {max_bn:>10} | {bs_64:>10} | {bs_209:>11} | {bs_488:>11} | {bs_1638:>12}")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--box", type=int, default=336, help="Reference volume box size (default: 336)")
    parser.add_argument("--pixel_size", type=float, default=1.27, help="Pixel size in Å (default: 1.27)")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU device ID (default: 0)")
    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}"

    # Import here so CLI args are parsed first
    from cryoPARES.projmatching.projMatcher import ProjectionMatcher

    props = torch.cuda.get_device_properties(args.gpu_id)
    total_vram_gb = props.total_memory / 1e9
    print(f"GPU {args.gpu_id}: {props.name}")
    print(f"Total VRAM: {total_vram_gb:.1f} GB")
    print(f"Box: {args.box}, pixel size: {args.pixel_size} Å")
    print()

    rows = sweep(args.box, args.pixel_size, device)

    fit_and_extrapolate(rows, args.box)

    # Raw CSV for spreadsheet
    print("\n=== RAW DATA (CSV) ===")
    print("n_rots,batch_size,B_times_N,peak_GB,status")
    for r in rows:
        print(f"{r[0]},{r[1]},{r[2]},{r[3]:.3f},{r[4]}")
