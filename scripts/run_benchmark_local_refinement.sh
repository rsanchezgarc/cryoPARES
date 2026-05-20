#!/usr/bin/env bash
# Benchmark: master_6_2 vs ilr_6_2 vs ilr_2stage vs ilr_so3interp on gdh_G2
#
# GPU 3 is reserved for training — do not use it.
#
# Branch isolation strategy:
#   pip uninstall cryopares removes the editable-install meta-path finder.
#   Both branches are run via "cd <srcdir> && python -m", so Python finds the
#   package through CWD (which is always in sys.path[0] for -m invocations).
#   ILR branch is checked out into /tmp/cryoPARES_ilr via git worktree.
#   On EXIT the editable install is restored from MASTER_SRC.
#
# ILR defaults: use_subpixel_shifts=True, zero_dc=True, use_so3_interpolation=True
# ilr_6_2 explicitly disables so3_interp to make it a distinct ablation point.
#
# Wave layout:
#   Wave 1 — master: master_6_2 (GPU 0)
#   Wave 2 — ILR:    ilr_6_2 (GPU 0) | ilr_2stage (GPU 1) | ilr_so3interp (GPU 2)
#
# Usage:
#   bash scripts/run_benchmark_local_refinement.sh           # 10K particles (default)
#   N_FIRST=100  bash scripts/run_benchmark_local_refinement.sh   # smoke test
#   N_FIRST=""   bash scripts/run_benchmark_local_refinement.sh   # full dataset

set -euo pipefail
ulimit -n 65536

# ── Paths ─────────────────────────────────────────────────────────────────────
PIP=/data/nagagpu05/not-backed-up/sanchezg/app/miniconda3/envs/cryopares/bin/pip
PYTHON=/data/nagagpu05/not-backed-up/sanchezg/app/miniconda3/envs/cryopares/bin/python
MASTER_SRC=/data/nagagpu05/not-backed-up/special-project/cryo/myProjects/cryoPARES
ILR_REPO=/data/nagagpu05/not-backed-up/sanchezg/cryo/myProjects/cryoPARES
ILR_TMP=/tmp/cryoPARES_ilr
ILR_OPT_TMP=/tmp/cryoPARES_ilr_opt

DATA_ROOT=/data/nagagpu05/not-backed-up/sanchezg/astex_data/data/preAlignedParticles
CKPT_ROOT=/data/nagagpu05/not-backed-up/sanchezg/local_storage/cryo/data/cryopares

OUTPUT_BASE=/data/nagagpu05/not-backed-up/sanchezg/cryo/benchmark_localref_$(date +%Y%m%d_%H%M%S)
N_FIRST=${N_FIRST:-10000}   # set N_FIRST="" for full dataset

# ── Dataset / checkpoint paths ────────────────────────────────────────────────
GDH_CKPT=$CKPT_ROOT/gdh/train/g1/200K_particles
GDH_STAR=$DATA_ROOT/gdh_G2/aligned_particles_float32.star
GDH_PDIR=$DATA_ROOT/gdh_G2
GDH_REF=$CKPT_ROOT/gdh/train/g1/200K_particles/half1/reconstructions/0.mrc

# ── Remove editable install so CWD-based imports work cleanly ─────────────────
echo ">>> Uninstalling cryopares editable install ..."
"$PIP" uninstall -y cryopares 2>/dev/null || true

# ── Checkout ILR branch into /tmp ─────────────────────────────────────────────
echo ">>> Setting up improve_local_refinement worktree at $ILR_TMP ..."
git -C "$ILR_REPO" fetch origin improve_local_refinement
if [[ -d "$ILR_TMP" ]]; then
    git -C "$ILR_REPO" worktree remove --force "$ILR_TMP" 2>/dev/null || rm -rf "$ILR_TMP"
fi
git -C "$ILR_REPO" worktree add "$ILR_TMP" origin/improve_local_refinement
echo ">>> ILR worktree ready ($(git -C "$ILR_TMP" log --oneline -1))"

echo ">>> Setting up ilr_speed_opt worktree at $ILR_OPT_TMP ..."
if [[ -d "$ILR_OPT_TMP" ]]; then
    git -C "$ILR_REPO" worktree remove --force "$ILR_OPT_TMP" 2>/dev/null || rm -rf "$ILR_OPT_TMP"
fi
git -C "$ILR_REPO" worktree add "$ILR_OPT_TMP" ilr_speed_opt
echo ">>> ILR-opt worktree ready ($(git -C "$ILR_OPT_TMP" log --oneline -1))"

# ── Restore editable install and remove worktrees on exit ─────────────────────
cleanup() {
    echo ">>> Restoring master editable install ..."
    "$PIP" install -q -e "$MASTER_SRC" 2>/dev/null || true
    echo ">>> Removing ILR worktrees ..."
    git -C "$ILR_REPO" worktree remove --force "$ILR_TMP" 2>/dev/null || true
    git -C "$ILR_REPO" worktree remove --force "$ILR_OPT_TMP" 2>/dev/null || true
}
trap cleanup EXIT

mkdir -p "$OUTPUT_BASE"
echo ">>> Output: $OUTPUT_BASE"
echo ">>> N_FIRST: ${N_FIRST:-'(all)'}"

# ── Verify branch isolation ────────────────────────────────────────────────────
echo ">>> master loads from: $( (cd "$MASTER_SRC" && "$PYTHON" -c "import cryoPARES; print(cryoPARES.__file__)") )"
echo ">>> ILR    loads from: $( (cd "$ILR_TMP"    && "$PYTHON" -c "import cryoPARES; print(cryoPARES.__file__)") )"

# ── run_one ───────────────────────────────────────────────────────────────────
# Usage: run_one <src> <label> <gpu_id> <ckpt> <star> <pdir> <ref> <sym> <batch_size> [key=val ...]
# batch_size: particles per batch.  8 = conservative (fits all GPUs); 32 = safe on 32 GB.
run_one() {
    local src=$1 label=$2 gpu=$3 ckpt=$4 star=$5 pdir=$6 ref=$7 sym=$8 batch_size=$9
    shift 9

    local out=$OUTPUT_BASE/$label
    mkdir -p "$out"
    echo ""
    echo "====== $label (GPU $gpu, bs=$batch_size, src=$(basename $src)) ======"

    local cfg_flags=()
    [[ $# -gt 0 ]] && cfg_flags=("--config" "$@")

    local nfirst_flags=()
    [[ -n "${N_FIRST:-}" ]] && nfirst_flags=("--n_first_particles" "$N_FIRST")

    local t0=$SECONDS
    local infer_ok=0
    (
        cd "$src"
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        CUDA_VISIBLE_DEVICES=$gpu \
        "$PYTHON" -m cryoPARES.inference.infer \
            --particles_star_fname  "$star"  \
            --checkpoint_dir        "$ckpt"  \
            --results_dir           "$out"   \
            --particles_dir         "$pdir"  \
            --reference_map         "$ref"   \
            --n_jobs 1               \
            --batch_size "$batch_size" \
            "${nfirst_flags[@]}"             \
            "${cfg_flags[@]}"
    ) 2>&1 | tee "$out/run.log" && infer_ok=1
    local elapsed=$(( SECONDS - t0 ))
    echo "$elapsed" > "$out/time.txt"

    if [[ $infer_ok -eq 0 ]]; then
        echo "ERROR: inference failed for $label (see run.log)" | tee -a "$out/run.log"
        return 0
    fi
    echo ">>> $label: ${elapsed}s"

    for pred in "$out"/*_half1.star "$out"/*_half2.star; do
        [[ -f "$pred" ]] || continue
        local half
        half=$(echo "$pred" | grep -oE 'half[12]')
        echo "--- compare_poses $half ---"
        (
            cd "$MASTER_SRC"
            "$PYTHON" -m cryoPARES.scripts.compare_poses \
                --starfile1 "$star"                        \
                --starfile2 "$pred"                        \
                --sym       "$sym"                         \
                --save_plots "$out/compare_poses_${half}"
        ) 2>&1 | tee "$out/compare_poses_${half}.log"
    done
}

# ── Config arrays ─────────────────────────────────────────────────────────────
MASTER_CFG=("projmatching.grid_distance_degs=6" "projmatching.grid_step_degs=2")

# Explicitly disable so3_interp (ILR default=True) to isolate other ILR changes
ILR_BASE_CFG=(
    "projmatching.grid_distance_degs=6" "projmatching.grid_step_degs=2"
    "projmatching.use_so3_interpolation=False"
)
ILR_2STAGE_CFG=(
    "projmatching.grid_distance_degs=6" "projmatching.grid_step_degs=2"
    "projmatching.use_two_stage_search=True"
    "projmatching.fine_grid_distance_degs=2.1"
    "projmatching.fine_grid_step_degs=0.7"
)
# so3_interp=True is already the ILR default; set explicitly for clarity
ILR_6_2_SO3I_CFG=(
    "projmatching.grid_distance_degs=6" "projmatching.grid_step_degs=2"
    "projmatching.use_so3_interpolation=True"
)
# Speed-optimised ILR with SO3 interpolation — same accuracy config as ilr_6_2_so3i
ILR_OPT_SO3I_CFG=(
    "projmatching.grid_distance_degs=6" "projmatching.grid_step_degs=2"
    "projmatching.use_so3_interpolation=True"
)
# G5+G9: zeros→empty + proj_sub_batch_size=64
ILR_OPT_SUB64_CFG=(
    "projmatching.grid_distance_degs=6" "projmatching.grid_step_degs=2"
    "projmatching.use_so3_interpolation=True"
    "projmatching.proj_sub_batch_size=64"
)
# G5+G9: zeros→empty + proj_sub_batch_size=128
ILR_OPT_SUB128_CFG=(
    "projmatching.grid_distance_degs=6" "projmatching.grid_step_degs=2"
    "projmatching.use_so3_interpolation=True"
    "projmatching.proj_sub_batch_size=128"
)
# All ILR features disabled — isolates always-active overhead (e.g. .contiguous() in correlate_dft_2d)
ILR_ALL_OFF_CFG=(
    "projmatching.grid_distance_degs=6" "projmatching.grid_step_degs=2"
    "projmatching.use_subpixel_shifts=False"
    "projmatching.zero_dc=False"
    "projmatching.use_so3_interpolation=False"
    "projmatching.use_two_stage_search=False"
    "projmatching.spectral_whitening=False"
)

# ── Wave 1: master (GPU 0) ────────────────────────────────────────────────────
echo ""
echo ">>> Wave 1: master_6_2 on GPU 0"
run_one "$MASTER_SRC" "master_6_2__gdh_G2" 0 \
    "$GDH_CKPT" "$GDH_STAR" "$GDH_PDIR" "$GDH_REF" D3 8 "${MASTER_CFG[@]}"
echo ">>> Wave 1 complete"

# ── Wave 2: ILR configs in parallel (GPUs 0,1,2) ─────────────────────────────
echo ""
echo ">>> Wave 2: ilr_6_2 | ilr_6_2_so3i | ilr_2stage on GPUs 0,1,2"
run_one "$ILR_TMP" "ilr_6_2__gdh_G2"      0 \
    "$GDH_CKPT" "$GDH_STAR" "$GDH_PDIR" "$GDH_REF" D3 8 "${ILR_BASE_CFG[@]}"      &
run_one "$ILR_TMP" "ilr_6_2_so3i__gdh_G2" 1 \
    "$GDH_CKPT" "$GDH_STAR" "$GDH_PDIR" "$GDH_REF" D3 8 "${ILR_6_2_SO3I_CFG[@]}" &
run_one "$ILR_TMP" "ilr_2stage__gdh_G2"   2 \
    "$GDH_CKPT" "$GDH_STAR" "$GDH_PDIR" "$GDH_REF" D3 8 "${ILR_2STAGE_CFG[@]}"   &
wait
echo ">>> Wave 2 complete"

# ── Wave 3: ILR all-features-off — overhead diagnostic (GPU 0) ───────────────
echo ""
echo ">>> Wave 3: ilr_all_off on GPU 0"
run_one "$ILR_TMP" "ilr_all_off__gdh_G2" 0 \
    "$GDH_CKPT" "$GDH_STAR" "$GDH_PDIR" "$GDH_REF" D3 8 "${ILR_ALL_OFF_CFG[@]}"
echo ">>> Wave 3 complete"

# ── Wave 4: ILR fully speed-optimised (DC+F2+G5+G10+G11+G12), batch_size=8 ──
# All speed-opt commits on ilr_speed_opt branch (see git log for details).
echo ""
echo ">>> Wave 4: ilr_speed_opt_so3i (bs=8) on GPU 0"
run_one "$ILR_OPT_TMP" "ilr_speed_opt_so3i__gdh_G2" 0 \
    "$GDH_CKPT" "$GDH_STAR" "$GDH_PDIR" "$GDH_REF" D3 8 "${ILR_OPT_SO3I_CFG[@]}"
echo ">>> Wave 4 complete"

# ── Wave 5: ILR fully speed-optimised (DC+F2+G5+G10+G11+G12), batch_size=32 ─
# Isolated hot-path: projection 10ms + correlation 28ms + CC max 4ms = 42ms/batch
# vs old ILR: projection 115ms + correlation 104ms = 219ms/batch (5.2× faster).
# Expected total: ~90-120s (vs 137s with previous ilr_speed_opt, 143s master).
echo ""
echo ">>> Wave 5: ilr_speed_opt_so3i (bs=32) on GPU 0"
run_one "$ILR_OPT_TMP" "ilr_speed_opt_so3i_bs32__gdh_G2" 0 \
    "$GDH_CKPT" "$GDH_STAR" "$GDH_PDIR" "$GDH_REF" D3 32 "${ILR_OPT_SO3I_CFG[@]}"
echo ">>> Wave 5 complete"

# ── Wave 6: G9 sub-batching ablation — (bs=32, GPUs 0 and 1) ─────────────────
# G9: candidate sub-batching to reduce peak projection buffer from 2.91 GB to
# 0.54 GB (sub=64) or 1.09 GB (sub=128).  After G10 (GEMM fix) the old 644 MB
# matmul intermediate is gone; peak is now only the projection buffer.
# Tests whether reducing HBM pressure improves in-context throughput.
echo ""
echo ">>> Wave 6: G9 sub-batching ablation (sub64 | sub128) in parallel on GPUs 0,1"
run_one "$ILR_OPT_TMP" "ilr_opt_sub64_bs32__gdh_G2" 0 \
    "$GDH_CKPT" "$GDH_STAR" "$GDH_PDIR" "$GDH_REF" D3 32 "${ILR_OPT_SUB64_CFG[@]}" &
run_one "$ILR_OPT_TMP" "ilr_opt_sub128_bs32__gdh_G2" 1 \
    "$GDH_CKPT" "$GDH_STAR" "$GDH_PDIR" "$GDH_REF" D3 32 "${ILR_OPT_SUB128_CFG[@]}" &
wait
echo ">>> Wave 6 complete"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=== BENCHMARK COMPLETE ==="
"$PYTHON" "$MASTER_SRC/scripts/summarize_benchmark.py" "$OUTPUT_BASE"
