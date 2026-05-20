#!/usr/bin/env bash
# Quick validation of ilr_speed_opt branch (G10/G11/G12 commits) against documented baselines.
#
# Runs only the speed-optimised waves (no master/ILR baselines — use previous run's data).
# Wave 1: ilr_speed_opt_so3i  bs=8  (GPU 0)
# Wave 2: ilr_speed_opt_so3i  bs=32 (GPU 0)
# Wave 3: sub-batching ablation: sub64 (GPU 0) | sub128 (GPU 1)  [parallel]
#
# Usage:
#   bash scripts/run_speedopt_validation.sh           # 10K particles (default)
#   N_FIRST=100  bash scripts/run_speedopt_validation.sh   # smoke test

set -euo pipefail
ulimit -n 65536

PIP=/data/nagagpu05/not-backed-up/sanchezg/app/miniconda3/envs/cryopares/bin/pip
PYTHON=/data/nagagpu05/not-backed-up/sanchezg/app/miniconda3/envs/cryopares/bin/python
MASTER_SRC=/data/nagagpu05/not-backed-up/special-project/cryo/myProjects/cryoPARES
ILR_REPO=/data/nagagpu05/not-backed-up/sanchezg/cryo/myProjects/cryoPARES
OPT_TMP=/tmp/cryoPARES_speedopt_val

DATA_ROOT=/data/nagagpu05/not-backed-up/sanchezg/astex_data/data/preAlignedParticles
CKPT_ROOT=/data/nagagpu05/not-backed-up/sanchezg/local_storage/cryo/data/cryopares

OUTPUT_BASE=/data/nagagpu05/not-backed-up/sanchezg/cryo/benchmark_speedopt_$(date +%Y%m%d_%H%M%S)
N_FIRST=${N_FIRST:-10000}

GDH_CKPT=$CKPT_ROOT/gdh/train/g1/200K_particles
GDH_STAR=$DATA_ROOT/gdh_G2/aligned_particles_float32.star
GDH_PDIR=$DATA_ROOT/gdh_G2
GDH_REF=$CKPT_ROOT/gdh/train/g1/200K_particles/half1/reconstructions/0.mrc

echo ">>> Current ilr_speed_opt HEAD: $(git -C "$ILR_REPO" log --oneline -1 ilr_speed_opt)"

# ilr_speed_opt is checked out in MASTER_SRC (main worktree) — run directly from there.
# Verify we're on the right branch/commit.
ACTUAL_BRANCH=$(git -C "$MASTER_SRC" branch --show-current)
ACTUAL_HEAD=$(git -C "$MASTER_SRC" log --oneline -1)
if [[ "$ACTUAL_BRANCH" != "ilr_speed_opt" ]]; then
    echo "ERROR: MASTER_SRC is on branch '$ACTUAL_BRANCH', expected 'ilr_speed_opt'" >&2
    exit 1
fi
echo ">>> Running from MASTER_SRC on $ACTUAL_BRANCH ($ACTUAL_HEAD)"
OPT_TMP="$MASTER_SRC"

echo ">>> Uninstalling cryopares editable install ..."
"$PIP" uninstall -y cryopares 2>/dev/null || true

cleanup() {
    echo ">>> Restoring editable install ..."
    "$PIP" install -q -e "$MASTER_SRC" 2>/dev/null || true
}
trap cleanup EXIT

mkdir -p "$OUTPUT_BASE"
echo ">>> Output: $OUTPUT_BASE"
echo ">>> N_FIRST: ${N_FIRST:-'(all)'}"

run_one() {
    local src=$1 label=$2 gpu=$3 ckpt=$4 star=$5 pdir=$6 ref=$7 sym=$8 batch_size=$9
    shift 9

    local out=$OUTPUT_BASE/$label
    mkdir -p "$out"
    echo ""
    echo "====== $label (GPU $gpu, bs=$batch_size) ======"

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
        echo "ERROR: inference failed for $label" | tee -a "$out/run.log"
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
                --sym       D3                             \
                --save_plots "$out/compare_poses_${half}"
        ) 2>&1 | tee "$out/compare_poses_${half}.log"
    done
}

# ── Configs ──────────────────────────────────────────────────────────────────
ILR_OPT_SO3I_CFG=(
    "projmatching.grid_distance_degs=6" "projmatching.grid_step_degs=2"
    "projmatching.use_so3_interpolation=True"
)
ILR_OPT_SUB64_CFG=(
    "projmatching.grid_distance_degs=6" "projmatching.grid_step_degs=2"
    "projmatching.use_so3_interpolation=True"
    "projmatching.proj_sub_batch_size=64"
)
ILR_OPT_SUB128_CFG=(
    "projmatching.grid_distance_degs=6" "projmatching.grid_step_degs=2"
    "projmatching.use_so3_interpolation=True"
    "projmatching.proj_sub_batch_size=128"
)

echo ""
echo ">>> Wave 1: ilr_speed_opt_so3i (bs=8) on GPU 0"
run_one "$OPT_TMP" "ilr_speed_opt_so3i__gdh_G2" 0 \
    "$GDH_CKPT" "$GDH_STAR" "$GDH_PDIR" "$GDH_REF" D3 8 "${ILR_OPT_SO3I_CFG[@]}"
echo ">>> Wave 1 complete"

echo ""
echo ">>> Wave 2: ilr_speed_opt_so3i (bs=32) on GPU 0"
run_one "$OPT_TMP" "ilr_speed_opt_so3i_bs32__gdh_G2" 0 \
    "$GDH_CKPT" "$GDH_STAR" "$GDH_PDIR" "$GDH_REF" D3 32 "${ILR_OPT_SO3I_CFG[@]}"
echo ">>> Wave 2 complete"

echo ""
echo ">>> Wave 3: sub-batching ablation (sub64 GPU 0 | sub128 GPU 1) in parallel"
run_one "$OPT_TMP" "ilr_opt_sub64_bs32__gdh_G2" 0 \
    "$GDH_CKPT" "$GDH_STAR" "$GDH_PDIR" "$GDH_REF" D3 32 "${ILR_OPT_SUB64_CFG[@]}" &
run_one "$OPT_TMP" "ilr_opt_sub128_bs32__gdh_G2" 1 \
    "$GDH_CKPT" "$GDH_STAR" "$GDH_PDIR" "$GDH_REF" D3 32 "${ILR_OPT_SUB128_CFG[@]}" &
wait
echo ">>> Wave 3 complete"

echo ""
echo "=== VALIDATION COMPLETE ==="
echo "Timing summary:"
for d in "$OUTPUT_BASE"/*/; do
    label=$(basename "$d")
    time_s=$(cat "$d/time.txt" 2>/dev/null || echo "N/A")
    echo "  $label: ${time_s}s"
done

"$PYTHON" "$MASTER_SRC/scripts/summarize_benchmark.py" "$OUTPUT_BASE"
