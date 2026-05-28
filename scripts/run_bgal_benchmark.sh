#!/usr/bin/env bash
# Benchmark: master vs ilr_speed_opt, bgal D2
#   Wave 1 — master_bs32         (GPU 2)
#   Wave 2 — ilr_single_bs32     (GPU 2, two_stage=False, default)
#   Wave 3 — ilr_two_stage_bs32  (GPU 2, two_stage=True)
#
# GPU 3 is reserved for training — do not use it.
#
# Usage:
#   bash scripts/run_bgal_benchmark.sh          # 10K particles (default)
#   N_FIRST=100 bash scripts/run_bgal_benchmark.sh   # smoke test
#   N_FIRST=""  bash scripts/run_bgal_benchmark.sh   # full dataset

set -euo pipefail
ulimit -n 65536

# ── Paths ─────────────────────────────────────────────────────────────────────
PIP=/data/nagagpu05/not-backed-up/sanchezg/app/miniconda3/envs/cryopares/bin/pip
PYTHON=/data/nagagpu05/not-backed-up/sanchezg/app/miniconda3/envs/cryopares/bin/python
MASTER_SRC=/data/nagagpu05/not-backed-up/special-project/cryo/myProjects/cryoPARES
ILR_REPO=/data/nagagpu05/not-backed-up/sanchezg/cryo/myProjects/cryoPARES
MASTER_TMP=/tmp/cryoPARES_master

DATA_ROOT=/data/nagagpu05/not-backed-up/sanchezg/astex_data/data/preAlignedParticles
CKPT_ROOT=/data/nagagpu05/not-backed-up/sanchezg/local_storage/cryo/data/cryopares

OUTPUT_BASE=/data/nagagpu05/not-backed-up/sanchezg/cryo/benchmark_bgal_$(date +%Y%m%d_%H%M%S)
N_FIRST=${N_FIRST:-10000}

BGAL_CKPT=$CKPT_ROOT/bgal/train/version_1
BGAL_STAR=$DATA_ROOT/bgal/lig_00892/Refine3D/run_data.star
BGAL_PDIR=$DATA_ROOT/bgal/lig_00892

# ── Verify MASTER_SRC is on ilr_speed_opt ─────────────────────────────────────
current_branch=$(git -C "$MASTER_SRC" branch --show-current)
if [[ "$current_branch" != "ilr_speed_opt" ]]; then
    echo "ERROR: MASTER_SRC is on '$current_branch', expected 'ilr_speed_opt'"
    exit 1
fi

# ── Ensure master worktree exists ─────────────────────────────────────────────
if [[ ! -d "$MASTER_TMP" ]]; then
    echo ">>> Creating master worktree at $MASTER_TMP ..."
    git -C "$ILR_REPO" worktree add "$MASTER_TMP" master
fi
echo ">>> master worktree: $(git -C "$MASTER_TMP" log --oneline -1)"
echo ">>> ilr_speed_opt:   $(git -C "$MASTER_SRC" log --oneline -1)"

# ── Remove editable install so CWD-based imports work cleanly ─────────────────
echo ">>> Uninstalling cryopares editable install ..."
"$PIP" uninstall -y cryopares 2>/dev/null || true

# ── Restore editable install and remove worktree on exit ──────────────────────
cleanup() {
    echo ">>> Restoring editable install ..."
    "$PIP" install -q -e "$MASTER_SRC" 2>/dev/null || true
    echo ">>> Removing master worktree ..."
    git -C "$ILR_REPO" worktree remove --force "$MASTER_TMP" 2>/dev/null || true
}
trap cleanup EXIT

mkdir -p "$OUTPUT_BASE"
echo ">>> Output: $OUTPUT_BASE"
echo ">>> N_FIRST: ${N_FIRST:-'(all)'}"

# ── Verify branch isolation ────────────────────────────────────────────────────
echo ">>> master  loads from: $( (cd "$MASTER_TMP" && "$PYTHON" -c "import cryoPARES; print(cryoPARES.__file__)") )"
echo ">>> ilr_opt loads from: $( (cd "$MASTER_SRC" && "$PYTHON" -c "import cryoPARES; print(cryoPARES.__file__)") )"

# ── run_one ───────────────────────────────────────────────────────────────────
# Usage: run_one <src> <label> <gpu_id> <ckpt> <star> <pdir> <sym> <batch_size> [key=val ...]
run_one() {
    local src=$1 label=$2 gpu=$3 ckpt=$4 star=$5 pdir=$6 sym=$7 batch_size=$8
    shift 8

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
            cd "$MASTER_TMP"
            "$PYTHON" -m cryoPARES.scripts.compare_poses \
                --starfile1 "$star"                        \
                --starfile2 "$pred"                        \
                --sym       "$sym"                         \
                --save_plots "$out/compare_poses_${half}"
        ) 2>&1 | tee "$out/compare_poses_${half}.log"
    done
}

# ── Wave 1: master, bs=32 ─────────────────────────────────────────────────────
echo ""
echo ">>> Wave 1: master_bs32 on GPU 2"
run_one "$MASTER_TMP" "master_bs32__bgal" 2 \
    "$BGAL_CKPT" "$BGAL_STAR" "$BGAL_PDIR" D2 32
echo ">>> Wave 1 complete"

# ── Wave 2: ilr_speed_opt, single-stage (default), bs=32 ─────────────────────
echo ""
echo ">>> Wave 2: ilr_single_bs32 on GPU 2"
run_one "$MASTER_SRC" "ilr_single_bs32__bgal" 2 \
    "$BGAL_CKPT" "$BGAL_STAR" "$BGAL_PDIR" D2 32
echo ">>> Wave 2 complete"

# ── Wave 3: ilr_speed_opt, two-stage, bs=32 ──────────────────────────────────
echo ""
echo ">>> Wave 3: ilr_two_stage_bs32 on GPU 2"
run_one "$MASTER_SRC" "ilr_two_stage_bs32__bgal" 2 \
    "$BGAL_CKPT" "$BGAL_STAR" "$BGAL_PDIR" D2 32 \
    "projmatching.use_two_stage_search=True"
echo ">>> Wave 3 complete"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=== BENCHMARK COMPLETE ==="
"$PYTHON" "$MASTER_SRC/scripts/summarize_benchmark.py" "$OUTPUT_BASE"
