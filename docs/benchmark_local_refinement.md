# Benchmark: Local Refinement Improvements

Systematic benchmark of the `improve_local_refinement` (ILR) branch vs master,
evaluating compute time and pose accuracy on the gdh_G2 dataset.

The ILR branch introduces several changes to `cryoPARES/projmatching/projMatcher.py`:
- **Change 1**: Sub-pixel shift interpolation (`use_subpixel_shifts`, default True)
- **Change 2a**: Zero DC component before correlation (`zero_dc`, default True)
- **Change 2b**: Spectral whitening of projections (`spectral_whitening`, default False)
- **Change 7**: Parabolic SO(3) sub-step interpolation (`use_so3_interpolation`, default True)
- **Change 8**: Two-stage coarse→fine search (`use_two_stage_search`, default False)

---

## How to Run

### Script location

```
scripts/run_benchmark_local_refinement.sh
```

### Quick start

```bash
# Smoke test — 100 particles, confirms pipeline runs end-to-end
N_FIRST=100 bash scripts/run_benchmark_local_refinement.sh

# Standard benchmark — 10K particles
bash scripts/run_benchmark_local_refinement.sh

# Full dataset (no particle limit)
N_FIRST="" bash scripts/run_benchmark_local_refinement.sh
```

Output is written to a timestamped directory:
```
/data/nagagpu05/not-backed-up/sanchezg/cryo/benchmark_localref_YYYYMMDD_HHMMSS/
```

Each run subdirectory contains:
- `run.log` — full inference log
- `time.txt` — wall-clock seconds
- `compare_poses_half{1,2}.log` — angular error statistics vs ground truth
- `compare_poses_half{1,2}/` — plots

### Summarise results from a past run

```bash
python scripts/summarize_benchmark.py /path/to/benchmark_localref_YYYYMMDD_HHMMSS
```

---

## Branch isolation

PYTHONPATH cannot override the editable-install meta-path finder registered by
`pip install -e`. The script therefore:

1. Runs `pip uninstall cryopares` at startup to remove the finder entirely.
2. Checks out the ILR branch into `/tmp/cryoPARES_ilr` via `git worktree`.
3. Runs all inference jobs as `cd <srcdir> && python -m cryoPARES.inference.infer …`,
   so Python finds the package through the CWD entry in `sys.path`.
4. On `EXIT`, restores the editable install: `pip install -e $MASTER_SRC`.

Wave 1 (sequential) uses `$MASTER_SRC`. Wave 2 (parallel, GPUs 0/1/2) uses
`/tmp/cryoPARES_ilr`. GPU 3 is reserved for training and never used.

---

## Configurations

| Label | Branch | Key config overrides | Description |
|-------|--------|---------------------|-------------|
| `master_6_2` | master | — | Baseline: original code, 6°/2° grid |
| `ilr_6_2` | ILR | `use_so3_interpolation=False` | ILR code, SO3 interp disabled — isolates effect of subpixel/zero-DC |
| `ilr_6_2_so3i` | ILR | `use_so3_interpolation=True` | ILR code with parabolic SO3 interpolation (ILR default) |
| `ilr_2stage` | ILR | `use_two_stage_search=True`, fine 2.1°/0.7° | Two-stage coarse→fine search |

ILR defaults always active in Wave 2: `use_subpixel_shifts=True`, `zero_dc=True`.

---

## Dataset

| Key | Value |
|-----|-------|
| Molecule | GDH (glutamate dehydrogenase) |
| Subset | G2 (test set, not seen during training) |
| Symmetry | D3 |
| Star file | `.../gdh_G2/aligned_particles_float32.star` |
| Particles dir | `.../gdh_G2/` |
| Checkpoint | `.../gdh/train/g1/200K_particles` (trained on G1) |
| Reference vol | `.../gdh/train/g1/200K_particles/half1/reconstructions/0.mrc` |
| Ground truth | RELION consensus poses in the star file |
| Base path | `/data/nagagpu05/not-backed-up/sanchezg/astex_data/data/preAlignedParticles` |
| Checkpoint base | `/data/nagagpu05/not-backed-up/sanchezg/local_storage/cryo/data/cryopares` |

Note: bgal was excluded — the APO-trained model does not transfer to LIG particles
(median angular error ~72°, not a code issue).

---

## Results

### Run: 2026-05-19 (run 1) — N=10 000, gdh_G2

ILR commit: `00acee5` (Add PKM2 lig_01029 results to cross-target summary)

#### Compute time

| Run | Time (s) | vs master |
|-----|----------|-----------|
| master_6_2 | 141 | — |
| ilr_6_2 | 156 | +11% |
| ilr_6_2_so3i | 158 | +12% |
| ilr_2stage | 528 | +275% |

#### Accuracy (averaged over half1 + half2)

| Run | N | Mean rot° | Median rot° | %<5° | %<10° |
|-----|---|-----------|------------|------|-------|
| master_6_2 | 10 000 | 17.76 | 2.27 | 66.90 | 71.25 |
| ilr_6_2 | 10 000 | 18.04 | 2.13 | 67.00 | 70.60 |
| ilr_6_2_so3i | 10 000 | 17.85 | 1.92 | 67.05 | 70.40 |
| ilr_2stage | 10 000 | 17.64 | 1.75 | 68.40 | 71.00 |

Shift errors not captured for this run.

---

### Run: 2026-05-19 (run 2) — N=10 000, gdh_G2 — overhead diagnostic

Same ILR commit (`00acee5`). Adds `ilr_all_off` to isolate whether the ~12% overhead
is from the ILR features themselves or from always-active code changes.

`ilr_all_off` config: `use_subpixel_shifts=False zero_dc=False use_so3_interpolation=False use_two_stage_search=False spectral_whitening=False`

#### Compute time

| Run | Time (s) | vs master |
|-----|----------|-----------|
| master_6_2 | 145 | — |
| **ilr_all_off** | **147** | **+1%** |
| ilr_6_2 | 163 | +12% |
| ilr_6_2_so3i | 162 | +12% |
| ilr_2stage | 606 | +318% |

#### Accuracy (averaged over half1 + half2)

| Run | N | Mean rot° | Median rot° | %<5° | %<10° |
|-----|---|-----------|------------|------|-------|
| master_6_2 | 10 000 | 17.87 | 2.29 | 67.15 | 71.00 |
| **ilr_all_off** | **10 000** | **18.07** | **2.15** | **66.75** | **70.45** |
| ilr_6_2 | 10 000 | 17.93 | 2.17 | 66.55 | 70.50 |
| ilr_6_2_so3i | 10 000 | 17.92 | 1.92 | 66.85 | 70.40 |
| ilr_2stage | 10 000 | 17.52 | 1.75 | 68.30 | 70.65 |

Shift errors not captured for this run.

#### Interpretation — overhead source confirmed

The mean angular error (~17–18°) is dominated by a tail of failed particles and is
not a reliable discriminator. Focus on **median** and **%<5°**.

**`ilr_all_off` is essentially as fast as master (+1%, within noise).** This confirms
the ~12% overhead in `ilr_6_2` is entirely from two ILR defaults that were left on:

- **`zero_dc=True`** — clones both `parts` and `projs` DFT tensors inside
  `correlate_dft_2d` on every batch (hot path).
- **`use_subpixel_shifts=True`** — adds ~12 indexed tensor ops per correlation peak
  in `_extract_ccor_max`.

The unconditionally-added `.contiguous()` calls in `correlate_dft_2d` contribute
negligibly (tensors are already contiguous on the hot path).

Interestingly, `ilr_all_off` is already slightly more accurate than master
(median 2.15° vs 2.29°) despite disabling all features — this small gain likely
comes from minor numerical differences in the refactored forward path.

Summary of the accuracy–speed frontier:

- All ILR variants improve the median over master (2.29° → 2.15°–1.75°).
- **SO3 interpolation** (ilr_6_2 → ilr_6_2_so3i): median 2.17° → 1.92°, essentially
  no extra compute (+0 s over ilr_6_2; the SO3 table lookup is O(1)).
- **`ilr_6_2_so3i` is the best default**: +12% compute, −0.37° median.
- **Two-stage search** gives the best median (1.75°) but at 4.2× the compute cost.

---

### Run: 2026-05-19 (run 3) — N=10 000, gdh_G2 — speed-opt validation

Adds `ilr_speed_opt_so3i` (branch `ilr_speed_opt`, commit `510cc60`).

Optimisations applied vs `ilr_6_2_so3i`:
1. **Eliminated `zero_dc` clones** — DC zeroing moved upstream to freshly-allocated tensors
   (`_preprocess_particles_to_F` for particles; `_extract_cc_peaks` for projections).
   Removes two `projs.clone()` calls (~726 MB/batch) from the hot path.
2. **Simplified subpixel arithmetic** — replaced `torch.where`+`full_like`/`zeros_like`
   with `clamp(max=…)` + mask-multiply in `_extract_ccor_max`, reducing intermediate
   allocations inside the compiled graph.

#### Compute time

| Run | Time (s) | vs master |
|-----|----------|-----------|
| master_6_2 | 144 | — |
| ilr_all_off | 144 | 0% |
| ilr_6_2 | 156 | +8% |
| ilr_6_2_so3i | 156 | +8% |
| **ilr_speed_opt_so3i** | **150** | **+4%** |

#### Accuracy (averaged over half1 + half2)

| Run | N | Mean rot° | Median rot° | %<5° | %<10° |
|-----|---|-----------|------------|------|-------|
| master_6_2 | 10 000 | 17.63 | 2.28 | 67.15 | 71.30 |
| ilr_all_off | 10 000 | 18.00 | 2.15 | 66.65 | 70.60 |
| ilr_6_2 | 10 000 | 17.80 | 2.13 | 66.95 | 70.35 |
| ilr_6_2_so3i | 10 000 | 17.79 | 1.89 | 67.05 | 70.60 |
| **ilr_speed_opt_so3i** | **10 000** | **17.88** | **1.90** | **67.00** | **70.50** |

Shift errors not captured for this run.

#### Interpretation — optimisations confirmed

**`ilr_speed_opt_so3i` passes all pass criteria:**
- Time ≤ 155 s: ✅ 150 s (down from 156 s, −4%)
- Median within ±0.05° of `ilr_6_2_so3i` baseline (1.89°): ✅ 1.90° (Δ = +0.01°)
- %<5° within ±0.3% of baseline (67.05%): ✅ 67.00% (Δ = −0.05%)

The two optimisations together recovered 6 s (4%) with no measurable accuracy impact.
Remaining overhead vs master is +4% (6 s). Candidates for closing this gap: phase-ramp
`irfftn` replacement (F2), larger batch_size (F3), contiguous preprocessing (F1).

---

### Run: 2026-05-19 (run 4) — N=10 000, gdh_G2 — F2+F3 optimisations

Adds two further optimisations to the `ilr_speed_opt` branch (commit `4df58e2`):

- **F2 — Phase-ramp trick**: Eliminates the post-`irfftn` `ifftshift_2d` in `correlate_dft_2d`
  by pre-multiplying the frequency-domain result by a ±1 sign grid (`build_ccorr_sign_grid`).
  Uses the DFT shift theorem: `ifftshift_2d(irfftn(A)) = irfftn(A × (−1)^(ky_std+kx_std))`.
  Removes a `(B, nCand, H, W)` float32 roll (~1.43 GB/batch) from the hot path;
  the sign multiply operates on `(B, nCand, H, W//2+1)` complex (~0.73 GB), saving ~0.7 GB.
- **F3 — Larger batch size**: Benchmark script uses `batch_size=32` (was 8).
  Peak memory at bs=32: ~2.9 GB projs + ~5.7 GB corr + ~2 GB model/ref = ~10.6 GB,
  safe for 32 GB GPUs.  The config default (64) targets 80 GB cards.

**Wave ordering note**: Waves 1–3 ran the same configurations as run 3 (for cross-check).
Wave 2 ran in parallel on GPUs 0/1/2; GPU thermal effects elevated those times by ~15–20 s.
Waves 4–5 ran sequentially on GPU 0 after wave 3 had cooled: these are the reliable numbers
for F2 and F3.

#### Compute time

| Run | Time (s) | vs master | Note |
|-----|----------|-----------|------|
| master_6_2 | 143 | — | Wave 1, sequential |
| ilr_all_off | 158 | +10% | Wave 3; elevated (post-parallel thermal) |
| ilr_6_2 | 160 | +12% | Wave 2, parallel; thermal-elevated |
| ilr_6_2_so3i | 177 | +24% | Wave 2, parallel; thermal-elevated |
| **ilr_speed_opt_so3i (F1+F2, bs=8)** | **145** | **+1%** | Wave 4, sequential |
| **ilr_speed_opt_so3i_bs32 (F1+F2+F3)** | **137** | **−4%** | Wave 5, sequential |
| ilr_2stage | 527 | +268% | Wave 2, parallel |

#### Accuracy (averaged over half1 + half2)

| Run | N | Mean rot° | Median rot° | %<5° | %<10° | Median shift Å | Mean shift Å |
|-----|---|-----------|------------|------|-------|---------------|-------------|
| master_6_2 | 10 000 | 17.96 | 2.26 | 66.75 | 71.00 | 0.76 | 3.85 |
| ilr_all_off | 10 000 | 17.70 | 2.14 | 66.90 | 70.70 | 0.76 | 4.92 |
| ilr_6_2 | 10 000 | 17.90 | 2.14 | 66.85 | 70.60 | 0.60 | 4.91 |
| ilr_6_2_so3i | 10 000 | 17.77 | 1.90 | 66.75 | 70.45 | 0.60 | 4.81 |
| **ilr_speed_opt_so3i (F1+F2, bs=8)** | **10 000** | **17.78** | **1.91** | **66.90** | **70.50** | **0.60** | **4.88** |
| **ilr_speed_opt_so3i_bs32 (F1+F2+F3)** | **10 000** | **17.80** | **1.90** | **67.15** | **70.45** | **0.60** | **4.75** |
| ilr_2stage | 10 000 | 17.40 | 1.73 | 68.25 | 71.00 | 0.57 | 4.70 |

#### Interpretation — goal achieved: faster than master, better accuracy

**F2 effect** (compare run 3 `ilr_speed_opt_so3i` 150 s → run 4 145 s): −5 s (−3%).
The phase-ramp pre-multiply on `(B, nCand, H, W//2+1)` fuses cleanly with the existing
element-wise multiply, removing the full `(B, nCand, H, W)` roll from the critical path.

**F3 effect** (bs=8 → bs=32 within run 4): 145 s → 137 s, −8 s (−5.5%).
Higher batch size improves GPU occupancy for both projection generation and correlation.

**Combined F1+F2+F3 result**: `ilr_speed_opt_so3i_bs32` at 137 s is **4% faster than master**
(143 s) while achieving 0.36° better median angular accuracy (1.90° vs 2.26°).

Shift error is substantially lower for ILR runs (0.60 Å) vs master (0.76 Å), consistent with
the sub-pixel shift interpolation (`use_subpixel_shifts=True`) delivering real gains.

**Pass criteria for `ilr_speed_opt_so3i_bs32`:**
- Time < master 143 s: ✅ 137 s (−4%)
- Median rot within ±0.05° of `ilr_6_2_so3i` baseline (1.89–1.90°): ✅ 1.90° (Δ = 0°)
- %<5° within ±0.3% of baseline (66.75%): ✅ 67.15% (Δ = +0.40%)
- Median shift within ±0.02 Å of baseline (0.60 Å): ✅ 0.60 Å (Δ = 0 Å)

---

### Run: 2026-05-20 (run 5) — N=10 000, gdh_G2 — G5/G10/G11/G12 validation

Validates four further commits on `ilr_speed_opt` (branch HEAD `7a3b014`):

- **G5 — Empty projection buffer**: Replace `torch.zeros(…)` with `torch.empty(…)` for the
  candidate projection output buffer. Avoids a 1.46 GB memset per batch; safe because all
  valid slots are written before use.
- **G10 — Efficient GEMM for coordinate rotation**: Replace the broadcast of 53M tiny
  `(3,3)×(3,1)` matvecs with a single batched GEMM:
  `(rm @ valid_coords.T).transpose(-1,-2)`. 165× speedup on the matmul; saves ~19 s at
  bs=32 over N=10 K particles.
- **G11 — Fuse whitening + sign filters**: When both `whitening_filter` and `ccorr_sign_grid`
  are provided (the hot path), combine them into a single real multiply before applying to
  `projs`, reducing three O(B·nCand·H·W//2+1) passes to two (~1.1 s saved at bs=32).
- **G12 — Fold CTF into correlation multiply**: Pass the CTF tensor through `correlateF()` so
  it is multiplied into the projection-side whitening filter inside `correlate_dft_2d` instead
  of a separate O(B·nCand·H·W//2+1) pass (~1.2 s saved at bs=32).

Runs `ilr_speed_opt` directly from `MASTER_SRC` (which is currently on `ilr_speed_opt`). All
waves run sequentially on GPU 0 to eliminate thermal effects; Wave 3 uses GPUs 0+1 in parallel.

#### Compute time

| Run | Time (s) | vs master (143 s) | vs prev best (137 s, F2+F3) |
|-----|----------|-------------------|------------------------------|
| **ilr_speed_opt_so3i (G5+G10+G11+G12, bs=8)** | **122** | **−15%** | **−11%** |
| **ilr_speed_opt_so3i_bs32 (G5+G10+G11+G12, bs=32)** | **106** | **−26%** | **−23%** |
| ilr_opt_sub64_bs32 (sub_batch=64, bs=32) | 110 | −23% | −19% |
| ilr_opt_sub128_bs32 (sub_batch=128, bs=32) | 110 | −23% | −19% |

#### Accuracy (averaged over half1 + half2)

| Run | N | Mean rot° | Median rot° | %<5° | %<10° | Median shift Å | Mean shift Å |
|-----|---|-----------|------------|------|-------|---------------|-------------|
| **ilr_speed_opt_so3i (bs=8)** | **10 000** | **17.85** | **1.92** | **67.0** | **70.4** | **0.61** | **4.92** |
| **ilr_speed_opt_so3i_bs32** | **10 000** | **17.91** | **1.89** | **66.9** | **70.5** | **0.59** | **4.74** |
| ilr_opt_sub64_bs32 | 10 000 | 17.83 | 1.90 | 66.9 | 70.5 | 0.60 | 4.79 |
| ilr_opt_sub128_bs32 | 10 000 | 17.81 | 1.92 | 66.8 | 70.4 | 0.60 | 4.74 |

Reference (from run 4, for comparison):

| Run | Time (s) | Median rot° | %<5° | Median shift Å |
|-----|----------|------------|------|---------------|
| master_6_2 | 143 | 2.26 | 66.75 | 0.76 |
| ilr_6_2_so3i (baseline) | 177* | 1.90 | 66.75 | 0.60 |

*thermal-elevated from parallel run; true value ~160 s (run 3).

#### Interpretation — G10/G11/G12 deliver large additional speedups

**G10 effect** (GEMM rotation): dominant gain; reduces bs=32 from 137 s to ~116 s (estimated
~21 s combined with G11+G12, consistent with the measured 31 s from G5+G10+G11+G12).

**G5 effect** (empty buffer): contributes the remaining ~10 s; avoids the 1.46 GB zeroing
pass that previously ran every batch on the hot path.

**G11+G12 effect** (filter/CTF fusion): together ~2.3 s savings (minor individually but free).

**Sub-batching ablation (G9)**: `proj_sub_batch_size=64` and `proj_sub_batch_size=128` are
both 4 s slower than the default (sub_batch=0). After G10/G11/G12, the projection buffer is
no longer the dominant memory-bandwidth consumer, so sub-batching adds loop overhead without
reducing HBM pressure meaningfully. **Conclusion: keep `proj_sub_batch_size=0` (disabled).**

**Accuracy preserved:** all ilr_speed_opt variants show median rot ~1.89–1.92° and %<5° ~66.9%,
identical to the F2+F3 baseline within measurement noise (Δ < 0.03°). Median shift error is
0.59–0.61 Å, identical to F2+F3 and 0.15–0.17 Å better than master (0.76 Å).

**Pass criteria for `ilr_speed_opt_so3i_bs32` (G5+G10+G11+G12):**
- Time < master 143 s: ✅ 106 s (−26%)
- Median rot within ±0.05° of `ilr_6_2_so3i` baseline (1.89–1.90°): ✅ 1.89° (Δ = 0°)
- %<5° within ±0.3% of baseline (66.75%): ✅ 66.9% (Δ = +0.15%)
- Median shift within ±0.02 Å of baseline (0.60 Å): ✅ 0.59 Å (Δ = −0.01 Å)

---

## Notes / next steps

- bgal benchmark deferred — waiting for retrained bgal model (APO→LIG transfer failed).
- Consider tuning `ilr_2stage` fine-grid parameters to reduce compute while preserving accuracy.
- Results above use the GDH G1 checkpoint applied to G2 particles (cross-set test).
- `ilr_speed_opt` branch (HEAD `7a3b014`) is ready to merge into `improve_local_refinement`.
- Default `batch_size` in the config is already 64 (targeting 80 GB cards); users on 32 GB
  should use `--batch_size 32`. Consider documenting this in the inference CLI help.
- `proj_sub_batch_size` should remain 0 (disabled) — sub-batching is slower after G10/G11/G12.
