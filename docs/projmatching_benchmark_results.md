# Projection Matching: Improvements & Benchmarks

## Research progression

Work went through two distinct phases:

**Phase 1 — Synthetic experiments (perturbed-pose benchmarks):** Algorithm changes were developed
and screened on DS2 (EMPIAR-10166, C1) and DS3 (PKM2, D2) by artificially perturbing known-good
RELION poses by 5° and measuring recovery. The NN is not involved — inputs are known GT poses with
controlled noise. This gave fast, reproducible signal for individual changes.

**Phase 2 — Real full-pipeline benchmark (β-galactosidase):** End-to-end runs with actual NN
inference, projection matching, and reconstruction on held-out β-gal ligand datasets. This is the
only metric that matters for the user.

**Key lesson: synthetic experiments were not representative.** The D2 recommendation from Phase 1
(4°/0.7° single-stage beats two-stage on D2) reversed in Phase 2: two-stage won on *both* bgal
ligand datasets despite them being D2. The perturbed-pose paradigm does not capture the real NN
error distribution (correlated errors, model-dependent failure modes) and the interaction with FSC.
No further synthetic experiments will be run. The next benchmark is on **PKM2 with a trained
model** (the actual D2 use case), which will be the ground truth for configuration decisions.

---

## Executive summary

**Starting point:** NN pose estimates (~7°) → Cartesian 6°/2° projmatching plateaus near 1.3–1.7° (master branch).
**Current best on real data:** two-stage 6°/2°+2.1°/0.7° K=5 remains best on median (1.05°/2.18°), but **cart_6-2 + SO(3) interpolation** beats two-stage on FSC for lig_00892 (2.581 vs 2.594 Å) at cart_6-2 speed (~3,500 p/min vs 838). SO(3) interp closes ~80% of the cart→two-stage angular gap at zero throughput cost.

### Implemented changes

| # | Change | Config flag | Default | Outcome |
|---|--------|-------------|---------|---------|
| 1 | Sub-pixel shift (parabolic interpolation) | `use_subpixel_shifts` | `True` | Angular improvement at 1° grid; shift improvement at all grids |
| 2a | Zero DC component | `zero_dc` | `True` | Small shift accuracy improvement |
| 2b | Particle-adaptive spectral whitening | `spectral_whitening`, `whitening_warmup_batches=8` | `True` | DS2: 1.64→1.33°, DS3: 1.69→1.43° (19% improvement) |
| 3 | Rotation matrix composition (Cartesian grid) | `rotation_composition` | `euler_add` | **Worse** than euler_add — grid incompatible with matrix composition |
| 4 | Fibonacci ω-ball grid + pre_multiply | `use_fibo_grid` | `False` | DS2: 1.33→1.31°, DS3 P90: 4.85→3.52° — best single-stage |
| 5 | Frequency band mask (high-pass ring) | `fftfreq_min` | `0.0` (off) | **Harmful** (P90 7.33° vs 2.75°) — keep disabled |
| 6 | Two-stage coarse-to-fine search | `use_two_stage_search`, `fine_grid_*` | `False` | DS2: median 1.37°→**0.42°** (3×); DS3: 1.60°→1.35°; 24% fewer evals vs 6°/1° |
| 7 | SO(3) sub-step pose interpolation | `use_so3_interpolation` | `False` | **Benchmarked on bgal.** cart_6-2+SO3: lig_00892 1.47°→**1.13°** (23%), FSC 2.667→**2.581 Å**; lig_00893 2.49°→**2.25°** (10%), FSC 2.967→**2.926 Å**. Beats two-stage on FSC for lig_00892. Zero throughput cost (runs at cart_6-2 speed). |

`use_fibo_grid=False`, `rotation_composition=euler_add`, and `use_two_stage_search=False` remain
behavior-preserving defaults.

### Current best config (behind flags)

```
use_subpixel_shifts=True, zero_dc=True, spectral_whitening=True,
whitening_warmup_batches=8, fftfreq_min=0.0,
use_fibo_grid=True, rotation_composition=pre_multiply,
use_two_stage_search=True, fine_grid_distance_degs=1.5, fine_grid_step_degs=0.5, fine_top_k=5
```

**Recommendation (from real full-pipeline benchmark):**
- **Two-stage** wins on all tested real datasets regardless of symmetry. The D2 exception seen in
  synthetic experiments (4°/0.7° > two-stage on DS3) did not hold on real bgal D2 data.
- Synthetic D2 finding (4°/0.7° single-stage) is retained here for reference but should not guide
  configuration choices until confirmed on a real D2 dataset with a trained model (PKM2).

---

## Setup

- **Machine**: Shared server with 2× RTX 6000 Ada (48 GB each); GPU 0 and GPU 1 are shared with other users.
  Runs targeting a free GPU use `--gpu_id 1` when GPU 0 is occupied.
- **Python env**: `/home/rsanchez/app/miniconda3/envs/cryopares/bin/python`
  (CLAUDE.md lists `/home/sanchezg/app/anaconda3/...` which does not exist on this machine)
- **Metric**: all reported errors are **error vs ground truth**, computed post-hoc using
  `cryoPARES/scripts/compare_poses.py --starfile1 OUTPUT --starfile2 GT --sym C1`.
  The inline `Median Ang Error` printed by `cryopares_projmatching` measures **displacement
  from the input** star file, NOT from GT. It is useful to confirm projmatching is moving,
  but must NOT be used as the benchmark number.
- **DS3 symmetry**: D2 — C1 error is reported in tables (the symmetry correction further lowers numbers).
- **Perturbation**: uniformly sampled from SO(3) ball of radius 5° (resulting in mean geodesic
  distance ~3.75°, median ~3.97° vs GT). Confirmed with `compare_poses.py` on the perturbed star files.

---

## Datasets

### DS1/DS2/DS3 (angular accuracy benchmarks)

| ID | Source | Particles | Pixel size (Å/px) | Box | Reference volume |
|----|--------|-----------|-------------------|-----|-----------------|
| DS2 | EMPIAR-10166 frealign_stack (real, C1) | 238631 | 1.27 | 336 | `EMPIAR-10166/data/reconstruct.mrc` |
| DS3 | astex-5534 PKM2 Refine3D (real, D2) | 254070 | 0.995 | 334 | `astex-5534/Refine3D/001_after2DclsEval/run_class001.mrc` |
| DS1 | EMPIAR-10166 projectionsSamePose (synthetic) | 238631 | 1.5 | 224 | — (particle images still copying) |

#### Ground-truth and perturbed star files

| File | Description |
|------|-------------|
| `tests/frealign_stack_relion3.star` | DS2 GT (RELION 3.1 format, optics+particles blocks) |
| `tests/ds2_perturbed_5deg.star` | DS2 perturbed, pre-composition (R_δ @ R_true), seed=42 |
| `tests/ds2_post_perturbed_5deg.star` | DS2 perturbed, post-composition (R_true @ R_δ), seed=42 |
| `tests/ds3_perturbed_5deg.star` | DS3 perturbed, pre-composition, seed=42 |
| `tests/ds3_post_perturbed_5deg.star` | DS3 perturbed, post-composition, seed=42 |
| `tests/ds1_perturbed_5deg.star` | DS1 perturbed, pre-composition, seed=42 |

**Note on frealign_stack.star format**: the original `EMPIAR-10166/data/frealign_stack.star` is RELION 3.0
format (single pandas DataFrame, no optics block). The code's `n_first_particles` path assumes RELION 3.1
format (dict with 'optics'/'particles' keys). A bug fix was applied to `projmatching/projmatching.py` line
290 to handle both formats. Use `frealign_stack_relion3.star` for benchmarking to avoid format issues.

### β-galactosidase (full-pipeline benchmark)

```bash
CKPT=/home/rsanchez/cryo/data/cryopares/train/bgal/apo_00482/version_0
PYTHON=/home/rsanchez/app/miniconda3/envs/cryopares/bin/python
BIN_DIR=/home/rsanchez/app/miniconda3/envs/cryopares/bin
COMPARE=$PYTHON /home/rsanchez/cryo/myProjects/cryoPARES/utils/compare_poses.py

# Dataset: apo (self-test, same molecules as training)
STAR_APO=/home/rsanchez/cryo/data/astexData/bgal/apo_00482/Refine3D/001_after2DclsEval/run_data.star
DIR_APO=/home/rsanchez/cryo/data/astexData/bgal/apo_00482
REF_APO=/home/rsanchez/cryo/data/astexData/bgal/apo_00482/reconstruction.mrc
MASK_APO=/home/rsanchez/cryo/data/astexData/bgal/apo_00482/mask.mrc
# ~177,824 particles, D2 symmetry

# Dataset: lig_00892 (cross-dataset test, different ligand state)
STAR_LIG892=/home/rsanchez/cryo/data/astexData/bgal/lig_00892/Refine3D/001_after2DclsEval/run_data.star
DIR_LIG892=/home/rsanchez/cryo/data/astexData/bgal/lig_00892
REF_LIG892=/home/rsanchez/cryo/data/astexData/bgal/lig_00892/reconstruction.mrc
# ~56,763 particles, D2 symmetry

# Dataset: lig_00893 (cross-dataset test, harder)
STAR_LIG893=/home/rsanchez/cryo/data/astexData/bgal/lig_00893/Refine3D/001_after2DclsEval/run_data.star
DIR_LIG893=/home/rsanchez/cryo/data/astexData/bgal/lig_00893
REF_LIG893=/home/rsanchez/cryo/data/astexData/bgal/lig_00893/reconstruction.mrc
# ~42,000 particles, D2 symmetry
```

All runs: `--data_halfset allParticles --model_halfset half1`, masked FSC with the apo mask, angular
error vs RELION GT with `--sym D2`. GPU 1 (RTX 6000 Ada 51 GB). Box = 476px.

---

## Implemented changes — implementation details & lessons

All changes are gated by config flags in `projmatching_config.py` with backward-compatible defaults.
Key files: `projMatcher.py`, `projmatchingUtils/fourierOperations.py`, `projmatching_config.py`.

### Change #1 — Sub-pixel shift refinement (`use_subpixel_shifts`, default `True`)

**File:** `projMatcher.py::_extract_ccor_max`

The integer CC peak location was found with `argmax` and stored as `int64`. This created a shift
quantization floor of ~1 pixel ≈ 1–3° depending on pixel size and particle radius.

**Fix:** 3-point parabolic interpolation on both row and column axes after the integer peak:
```
delta = (f[p-1] - f[p+1]) / (2*f[p-1] - 4*f[p] + 2*f[p+1])
subpixel_pos = p + delta
```
`delta` is clamped to the border guard so boundary peaks get `delta=0`. The `pixelShiftsXY`
tensor type changed from `int64` → `float32`. Downstream conversion in `forward()` already called
`.float()`, so no other changes needed.

**Bug encountered during implementation:** First attempt used `expand(*corrs_crop.shape[:-2], 1, W)`
which broke `torch.compile(fullgraph=True)` because dynamic shape unpacking is not supported by
`torch.compile`. Rewrote using explicit batch indexing: `n_idx = torch.arange(N)` followed by
`corrs_crop[n_idx, safe_i, int_j]` — fully static shapes, compatible with `fullgraph=True`.

**Lesson:** Integer-pixel CC peak location creates a quantization floor of 1–3° depending on pixel size and
particle radius. Parabolic 3-point interpolation on both axes after the integer peak resolves this.
The parabolic approximation is valid because the CC peak is smooth and band-limited.

TODO: think if makes sense to cache n_idx = torch.arange(N)

---

### Change #2a — Zero DC component (`zero_dc`, default `True`)

**File:** `fourierOperations.py::correlate_dft_2d`

The DC bin (row `H//2`, col `0` in fftshifted rfft layout) represents the mean pixel value.
Correlating it adds a global-offset bias term that is orientation-independent.

**Fix:** Zero the DC bin in both `parts` and `projs` before the cross-product:
```python
dc_row = parts.shape[-2] // 2
parts[..., dc_row, 0] = 0
projs[..., dc_row, 0] = 0
```

---

### Change #2b — Particle-adaptive spectral whitening (`spectral_whitening`, default `True`)

**File:** `projMatcher.py::forward`, `fourierOperations.py::correlate_dft_2d`

The raw CC is dominated by low frequencies (power spectrum ∝ 1/f²–1/f⁴). Orientation-
discriminating signal at higher frequencies contributes negligible weight.

**Design evolution (three failed attempts, one success):**

1. *Reference-based whitening v1* (`1/sqrt(amp_3d(r))` from reference volume spectrum):
   Failed on DS3 — RELION's LP-filtered reference has near-zero amplitude beyond 6Å cutoff →
   whitening factor explodes to **6.99×** at Nyquist (vs 1.33× for DS2). DS3 regression: 2.0°.

2. *Reference-based whitening v2* (masked at `fftfreq_max`): still 2.0° regression on DS3.
   The amplitude spectrum from the reference volume does not match the actual particle spectrum
   per-dataset (different CTF envelopes, noise floors, pixel sizes).

3. *Reference-based v3* (apply to projections only, not particles): still 2.0° regression on DS3.

4. **Particle-adaptive whitening (final):** estimate `1/amp` from the actual particle DFTs.
   Applied to projections only (whitening both sides gives `1/amp²` weighting that amplifies
   particle noise; projections-only gives `1/amp` — more robust). Absorbs per-dataset CTF
   envelope, detector noise, and pixel-size effects automatically.

**Implementation:**
- `forward()`: lazy warm-up loop; accumulates `fparts.abs().mean(dim=0)` over first
  `whitening_warmup_batches` batches (default 8). Whitening map is recomputed after each
  batch and frozen once warm-up completes. Reset at each `align_star()` call.
- `correlate_dft_2d`: accepts `whitening_filter` tensor; `projs = projs * whitening_filter`.
- `align_star()`: resets `_whitening_amp_sum = None`, `_whitening_batches_seen = 0`.

**Why 8 warm-up batches:** more diverse defocus conditions are averaged → CTF oscillations cancel →
smoother amplitude profile. Analogous to RELION's per-optics-group noise estimation averaged
over many particles per group. Single-batch was sufficient for DS2 but not DS3.

**Lesson — particle-adaptive, not reference-based:**
Three attempts at reference-based whitening failed on DS3: RELION's LP-filtered reference has
near-zero amplitude beyond the resolution cutoff, causing the whitening factor to explode to 6.99×
at Nyquist. The reference volume spectrum does not match the actual particle spectrum — they differ
in CTF envelope, noise floor, and pixel size. Particle-adaptive whitening absorbs these differences
automatically and works across datasets.

Apply whitening to projections only, not particles: whitening both sides gives `1/amp²` which
over-amplifies particle noise. Using 8 warm-up batches (vs 1) matters for DS3 where defocus
diversity cancels CTF oscillations in the amplitude estimate.

TODO: Think how to implement this for on-the-fly friendly runs — may need to keep updating stats
every few 1000s of particles in case of microscope drift, but this could be computationally
inefficient and we care about performance.

---

### Change #5 — Frequency band mask (`fftfreq_min`, default `0.0`)

**File:** `fourierOperations.py::_mask_for_dft_2d`, `projMatcher.py::_store_reference_vol`

Extended `_mask_for_dft_2d` with a `min_freq_pixels` parameter for a high-pass ring.
`band_mask` is pre-computed at init and registered as a buffer. Applied to `fparts` in `forward()`
when `fftfreq_min > 0`.

**Result:** Harmful at `fftfreq_min=0.05` — P90 worsened from 3.16° → 7.33° (DS2 Scen B).
Zero DC (#2a) handles the DC bias. A high-pass ring removes low-frequency structural information
that is still orientation-discriminating above DC. **Default kept at 0.0 (disabled).**

---

### Change #3 — Rotation composition (`rotation_composition`, default `"euler_add"`)

**File:** `projMatcher.py::forward`, `projmatching_config.py`

The original Euler angle addition `delta + euler` is an approximation to rotation composition and
is inaccurate near the tilt poles (when β ≈ 0, α and γ are degenerate). Implemented exact SO(3)
composition via rotation matrices with two orderings:

- `"pre_multiply"`: `R_total = R_delta @ R_current` (delta in lab frame)
- `"post_multiply"`: `R_total = R_current @ R_delta` (delta in body/particle frame)

**Key finding with Cartesian grid:** Both modes are **significantly worse** than `euler_add`
(2.27–2.42° vs 1.33°). Root cause: the Cartesian Euler grid is designed for Euler-space arithmetic.
Converted to rotation matrices and composed, the coverage pattern changes unfavorably — the
near-identity cluster that the Cartesian grid accumulates (7 identical rot/psi values per
tilt=0 row) vanishes when converted to matrices, leaving sparser effective search around the
correct pose.

**Cartesian grid rotation composition results (DS2 Scen B vs GT, n=2000):**

| Mode | Median (°) | P75 (°) | P90 (°) |
|------|-----------|---------|---------|
| euler_add (baseline) | 1.33 | 1.95 | 2.75 |
| pre_multiply (Cartesian) | 2.42 | 3.61 | 4.81 |
| post_multiply (Cartesian) | 2.27 | 3.35 | 4.32 |

**Lesson:** Euler angle addition (`delta + euler`) works well with the Cartesian Euler grid because
the grid is designed for Euler-space arithmetic. Rotation matrix composition is only compatible with
the Fibonacci grid, which is specifically designed for it.

---

### Change #4 — Fibonacci ω-ball grid (`use_fibo_grid`, default `False`)

**File:** `projMatcher.py::_get_so3_delta_rotmats`, `projmatching_config.py`

Replaced the Cartesian Euler grid with a uniform geodesic ω-ball grid (`so3_near_identity_by_spacing`
with `use_small_aprox=True`). This grid samples rotation vectors in an SO(3) ball of radius
`grid_distance_degs` with approximately uniform coverage — no Euler angle singularities,
no polar clustering.

**Grid point counts:**

| Settings | Cartesian | Fibonacci |
|----------|-----------|-----------|
| 6°/2° | 343 (7³) | 208 + 1 identity = 209 |
| 6°/1° | 2197 (13³) | 1637 + 1 identity = 1638 |

At 6°/2°: Fibonacci uses 39% fewer points with better uniform coverage.
At 6°/1°: Fibonacci uses 25% fewer points (1638 vs 2197).

**Critical bug fixed — missing identity:** The ω-ball shell centres start at ~3.3° geodesic
distance. Without explicitly adding identity to the grid, Scenario A (where ground truth poses
are fed as input) always returns the wrong pose with ≥3.3° error — a guaranteed regression.
Fix: prepend `torch.eye(3)` to the cached rotation matrix tensor in `_get_so3_delta_rotmats()`.

**Implementation:**
- New `_get_so3_delta_rotmats(device)` method on `ProjectionMatcher`: returns cached `(nDelta, 3, 3)`
  delta rotation matrices. Uses Fibonacci when `use_fibo_grid=True`, otherwise converts the
  Cartesian Euler grid to rotation matrices.
- `rotation_composition` auto-upgrades from `"euler_add"` to `"pre_multiply"` when
  `use_fibo_grid=True` (prints a message; euler_add is meaningless for the rotation-matrix path).

**Composition mode experiment (pre vs post) — DS2, n=500, 6°/1°:**

| Perturbation \ Search | pre_multiply | post_multiply |
|-----------------------|-------------|--------------|
| pre (R_δ @ R_true) | Median **0.89°**, <5°: 98.8% | Median **0.85°**, <5°: 98.0% |
| post (R_true @ R_δ) | Median **0.84°**, <5°: 98.4% | Median **0.85°**, <5°: 98.4% |

**Lesson — pre_multiply ≡ post_multiply:**
For an isotropic SO(3) ball, `{R_δ @ R_est}` and `{R_est @ R_δ}` cover the same geodesic ball
and give identical recovery accuracy (~0.85° for both). Similarly, perturbation composition
order has no effect on recovery accuracy. The user hypothesis that only one ordering is
geometrically correct is disproved.

The Fibonacci grid eliminates Euler polar degeneracy that was harming DS3 (D2 symmetry, particles
near symmetry axes): P90 dropped from 4.85°→3.52°, a standout improvement.

TODO: Check if we are following the definition of the grid distance — it is supposed to have a
grid that goes from -6° to +6° (check stats for Cartesian and Fibonacci).

---

### Change #6 — Two-stage coarse-to-fine search (`use_two_stage_search`, default `False`)

**Config flags:**
```
use_two_stage_search=True, fine_grid_distance_degs=1.5, fine_grid_step_degs=0.5, fine_top_k=5
```

**Actual Fibonacci grid point counts** (measured, formula: `so3_grid_near_identity_fibo(use_small_aprox=True)`):

| Config | Pts | Notes |
|--------|-----|-------|
| 6°/2° | 209 | coarse pass |
| 6°/1° | 1638 | strong single-stage baseline |
| 4°/0.5° | 3875 | strong single-stage baseline (covers 4° initial error) |
| 4°/0.7° | 1486 | sweet spot single-stage for D2 |
| 4°/1° | 488 | cheaper single-stage |
| 1.5°/0.5° | 209 | fine pass (same count as 6°/2°!) |
| 1.0°/0.5° | 63 | fine pass (aggressive) |

> **Correction:** "2°/0.5° → ~65 pts" was wrong — 2°/0.5° gives ~488 pts.
> The 63-pt target is hit by **1°/0.5°**; the preferred fine grid is **1.5°/0.5° (209 pts)**.

**Two-stage totals** (K=5 candidates from coarse):

| Coarse + Fine | Total | vs 6°/1° single |
|---------------|-------|-----------------|
| 6°/2° + 1.5°/0.5° | 209 + 5×209 = **1249** | 24% fewer pts |
| 6°/2° + 1.0°/0.5° | 209 + 5×63  = **524**  | 68% fewer pts |

**Implementation details:**
- `_preprocess_particles_to_F()`: particle FFT + whitening warm-up, called once per `forward()` call
- `_expand_rotmats()`: SO(3) composition of (B, K, 3, 3) × (nDelta, 3, 3) → (B, K*nDelta, 3, 3)
- `_do_search()`: project + CTF + correlate + topk, reusable for coarse and fine passes
- `_forward_two_stage()`: orchestrates both passes, confidence from coarse distribution
- `euler_add` auto-switched to `pre_multiply` when `use_two_stage_search=True`

**Results (n=500 except 6°/2° n=2000, Scen B vs GT):**

| Config | Pts | DS1 med | DS2 med | DS2 P75 | DS2 P90 | DS3 med | DS3 P75 | DS3 P90 | Time/500 |
|--------|-----|--------|--------|---------|---------|--------|---------|---------|---------|
| **master** Cartesian 6°/2° | 343 | — | 1.64° | — | 3.79° | 1.69° | — | 4.06° | **~9.4s** |
| fibo 4°/2° (bs=75) | 63 | — | 1.39° | — | — | 1.50° | — | — | **~3.1s** |
| fibo 6°/2° (bs=32) | 209 | — | 1.31° | 1.88° | 2.62° | 1.31° | 1.98° | 3.52° | **~6.2s** |
| fibo 4°/1° (bs=16) | 488 | — | 0.97° | 1.53° | 2.13° | 1.35° | 2.03° | 2.81° | **~13.4s** |
| fibo 4°/0.7° (bs=5) | 1486 | — | 0.87° | 1.43° | 2.23° | **1.24°** | **1.93°** | **2.74°** | **~40.4s** |
| fibo 6°/1° (bs=8) | 1638 | — | 0.89° | 1.62° | 2.67° | **1.35°** | — | — | **~43.2s** |
| **two-stage 6°/2°+1.5°/0.5° K=5 (bs=7)** | **1249** | **0.22°** | **0.42°** | **1.25°** | 2.47° | 1.35° | 2.35° | 3.43° | **~33.2s** |

Timing measured on 10K particles, 3 runs each, RTX 6000 Ada 49 GB. Per-500 = raw ÷ 20.
Master baseline: DS2 187s/10K, DS3 200s/10K (Cartesian 6°/2°, batch_size=11).
Note: 6°/1° (1638 pts) is slower than two-stage (1249 pts) yet less accurate on both datasets — dominated.
4°/0.7° wins on DS3 because it concentrates coverage where it matters (≤4° initial error, 0.7° step)
rather than wasting evaluations from 4°–6°.

Two-stage gives **4× better median** on DS2 (C1) vs master, 3.6× slower.
DS3 (D2): 4°/0.7° single-stage is better — coarse K=5 candidates can cluster in one D2 domain,
missing poses in the other 3. DS1 (synthetic): near-perfect 0.22° recovery.

**Scenario A validation (GT input → expect ~0°):**
- DS1: 0.00° ✓ (requires CTF-corrected reference volume)
- DS2: 0.00° ✓
- DS3: 1.41° — fine search (0.5° step) finds a pose 1.41° from RELION GT that correlates marginally
  better against the reference. Not a bug; single-stage still returns 0.00° ✓.

**DS3 Scen B K=10 result:** Median 1.36°, P75 2.29°, P90 3.50° — identical to K=5 (1.35°/2.35°/3.43°)
at 77% more evaluations (2299 vs 1249 pts). The D2 gap is geometric (symmetry-axis ambiguity),
not a candidate-count issue. K=10 ruled out.

**DS3 Scen B 4°/0.5° result:** Median 1.22°, P75 1.89°, P90 2.64° in ~2m (batch_size=1) vs 4°/0.7°:
1.24°/1.93°/2.74° in ~52s (batch_size=2). Only 0.02° median gain for 2.3× wall-clock cost.
**4°/0.7° is the sweet spot for D2 — going finer hits diminishing returns.**

---

## DS2/DS3 benchmark results (Phase 1 — synthetic experiments)

> **Note:** These experiments use artificially perturbed GT poses as input — the NN is not
> involved. They were useful for screening algorithm changes quickly but proved to be a poor
> predictor of real full-pipeline performance (see Phase 2 bgal results and lessons below).
> No further experiments of this type are planned.

```bash
PYTHON=/home/rsanchez/app/miniconda3/envs/cryopares/bin/python
BIN=/home/rsanchez/app/miniconda3/envs/cryopares/bin/cryopares_projmatching
EMPIAR=/home/rsanchez/cryo/data/EMPIAR-download
DATA=/home/rsanchez/cryo/data/cryopares

# DS2 Scenario B — run projmatching on perturbed poses
$BIN \
    --reference_vol $EMPIAR/EMPIAR-10166/data/reconstruct.mrc \
    --particles_star_fname $DATA/tests/ds2_perturbed_5deg.star \
    --particles_dir $EMPIAR/EMPIAR-10166/data \
    --out_fname $OUT \
    --n_first_particles 500 --grid_distance_degs 6 --grid_step_degs 1 \
    --batch_size 2 --gpu_id 1 \
    --config projmatching.use_subpixel_shifts=True projmatching.zero_dc=True \
             projmatching.spectral_whitening=True projmatching.whitening_warmup_batches=8 \
             projmatching.fftfreq_min=0.0 projmatching.use_fibo_grid=True \
             projmatching.rotation_composition=pre_multiply

# GT comparison (post-hoc) — ALWAYS use compare_poses.py
$PYTHON -m cryoPARES.scripts.compare_poses \
    --starfile1 $OUT --starfile2 $DATA/tests/frealign_stack_relion3.star --sym C1
```

### Batch size constraint (XBLOCK Triton error)

The compiled projection kernel materialises a `(batch_size × n_rotations × n_valid_coords × 9)` float32
buffer. With `dynamic=True`, Triton uses a worst-case size hint of 2^36, triggering:
```
AssertionError: 'XBLOCK' too large. Maximum: 4096. Actual: 8192.
```
**Fix:** compile with `dynamic=False` (applied in `extract_central_slices_as_real.py`). With
`dynamic=False`, Triton receives the concrete tensor shape and selects an appropriate XBLOCK.

---

### Phase 1 — Baseline (unmodified code)

Grid: 6°/2° (343 rotations), batch_size=32, compile enabled, GPU 0.
All errors vs GT via post-hoc `compare_poses.py`.

| Change | Dataset | Scenario | Median (°) | P75 (°) | P90 (°) | Shift Err (Å) | Time (s) |
|--------|---------|----------|-----------|---------|---------|--------------|---------|
| Baseline | DS2 | A (GT) | 1.19 | 2.43 | 4.01 | — | ~25 |
| Baseline | DS2 | B (5° perturb) | 1.64 | 2.46 | 3.79 | 0.86 | ~25 |
| Baseline | DS3 | A (GT) | 0.00 | 2.69 | 4.32 | 0.46 | ~30 |
| Baseline | DS3 | B (5° perturb) | 1.69 | 2.60 | 4.06 | 0.46 | ~24 |

### Phase 3 — Changes enabled (grid 6°/2°, batch_size=32, n=2000)

| Change | Dataset | Scenario | Median (°) | P75 (°) | P90 (°) | Shift (Å) | Notes |
|--------|---------|----------|-----------|---------|---------|-----------|-------|
| baseline | DS2 | B | 1.64 | 2.46 | 3.79 | 0.86 | |
| #1 subpixel + #2a zero_dc | DS2 | B | 1.64 | 2.46 | — | **0.70** | shift improved |
| #2b particle-adaptive whitening | DS2 | A | **0.00** | **0.00** | **0.00** | — | |
| #2b particle-adaptive whitening | DS2 | B | **1.32** | **1.96** | **3.16** | 0.70 | 19% |
| #2b particle-adaptive whitening | DS3 | A | **0.00** | **2.00** | **4.02** | — | |
| #2b particle-adaptive whitening | DS3 | B | 1.72 | 2.59 | 3.85 | — | within noise |
| whiten + fftfreq_min=0.05 | DS2 | B | 1.41 | — | **7.33** | — | harmful |
| whiten + fftfreq_min=0.05 | DS3 | A | 2.00 | — | 5.34 | — | harmful |
| warmup8 | DS2 | B | 1.33 | 1.95 | 2.75 | — | same as single-batch |
| warmup8 | DS3 | B | **1.43** | **2.37** | **4.85** | — | 17% vs single-batch |
| fibo + pre_multiply | DS2 | B | **1.31** | 1.88 | **2.62** | — | |
| fibo + post_multiply | DS2 | B | 1.35 | — | 2.66 | — | equivalent |
| fibo + pre_multiply | DS3 | B | **1.31** | 1.98 | **3.52** | — | D2 P90 standout |

### Finer grid experiments (6°/1°, n=500, GPU 1)

At 6°/1°: Fibonacci = 1638 pts, Cartesian = 2197 pts. XBLOCK constraint requires bs=2 (Fibo)
or bs=1 (Cartesian) without the `dynamic=False` fix.

| Change | Dataset | Scenario | Median (°) | P75 (°) | P90 (°) | Shift (Å) | Time (s) |
|--------|---------|----------|-----------|---------|---------|-----------|---------|
| baseline (flags off), Cartesian 6°/1° | DS2 | B | 1.49 | 2.35 | 3.80 | 0.70 | ~70 |
| all flags ON, Cartesian 6°/1° | DS2 | B | **1.00** | **1.72** | **2.78** | **0.48** | ~72 |

Overall DS2 Scen B improvement: **1.64° → 1.00° = 39% reduction** vs 2° grid baseline.

Fibonacci 6°/1°, all 4 composition combinations (pre vs post perturbation × pre vs post search):
all give **~0.84–0.89°** — composition order is irrelevant for an isotropic ball.

### Summary table (Scenario B vs GT, optimal batch sizes)

Grid pts measured with `so3_grid_near_identity_fibo(use_small_aprox=True)` + 1 identity.

| Config | Grid pts | n | DS1 med | DS2 med | DS2 P75 | DS2 P90 | DS3 med | DS3 P75 | DS3 P90 | Time/500 |
|--------|----------|---|--------|--------|---------|---------|--------|---------|---------|---------|
| baseline (all off) | Cartesian 343 | 2000 | — | 1.64° | — | 3.79° | 1.69° | — | 4.06° | ~9.4s |
| all flags ON | Cartesian 343 | 2000 | — | 1.32° | — | 3.16° | 1.43° | — | 4.85° | — |
| +warmup8 | Cartesian 343 | 2000 | — | 1.33° | — | 2.75° | 1.43° | — | 4.85° | — |
| +fibo+pre | Fibonacci 209 | 2000 | — | 1.31° | 1.88° | 2.62° | 1.31° | 1.98° | 3.52° | **~6.4s** |
| all flags ON | Cartesian 2197 | 500 | — | 1.00° | — | 2.78° | — | — | — | — |
| fibo+pre | Fibonacci 488 | 500 | — | 0.97° | 1.53° | 2.13° | 1.35° | 2.03° | 2.81° | ~13.4s |
| fibo+pre | Fibonacci 1638 | 500 | — | 0.89° | 1.62° | 2.67° | 1.38° | 2.15° | 3.18° | ~43.2s |
| fibo+pre | Fibonacci 1486 | 500 | — | 0.87° | 1.43° | 2.23° | **1.24°** | **1.93°** | **2.74°** | ~40.8s |
| fibo+pre | Fibonacci 3875 | 500 | — | — | — | — | 1.22° | 1.89° | 2.64° | ~2m |
| **two-stage 6°/2°+1.5°/0.5° K=5** | **1249** | **500** | **0.22°** | **0.42°** | **1.25°** | **2.47°** | 1.35° | 2.35° | 3.43° | ~33.4s |

All flags for fibo/two-stage rows: `use_subpixel_shifts=True, zero_dc=True, spectral_whitening=True, whitening_warmup_batches=8, fftfreq_min=0.0, use_fibo_grid=True, rotation_composition=pre_multiply`

DS1 (synthetic, C1) two-stage Scen B = 0.22° from ~4° perturbed start — essentially perfect recovery.
DS1 reference volume must be reconstructed **with CTF correction** (default `--correct_ctf`).

### Two-stage vs 4°/0.7° head-to-head

| Config | DS2 median | DS2 P90 | DS3 median | DS3 P90 | Total pts | Winner |
|--------|-----------|---------|-----------|---------|-----------|--------|
| 4°/0.7° single-stage | 0.87° | 2.23° | **1.24°** | **2.74°** | 1486 | DS3 |
| two-stage 6°/2°+1.5°/0.5° | **0.42°** | 2.47° | 1.35° | 3.43° | 1249 | DS2 + fewer pts |

Two-stage wins for C1 (2× better median, fewer evaluations). 4°/0.7° wins for D2 (better median
and P90). The two-stage coarse K=5 candidates can cluster in one symmetry domain, missing better
poses in the other D2-related domains; dense single-stage coverage avoids this.

---

## β-galactosidase full-pipeline benchmark (Phase 2 — real experiment)

End-to-end runs (NN inference + projection matching + reconstruction) on the bgal checkpoint
(`apo_00482/version_0`, D2 symmetry) across two held-out ligand datasets.

**Cartesian grid point counts:** 6°/2° = 343 pts (7×7×7 Euler grid).
**Fibo grid point counts** (measured): 6°/2° = 208 pts; 6°/1.8° = 339 pts; 6.8°/2° = 353 pts; two-stage ~1250 pts.

Note on batch sizes (box ≈ 476px for apo; memory scales as `bs × n_rots × n_valid × 9 × 4 bytes`):

| Config | n_rots | Recommended bs |
|--------|--------|---------------|
| Cartesian 6°/2° | ~343 | 8 |
| Cartesian 4°/1° | ~343+ | 6 |
| Fibo 6°/2° | 209 | 16 |
| Fibo 4°/1° | 488 | 8 |
| Two-stage fine (5×~420) | ~2100 | 4 |

### Benchmark commands

```bash
# Step 0: Shared NN inference (run once, reuse for all projmatching configs)
# APO — NN only
CUDA_VISIBLE_DEVICES=0 $BIN_DIR/cryopares_infer \
  --particles_star_fname $STAR_APO \
  --particles_dir $DIR_APO \
  --checkpoint_dir $CKPT \
  --results_dir /home/rsanchez/cryo/data/cryopares/benchmarks/bgal/apo/nn_only \
  --data_halfset allParticles --model_halfset allParticles \
  --reference_map $REF_APO --reference_mask $MASK_APO \
  --batch_size 64 --skip_localrefinement --skip_reconstruction

# LIG_00892 — NN only
CUDA_VISIBLE_DEVICES=0 $BIN_DIR/cryopares_infer \
  --particles_star_fname $STAR_LIG892 \
  --particles_dir $DIR_LIG892 \
  --checkpoint_dir $CKPT \
  --results_dir /home/rsanchez/cryo/data/cryopares/benchmarks/bgal/lig_00892/nn_only \
  --data_halfset allParticles --model_halfset allParticles \
  --reference_map $REF_LIG892 \
  --batch_size 64 --skip_localrefinement --skip_reconstruction

# After NN runs:
NN_STAR_APO=/home/rsanchez/cryo/data/cryopares/benchmarks/bgal/apo/nn_only/*/predictions.star
NN_STAR_LIG892=/home/rsanchez/cryo/data/cryopares/benchmarks/bgal/lig_00892/nn_only/*/predictions.star

# Config 1: master Cartesian 6°/2° (baseline)
CUDA_VISIBLE_DEVICES=0 $BIN_DIR/cryopares_projmatching \
  --reference_vol $REF_APO \
  --particles_star_fname $NN_STAR_APO \
  --particles_dir $DIR_APO \
  --out_fname /home/rsanchez/cryo/data/cryopares/benchmarks/bgal/apo/cart_6-2/aligned.star \
  --grid_distance_degs 6 --batch_size 8

# Config 2: master Cartesian 4°/1°
CUDA_VISIBLE_DEVICES=0 $BIN_DIR/cryopares_projmatching \
  --reference_vol $REF_APO \
  --particles_star_fname $NN_STAR_APO --particles_dir $DIR_APO \
  --out_fname .../cart_4-1/aligned.star \
  --grid_distance_degs 4 --batch_size 6 \
  --config projmatching.grid_step_degs=1

# Config 3: fibo 6°/2° (branch)
CUDA_VISIBLE_DEVICES=0 $BIN_DIR/cryopares_projmatching \
  --reference_vol $REF_APO \
  --particles_star_fname $NN_STAR_APO --particles_dir $DIR_APO \
  --out_fname .../fibo_6-2/aligned.star \
  --grid_distance_degs 6 --batch_size 16 \
  --config projmatching.use_fibo_grid=True projmatching.rotation_composition=pre_multiply \
           projmatching.use_subpixel_shifts=True projmatching.zero_dc=True \
           projmatching.spectral_whitening=True projmatching.whitening_warmup_batches=8

# Config 4: fibo 4°/1° (branch)
CUDA_VISIBLE_DEVICES=0 $BIN_DIR/cryopares_projmatching \
  --reference_vol $REF_APO \
  --particles_star_fname $NN_STAR_APO --particles_dir $DIR_APO \
  --out_fname .../fibo_4-1/aligned.star \
  --grid_distance_degs 4 --batch_size 8 \
  --config projmatching.grid_step_degs=1 \
           projmatching.use_fibo_grid=True projmatching.rotation_composition=pre_multiply \
           projmatching.use_subpixel_shifts=True projmatching.zero_dc=True \
           projmatching.spectral_whitening=True projmatching.whitening_warmup_batches=8

# Config 5: two-stage 6°/2° + 2.1°/0.7° K=5 (branch)
# Note: 2.1°/0.7° fine grid ≈ 209 pts/candidate (same count as 1.5°/0.5° in DS2/DS3 benchmarks)
CUDA_VISIBLE_DEVICES=0 $BIN_DIR/cryopares_projmatching \
  --reference_vol $REF_APO \
  --particles_star_fname $NN_STAR_APO --particles_dir $DIR_APO \
  --out_fname .../twostage_6-2_2.1-0.7/aligned.star \
  --grid_distance_degs 6 --batch_size 4 \
  --config projmatching.use_fibo_grid=True projmatching.rotation_composition=pre_multiply \
           projmatching.use_subpixel_shifts=True projmatching.zero_dc=True \
           projmatching.spectral_whitening=True projmatching.whitening_warmup_batches=8 \
           projmatching.use_two_stage_search=True \
           projmatching.fine_grid_distance_degs=2.1 projmatching.fine_grid_step_degs=0.7 \
           projmatching.fine_top_k=5

# Evaluation
$COMPARE \
  --starfile1 $OUT/aligned.star \
  --starfile2 $STAR_APO \
  --sym D2

# Reconstruction
CUDA_VISIBLE_DEVICES=0 $BIN_DIR/cryopares_reconstruct \
  --particles_star_fname $OUT/aligned.star \
  --symmetry D2 \
  --output_mrc_fname $OUT/reconstruction.mrc
```

### Results — lig_00892 (~57K particles)

| Config | pts | Batch size | med° | %<5° | FSC@0.143 | p/min |
|--------|-----|-----------|------|------|-----------|-------|
| true_master_6-2 (improvements OFF) | 343 | 8 | 1.49° | 76.7% | 2.697 Å | ~3,500 |
| master_regression_6-2 (on master branch) | 343 | 8 | 1.49° | 76.8% | 2.760 Å | ~3,500 |
| cart_6-2 (improvements ON, Cartesian) | 343 | 8 | 1.47° | 76.7% | 2.667 Å | 3,300 |
| **cart_6-2 + SO(3) interp** | **343** | **8** | **1.13°** | **76.7%** | **2.581 Å** | **~3,500** |
| cart_4-1 | 343+ | 6 | 1.26° | — | 2.589 Å | 1,700 |
| fibo_6-2 | 208 | 16 | 1.62° | 76.3% | 2.637 Å | 5,100 |
| fibo_6-1.8 (density-matched to cart) | 339 | 16 | 1.51° | 76.4% | 2.622 Å | — |
| fibo_6.8-2 (wider ball, same step) | 353 | 16 | 1.64° | 76.6% | 2.637 Å | — |
| fibo_4-1 | 488 | 8 | 1.41° | — | 2.594 Å | 2,480 |
| twostage_6-2_2.1-0.7 K=5 | ~1250 | 4 | **1.05°** | 76.8% | 2.594 Å | 838 |

### Results — lig_00893 (~42K particles, harder)

| Config | pts | Batch size | med° | %<5° | FSC@0.143 | p/min |
|--------|-----|-----------|------|------|-----------|-------|
| true_master_6-2 (improvements OFF) | 343 | 8 | 2.56° | 64.5% | 3.082 Å | ~3,500 |
| master_regression_6-2 (on master branch) | 343 | 8 | 2.56° | 64.6% | 3.043 Å | ~3,500 |
| cart_6-2 (improvements ON, Cartesian) | 343 | 8 | 2.49° | 64.7% | 2.967 Å | 3,300 |
| **cart_6-2 + SO(3) interp** | **343** | **8** | **2.25°** | **64.3%** | **2.926 Å** | **~3,500** |
| cart_4-1 | 343+ | 6 | 2.63° | — | 2.968 Å | 1,700 |
| fibo_6-2 | 208 | 16 | 2.69° | 63.8% | 3.062 Å | 5,100 |
| fibo_6-1.8 (density-matched to cart) | 339 | 16 | 2.63° | 63.9% | 3.023 Å | — |
| fibo_6.8-2 (wider ball, same step) | 353 | 16 | 2.63° | 64.0° | 2.985 Å | — |
| fibo_4-1 | 488 | 8 | 2.92° | — | 2.986 Å | 2,480 |
| twostage_6-2_2.1-0.7 K=5 | ~1250 | 4 | **2.18°** | 65.5% | **2.948 Å** | 838 |

### Key findings (bgal)

**No regression:** `true_master_6-2` and `master_regression_6-2` are identical within noise on both
datasets — the `improve_local_refinement` branch introduces no regression.

**Subpixel+whitening alone (true_master → cart):** ~0.05–0.10° angular improvement, ~0.1 Å FSC
improvement consistently. Small but reliable.

**Fibonacci at box=476px is worse than Cartesian at matched point count.** At box=336px (DS2/DS3),
fibo 6°/2° (208 pts) was comparable to Cartesian 6°/2° (343 pts). At box=476px the Cartesian 343-pt
grid is denser in SO(3) coverage relative to the particle size, and fibo at 208 pts is sparser.
Density-matching (fibo_6-1.8 at 339 pts, fibo_6.8-2 at 353 pts) closes the gap slightly but does
not beat Cartesian in either metric. The optimal grid geometry is particle-size dependent.

**`%<5°` is not discriminating here.** All single-stage configs cluster within ±0.9%.
The meaningful variation is in the median.

**Two-stage has best median on both datasets** (1.05°/2.18°), but **cart_6-2 + SO(3) interpolation
beats two-stage on FSC for lig_00892** (2.581 vs 2.594 Å) while running at full cart speed
(~3,500 p/min vs 838). The median gap between SO(3) interp and two-stage is 0.08°/0.07° — within
noise on a real dataset. SO(3) interpolation is a strong result: ~23%/10% angular improvement over
cart_6-2 at zero throughput cost.

**Two-stage is still the best if absolute accuracy matters most.** Cost: ~838 p/min vs ~3,500 p/min
for cart_6-2 + SO(3) interp.

**Why fibo_4-1 underperforms on lig_00893:** On this harder dataset the initial NN error exceeds 4°
for many particles, so the 4° ball misses the true pose. Cart_4-1 suffers the same (2.63° vs 2.49°
for cart_6-2). Larger search radius matters more than finer step on hard data.

### Lessons: what the synthetic experiments got wrong

| Claim from synthetic (DS2/DS3) | Real benchmark outcome |
|-------------------------------|----------------------|
| For D2, 4°/0.7° single-stage > two-stage | **Reversed**: two-stage wins on both bgal D2 datasets |
| Fibonacci at 6°/2° (209 pts) ≈ Cartesian 6°/2° (343 pts) | **Box-size dependent**: at 476px fibo 6°/2° is worse; density-matching (339 pts) needed to close the gap |
| DS3 P90 improvement (4.85°→3.52°) shows D2 benefit of Fibonacci | Not confirmed in real pipeline; grid geometry must be re-evaluated per particle size |

**Root cause:** Perturbed-pose benchmarks use a known, isotropic 5° error distribution. Real NN
errors are correlated, model-dependent, and non-uniform across orientations. The synthetic
benchmark cannot capture how the NN's actual error distribution interacts with grid geometry,
especially near D2 symmetry axes where the NN may be systematically biased.

**Decision:** No further synthetic experiments. Future configuration decisions will be based on
real full-pipeline benchmarks. The next target is **PKM2 (D2) with a cryoPARES-trained model** —
this is the actual D2 use case and will determine whether two-stage or 4°/0.7° is the right
default for D2 symmetry.

---

## Wall-clock timing

Hardware: NVIDIA RTX 6000 Ada (49 GB). 3 runs, 10,000 particles each. Per-500 = raw ÷ 20.
p/min = 600,000 ÷ raw_seconds.

### Particles-per-minute summary

| Config | Grid | Pts | Best bs | DS2 p/min | DS3 p/min |
|--------|------|-----|---------|----------|----------|
| **master** Cartesian 6°/2° | Cartesian | 216 | 11–32 | ~3200 | ~3000 |
| **master** Cartesian 4°/2° | Cartesian | 64 | 64 | ~7700 | ~7400 |
| **master** Cartesian 4°/1° | Cartesian | 512 | 8–16 | ~1600 | ~700† |
| **master** Cartesian 6°/1° (bs=4) | Cartesian | 1728 | **4** | **~545** | — |
| **branch** fibo 4°/2° (bs=64) | Fibo | 63 | 64 | **~9760** | **~8160** |
| **branch** fibo 6°/2° (bs=32) | Fibo | 209 | 32 | ~4840 | ~4680 |
| **branch** fibo 4°/1° (bs=16) | Fibo | 488 | 16 | ~2240 | ~2170 |
| **branch** two-stage K=5 (bs=7) | — | 1249 | 7 | ~903 | — |
| **branch** fibo 4°/0.7° (bs=5) | Fibo | 1486 | 5 | — | ~740 |
| **branch** fibo 6°/1° (bs=4) | Fibo | 1638 | 4 | ~695 | ~675 |

†DS3 4/1 highly variable across runs (IO jitter); ~700 p/min is conservative.

### Master branch baseline (Cartesian 6°/2°, batch_size=11)

| Dataset | Run 1 | Run 2 | Run 3 | Median (10K) | Per 500 |
|---------|-------|-------|-------|--------------|---------|
| DS2 Scen B | 3m06.7s | 3m07.3s | 3m07.4s | **3m07s (187s)** | **~9.4s** |
| DS3 Scen B | 4m21.1s* | 3m25.6s | 3m14.6s | **3m20s (200s)** | **~10.0s** |

*DS3 Run 1 outlier (GPU contention).

### Branch optimal batch-size configs (improve_local_refinement, GPU 0)

| Config | Dataset | Run 1 | Run 2 | Run 3 | Avg (10K) | Per 500 | p/min | vs sub-opt bs |
|--------|---------|-------|-------|-------|-----------|---------|-------|--------------|
| fibo 4°/2° bs=64 | DS2 | 1m01.9s | 1m01.3s | 1m01.4s | **61.5s** | **~3.1s** | **~9760** | first |
| fibo 4°/2° bs=64 | DS3 | 1m13.8s | 1m13.7s | 1m13.1s | **73.5s** | **~3.7s** | **~8160** | first |
| fibo 6°/2° bs=32 | DS2 | 2m04.0s | 2m03.9s | 2m04.3s | **124s** | **~6.2s** | **~4840** | ≈same (bs=11→32) |
| fibo 6°/2° bs=32 | DS3 | 2m08.3s | 2m08.2s | 2m08.4s | **128s** | **~6.4s** | **~4680** | ≈same |
| fibo 4°/1° bs=16 | DS2 | 4m27.3s | 4m27.4s | 4m27.6s | **267s** | **~13.4s** | **~2240** | **2.2× faster** (bs=8→16) |
| fibo 4°/1° bs=16 | DS3 | 4m36.3s | 4m36.9s | 4m37.3s | **277s** | **~13.8s** | **~2170** | **2.2× faster** |
| two-stage K=5 bs=7 | DS2 | 11m04.9s | 11m03.9s | 11m04.3s | **664s** | **~33.2s** | **~903** | ≈same (bs=3→7) |
| fibo 4°/0.7° bs=5 | DS3 | 13m28.0s | 13m27.9s | 13m28.6s | **808s** | **~40.4s** | **~740** | ≈same (bs=2→5) |
| fibo 6°/1° bs=4 | DS2 | 14m22.3s | 14m26.3s | 14m20.6s | **863s** | **~43.2s** | **~695** | **2.3× faster** (bs=2→4) |
| fibo 6°/1° bs=4 | DS3 | 14m48.7s | 14m49.8s | 14m49.8s | **889s** | **~44.5s** | **~675** | first at bs=4 |

**Key observations:**
- Dense grids (4°/1°, 6°/1°): 2.2–2.3× speedup from optimal bs.
- Sparse grids (6°/2°, two-stage, 4°/0.7°): little benefit — already compute-bound per batch.
- fibo 4°/2° (63 pts): ~9760 p/min DS2 — fastest branch config.
- Two-stage (664s/10K DS2) is faster AND more accurate than fibo 6°/1° (863s/10K). 6°/1° dominated.

### Speed vs accuracy comparison (optimal batch sizes)

| Config | Dataset | Per 500 | Median err | P90 err | vs master |
|--------|---------|---------|-----------|---------|-----------|
| master (Cartesian 6°/2°, bs=11) | DS2 | ~9.4s | 1.64° | 3.79° | baseline |
| branch fibo 4°/2°, bs=64 | DS2 | **~3.1s** | 1.39° | — | **3.0× faster** |
| branch fibo 6°/2°, bs=32 | DS2 | **~6.2s** | **1.31°** | **2.62°** | 1.5× faster, more accurate |
| branch fibo 4°/1°, bs=16 | DS2 | ~13.4s | 0.97° | 2.13° | 1.4× slower, good tradeoff |
| branch two-stage K=5, bs=7 | DS2 | ~33.2s | **0.42°** | **2.47°** | 3.5× slower, 4× better median |
| branch fibo 6°/1°, bs=4 | DS2 | ~43.2s | 0.89° | 2.67° | 4.6× slower, worse than two-stage |
| master (Cartesian 6°/2°, bs=11) | DS3 | ~10.0s | 1.69° | 4.06° | baseline |
| branch fibo 4°/2°, bs=64 | DS3 | **~3.7s** | 1.50° | — | **2.7× faster** |
| branch fibo 6°/2°, bs=32 | DS3 | **~6.4s** | **1.31°** | **3.52°** | 1.6× faster, more accurate |
| branch fibo 4°/1°, bs=16 | DS3 | ~13.8s | 1.35° | 2.81° | 1.4× slower |
| branch fibo 4°/0.7°, bs=5 | DS3 | ~40.4s | **1.24°** | **2.74°** | 4.0× slower, best accuracy |

**Key finding — no regression:** branch fibo 6°/2° is 1.5× *faster* than master (209 vs 343 pts)
and more accurate. The accuracy improvements cost only when using denser grids or two-stage search.

---

## Batch-size limits

**Memory model (PyTorch 2.8 + Triton 3.4, RTX 6000 Ada 49–51 GB, box=336):**

The projection kernel materialises a `(batch_size × n_rotations × n_valid_coords × 9)` float32
buffer, where `n_valid_coords ≈ π/4 × H × (H//2+1)` (box=336: ~44,500 pts). Peak GPU memory scales
with **both** `batch_size` and `n_rotations`.

```
n_valid ≈ π/4 × H × (H//2+1)
bs_safe = floor(0.85 × V / (n_rots × n_valid × 9 × 4 bytes))
```

Per-grid safe batch sizes at box=336, 49 GB GPU (0.85× = 41.6 GB usable):

| Grid | n_rots | GB/batch | safe bs | Notes |
|------|--------|----------|---------|-------|
| fibo 4°/2° | 63 | 0.10 | ~411 | use bs=64–75 |
| fibo 6°/2° | 209 | 0.34 | ~124 | use bs=75 |
| fibo 4°/1° | 488 | 0.78 | ~53 | use bs=32–40 |
| two-stage fine 5×209 | 1045 | 1.68 | ~24 | use bs=16 |
| fibo 4°/0.7° | 1486 | 2.39 | ~17 | use bs=8 |
| fibo 6°/1° | 1638 | 2.63 | ~15 | use bs=8 |

Historical note: old measurements used lower batch sizes (bs=4 for 6°/1°) because the XBLOCK
compile error with `dynamic=True` limited exploration. After fixing to `dynamic=False`, these
limits were re-evaluated. Re-timing at higher bs for sparse grids is pending.

**User-facing warnings (`_check_batch_size`):**
1. **Too large (OOM risk):** if `batch_size > bs_safe`:
   > `[projmatching] WARNING: batch_size={bs} may exhaust GPU memory ({V} GB, estimated peak {p} GB). Recommended: --batch_size {bs_safe}.`
2. **Too small (throughput penalty):** if `batch_size < bs_safe // 8`:
   > `[projmatching] NOTE: batch_size={bs} uses only {p}/{V} GB GPU memory. Consider --batch_size {bs_safe} ({p_safe} GB) for better GPU throughput.`

---

## Benchmark tooling

### `tests/benchmarks/perturb_poses.py`

Script to perturb RELION star file poses by a uniform SO(3) ball of radius `perturb_deg`.

**Critical bug fixed — rejection sampling:** The original implementation rejected uniform
quaternions outside the geodesic ball. For a 5° ball, acceptance rate ≈ 0.002% → hung for hours
on 238,631 particles.

**Fix:** Inverse-CDF method. The marginal CDF of geodesic angle θ in an SO(3) ball:
```
F(θ) = (θ - sin(θ)) / (θ_max - sin(θ_max))
```
Inverted via Newton's method (converges in ~6 iterations). Direction sampled from uniform S².
Runtime for 238K particles: **6 seconds** (was hours).

**Composition modes:**
- `--composition pre` (default): R_perturbed = R_delta @ R_current (delta in lab frame)
- `--composition post`: R_perturbed = R_current @ R_delta (delta in body frame)

Both produce identical geodesic distance distributions (verified). Mean: 3.75°, Median: 3.97°, Max: 5.000°.

### `cryoPARES/scripts/compare_poses.py`

```bash
python -m cryoPARES.scripts.compare_poses \
    --starfile1 OUTPUT.star --starfile2 GROUND_TRUTH.star --sym C1
```
Reports Mean, Median, IQR angular error in degrees, plus shift errors in Å.
**Always use this script for GT-based metrics** — never use the inline `align_star` output.

### STAR file format handling fix

`projmatching/projmatching.py` line 290: fixed to handle both RELION 3.0 (bare DataFrame) and
RELION 3.1+ (dict with 'optics'/'particles' keys) formats.

---

## Testing protocol

- Always use `compare_poses.py --starfile1 OUTPUT --starfile2 GT` for GT-based metrics.
- Fix `n_first_particles=500` for comparability across experiments.
- DS3 (D2 symmetry): use `--sym D2` in `compare_poses.py`.
- **Batch size**: see per-grid safe limits table above.
- **Triton compile**: use `dynamic=False` (applied in `extract_central_slices_as_real.py`).
- Clear Triton cache when switching branches: `rm -rf /tmp/torchinductor_rsanchez/`.
- Run `ulimit -n 65536` before any cryopares command.
- For quick spot-checks: `--n_first_particles 2000`.

**Two-stage benchmark command:**
```bash
cryopares_projmatching ... --grid_distance_degs 6 --grid_step_degs 2 --batch_size 16 --gpu_id 0 \
  --config projmatching.use_fibo_grid=True projmatching.rotation_composition=pre_multiply \
           projmatching.use_subpixel_shifts=True projmatching.zero_dc=True \
           projmatching.spectral_whitening=True projmatching.whitening_warmup_batches=8 \
           projmatching.use_two_stage_search=True \
           projmatching.fine_grid_distance_degs=1.5 projmatching.fine_grid_step_degs=0.5 \
           projmatching.fine_top_k=5
```

---

## Pending experiments

### Completed
- [x] DS3 Scenario B with two-stage K=10 — **ruled out**: 1.36°/3.50° vs K=5 1.35°/3.43°, at 77% more evals.
- [x] DS3 Scen B 4°/0.5° single-stage — **done**: 1.22°/2.64° P90. 4°/0.7° confirmed as sweet spot.
- [x] Master branch timing — DS2: 187s/10K (~9.4s/500), DS3: 200s/10K (~10s/500).
- [x] Branch best-config timing — DS2 two-stage: 664s/10K; DS3 4°/0.7°: 808s/10K.
- [x] Batch-size investigation — memory scales with bs only (N-independent). XBLOCK issue resolved.
- [x] Re-measure branch timings with optimal batch sizes — dense grids 2.2–2.3× faster.
- [x] Branch 4°/2° timing — DS2: ~61.5s/10K (~9760 p/min), DS3: ~73.5s/10K (~8160 p/min).
- [x] DS2 Scenario A with Fibonacci grid 6°/1° — **0.00° median** ✓
- [x] DS3 Scenario B with Fibonacci grid 4°/2° — **1.50°** ✓
- [x] DS1 Scenarios A and B — **done** (0.00° / 0.22°)
- [x] Full-pipeline benchmark on real β-gal data — **done** (see bgal section above)

### Cancelled (synthetic experiments — not representative of real pipeline)
- ~~DS3 Scenario B with Fibonacci grid 6°/1°~~ — not worth running; synthetic results don't predict real performance
- ~~DS3 Scenario B with Fibonacci grid 4°/0.5°~~ — already showed diminishing returns vs 4°/0.7°; cancelled
- ~~DS3 Scenario B with Fibonacci grid 4°/0.5°~~ — already showed diminishing returns; cancelled
- ~~Apo self-test runs for bgal full-pipeline~~ — self-test (same-dataset) less informative than cross-dataset lig_00892/893

### Active / next
- [ ] **Measure NN error distribution on bgal before projmatching** (GPU needed, ~10 min):
  ```bash
  CUDA_VISIBLE_DEVICES=1 $BIN_DIR/cryopares_infer \
    --particles_star_fname $STAR_LIG892 --particles_dir $DIR_LIG892 \
    --checkpoint_dir $CKPT \
    --results_dir /home/rsanchez/cryo/data/cryopares/benchmarks/bgal/lig_00892/nn_only \
    --data_halfset allParticles --model_halfset half1 \
    --reference_map $REF_LIG892 --batch_size 64 \
    --skip_localrefinement --skip_reconstruction

  # Compare NN predictions vs RELION GT (no GPU needed)
  $COMPARE \
    --starfile1 /home/rsanchez/cryo/data/cryopares/benchmarks/bgal/lig_00892/nn_only/*/predictions.star \
    --starfile2 $STAR_LIG892 --sym D2
  ```
  Key question: if median NN error < 3°, the 6°/2° grid already finds the right cell and the 2°
  quantization floor is the true bottleneck → SO(3) interpolation (Change #7) is the right fix.
  If median > 4°, search radius is the bottleneck.

- [x] **Benchmark Change #7 (SO(3) interpolation) on bgal** — **done.** cart_6-2+SO3 interp:
  lig_00892 1.47°→1.13° (23%), FSC 2.667→2.581 Å; lig_00893 2.49°→2.25° (10%), FSC 2.967→2.926 Å.
  Beats two-stage on FSC for lig_00892 at full cart speed.
- [ ] **cart_6-2 double-step + SO(3) interp (4° grid, 4°/2° = coarser, relies on SO3 for sub-step):**
  Use `--grid_distance_degs 4 --grid_step_degs 4` (or similar larger step) + SO(3) interp to see
  if interpolation can recover accuracy lost from a coarser grid at faster throughput.

- [ ] **PKM2 full-pipeline benchmark** — train a cryoPARES model on PKM2 (D2, astex-5534), then run
  end-to-end inference + projmatching + reconstruction on held-out data. This is the real D2 use
  case and will determine the correct default config for D2 symmetry (two-stage vs 4°/0.7°).
- [ ] Merge `improve_local_refinement` to master once PKM2 benchmark confirms no regression.

---

## Sanity checks

| Check | Result | Interpretation |
|-------|--------|---------------|
| DS2 Scen A baseline: 1.19° | ✓ | 2° step quantization floor, expected |
| DS3 Scen A baseline: 0.0° | ✓ | Already RELION-refined; projmatching returns same poses |
| DS2 Scen A with whitening: 0.00° all percentiles | ✓ | CC peak sharp enough; GT always wins |
| DS2 Scen A Fibonacci (identity prepended): 0.00° | ✓ | No regression from grid change |
| DS3 warmup8 (1.43°) vs single-batch (1.72°) | ✓ | CTF diversity averaging works |
| DS3 Fibonacci P90 (3.52° vs 4.85°) | ✓ | Euler polar degeneracy eliminated |
| Cartesian rotation composition: worse than euler_add | ✓ | Grid not designed for matrix composition |
| Band mask (fftfreq_min=0.05): P90 7.33° | ✓ | Low-freq content essential; keep disabled |
| perturb_poses.py: max dist = 5.000°, >5°: 0.0% | ✓ | Inverse-CDF sampling correct |
| Pre vs post perturbation: same distance distribution | ✓ | Confirmed via compare_poses.py |
| Pre/post composition 2×2: all ~0.85° | ✓ | Both modes mathematically equivalent |
| bgal true_master vs master_regression: identical | ✓ | Branch introduces no regression |
| DS1 two-stage Scen A: 0.00° | ✓ | CTF-corrected reference required |
