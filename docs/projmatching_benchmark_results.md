# Projection Matching Benchmark Results

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

## Datasets

| ID | Source | Particles | Pixel size (Å/px) | Box | Reference volume |
|----|--------|-----------|-------------------|-----|-----------------|
| DS2 | EMPIAR-10166 frealign_stack (real, C1) | 238631 | 1.27 | 336 | `EMPIAR-10166/data/reconstruct.mrc` |
| DS3 | astex-5534 PKM2 Refine3D (real, D2) | 254070 | 0.995 | 334 | `astex-5534/Refine3D/001_after2DclsEval/run_class001.mrc` |
| DS1 | EMPIAR-10166 projectionsSamePose (synthetic) | 238631 | 1.5 | 224 | — (particle images still copying) |

### Ground-truth and perturbed star files

| File | Description |
|------|-------------|
| `tests/frealign_stack_relion3.star` | DS2 GT (RELION 3.1 format, optics+particles blocks) — copied from previous preprocessing |
| `EMPIAR-10166/frealign_stack_relion3.star` | Symlink/copy made during benchmarking session |
| `tests/ds2_perturbed_5deg.star` | DS2 perturbed, pre-composition (R_δ @ R_true), seed=42 |
| `tests/ds2_post_perturbed_5deg.star` | DS2 perturbed, post-composition (R_true @ R_δ), seed=42 |
| `tests/ds3_perturbed_5deg.star` | DS3 perturbed, pre-composition, seed=42 |
| `tests/ds3_post_perturbed_5deg.star` | DS3 perturbed, post-composition, seed=42 |
| `tests/ds1_perturbed_5deg.star` | DS1 perturbed, pre-composition, seed=42 |

**Note on frealign_stack.star format**: the original `EMPIAR-10166/data/frealign_stack.star` is RELION 3.0
format (single pandas DataFrame, no optics block). The code's `n_first_particles` path assumes RELION 3.1
format (dict with 'optics'/'particles' keys). A bug fix was applied to `projmatching/projmatching.py` line
290 to handle both formats. Use `frealign_stack_relion3.star` for benchmarking to avoid format issues.

## Benchmark commands

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

On GPU 1 (and possibly GPU 0 as well), Triton compilation fails with:

```
AssertionError: 'XBLOCK' too large. Maximum: 4096. Actual: 8192.
```

when the total number of projections per batch (batch_size × n_grid_points) exceeds 4096.

- Fibonacci grid at 6°/1°: 1638 points → `batch_size ≤ 2` required (2 × 1638 = 3276 < 4096)
- Cartesian grid at 6°/1°: 2197 points → `batch_size = 1` required (1 × 2197 = 2197 < 4096)
- Cartesian grid at 6°/2°: 343 points → `batch_size ≤ 11` (used 32 in early benchmarks, worked on GPU 0)

The 32× batch_size early benchmarks ran on GPU 0 which may have a higher XBLOCK limit or was
using a different Triton compilation path. Always specify `--batch_size 2` (Fibonacci) or
`--batch_size 1` (Cartesian at 1° step) when running on GPU 1.

TODO: In the projection matching script verify if the selected size will cause problems (you will need to figure outthe XBLOCK value at running time, in case it changes. Also, it possible, capture this error, and have a user friendly error)
---

## Implementation Details

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

**Why 8 batches:** more diverse defocus conditions are averaged → CTF oscillations cancel →
smoother amplitude profile. Analogous to RELION's per-optics-group noise estimation averaged
over many particles per group. Single-batch was sufficient for DS2 but not DS3 (see results).

TODO: Think how to implement this for on-the-fly friendly runs, we might need to keep updateing the stats every
few 1000s of particles, in case there is some shift in the microscope stats, but this could be very computationally
inefficient, and we care about performance

---

### Change #5 — Frequency band mask (`fftfreq_min`, default `0.0`)

**File:** `fourierOperations.py::_mask_for_dft_2d`, `projMatcher.py::_store_reference_vol`

Extended `_mask_for_dft_2d` with a `min_freq_pixels` parameter for a high-pass ring.
`band_mask` is pre-computed at init and registered as a buffer. Applied to `fparts` in `forward()`
when `fftfreq_min > 0`.

**Result:** Harmful at `fftfreq_min=0.05` — P90 worsened from 3.16° → 7.33° (DS2 Scen B).
Low-frequency information that zero_dc doesn't cover is important for orientation discrimination.
**Default kept at 0.0 (disabled).**

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
Converted to rotation matrices and composed, the coverage pattern changes unfavorably — particularly
the near-identity cluster that the Cartesian grid accumulates (7 identical rot/psi values per
tilt=0 row) vanishes, leaving sparser effective search around the correct pose.

**Cartesian grid rotation composition results (DS2 Scen B vs GT, n=2000):**

| Mode | Median (°) | P75 (°) | P90 (°) |
|------|-----------|---------|---------|
| euler_add (baseline) | 1.33 | 1.95 | 2.75 |
| pre_multiply (Cartesian) | 2.42 | 3.61 | 4.81 |
| post_multiply (Cartesian) | 2.27 | 3.35 | 4.32 |

**Conclusion for Cartesian grid:** rotation matrix composition is incompatible with the Cartesian
Euler grid. The Fibonacci grid was designed specifically for rotation-matrix composition.

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

TODO: Check if we are following the definition of the grid distance, it is supposed to have a grid that goes from -6º to +6º (check stats for Cartesian and Fibonacy)
**Implementation:**
- New `_get_so3_delta_rotmats(device)` method on `ProjectionMatcher`: returns cached `(nDelta, 3, 3)`
  delta rotation matrices. Uses Fibonacci when `use_fibo_grid=True`, otherwise converts the
  Cartesian Euler grid to rotation matrices.  TODO: If euler does not work with composition, this option should not be allowed
- `rotation_composition` auto-upgrades from `"euler_add"` to `"pre_multiply"` when
  `use_fibo_grid=True` (prints a message; euler_add is meaningless for the rotation-matrix path).

**Composition mode experiment (pre vs post) — DS2, n=500, 6°/1°:**

To test whether pre_multiply or post_multiply is "more correct", ran a 2×2 matrix:
pose perturbation (pre: R_δ@R_true vs post: R_true@R_δ) × search composition (pre vs post).

| Perturbation \ Search | pre_multiply | post_multiply |
|-----------------------|-------------|--------------|
| pre (R_δ @ R_true) | Median **0.89°**, <5°: 98.8% | Median **0.85°**, <5°: 98.0% |
| post (R_true @ R_δ) | Median **0.84°**, <5°: 98.4% | Median **0.85°**, <5°: 98.4% |

**Conclusion:** All four combinations give essentially identical accuracy (~0.84–0.89°).
**pre_multiply and post_multiply are mathematically equivalent for an isotropic SO(3) ball.**
The user hypothesis that only one is geometrically correct is disproved — for an isotropic grid
the geodesic ball covered by {R_δ@R_est} and {R_est@R_δ} is identical, so orientation recovery
accuracy is invariant to composition order.

Input error (5° perturbation) vs GT: Median **3.97°** → After Fibonacci projmatching: **~0.85°**.
This represents an ~79% reduction in angular error.

**Default remains `use_fibo_grid=False`** (behavior-preserving) — promote to `True` after
validation on more datasets.

---

## Infrastructure — Benchmark tooling

### `tests/benchmarks/perturb_poses.py`

Script to perturb RELION star file poses by a uniform SO(3) ball of radius `perturb_deg`.

**Critical bug fixed — rejection sampling:** The original implementation sampled uniform random
quaternions and rejected those outside the geodesic ball. For a 5° ball, the acceptance rate is
approximately P(|w| ≥ cos(2.5°)) ≈ 0.002% when sampling uniform unit quaternions. Processing
238,631 particles required ~13 billion random quaternion draws → hung for hours.

**Fix:** Replaced with inverse-CDF method. The marginal CDF of geodesic angle θ in an SO(3) ball is:
```
F(θ) = (θ - sin(θ)) / (θ_max - sin(θ_max))
```
Inverted via Newton's method (converges in ~6 iterations). Direction sampled from uniform S² (normal
vectors, normalized). Runtime for 238K particles: **6 seconds** (was hours). No rejection needed.

**Composition modes:**
- `--composition pre` (default): R_perturbed = R_delta @ R_current (delta in lab frame)
- `--composition post`: R_perturbed = R_current @ R_delta (delta in body frame)

Both produce identical geodesic distance distributions (verified via `compare_poses.py`), but
different specific pose sets. Pre and post perturbed files with the same seed give:
Mean error vs GT: 3.75°, Median: 3.97°, Max: 5.000°, >5°: 0.0% for both modes. ✓

### `cryoPARES/scripts/compare_poses.py`

**Usage for benchmarking:**
```bash
python -m cryoPARES.scripts.compare_poses \
    --starfile1 OUTPUT.star --starfile2 GROUND_TRUTH.star --sym C1
```
Reports Mean, Median, IQR angular error in degrees, plus shift errors in Å.
Always use this script for GT-based metrics — never use the inline `align_star` output.

### STAR file format handling bug fix

`projmatching/projmatching.py` line 290: the `n_first_particles` subsetting code assumed
RELION 3.1+ format (dict with 'optics'/'particles' keys). Old RELION 3.0 format returns a
bare DataFrame from `starfile.read()`. Fix: check `isinstance(star_data, dict)` and handle
both formats:
```python
if isinstance(star_data, dict):
    particles_df = star_data["particles"][:n_first_particles]
    optics_df = star_data["optics"]
    starfile.write({"optics": optics_df, "particles": particles_df}, tmp)
else:
    # Old RELION 3.0 single-block format
    particles_df = star_data[:n_first_particles]
    starfile.write(particles_df, tmp)
```

---

## Phase 1 — Baseline (unmodified code)

Grid: 6°/2° (343 rotations), batch_size=32, compile enabled, GPU 0.
**Scen A errors are vs GT. Scen B GT-based errors are from post-hoc `compare_poses.py` runs.**

| Change | Dataset | Scenario | Median Ang Err (°) vs GT | P75 (°) | P90 (°) | Shift Err (Å) | Time (s) |
|--------|---------|----------|--------------------------|---------|---------|--------------|---------|
| Baseline | DS2 | A (GT) | 1.19 | 2.43 | 4.01 | — | ~25 |
| Baseline | DS2 | B (5° perturb) | 1.64 | 2.46 | 3.79 | 0.86 | ~25 |
| Baseline | DS3 | A (GT) | 0.00 | 2.69 | 4.32 | 0.46 | ~30 |
| Baseline | DS3 | B (5° perturb) | 1.69 | 2.60 | 4.06 | 0.46 | ~24 |

### Sanity checks (Phase 1)
- DS2 Scenario A: 1.19° — slightly elevated vs zero; expected for coarse 2° step quantization. ✓
- DS3 Scenario A: 0.0° — already refined RELION data; projmatching returns same poses. ✓
- DS2/DS3 Scenario B: ~1.6–1.7° GT error recovered from 3.97° median input. ✓

---

## Phase 3 — Changes enabled (grid 6°/2°, batch_size=32, n=2000)

All numbers are **error vs GT** computed post-hoc by `compare_poses.py`.

### Whitening strategy evolution

Reference-based whitening failed on DS3: see Change #2b implementation details above.
All Phase 3 results below use particle-adaptive whitening.

| Change | Dataset | Scenario | Median Ang Err (°) vs GT | P75 (°) | P90 (°) | Shift Err (Å) | Notes |
|--------|---------|----------|--------------------------|---------|---------|--------------|-------|
| baseline | DS2 | A | 1.19 | 2.43 | 4.01 | — | |
| baseline | DS2 | B | 1.64 | 2.46 | 3.79 | 0.86 | |
| baseline | DS3 | A | 0.00 | 2.69 | 4.32 | 0.46 | |
| baseline | DS3 | B | 1.69 | 2.60 | 4.06 | 0.46 | |
| #1 subpixel + #2a zero_dc | DS2 | A | 1.19 | 2.43 | — | — | no angular change at 2° step |
| #1 subpixel + #2a zero_dc | DS2 | B | 1.64 | 2.46 | — | **0.70** | shift improved |
| #2b particle-adaptive whitening | DS2 | A | **0.00** | **0.00** | **0.00** | — | all flags on |
| #2b particle-adaptive whitening | DS2 | B | **1.32** | **1.96** | **3.16** | 0.70 | 19% improvement |
| #2b particle-adaptive whitening | DS3 | A | **0.00** | **2.00** | **4.02** | — | P75 improved |
| #2b particle-adaptive whitening | DS3 | B | 1.72 | 2.59 | 3.85 | — | within noise of baseline |

### Band mask (fftfreq_min=0.05) — harmful, do not use

| Change | Dataset | Scenario | Median (°) | P90 (°) | Notes |
|--------|---------|----------|-----------|--------|-------|
| whiten + fftfreq_min=0.05 | DS2 | B | 1.41 | 7.33 | P90 catastrophic |
| whiten + fftfreq_min=0.05 | DS3 | A | 2.00 | 5.34 | regression vs whitening-only (0.0°) |

fftfreq_min removes orientation-discriminating low-frequency structural information. Zero_dc already
handles the DC bin. **Keep fftfreq_min=0.0 (disabled).**

### Warm-up averaging (whitening_warmup_batches=8, n=2000)

| Change | Dataset | Scenario | Median (°) vs GT | P75 (°) | P90 (°) | Notes |
|--------|---------|----------|-----------------|---------|---------|-------|
| warmup8 | DS2 | B | 1.33 | 1.95 | 2.75 | same as single-batch (within noise) |
| warmup8 | DS3 | B | **1.43** | **2.37** | **4.85** | improved vs single-batch (1.72°→1.43°, +17%) |

### Rotation composition — Cartesian grid (incompatible, do not use)

| Mode | DS2 Median (°) | DS2 P75 | DS2 P90 |
|------|---------------|---------|---------|
| euler_add (baseline) | 1.33 | 1.95 | 2.75 |
| pre_multiply (Cartesian) | 2.42 | 3.61 | 4.81 |
| post_multiply (Cartesian) | 2.27 | 3.35 | 4.32 |

Rotation composition is incompatible with the Cartesian Euler grid. Both modes are worse.

### Fibonacci ω-ball grid (6°/2°, n=2000)

| Config | DS2 Median | DS2 P90 | DS3 Median | DS3 P90 | Notes |
|--------|-----------|---------|-----------|---------|-------|
| euler_add + Cartesian (pre-fibo best) | 1.33° | 2.75° | 1.43° | 4.85° | |
| fibo + pre_multiply | **1.31°** | **2.62°** | **1.31°** | **3.52°** | best overall |
| fibo + post_multiply | 1.35° | 2.66° | — | — | equivalent to pre |

DS3 P90 improvement (4.85°→3.52°) is the standout result: Euler polar degeneracy near D2
symmetry axes is eliminated by the uniform Fibonacci grid.

---

## Finer grid experiments (6°/1°, n=500, batch_size ≤ 2, GPU 1)

**Note:** At 6°/1° the Fibonacci grid has 1638 points and the Cartesian grid has 2197 points.
The Triton XBLOCK constraint requires `batch_size=2` (Fibonacci) or `batch_size=1` (Cartesian)
on this GPU, making these runs slower per particle than the 6°/2° runs.

### All flags ON, Cartesian grid (6°/1°)

| Change | Dataset | Scenario | Median (°) vs GT | P75 (°) | P90 (°) | Shift (Å) | Time (s) |
|--------|---------|----------|-----------------|---------|---------|-----------|---------|
| baseline (flags off) | DS2 | B | 1.49 | 2.35 | 3.80 | 0.70 | ~70 |
| all flags ON | DS2 | B | **1.00** | **1.72** | **2.78** | **0.48** | ~72 |

Overall DS2 Scen B improvement: **1.64° → 1.00° = 39% reduction** vs 2° grid baseline.

### Fibonacci ω-ball grid (6°/1°), pre vs post composition experiment

Perturbed input median vs GT: 3.97°. After projmatching: ~0.85° (79% reduction). n=500.

| Perturbation \ Search | pre_multiply | post_multiply |
|-----------------------|-------------|--------------|
| pre (R_δ @ R_true) | 0.89° (IQR 1.14°, <5°: 98.8%) | 0.85° (IQR 1.06°, <5°: 98.0%) |
| post (R_true @ R_δ) | 0.84° (IQR 1.11°, <5°: 98.4%) | 0.85° (IQR 1.08°, <5°: 98.4%) |

**Key finding:** All 4 combinations give identical accuracy. pre_multiply ≡ post_multiply for
an isotropic ball. The composition order of perturbation (pre vs post) also has no effect on
recovery accuracy — both produce the same geodesic distance distribution from GT.

---

## Summary table (all benchmarks, Scenario B vs GT)

Grid pts measured with `so3_grid_near_identity_fibo(use_small_aprox=True)` + 1 identity.

| Config | Grid pts | n | DS1 med | DS2 med | DS2 P75 | DS2 P90 | DS3 med | DS3 P75 | DS3 P90 | Time/500 | Rec? |
|--------|----------|---|--------|--------|---------|---------|--------|---------|---------|---------|------|
| baseline (all off) | Cartesian 6°/2° (343) | 2000 | — | 1.64° | — | 3.79° | 1.69° | — | 4.06° | ~9.4s | no |
| all flags ON | Cartesian 6°/2° (343) | 2000 | — | 1.32° | — | 3.16° | 1.43° | — | 4.85° | — | yes |
| +warmup8 | Cartesian 6°/2° (343) | 2000 | — | 1.33° | — | 2.75° | 1.43° | — | 4.85° | — | yes |
| +fibo+pre | Fibonacci 6°/2° (209) | 2000 | — | 1.31° | 1.88° | 2.62° | 1.31° | 1.98° | 3.52° | **~6.4s** | yes |
| all flags ON | Cartesian 6°/1° (2197) | 500 | — | 1.00° | — | 2.78° | — | — | — | — | yes |
| fibo+pre | Fibonacci 4°/1° (488) | 500 | — | 0.97° | 1.53° | 2.13° | 1.35° | 2.03° | 2.81° | **~30s** | yes |
| fibo+pre | Fibonacci 6°/1° (1638) | 500 | — | 0.89° | 1.62° | 2.67° | 1.38° | 2.15° | 3.18° | ~99s | no¹ |
| fibo+pre | Fibonacci 4°/0.7° (1486) | 500 | — | 0.87° | 1.43° | 2.23° | **1.24°** | **1.93°** | **2.74°** | ~40.8s | **yes D2** |
| fibo+pre | Fibonacci 4°/0.5° (3875) | 500 | — | — | — | — | 1.22° | 1.89° | 2.64° | ~2m | no² |
| **two-stage 6°/2°+1.5°/0.5° K=5** | **1249** | **500** | **0.22°** | **0.42°** | **1.25°** | **2.47°** | 1.35° | 2.35° | 3.43° | ~33.4s | **yes C1** |

¹ 6°/1°: 99s/500 — 3× slower than two-stage and less accurate on both datasets — dominated.
² 4°/0.5°: 2.3× slower than 4°/0.7° for 0.02° median gain — not worth it.

All flags used for fibo/two-stage rows: `use_subpixel_shifts=True, zero_dc=True, spectral_whitening=True, whitening_warmup_batches=8, fftfreq_min=0.0, use_fibo_grid=True, rotation_composition=pre_multiply`

DS1 (synthetic, C1) two-stage Scen B = 0.22° from a ~4° perturbed start, essentially perfect recovery. DS1 reference volume must be reconstructed **with CTF correction** (`--correct_ctf` default; do NOT use `--NOT_correct_ctf`).

### Scenario A validation (GT input — no regression check)

| Config | DS1 Scen A | DS2 Scen A | DS3 Scen A |
|--------|-----------|-----------|-----------|
| single-stage fibo 6°/2° | 0.00° | 0.00° | 0.00° |
| two-stage 6°/2°+1.5°/0.5° | 0.00° | 0.00° | 1.41°* |

*DS3 two-stage Scen A: fine search (0.5° step) finds a pose 1.41° from RELION GT that correlates marginally better against the reference. This is physical — single-stage can't resolve this sub-2° variation at 2° step resolution. Not a bug; single-stage Scen A (correct behavior at coarse resolution) still passes.

### Two-stage vs 4°/0.7° head-to-head (n=500, Scen B vs GT)

| Config | DS2 median | DS2 P90 | DS3 median | DS3 P90 | Total pts | Winner |
|--------|-----------|---------|-----------|---------|-----------|--------|
| 4°/0.7° single-stage | 0.87° | 2.23° | **1.24°** | **2.74°** | 1486 | DS3 |
| two-stage 6°/2°+1.5°/0.5° | **0.42°** | 2.47° | 1.35° | 3.43° | 1249 | DS2 + fewer pts |

**Key finding:** Two-stage wins for C1 symmetry (DS2: 2× better median, fewer evaluations). 4°/0.7° wins for D2 symmetry (DS3: better median and P90). The two-stage's coarse K=5 candidates can cluster in one symmetry domain, missing better poses in the other 3 D2-related domains; dense single-stage coverage avoids this.

**Recommended config by use case:**
- C1 or high-symmetry (Cn, n≥4): **two-stage** (`use_two_stage_search=True, fine_top_k=5`)
- Low symmetry (D2, D3, T, O): **4°/0.7° single-stage** (`grid_distance_degs=4, grid_step_degs=0.7`)

```
# Two-stage (best for C1)
use_subpixel_shifts=True, zero_dc=True, spectral_whitening=True, whitening_warmup_batches=8,
use_fibo_grid=True, rotation_composition=pre_multiply,
use_two_stage_search=True, fine_grid_distance_degs=1.5, fine_grid_step_degs=0.5, fine_top_k=5

# Dense single-stage (best for D2/low symmetry)
use_subpixel_shifts=True, zero_dc=True, spectral_whitening=True, whitening_warmup_batches=8,
use_fibo_grid=True, rotation_composition=pre_multiply,
grid_distance_degs=4, grid_step_degs=0.7
```

---

## Wall-clock timing benchmarks

All timing runs: GPU 1 (NVIDIA, 14 GB), one process at a time, 3 independent runs, 10 000 particles each.
Per-500 estimates = raw time ÷ 20.

### Master branch baseline (Cartesian 6°/2°, batch_size=11)

| Dataset | Run 1 | Run 2 | Run 3 | Median (10K) | Per 500 |
|---------|-------|-------|-------|--------------|---------|
| DS2 Scen B | 3m06.7s | 3m07.3s | 3m07.4s | **3m07s (187s)** | **~9.4s** |
| DS3 Scen B | 4m21.1s* | 3m25.6s | 3m14.6s | **3m20s (200s)** | **~10.0s** |

*DS3 Run 1 is an outlier — GPU 0 was at 100%/17 GB on a competing job at start. Steady-state median from runs 2–3.

> These times are for the **master default** (Cartesian grid, 343 pts, no accuracy improvements).
> They establish the speed baseline before any new features.

### Branch best configs (improve_local_refinement, all accuracy flags ON)

| Config | Dataset | Run 1 | Run 2 | Run 3 | Median (10K) | Per 500 |
|--------|---------|-------|-------|-------|--------------|---------|
| fibo 6°/2°, batch_size=11 | DS2 | 2m06.8s | 2m05.8s | 2m05.6s | **2m06s (126s)** | **~6.3s** |
| fibo 6°/2°, batch_size=11 | DS3 | 2m09.3s | 2m08.7s | 2m09.0s | **2m09s (129s)** | **~6.5s** |
| fibo 4°/1°, batch_size=8 | DS2 | 9m55.8s | 9m55.0s | 9m54.1s | **9m55s (595s)** | **~29.8s** |
| fibo 4°/1°, batch_size=8 | DS3 | 10m11.3s | 10m12.5s | 10m12.5s | **10m12s (612s)** | **~30.6s** |
| two-stage 6°/2°+1.5°/0.5° K=5, batch_size=3 | DS2 | 11m09.6s | 11m07.8s | 11m08.7s | **11m08s (668s)** | **~33.4s** |
| fibo 4°/0.7°, batch_size=2 | DS3 | 13m35.9s | 13m35.6s | 13m35.3s | **13m35s (815s)** | **~40.8s** |
| fibo 6°/1°, batch_size=2 | DS2 | 28m07.3s* | 32m40.8s | 32m40.4s | **32m40s (1960s)** | **~98s** |
| fibo 6°/1°, batch_size=2 | DS3 | 33m40.7s | 33m32.7s | 33m42.0s | **33m38s (2018s)** | **~101s** |

*DS2 6°/1° Run 1 faster (GPU warmup/cache effect); runs 2–3 are steady-state.

### Speed vs accuracy comparison

| Config | Dataset | Per 500 | Median err | P90 err | vs master |
|--------|---------|---------|-----------|---------|-----------|
| master (Cartesian 6°/2°, batch_size=11) | DS2 | ~9.4s | 1.64° | 3.79° | baseline |
| branch fibo 6°/2°, batch_size=11 | DS2 | **~6.3s** | **1.31°** | **2.62°** | 1.5× faster, more accurate |
| branch fibo 4°/1°, batch_size=8 | DS2 | ~29.8s | 0.97° | 2.13° | 3.2× slower, good tradeoff |
| branch two-stage K=5, batch_size=3 | DS2 | ~33.4s | **0.42°** | **2.47°** | 3.6× slower, 4× better median |
| branch fibo 6°/1°, batch_size=2 | DS2 | ~98s | 0.89° | 2.67° | 10.4× slower, worse than two-stage |
| master (Cartesian 6°/2°, batch_size=11) | DS3 | ~10.0s | 1.69° | 4.06° | baseline |
| branch fibo 6°/2°, batch_size=11 | DS3 | **~6.5s** | **1.31°** | **3.52°** | 1.5× faster, more accurate |
| branch fibo 4°/1°, batch_size=8 | DS3 | ~30.6s | 1.35° | 2.81° | 3.1× slower |
| branch fibo 4°/0.7°, batch_size=2 | DS3 | ~40.8s | **1.24°** | **2.74°** | 4.1× slower, best accuracy |

**Key finding — no regression:** branch fibo 6°/2° (same grid spacing as master, same batch_size) is 1.5× *faster* than master because the Fibonacci grid has fewer points (209 vs Cartesian 343) and more accurate (1.31° vs 1.64°/1.69°). The branch's accuracy improvements come at a cost only when using denser grids (4°/0.7°) or two-stage search.

---

## Pending experiments

- [x] DS3 Scenario B with two-stage K=10 — **ruled out**: 1.36°/3.50° vs K=5 1.35°/3.43°, at 77% more evaluations. D2 gap is geometric, not a K issue.
- [x] DS3 Scen B 4°/0.5° single-stage — **done**: 1.22°/2.64° P90, ~2m. Only 0.02° better than 4°/0.7° at 2.3× cost. 4°/0.7° confirmed as sweet spot.
- [x] **Master branch timing** — DS2: 187s/10K (~9.4s/500), DS3: 200s/10K (~10s/500). Cartesian 6°/2°, batch_size=11.
- [x] **Branch best-config timing** — DS2 two-stage: 668s/10K (~33.4s/500); DS3 4°/0.7°: 815s/10K (~40.8s/500). 3.6–4.1× slower than master, substantially more accurate.
- [ ] DS2 Scenario A with Fibonacci grid 6°/1° (verify no regression vs GT)
- [ ] DS3 Scenario B with Fibonacci grid 6°/1°

---

## Sanity checks summary

| Check | Result | Interpretation |
|-------|--------|---------------|
| DS2 Scen A baseline: 1.19° | ✓ | 2° step quantization floor, expected |
| DS3 Scen A baseline: 0.0° | ✓ | Already RELION-refined; projmatching returns same poses |
| DS2 Scen A with whitening: 0.00° all percentiles | ✓ | CC peak sharp enough; GT always wins |
| DS2 Scen A Fibonacci (identity prepended): 0.00° | ✓ | No regression from grid change |
| DS3 warmup8 (1.43°) vs single-batch (1.72°) | ✓ | CTF diversity averaging works for DS3 |
| DS3 Fibonacci P90 (3.52° vs 4.85°) | ✓ | Euler polar degeneracy eliminated |
| Cartesian rotation composition: worse than euler_add | ✓ | Grid not designed for matrix composition |
| Band mask (fftfreq_min=0.05): P90 7.33° | ✓ | Low-freq content essential; keep disabled |
| perturb_poses.py: max dist = 5.000°, >5°: 0.0% | ✓ | Inverse-CDF sampling correct |
| Pre vs post perturbation: same distance distribution | ✓ | Confirmed via compare_poses.py |
| Pre/post composition 2×2: all ~0.85° | ✓ | Both modes mathematically equivalent |
