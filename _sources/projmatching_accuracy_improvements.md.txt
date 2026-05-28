# Projection Matching Accuracy Improvement Ideas

Analysis of bottlenecks in `cryoPARES/projmatching/projMatcher.py` and potential solutions.
The code improves NN pose estimates (~7°) to ~3° but plateaus there even with fine 1° grids.

Previously tried with no improvement: rotation matrix composition, Fibonacci grid, normalized CC.

---

## Testing Protocol

All changes must be validated through systematic A/B benchmarks before being promoted to default.

### Datasets

| # | Particles | Reference volume | Type | Symmetry | Ground truth |
|---|-----------|-----------------|------|----------|-------------|
| 1 | `EMPIAR-10166/data/projectionsSamePose/proj_0.star` | `EMPIAR-10166/data/reconstruct.mrc` | Synthetic | C1 | Exact known poses in star file |
| 2 | `EMPIAR-10166/data/frealign_stack.star` | `EMPIAR-10166/data/reconstruct.mrc` | Real, full size | C1 | Frealign/RELION consensus poses |
| 3 | `astex-5534/Refine3D/001_after2DclsEval/run_data.star` | Reference vol in same directory | Real, PKM2 | **D2** | RELION consensus poses |

All paths are under `~/cryo/data/EMPIAR-download/`.
For dataset 3, locate the reference `.mrc` in `astex-5534/Refine3D/` before running.

### Two test scenarios

**Scenario A — Recovery from exact ground truth poses (regression test):**
Feed projmatching the exact ground truth poses. The code must not degrade them.
Any change that raises the median error vs baseline in Scenario A is a regression. Code that
improves the baseline is promissing.

**Scenario B — Refinement from perturbed poses (real-world simulation):**
Perturb ground truth poses by 4–6° random rotation per particle, feed those as input.
Measure angular error after projmatching vs original ground truth.
This is the real use case: NN predicts with ~4–7° error, projmatching should recover.

### Pose perturbation helper

**New file:** `tests/benchmarks/perturb_poses.py`

Reads a star file, applies a random rotation (uniformly sampled from an SO3 ball of radius `perturb_deg`) to each particle's pose, saves a new star file.

Key implementation note: perturbation must use **rotation matrix composition**, not Euler angle addition:
```python
from scipy.spatial.transform import Rotation
R_current = Rotation.from_euler("ZYZ", current_euler_angles, degrees=True)
R_delta = Rotation.random(n_particles)  # or sample from SO3 ball of given radius
R_perturbed = R_delta * R_current
perturbed_eulers = R_perturbed.as_euler("ZYZ", degrees=True)
```

### Error metrics
- **Median angular error (degrees)** — primary metric; already printed by `align_star`
- 75th and 90th percentile angular error — tracks outlier fraction
- Median shift error (Å)
- **Dataset 3 (D2):** always use `rotation_error_with_sym(symmetry="D2")` — already supported

### Comparison protocol
- Fix `n_first_particles=500`, `grid_distance_degs=6`, `grid_step_degs=1` for all runs
- Enable one change at a time; always compare against the current-default baseline
- Report Scenario A and B results side-by-side

---

## Proposed Changes

### 1. Sub-pixel shift refinement ★ (untried, high confidence)

**File:** `projMatcher.py:503–512` — `_extract_ccor_max`
**Config flag:** `use_subpixel_shifts: bool = True`

**Problem:** Shifts are found at integer pixel accuracy (`dtype=torch.int64`). Consequence:
- 1.5 Å/px, 80 Å particle radius → 1 pixel ≈ **1.1°**
- 4.0 Å/px, 80 Å radius → 1 pixel ≈ **2.9°**

This integer-pixel floor likely explains why sub-1° improvement stops happening regardless of the angular grid.

**Fix:** 3-point parabolic sub-pixel interpolation on both shift axes after finding the integer peak:
```python
# For a 1D peak at index p with values f[p-1], f[p], f[p+1]:
delta = (f[p-1] - f[p+1]) / (2*f[p-1] - 4*f[p] + 2*f[p+1])
subpixel_pos = p + delta
```
- Change `dtype=torch.int64` → `dtype=torch.float32` in the shift tensor
- Guard: skip correction if the peak is at the boundary of the valid shift region
- `forward()` already calls `.float()` on shifts — no downstream changes needed

**Risk:** Low. The CC peak is smooth and band-limited, making parabolic fitting reliable. Well-established in image registration literature.

---

### 2. Raw CC dominated by low frequencies (untried)

**File:** `fourierOperations.py:135`

**Problem:** `result = parts * conj(projs)` weights each frequency by its power. Cryo-EM images have a steeply falling power spectrum (~1/f² to 1/f⁴), so the CC score is dominated by large-scale blobs (DC–1/20 Å⁻¹) where all orientations look similar. The orientation-discriminating signal at higher frequencies contributes negligible weight.

**Key distinction from NCC (already tried):** NCC normalizes the total amplitude but does not change the per-frequency weighting. That is why NCC had no effect. The changes below rebalance frequency contributions.

#### 2a. Zero DC (trivial, nearly free)
**Config flag:** `zero_dc: bool = True`

Set the DC bin to 0 before correlating. Removes background mean bias. Single-line change in `correlate_dft_2d`.

#### 2b. Spectral whitening
**Config flag:** `spectral_whitening: bool = False`

Pre-multiply each Fourier product by `1 / sqrt(ref_power_spectrum(|freq|) + ε)`. This makes all frequency shells contribute equally to the correlation score.

Implementation:
1. In `_store_reference_vol()`: compute the 1D radially-averaged amplitude spectrum of `reference_vol` in Fourier space and store as a buffer
2. Build a 2D whitening map by evaluating at each pixel's `|freq|`
3. In `correlate_dft_2d`: accept optional `whitening_filter` argument; multiply both `parts` and `projs` by it before taking the cross-product

The whitening map is computed once at init — no per-batch overhead beyond one extra element-wise multiply.

**Risk:** Medium. Whitening amplifies noise beyond the resolution limit. Always combine with the existing `fftfreq_max` mask.

---

### 3. Euler angle addition is not rotation composition (tried, had problems)

**File:** `projMatcher.py:295–303`
**Config flag:** `rotation_composition_mode: str = "euler_add"` | `"pre_multiply"` | `"post_multiply"`

**Problem:**
```python
expanded_eulerDegs = self._get_so3_delta(...) + eulerDegs.unsqueeze(2)
```
Adding Euler angles does not compose rotations. The error is O(sin(δ) × sin(β₀)) and is significant for grid ranges ≥ 3°, particularly near the pole (small β₀) where α and γ become degenerate.

**Why it failed before:** The previous attempt likely used only one composition order, or the Cartesian grid's near-duplicate entries near identity caused the composed search grid to not cover the intended angular neighborhood.

**Proposed fix:** Compose rotation matrices. Two orderings must both be tested:
- `R_total = R_delta @ R_current` — delta applied in lab frame (pre-multiply)
- `R_total = R_current @ R_delta` — delta applied in body frame (post-multiply)

**Implementation:**
1. Add `_get_so3_delta_rotmats(device)`: convert the Cartesian Euler grid to rotation matrices, cache
2. In `forward()`: add conditional branch for each composition mode
3. After finding best poses, index directly into `expanded_rotmats` (avoid Euler round-trip)

**Critical requirement:** Do not change the default (`euler_add`) without a clear improvement on **both** Scenario A and B across all datasets.

---

### 4. Non-uniform SO3 grid wastes sampling points (tried, had problems)

**File:** `projMatcher.py:122–144`, `grids.py:96–110`
**Config flag:** `use_fibo_grid: bool = False`

**Problem:** `so3_near_identity_grid_cartesianprod` creates n³ points from independent Euler angle ranges. Near identity (small β), α and γ are nearly degenerate — the grid wastes the majority of points sampling the same in-plane rotation. A `13³=2197` point grid (for 6°/1° settings) may effectively cover only ~100–150 distinct rotations.

**Better alternative already in the codebase:** `so3_grid_near_identity_fibo(use_small_aprox=True, output="matrix")` at `grids.py:142`. Uses tangent-space rotation vector sampling with a Fibonacci sphere — uniform geodesic coverage, no Euler singularities, no near-duplicate entries.

**Why it failed before:** The Cartesian grid's accidental over-representation of the identity (due to duplicates) may have compensated for the wrong Euler addition. Once Change 3 is fixed, the Fibonacci grid may behave correctly.

**Caveat:** The Fibonacci grid may produce a different number of points and different distribution than the Cartesian grid. Verify the count and coverage before enabling. Check for regressions in `tests/test_projmatching_distrib.py`.

**Only test after Change 3 is resolved.**

---

### 5. Frequency band masking (easy, related to #2)

**File:** `fourierOperations.py`
**Config flag:** `fftfreq_min: float = 0.0`

A `_mask_for_dft_2d` helper already exists at `fourierOperations.py:89` but is never used in the correlation path. Adding a minimum frequency cutoff (high-pass ring) removes dominant low-frequency content without requiring full spectral whitening.

Apply band mask `fftfreq_min < |freq| < fftfreq_max` to both `parts` and `projs` before correlating.

**Risk:** Low. Simple extension of existing infrastructure.

---

### 6. Multi-scale coarse-to-fine refinement (architectural)

**Problem:** Single-pass grid search with fine step is expensive. For 6°/1° settings, the Cartesian grid has `13³ = 2197` points (most redundant); the Fibonacci grid has ~150 points. Both are searched exhaustively.

**Fix:** Two-pass approach:
1. Coarse pass: 3° steps over ±6° → keep top-K candidates per particle
2. Fine pass: 0.5° steps over ±2° → search around each candidate → keep best

Achieves fine-grid accuracy at lower cost. Already partially supported via `top_k_poses_localref` — could be exposed as a two-stage calling convention at the `align_star` level without changes to the core pipeline.

---

### 7. FFT upsampling for sub-pixel shifts (alternative to #1)

Instead of parabolic interpolation, upsample the CC map in the region around the integer peak using FFT zero-padding (the CC map is band-limited, so upsampling is exact in theory).

**Advantage over parabolic:** More accurate for non-Gaussian peaks.
**Disadvantage:** Harder to implement efficiently in batched GPU code. Change #1 (parabolic) is much simpler and usually sufficient.

---

## Summary

| # | Change | Tried | Risk | Likely impact | Order |
|---|--------|-------|------|--------------|-------|
| 1 | Sub-pixel shift | No | Low | **High** | 1st |
| 2a | Zero DC | No | Trivial | Medium | 1st (bundle) |
| 2b | Spectral whitening | No | Medium | **High** | 2nd |
| 3 | Rotation composition | Yes — failed | Medium | High if correct | 3rd |
| 4 | Fibonacci grid | Yes — failed | Medium | Medium (after #3) | 4th |
| 5 | Frequency band mask | No | Low | Medium | 2nd (bundle) |
| 6 | Multi-scale search | No | Low | Speed/accuracy tradeoff | Later |
| 7 | FFT upsampling | No | Medium | Medium (alt to #1) | Skip if #1 works |

**Recommended "fast+safe" default:** Sub-pixel shifts + zero DC (changes 1 + 2a)
**Recommended "accurate" mode (behind config):** + spectral whitening + Fibonacci grid (once validated)