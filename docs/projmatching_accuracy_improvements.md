# Projection Matching Accuracy — Strategy & Lessons Learned

Summary of bottlenecks identified in `cryoPARES/projmatching/projMatcher.py`, what was tried,
what we learned, and what to try next. Detailed benchmark numbers and implementation notes live in
`docs/projmatching_benchmark_results.md`.

**Starting point:** NN pose estimates (~7°) → projmatching plateaus near 1.3–1.7° at 6°/2° grid.
**Current best (6°/1° Fibonacci):** ~0.85° median (79% reduction from 5° perturbation input).

---

## Implemented changes (all gated by config flags)

| # | Change | Config flag | Default | Outcome |
|---|--------|-------------|---------|---------|
| 1 | Sub-pixel shift (parabolic interpolation) | `use_subpixel_shifts` | `True` | Angular improvement at 1° grid; shift improvement at all grids |
| 2a | Zero DC component | `zero_dc` | `True` | Small shift accuracy improvement |
| 2b | Particle-adaptive spectral whitening | `spectral_whitening`, `whitening_warmup_batches=8` | `True` | DS2: 1.64→1.33°, DS3: 1.69→1.43° (19% improvement) |
| 3 | Rotation matrix composition (Cartesian grid) | `rotation_composition` | `euler_add` | **Worse** than euler_add — grid incompatible with matrix composition |
| 4 | Fibonacci ω-ball grid + pre_multiply | `use_fibo_grid` | `False` | DS2: 1.33→1.31°, DS3 P90: 4.85→3.52° — best overall |
| 5 | Frequency band mask (high-pass ring) | `fftfreq_min` | `0.0` (off) | **Harmful** (P90 7.33° vs 2.75°) — keep disabled |
| 6 | Two-stage coarse-to-fine search | `use_two_stage_search`, `fine_grid_*` | `False` | DS2: median 1.37°→**0.42°** (3×); DS3: 1.60°→1.35°; 24% fewer evals vs 6°/1° |

**Current best config (behind flags):**
```
use_subpixel_shifts=True, zero_dc=True, spectral_whitening=True,
whitening_warmup_batches=8, fftfreq_min=0.0,
use_fibo_grid=True, rotation_composition=pre_multiply,
use_two_stage_search=True, fine_grid_distance_degs=1.5, fine_grid_step_degs=0.5, fine_top_k=5
```

`use_fibo_grid=False`, `rotation_composition=euler_add`, and `use_two_stage_search=False` remain
the behavior-preserving defaults. DS1 validation is complete (see two-stage section for results).

**Starting point:** NN pose estimates (~7°) → projmatching plateaus near 1.3–1.7° at 6°/2° grid.
**Current best (two-stage 6°/2°+1.5°/0.5° K=5):** ~0.42° DS2 median (89% reduction from 5° perturbation input).

---

## Key lessons learned

### Sub-pixel shifts (#1)
Integer-pixel CC peak location creates a quantization floor of 1–3° depending on pixel size and
particle radius. Parabolic 3-point interpolation on both axes after the integer peak resolves this.
The parabolic approximation is valid because the CC peak is smooth and band-limited.

### Spectral whitening (#2b) — particle-adaptive, not reference-based
Three attempts at reference-based whitening (`1/sqrt(amp_3d(r))` from reference volume) failed on
DS3: RELION's LP-filtered reference has near-zero amplitude beyond the resolution cutoff, causing
the whitening factor to explode to 6.99× at Nyquist and degrade accuracy by ~0.3°.

**Root cause:** The reference volume spectrum does not match the actual particle spectrum — they
differ in CTF envelope, noise floor, and pixel size. Particle-adaptive whitening (estimating
`1/amp` from the actual particle DFTs over the first 8 warm-up batches) absorbs these differences
automatically and works across datasets.

Apply whitening to projections only, not particles: whitening both sides gives `1/amp²` which over-
amplifies particle noise; projections-only gives `1/amp` — more robust. Using 8 warm-up batches
(vs 1) matters for DS3 where defocus diversity cancels CTF oscillations in the amplitude estimate.

### Frequency band mask (#5) — harmful
Zero DC (#2a) handles the DC bias. A high-pass ring (`fftfreq_min=0.05`) removes low-frequency
structural information that is still orientation-discriminating above DC. Result: catastrophic P90
regression (3.16°→7.33° on DS2). **Do not use.**

### Rotation composition (#3) — grid-dependent
Euler angle addition (`delta + euler`) is an approximation that breaks near tilt poles, but it
works well with the Cartesian Euler grid because the grid is designed for Euler-space arithmetic.
Converting to rotation matrix composition with the same Cartesian grid made results significantly
worse (1.33°→2.27–2.42°): the near-identity cluster that the Cartesian grid accumulates from
Euler degeneracy (multiple (α, 0, γ) entries collapse to the same rotation) vanishes when
converted to matrices, leaving sparser effective search near the correct pose.

**Conclusion:** rotation matrix composition is only compatible with the Fibonacci grid, which is
specifically designed for it.

### Fibonacci grid (#4) — pre_multiply ≡ post_multiply
The hypothesis that only one composition order (pre or post) is "geometrically correct" is
disproved: for an isotropic SO(3) ball, `{R_δ @ R_est}` and `{R_est @ R_δ}` cover the same
geodesic ball and give identical recovery accuracy (~0.85° for both in 2×2 experiment).
Similarly, perturbation composition order (pre vs post) has no effect on recovery accuracy —
both produce identical geodesic distance distributions from ground truth.

The Fibonacci grid eliminates Euler polar degeneracy that was harming DS3 (D2 symmetry, particles
near symmetry axes): P90 dropped from 4.85°→3.52°, a standout improvement.

**Critical implementation detail:** The ω-ball shell starts at ~3.3° geodesic distance, so the
identity rotation must be explicitly prepended to the grid. Without it, Scenario A (GT input)
always returns a pose ≥3.3° wrong — a guaranteed regression.

### Whitening warmup and on-the-fly inference
The current warmup strategy accumulates particle DFTs over the first 8 batches then freezes the
map. For long-running or streaming inference sessions, microscope conditions (defocus, detector
noise) may drift. Updating the whitening estimate periodically (e.g., every few thousand
particles) could help accuracy but has a computational cost — not yet addressed.

---

## Pending validation

- [ ] DS2 Scenario A with Fibonacci grid 6°/1° — confirm no regression vs GT
- [ ] DS3 Scenario B with Fibonacci grid 6°/1°
- [x] DS1 (synthetic, EMPIAR-10166 projectionsSamePose) Scenarios A and B — **done** (see two-stage section)
- [ ] DS3 Scenario B with Fibonacci grid 4°/0.5°  - confirm if error can be increased further

---

## Implemented: Two-stage coarse-to-fine (#6)

**Config flags** (all default-off):
```
use_two_stage_search=True, fine_grid_distance_degs=1.5, fine_grid_step_degs=0.5, fine_top_k=5
```

**Actual Fibonacci grid point counts** (measured, formula: `so3_grid_near_identity_fibo(use_small_aprox=True)`):

| Config | Pts | Notes |
|--------|-----|-------|
| 6°/2° | 209 | coarse pass |
| 6°/1° | 1638 | previous best single-stage |
| 4°/0.5° | 3875 | strong single-stage baseline (covers 4° initial error) |
| 4°/0.7° | 1486 | cheaper single-stage baseline |
| 1.5°/0.5° | 209 | fine pass (same count as 6°/2°!) |
| 1.0°/0.5° | 63 | fine pass (aggressive) |

> **Correction of earlier doc:** "2°/0.5° → ~65 pts" was wrong — 2°/0.5° gives ~488 pts.
> The 63-pt target is hit by **1°/0.5°**; the preferred fine grid is **1.5°/0.5° (209 pts)**.

**Two-stage totals** (K=5 candidates from coarse):

| Coarse + Fine | Total | vs 6°/1° single |
|---------------|-------|-----------------|
| 6°/2° + 1.5°/0.5° | 209 + 5×209 = **1249** | 24% fewer pts |
| 6°/2° + 1.0°/0.5° | 209 + 5×63  = **524**  | 68% fewer pts |

**Expected accuracy hypothesis (to validate on DS2/DS3):**
- Two-stage 6°/2° + 1.5°/0.5° should beat single-stage 6°/1° (finer 0.5° step, same cost class)
- Benchmark `4°/0.5°` and `4°/0.7°` as strong single-stage baselines (4° covers typical initial error)

**XBLOCK benefit:** Coarse max=209 pts, fine max=5×209=1045 pts — both allow `batch_size≥3` on GPU 1
(vs `batch_size=2` required for flat 6°/1°).

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
| fibo 6°/2° (bs=32) | 209 | — | 1.31° | 1.88° | 2.62° | 1.31° | 1.98° | 3.52° | **~6.2s** |
| fibo 4°/1° (bs=16) | 488 | — | 0.97° | 1.53° | 2.13° | 1.35° | 2.03° | 2.81° | **~13.4s** |
| fibo 4°/0.7° (bs=5) | 1486 | — | 0.87° | 1.43° | 2.23° | **1.24°** | **1.93°** | **2.74°** | **~40.4s** |
| fibo 6°/1° (bs=4) | 1638 | — | 0.89° | 1.62° | 2.67° | 1.38° | 2.15° | 3.18° | **~43.2s** |
| **two-stage 6°/2°+1.5°/0.5° K=5 (bs=7)** | **1249** | **0.22°** | **0.42°** | **1.25°** | 2.47° | 1.35° | 2.35° | 3.43° | **~33.2s** |

Timing measured on 10K particles, 3 runs each, RTX 6000 Ada 49 GB. Per-500 = raw ÷ 20. All branch times use optimal batch sizes (OOM limit ~8192 total projs/batch).
Master baseline: DS2 187s/10K, DS3 200s/10K (Cartesian 6°/2°, batch_size=11).
Fibo 6°/2° (bs=32): DS2 124s/10K, DS3 128s/10K — **faster than master** (fewer grid pts: 209 vs 343). No regression.
Fibo 4°/1° (bs=16): DS2 267s/10K, DS3 277s/10K — **2.2× faster than prior bs=8 measurement** (595s/612s).
Fibo 6°/1° (bs=4): DS2 863s/10K, DS3 889s/10K — **2.3× faster than prior bs=2 measurement** (1960s/2018s).
Two-stage (bs=7): DS2 664s/10K. Fibo 4°/0.7° (bs=5): DS3 808s/10K — both unchanged from prior sub-optimal bs (sparse grids are compute-bound per batch, not launch-overhead-bound).

Note: 6°/1° (1638 pts) is slower than two-stage (1249 pts) yet less accurate on both datasets — dominated.
4°/0.7° wins on DS3 because it concentrates coverage where it matters (≤4° initial error, 0.7° step)
rather than wasting evaluations from 4°–6°.

Two-stage gives **4× better median** on DS2 (C1) vs master, 3.6× slower. DS3 (D2): 4°/0.7° single-stage is better (coarse K=5 candidates can cluster in one D2 domain, missing poses in the other 3). DS1 (synthetic): near-perfect 0.22° recovery.

**Scenario A validation (GT input → expect ~0°):**
- DS1: 0.00° ✓ (requires CTF-corrected reference volume)
- DS2: 0.00° ✓
- DS3: 1.41° — fine search finds marginally better-correlating pose than RELION GT at 0.5° resolution; single-stage still returns 0.00° ✓

**Recommendation by symmetry:**
- C1 / high symmetry: **two-stage** (best median, fewer evaluations)
- D2 / low symmetry: **4°/0.7° single-stage** (robust spatial coverage avoids symmetry-domain clustering)

**DS3 Scen B K=10 result:** Median 1.36°, P75 2.29°, P90 3.50° — identical to K=5 (1.35°/2.35°/3.43°) at 77% more evaluations (2299 vs 1249 pts, ~77s vs ~52s). The D2 gap is geometric (symmetry-axis ambiguity), not a candidate-count issue. K=10 ruled out.

**DS3 Scen B 4°/0.5° result:** Median 1.22°, P75 1.89°, P90 2.64° in ~2m (batch_size=1) vs 4°/0.7°: 1.24°/1.93°/2.74° in ~52s (batch_size=2). Only 0.02° median gain for 2.3× wall-clock cost. **4°/0.7° is the sweet spot for D2 — going finer hits diminishing returns.**

**Pending validation:**
- [x] Master branch comparison + wall-clock timing — done (see timing table above)
- [x] Branch fibo 6°/2° timing — **no regression**: 6.4s/500 vs master 9.4s/500; faster and more accurate

---

## Testing protocol (reference)

See `docs/projmatching_benchmark_results.md` for datasets, star file paths, benchmark commands,
and the XBLOCK batch-size constraint on GPU 1. Key reminders:

- Always use `compare_poses.py --starfile1 OUTPUT --starfile2 GT` for GT-based metrics; the
  inline `align_star` output measures displacement from input, not from GT.
- Fix `n_first_particles=500` for comparability.
- DS3 (D2 symmetry): use `--sym D2` in `compare_poses.py`.
- Fibonacci 6°/1°: requires `batch_size=2` on GPU 1 due to Triton XBLOCK limit.
- **New single-stage baselines** (use `grid_distance_degs=4`, `grid_step_degs=0.5` or `0.7`).
- **Two-stage benchmark command:**
  ```bash
  cryopares_projmatching ... --grid_distance_degs 6 --grid_step_degs 2 --batch_size 4 --gpu_id 1 \
    --config projmatching.use_fibo_grid=True projmatching.rotation_composition=pre_multiply \
             projmatching.use_subpixel_shifts=True projmatching.zero_dc=True \
             projmatching.spectral_whitening=True projmatching.whitening_warmup_batches=8 \
             projmatching.use_two_stage_search=True \
             projmatching.fine_grid_distance_degs=1.5 projmatching.fine_grid_step_degs=0.5 \
             projmatching.fine_top_k=5
  ```
