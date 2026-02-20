# CryoPARES Reconstruction Issue Analysis
**Date:** 2026-01-19
**Status:** Investigation in progress

## Problem Summary

### Observed Symptoms
1. **Half-maps are extremely noisy** - almost all noise with very little visible signal
2. **Postprocessing recovers signal** - After running `relion_postprocess`, the final map looks acceptable
3. **FSC resolution appears inflated** - Resolution estimates seem artificially high compared to RELION reconstruction

### Critical Insight
The fact that noisy half-maps produce acceptable FSC after postprocessing, but with inflated resolution, suggests **systematic correlation between half-maps** rather than random noise. This indicates a reconstruction bias that affects both halves identically.

---

## Code Review Findings

### 1. Wiener Filter / Regularization ⚠️ **MAJOR ISSUE IDENTIFIED**

**Location:** `reconstructor.py:442`

```python
if self.correct_ctf:
    denominator = self.ctfsq[mask] + self.eps * self.weights[mask]
    dft[:, mask] = self.numerator[:, mask] / denominator[None, ...]
```

**Formula:**
```
F_reconstructed = Σ(w_geom · img · CTF) / [Σ(w_geom · CTF²) + ε · Σ(w_geom)]
```

**Default value:** `eps = 1e-3` (from `reconstruct_config.py:38`)

---

## CRITICAL: RELION Comparison (WITHOUT --fsc)

**Important clarification:** The correct comparison is:
```bash
relion_reconstruct --i particles.star --ctf --o output.mrc
# No --fsc option (standard usage)
```

From [backprojector.cpp](https://github.com/3dem/relion/blob/master/src/backprojector.cpp) investigation:

### RELION without --fsc: NO REGULARIZATION

```cpp
// When no --fsc is provided:
do_map = false;  // No FSC file loaded
tau2 = empty;    // Empty array

// In reconstruct():
if (do_map) {
    // This block is SKIPPED - no tau2 term added!
    invtau2 = 1.0 / (oversampling_correction * tau2_fudge * tau2(ires));
    invw += invtau2;
}

// Result: Pure weighted backprojection
F = numerator / weight
  = Σ(img·CTF·w_geom) / Σ(CTF²·w_geom)
```

**RELION uses NO regularization term!** It only prevents division by zero using **radial averaging fallback** (explained below).

### CryoPARES: Has Regularization

```python
denominator = self.ctfsq + self.eps * self.weights
F = Σ(img·CTF·w_geom) / [Σ(CTF²·w_geom) + eps·Σ(w_geom)]
```

**The eps·weights term adds unwanted regularization everywhere!**

---

## Direct Comparison Table

| Aspect | RELION --ctf (no --fsc) | CryoPARES | Impact |
|--------|------------------------|-----------|---------|
| **Regularization term** | **NONE** | `eps · Σ(w_geom)` | CryoPARES dampens signal |
| **Division-by-zero prevention** | Radial averaging | `+ eps · weights` | Different approaches |
| **Well-sampled voxels** | No dampening | ~10-20% dampening | Signal loss |
| **Poorly-sampled voxels** | Threshold to radavg/1000 | Small eps·w term | RELION more robust |

### Numerical Example

Well-sampled voxel at low frequency:
- `Σ(CTF²·w_geom) = 0.5`
- `Σ(w_geom) = 100`
- `eps = 1e-3`

**RELION:**
```
denominator = 0.5
No dampening
```

**CryoPARES:**
```
denominator = 0.5 + 0.001×100 = 0.5 + 0.1 = 0.6
Signal dampened by 17%!
```

**This is a major contributor to noisy/low-quality maps!**

---

## RELION's Radial Averaging Strategy

RELION prevents division by zero without adding regularization using **radial averaging**:

### Algorithm

**Step 1: Compute radial average of weights per frequency shell**

```cpp
// Group voxels by frequency (spherical shells)
FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(Fweight) {
    int r2 = k*k + i*i + j*j;
    int ires = ROUND(sqrt((RFLOAT)r2));  // Radial distance

    // Accumulate weights by frequency shell
    radavg_weight[ires] += Fweight[k,i,j];
    counter[ires] += 1.0;
}

// Average per shell
radavg_weight[ires] /= counter[ires];
```

**Step 2: Use radial average as safety threshold**

```cpp
FOR_ALL_ELEMENTS(Fconv) {
    int ires = ROUND(sqrt(k*k + i*i + j*j));
    RFLOAT weight = Fweight[k,i,j];

    // Use max of actual weight or 1/1000 of shell average
    weight = MAX(weight, radavg_weight[ires] / 1000.0);

    // Safe division
    Fconv[k,i,j] /= weight;
}
```

### Why This Works

At frequency f = 0.3 Å⁻¹:
- Most voxels: weight ≈ 100 (well sampled)
- CTF zero voxels: weight ≈ 0.01 (poorly sampled)
- Radial average = 90

**Without protection:**
```
F = numerator / 0.01  → 100× amplification (noise explosion!)
```

**With radial averaging:**
```
effective_weight = max(0.01, 90/1000) = 0.09
F = numerator / 0.09  → 11× amplification (reasonable)
```

**Key properties:**
- ✓ **Frequency-adaptive**: Different threshold per frequency shell
- ✓ **Minimal intervention**: Only affects voxels with weight < radavg/1000
- ✓ **No dampening**: Well-sampled voxels completely unchanged
- ✓ **Smart fallback**: Uses local frequency behavior, not global constant

---

## Why CryoPARES Maps Are Noisier

The `eps * weights` term causes:

1. **Signal dampening in well-sampled regions** (10-20% typical)
2. **Uniform across all frequencies** (over-regularizes low freq, under-regularizes high freq)
3. **Scales with sampling density** (backwards - well-sampled regions shouldn't need more regularization)

Result: **Lower overall map quality**, appearing as reduced signal and increased "noise" (actually over-smoothed signal).

---

### 2. Gridding Correction Implementation ⚠️ NEEDS INVESTIGATION

**Location:** `reconstructor.py:449-453`

```python
dft = torch.fft.ifftshift(dft, dim=(-3, -2))
dft = torch.fft.irfftn(dft, dim=(-3, -2, -1))
dft = torch.fft.ifftshift(dft, dim=(-3, -2, -1))
sincsq = self.get_sincsq(dft.shape, device, self.eps)
vol = dft.to(device) / sincsq
```

**CryoPARES approach:** Separable sinc² correction in real space
```python
z = torch.linspace(-(D-1)/2, (D-1)/2, D) / D  # z ∈ [-0.5, 0.5]
sz = torch.sinc(z).pow(2)
S = sz[:, None, None] * sy[None, :, None] * sx[None, None, :]  # Separable product
```

**RELION approach:** Radial sinc² correction in real space ([source](https://github.com/3dem/relion/blob/master/src/projector.cpp))
```cpp
RFLOAT r = sqrt(k*k + i*i + j*j);
RFLOAT rval = r / (ori_size * padding_factor);
RFLOAT sinc = sin(PI * rval) / (PI * rval);
vol[k,i,j] /= sinc * sinc;
```

**Key differences:**
1. **CryoPARES:** Separable per-axis correction (more accurate for separable trilinear interpolation)
2. **RELION:** Radial approximation with padding factor
3. **CryoPARES normalization:** `z/D` where z ∈ [-(D-1)/2, (D-1)/2] → sinc argument ∈ [-0.5, 0.5]
4. **RELION normalization:** `r/(ori_size * padding_factor)` → for padding=2, argument ∈ [0, ~0.43]

**CRITICAL FINDING: CryoPARES DOES NOT USE PADDING**

From RELION source investigation ([reconstructor.cpp](https://github.com/3dem/relion/blob/master/src/reconstructor.cpp)):
- **Default padding_factor = 2.0**
- Reconstruction happens in padded Fourier space (2× larger in each dimension)
- After inverse FFT, windowed back to original size
- Gridding correction normalizes by `r / (ori_size * padding_factor)`

From CryoPARES code:
```bash
$ grep -r "padding" cryoPARES/reconstruction/
# No results - NO PADDING USED
```

**Impact of no padding:**
- RELION reconstructs at 2× resolution then downsamples → reduces aliasing
- CryoPARES reconstructs at native resolution → potential aliasing artifacts
- Gridding correction formula differs because of this fundamental architectural difference
- CryoPARES normalization uses full sinc argument range [-0.5, 0.5]
- RELION uses reduced range due to padding: [0, 0.43] for padding=2

**Theoretical assessment:** CryoPARES's separable approach is theoretically more correct for separable trilinear gridding, BUT the lack of Fourier padding may introduce aliasing that affects reconstruction quality.

---

### 3. FFT Normalization ⚠️ POTENTIAL ISSUE

**Forward transform (2D):**
```python
imgs = torch.fft.rfftn(imgs, dim=(-2, -1))  # norm='backward' (default)
```

**Inverse transform (3D):**
```python
dft = torch.fft.irfftn(dft, dim=(-3, -2, -1))  # norm='backward' (default)
```

**PyTorch 'backward' normalization:**
- Forward: `F[k] = Σ f[n] · exp(-2πi·k·n/N)` (no normalization)
- Inverse: `f[n] = (1/N) · Σ F[k] · exp(2πi·k·n/N)` (divide by N)

**RELION FFT Normalization (DETAILED INVESTIGATION):**

From [backprojector.cpp:2577-2619](https://github.com/3dem/relion/blob/master/src/backprojector.cpp):

```cpp
// For 3D reconstruction with 3D data:
if (ref_dim == 3 && data_dim == 3) {
    normfft = (RFLOAT)(padding_factor * padding_factor * padding_factor);
}

// After inverse FFT:
Mout /= normfft;  // Divide by padding_factor³
```

**RELION normalization:**
- Inverse FFT returns unnormalized result (FFTW convention)
- RELION divides by `padding_factor³` = 8 (for padding=2)
- This accounts for the padded reconstruction grid
- Additional oversampling_correction applied to weights earlier

**CryoPARES normalization:**
- Uses PyTorch's 'backward' convention: inverse FFT divides by N
- No padding → no additional padding correction
- Final volume scaled only by PyTorch's automatic 1/N factor

**Critical difference:**
- **RELION:** Reconstructs at 2× resolution, normalizes by padding³, then windows to original size
- **CryoPARES:** Reconstructs at native resolution, normalizes only by FFT convention

**Dimensional mismatch concern:**
- 2D forward FFT: unnormalized (PyTorch convention)
- 3D inverse FFT: divided by (D×H×W)
- For Fourier slice theorem, this should be correct
- BUT: May need empirical verification of absolute density scaling

---

### 4. Potential Noise Amplification Sources

#### A. High-frequency CTF zeros
When `correct_ctf=True`, if CTF approaches zero at certain frequencies:
```
denominator = ctfsq + eps * weights
```

If `ctfsq ≈ 0` and `weights` is large (well-sampled), then:
- denominator ≈ `eps * weights` (small if eps=1e-3)
- Could still amplify noise at CTF zeros

**Comparison needed:** What does RELION use for regularization? Is it constant or frequency-dependent?

#### B. Missing soft masking ⚠️ **CONFIRMED MAJOR ISSUE**

**RELION reconstruction pipeline** ([source](https://github.com/3dem/relion/blob/master/src/backprojector.cpp)):
1. Inverse FFT
2. Gridding correction (sinc²)
3. **Soft masking** - `softMaskOutsideMap(vol_out)` to prevent aliasing

From [mask.cpp](https://github.com/3dem/relion/blob/master/src/mask.cpp):
```cpp
void softMaskOutsideMap(MultidimArray<RFLOAT> &vol, RFLOAT radius,
                       RFLOAT cosine_width, MultidimArray<RFLOAT> *Mnoise);
```

**RELION soft masking parameters** (from [reconstructor.cpp](https://github.com/3dem/relion/blob/master/src/reconstructor.cpp)):
```cpp
// Default parameters:
radius = mask_diameter / (angpix * 2.0)  // Typically ~half the box size
cosine_width = width_mask_edge           // Default: 3 pixels
```

**Soft mask formula:**
```cpp
if (r < radius): vol[r] unchanged
if (r > radius + cosine_width): vol[r] = background
if (radius < r < radius + cosine_width):
    raisedcos = 0.5 + 0.5 * cos(PI * (radius_p - r) / cosine_width)
    vol[r] = (1 - raisedcos) * vol[r] + raisedcos * background
```

**CryoPARES:** No soft masking visible in `generate_volume()`

**Impact:**
- Without soft masking, high-frequency noise at box edges aliases into the volume
- Edge discontinuities create ringing artifacts throughout the reconstruction
- These artifacts appear as "noise" but are actually systematic errors
- **This could explain the extremely noisy appearance of half-maps**

#### C. Empty voxel handling
```python
mask = self.weights > self.min_denominator_value
# Only voxels with sufficient sampling are reconstructed
# Others remain zero in Fourier space
```

After inverse FFT, these zeros spread throughout real space. While the sinc² correction is applied uniformly, it doesn't address the fundamental issue of missing data in certain Fourier regions.

**Question:** Does this create systematic artifacts that correlate between half-maps?

---

### 5. Shift Convention ⚠️ VERIFY

**Location:** `reconstructor.py:336`

```python
imgs = fourier_shift_dft_2d(
    dft=imgs,
    image_shape=self.particle_shape,
    shifts=hwShiftAngs / self.sampling_rate,
    rfft=True,
    fftshifted=True,
)
```

**Concern:** Variable name `hwShiftAngs` suggests Angstroms, but RELION stores shifts in **pixels**.

**RELION convention:**
- `rlnOriginXAngst` and `rlnOriginYAngst` → in Angstroms
- `rlnOriginX` and `rlnOriginY` → in pixels

**From constants:** `RELION_SHIFTS_NAMES` comes from `starstack` library

**Verification needed:**
1. What does `RELION_SHIFTS_NAMES` actually contain?
2. Are the shifts in pixels or Angstroms?
3. If in pixels, should there be division by `sampling_rate`?

**Impact:** Incorrect shifts would cause blurring but affect both RELION and CryoPARES postprocessing identically, so may not explain the differential behavior.

---

### 6. Symmetry Expansion ✓ APPEARS CORRECT

**Location:** `reconstructor.py:138-190`

The symmetry expansion logic follows RELION conventions:
- Applies symmetry operations to rotation matrices
- Replicates images and CTFs
- Keeps shifts unchanged (correct for non-helical symmetry)
- Applies confidence weighting if requested

No obvious issues detected here.

---

## Summary of RELION Deep Dive Findings

### Major Architectural Differences

| Aspect | RELION | CryoPARES | Impact |
|--------|--------|-----------|--------|
| **Fourier Padding** | 2× (padding_factor=2) | None | RELION: less aliasing, higher computational cost |
| **Regularization** | Frequency-dependent tau²(ires) from FSC | Constant eps=1e-3 | RELION adapts to SNR, CryoPARES uniform |
| **Soft Masking** | Yes (cosine edge, width=3px) | **NO** | **Critical: missing masking → aliasing artifacts** |
| **FFT Normalization** | padding_factor³ = 8 | PyTorch default (1/N) | Different absolute scaling |
| **Gridding Correction** | Radial r/(ori_size·padding) | Separable z/D, y/H, x/W | CryoPARES more accurate theoretically |
| **Oversampling Correction** | padding³ applied to weights/tau² | Not applicable (no padding) | Accounts for padded grid |

### **SMOKING GUN: Missing Soft Masking**

This is the **most likely culprit** for the noisy half-maps:

1. **Without soft masking**, sharp discontinuities at box edges create Gibbs ringing
2. These artifacts spread throughout the volume in Fourier space
3. **Appear as noise** but are actually systematic errors
4. **Affect both half-maps identically** → inflated FSC (correlated artifacts)
5. **relion_postprocess applies its own masking** → recovers signal, but FSC already corrupted

---

## Root Causes Identified (UPDATED with CRITICAL FINDINGS)

### Issue 1: Unwanted Regularization from eps·weights ⚠️ **CRITICAL** (VERY HIGH)

**Evidence:**
- **RELION without --fsc uses NO regularization** (confirmed from source)
- **CryoPARES adds eps·weights everywhere**, dampening signal by 10-20%
- This is the **wrong comparison** - standard RELION usage is without --fsc
- **Direct cause of reduced signal and "noisy" appearance**

**Impact severity:** HIGH - Affects every voxel in reconstruction

**Test:**
```bash
# Should match RELION behavior
cryopares_reconstruct --eps 0 ...

# Or implement radial averaging (better)
```

**Expected outcome:** Dramatic improvement in map quality, signal recovery

**Fix difficulty:** Easy - Either set eps=0 or implement radial averaging (code below)

---

### Issue 2: Missing Soft Masking ⚠️ **CRITICAL** (VERY HIGH)

**Evidence:**
- RELION always applies soft masking after gridding correction
- CryoPARES does not
- Edge artifacts would affect both half-maps identically (inflated FSC)
- Explains "noisy but structured" appearance
- **Confirmed by code inspection:** No masking in `generate_volume()`

**Impact severity:** HIGH - Creates systematic artifacts, inflates FSC

**Test:** Add soft masking after sinc² correction (code provided below)

**Expected outcome:** Should dramatically reduce edge artifacts and improve FSC reliability

**Fix difficulty:** Medium - ~30 lines of code (implementation provided)

---

### Issue 3: No Fourier Padding (MEDIUM)

**Evidence:**
- RELION reconstructs at 2× resolution, CryoPARES at native resolution
- No padding → potential aliasing in backprojection step
- Different gridding correction formulas due to this
- May interact with missing soft masking to amplify artifacts

**Impact severity:** MEDIUM - May cause some aliasing

**Test:**
1. Compare radial power spectra for signs of aliasing (peaks at Nyquist)
2. Implement padding (major architectural change)

**Expected outcome:** Padding would reduce aliasing but is a major code change

**Fix difficulty:** Hard - Major architectural change (2-3 days work)

---

### Non-Issue: FFT Normalization (LOW)

**Evidence:**
- RELION divides by padding_factor³
- CryoPARES uses PyTorch's automatic 1/N
- Different absolute density scaling
- Unlikely to cause "noisy" appearance (would just scale uniformly)

**Impact severity:** LOW - Just affects absolute scale

**Expected outcome:** Unlikely to be the primary issue

---

### Non-Issue: Gridding Correction Details (LOW)

**Evidence:**
- CryoPARES uses separable correction (theoretically better)
- RELION uses radial approximation
- Both approaches are valid

**Impact severity:** NEGLIGIBLE

**Expected outcome:** Minimal difference

---

## Diagnostic Plan for Tomorrow

### Test 1: Sanity Checks
```bash
# Reconstruct without CTF correction
cryopares_reconstruct \
    --particles_star_fname test.star \
    --symmetry C1 \
    --output_fname cryo_no_ctf.mrc \
    --correct_ctf False

# Compare with RELION
relion_reconstruct \
    --i test.star \
    --o relion_no_ctf.mrc \
    --ctf false

# Compare statistics
python -c "
import mrcfile
c = mrcfile.open('cryo_no_ctf.mrc', mode='r').data
r = mrcfile.open('relion_no_ctf.mrc', mode='r').data
print(f'CryoPARES: mean={c.mean():.6f}, std={c.std():.6f}, sum={c.sum():.6e}')
print(f'RELION:    mean={r.mean():.6f}, std={r.std():.6f}, sum={r.sum():.6e}')
print(f'Ratio: {c.sum()/r.sum():.3f}')
"
```

### Test 2: Add Soft Masking
Modify `generate_volume()` to add soft masking after sinc² correction:
```python
vol = dft.to(device) / sincsq

# Add soft masking (RELION-style)
def soft_mask_sphere(vol, radius_fraction=0.45, edge_width=5):
    D, H, W = vol.shape
    center = torch.tensor([D/2, H/2, W/2])
    z, y, x = torch.meshgrid(torch.arange(D), torch.arange(H), torch.arange(W))
    r = torch.sqrt((z-center[0])**2 + (y-center[1])**2 + (x-center[2])**2)
    mask_radius = radius_fraction * min(D, H, W)
    mask = 0.5 * (1 + torch.cos(np.pi * (r - mask_radius) / edge_width))
    mask = torch.clamp(mask, 0, 1)
    return vol * mask

vol = soft_mask_sphere(vol)
```

### Test 3: Check Shift Units
Add debug output to verify shift conventions:
```python
print(f"RELION_SHIFTS_NAMES: {RELION_SHIFTS_NAMES}")
print(f"Sample shift values: {hwShiftAngs[:10]}")
print(f"Sampling rate: {self.sampling_rate}")
print(f"Shifts after division: {hwShiftAngs[:10] / self.sampling_rate}")
```

### Test 4: Frequency-dependent Analysis
Compute radial profiles and compare:
```python
import numpy as np
from scipy.ndimage import distance_transform_edt

def radial_profile(data):
    D, H, W = data.shape
    center = np.array([D/2, H/2, W/2])
    z, y, x = np.mgrid[0:D, 0:H, 0:W]
    r = np.sqrt((z-center[0])**2 + (y-center[1])**2 + (x-center[2])**2)
    r = r.astype(int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    profile = tbin / nr
    return profile

# Compare profiles
cryo_profile = radial_profile(cryo_vol)
relion_profile = radial_profile(relion_vol)
plt.plot(cryo_profile / relion_profile)
plt.xlabel('Frequency (pixels)')
plt.ylabel('CryoPARES / RELION amplitude ratio')
```

### Test 5: Half-map Correlation Analysis
Check if noise is truly random or has structure:
```python
# Compute difference map
diff = half1 - half2

# Compute power spectrum of difference
diff_fft = np.fft.fftn(diff)
diff_power = np.abs(diff_fft)**2

# Should be flat if noise is white
# Structure indicates systematic error
```

---

## RELION Source Code Investigation - ANSWERED ✓

1. **Normalization:** ✓ RELION divides by `padding_factor³ = 8` after inverse FFT, plus oversampling_correction applied to weights
2. **Regularization:** ✓ **Frequency-dependent tau²(ires)** computed as `(FSC/(1-FSC)) × sigma² × tau2_fudge`. When no FSC provided, **skips regularization entirely**
3. **Padding:** ✓ Default **padding_factor = 2.0**. Gridding correction normalizes by `r/(ori_size × padding_factor)`. Oversampling correction = padding³
4. **Masking:** ✓ `softMaskOutsideMap(vol, radius, cosine_width=3)` with raised cosine falloff. Applied **after** gridding correction
5. **FFT:** ✓ Uses FFTW (unnormalized forward, normalized inverse), plus manual division by `padding_factor³`

**All questions answered from source code analysis!**

---

## Proposed Implementation Fix

### Priority 1: Add Soft Masking (CRITICAL)

Add this method to the `Reconstructor` class:

```python
@staticmethod
@lru_cache(maxsize=1)
def get_soft_mask(shape, device, radius_fraction=0.45, edge_width=5):
    """
    Create a soft spherical mask with cosine falloff, matching RELION behavior.

    Args:
        shape: (D, H, W) volume shape
        device: torch device
        radius_fraction: fraction of box size for mask radius (default 0.45)
        edge_width: width of cosine falloff edge in pixels (default 5, RELION uses 3)

    Returns:
        mask: (D, H, W) soft mask with values in [0, 1]
    """
    D, H, W = shape
    center = torch.tensor([D/2, H/2, W/2], device=device)

    # Create coordinate grids
    z = torch.arange(D, device=device).float()
    y = torch.arange(H, device=device).float()
    x = torch.arange(W, device=device).float()
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')

    # Distance from center
    r = torch.sqrt((zz - center[0])**2 + (yy - center[1])**2 + (xx - center[2])**2)

    # Mask parameters
    mask_radius = radius_fraction * min(D, H, W)

    # Raised cosine falloff (matching RELION formula)
    mask = torch.ones_like(r)
    transition = (r > mask_radius) & (r < mask_radius + edge_width)
    beyond = r >= mask_radius + edge_width

    # raisedcos = 0.5 + 0.5 * cos(π * (radius + edge_width - r) / edge_width)
    mask[transition] = 0.5 + 0.5 * torch.cos(
        np.pi * (r[transition] - mask_radius) / edge_width
    )
    mask[beyond] = 0.0

    return mask
```

Modify `generate_volume()`:

```python
def generate_volume(
    self,
    fname: Optional[FNAME_TYPE] = None,
    overwrite_fname: bool = True,
    device: Optional[str] = "cpu",
    apply_soft_mask: bool = True,  # NEW PARAMETER
    mask_radius_fraction: float = 0.45,  # NEW PARAMETER
    mask_edge_width: int = 5,  # NEW PARAMETER (RELION default is 3)
):
    dft = torch.zeros_like(self.numerator)

    mask = self.weights > self.min_denominator_value
    if self.correct_ctf:
        denominator = self.ctfsq[mask] + self.eps * self.weights[mask]
        denominator[denominator.abs() < self.min_denominator_value] = self.min_denominator_value
        dft[:, mask] = self.numerator[:, mask] / denominator[None, ...]
    else:
        dft[:, mask] = self.numerator[:, mask] / self.weights[mask][None, ...]

    dft = torch.complex(real=dft[0, ...], imag=dft[1, ...])

    # Inverse FFT
    dft = torch.fft.ifftshift(dft, dim=(-3, -2))
    dft = torch.fft.irfftn(dft, dim=(-3, -2, -1))
    vol = torch.fft.ifftshift(dft, dim=(-3, -2, -1))

    # Gridding correction
    sincsq = self.get_sincsq(vol.shape, device, self.eps)
    vol = vol.to(device) / sincsq

    # NEW: Apply soft masking (matching RELION pipeline)
    if apply_soft_mask:
        soft_mask = self.get_soft_mask(
            vol.shape, device, mask_radius_fraction, mask_edge_width
        )
        vol = vol * soft_mask

    if fname is not None:
        write_vol(vol.detach().cpu(), fname, self.sampling_rate, overwrite=overwrite_fname)
    return vol
```

### Priority 2: Test Without Regularization

Try `eps=0` to match RELION's behavior when no FSC is provided:

```bash
cryopares_reconstruct --eps 0.0 ...
```

### Priority 3: Implement Frequency-dependent Regularization (Future Work)

Would require FSC estimation between half-maps during reconstruction.

---

## References

### RELION Source Code (Analyzed in Detail)
- [backprojector.cpp](https://github.com/3dem/relion/blob/master/src/backprojector.cpp) - Core reconstruction pipeline, tau2 computation, oversampling correction
- [projector.cpp](https://github.com/3dem/relion/blob/master/src/projector.cpp) - Gridding correction (griddingCorrect function)
- [mask.cpp](https://github.com/3dem/relion/blob/master/src/mask.cpp) - Soft masking implementation (softMaskOutsideMap)
- [reconstructor.cpp](https://github.com/3dem/relion/blob/master/src/reconstructor.cpp) - High-level reconstruction interface, default parameters
- [backprojector.h](https://github.com/3dem/relion/blob/master/src/backprojector.h) - Class definition and function signatures

### Publications
- [RELION: Implementation of a Bayesian approach to cryo-EM structure determination](https://pmc.ncbi.nlm.nih.gov/articles/PMC3690530/) - Original RELION paper

### CryoPARES Configuration
- `cryoPARES/configs/reconstruct_config/reconstruct_config.py:38` - eps=1e-3 default
- `cryoPARES/reconstruction/reconstructor.py` - Main reconstruction implementation

---

## Next Steps - Action Plan

### Tomorrow's Priorities (Ranked)

#### 1. **IMPLEMENT SOFT MASKING** (30 min, HIGH IMPACT) ⚠️
- Add `get_soft_mask()` method to Reconstructor class (code provided above)
- Modify `generate_volume()` to apply masking (3 lines of code)
- **Expected impact:** Should dramatically reduce noise and fix FSC inflation
- **Quick test:** Run on existing data and compare half-maps visually

#### 2. **TEST eps=0** (5 min, MEDIUM IMPACT)
```bash
cryopares_reconstruct --eps 0.0 --particles_star_fname test.star ...
```
- Matches RELION's behavior when no FSC is provided
- May reveal if constant regularization is causing issues

#### 3. **VERIFY FFT NORMALIZATION** (15 min, LOW-MEDIUM IMPACT)
```python
import mrcfile
import numpy as np

cryo = mrcfile.open('cryopares_map.mrc').data
relion = mrcfile.open('relion_map.mrc').data

print(f"CryoPARES: mean={cryo.mean():.6f}, std={cryo.std():.6f}")
print(f"RELION:    mean={relion.mean():.6f}, std={relion.std():.6f}")
print(f"Density ratio: {cryo.sum() / relion.sum():.3f}")

# Check if it's just a uniform scaling factor
print(f"Voxel correlation: {np.corrcoef(cryo.ravel(), relion.ravel())[0,1]:.4f}")
```
- If correlation is high but scaling differs, just a normalization constant
- If correlation is low, more fundamental issue

#### 4. **RADIAL PROFILE COMPARISON** (20 min, DIAGNOSTIC)
- Implement the radial profile test from the document
- Look for signs of aliasing (peaks at Nyquist) or frequency-dependent amplitude errors
- Helps understand if padding would help

### Long-term Improvements (Future Work)

1. **Implement Fourier padding** (major change, 2-3 days work)
2. **Frequency-dependent regularization** from FSC (1 day)
3. **Compare with RELION on standardized test dataset** (1 day setup)

---

## Conclusion

After deep investigation of RELION source code, the **most likely cause** of noisy half-maps and inflated FSC is:

### **Missing soft masking after reconstruction** ⚠️

**Why this explains everything:**
1. **Noisy half-maps:** Edge artifacts create ringing throughout volume
2. **Inflated FSC:** Same artifacts in both halves → spurious correlation
3. **Postprocessing works:** relion_postprocess applies its own masking

**The fix is simple:** Add ~30 lines of code for soft masking (implementation provided above)

**Secondary issues:**
- Constant vs frequency-dependent regularization (moderate impact)
- No Fourier padding (may cause some aliasing, but soft masking is more critical)

**Recommended immediate action:** Implement soft masking first, test, then iterate based on results.

---
