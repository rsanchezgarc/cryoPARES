# Utility Scripts Guide

CryoPARES includes several utility scripts for analysis, visualization, and model management. This guide documents all available scripts, their use cases, and whether they're used automatically by the main pipeline.

## Table of Contents

- [GMM Histogram Analysis](#gmm-histogram-analysis) - Automatic (used by training)
- [FSC Computation](#fsc-computation) - Automatic (used by inference)
- [Pose Comparison](#pose-comparison) - Manual
- [Learning Curve Visualization](#learning-curve-visualization) - Manual
- [STAR File Histograms](#star-file-histograms) - Manual
- [Checkpoint Compactification](#checkpoint-compactification) - Manual

---

## GMM Histogram Analysis

**Script:** `cryoPARES.scripts.gmm_hists`
**Automatic Usage:** ✅ Yes - Called automatically during training to estimate confidence thresholds
**Manual Usage:** ✅ Yes - Can be run standalone for analysis

### Overview

Analyzes and compares score distributions between "good" (aligned) and "bad" (misaligned) particle populations using Gaussian Mixture Models (GMMs). Automatically estimates optimal thresholds for filtering low-quality particles.

### Automatic Usage in Training

During training, `compare_prob_hists()` is automatically called when `--junk_particles_star_fname` is provided:

```python
# Called in train.py
from cryoPARES.scripts.gmm_hists import compare_prob_hists

threshold, gmms, method = compare_prob_hists(
    fname_good=["aligned_particles.star"],
    fname_bad=["junk_particles.star"],
    score_name="rlnDirectionalZScore",
    compute_gmm=True
)
```

This generates diagnostic plots and estimates a confidence threshold for filtering during inference.

### Manual CLI Usage

Run as a standalone script to analyze score distributions:

```bash
python -m cryoPARES.scripts.gmm_hists \
    --fname_good aligned.star \
    --fname_all all_particles.star \
    --plot_fname results/distributions.png \
    --low_pct 1.0 \
    --up_pct 99.0 \
    --fn_cost 2.0
```

### Three Operating Modes

1. **GOOD + ALL mode**: Provide good particles and all particles; bad = all \\ good
   ```bash
   python -m cryoPARES.scripts.gmm_hists \
       --fname_good aligned.star \
       --fname_all all_particles.star
   ```

2. **GOOD + BAD mode**: Provide good and bad particles directly
   ```bash
   python -m cryoPARES.scripts.gmm_hists \
       --fname_good aligned.star \
       --fname_bad misaligned.star
   ```

3. **Symmetry-based inference**: Infer good/bad from angular error
   ```bash
   python -m cryoPARES.scripts.gmm_hists \
       --fname_good particles_with_poses.star \
       --symmetry C1 \
       --degs_error_thr 3.0
   ```

### Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--fname_good` | List[str] | **Required** | .star file(s) with GOOD particles |
| `--fname_all` | List[str] | None | .star file(s) with ALL particles (superset of GOOD) |
| `--fname_bad` | List[str] | None | .star file(s) with BAD particles (disjoint from GOOD) |
| `--score_name` | str | `rlnDirectionalZScore` | Score column to analyze |
| `--plot_fname` | str | None | Path to save histogram plot |
| `--low_pct` | float | 2.5 | Lower percentile for outlier clipping |
| `--up_pct` | float | 97.5 | Upper percentile for outlier clipping |
| `--fn_cost` | float | 2.0 | False negative cost multiplier (>1 = conservative) |
| `--fallback_method` | str | "auto" | GMM fallback: "auto" (ROC), "manual", "none" |
| `--symmetry` | str | None | Symmetry for angular error-based inference |
| `--degs_error_thr` | float | 3.0 | Angular error threshold (degrees) for GOOD |

### Algorithm

1. **Outlier clipping**: Remove extreme outliers using percentiles
2. **Adaptive GMM fitting**: Try 2-4 components, select best by BIC
3. **Robust component selection**:
   - Good: higher-mean component when weights comparable
   - Bad: lower-mean component when weights comparable
4. **Threshold computation**: Find weighted Gaussian intersection
5. **ROC fallback**: If GMM fails, use weighted Youden's J

### Output Files

- `plot_fname.png`: Main histogram (bad vs good overlay)
- `plot_fname_gmm.png`: GMM components and threshold visualization

### Quality Metrics

- **d-prime (d')**: Signal detection metric
  - d' > 2.0: Excellent separation
  - d' > 1.0: Good separation
  - d' < 1.0: Poor separation (warning issued)

### See Also

- Module documentation: `cryoPARES/scripts/gmm_hists.py` (lines 1-135)
- Training guide: [training_guide.md](training_guide.md)

---

## FSC Computation

**Script:** `cryoPARES.scripts.computeFsc`
**Automatic Usage:** ✅ Yes - Called automatically during inference when reconstructing volumes
**Manual Usage:** ✅ Yes - Can be run standalone for FSC analysis

### Overview

Computes Fourier Shell Correlation (FSC) between two 3D volumes to assess resolution. Includes robust crossing detection with bounce checking to avoid artifacts from noisy high-frequency regions.

### Automatic Usage in Inference

Automatically called when `cryopares_infer` processes half-sets:

```python
# Called in infer.py and inferencer.py
from cryoPARES.scripts.computeFsc import compute_fsc

fsc_curve, resolution_143, resolution_05 = compute_fsc(
    fname_vol_half1="reconstruction_half1.mrc",
    fname_vol_half2="reconstruction_half2.mrc",
    fname_fsc_out="fsc.txt",
    fname_mask=mask_path
)
```

Results are saved to `results_dir/fsc.txt` and used to report final resolution.

### Manual CLI Usage

```bash
python -m cryoPARES.scripts.computeFsc \
    --fname_vol_half1 half1.mrc \
    --fname_vol_half2 half2.mrc \
    --fname_fsc_out fsc_curve.txt \
    --fname_mask mask.mrc \
    --show_plot
```

### Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--fname_vol_half1` | str | **Required** | First half-map volume (.mrc) |
| `--fname_vol_half2` | str | **Required** | Second half-map volume (.mrc) |
| `--fname_fsc_out` | str | None | Output file for FSC curve (.txt) |
| `--fname_mask` | str | None | Mask for computing masked FSC |
| `--fname_plot_out` | str | None | Save FSC plot image |
| `--show_plot` | bool | False | Display FSC plot interactively |
| `--apply_cosine_edge` | bool | True | Apply cosine edge to mask |
| `--threshold_143` | float | 0.143 | Gold-standard FSC threshold (0.143) |
| `--threshold_05` | float | 0.5 | Alternative FSC threshold (0.5) |

### Bounce-Resistant Crossing Detection

The FSC computation uses sophisticated logic to avoid false crossings:

- **Persistence check**: Requires FSC to stay below threshold for 3+ bins
- **Rebound detection**: Ignores crossings that bounce back above threshold
- **Cutoff region**: Immediately accepts crossings at ≤15Å (high-confidence region)
- **Noisy region**: Applies stricter checks for worse-than-15Å crossings

### Output Format

The FSC curve file contains:

```
# Resolution(A)  FSC
50.0  0.998
25.0  0.987
12.5  0.942
6.25  0.856
3.12  0.143  <- Resolution at 0.143 threshold
```

### See Also

- FSC theory: Gold-standard FSC (Scheres & Chen, 2012)
- Inference output: [cli.md - Inference Output Files](cli.md#output-files)

---

## Pose Comparison

**Script:** `cryoPARES.scripts.compare_poses`
**Automatic Usage:** ❌ No - Manual analysis tool only
**Manual Usage:** ✅ Yes

### Overview

Compares predicted particle orientations between two STAR files to evaluate pose accuracy. Computes angular errors, generates histograms, and reports statistics. Essential for validating model performance.

### CLI Usage

```bash
python -m cryoPARES.scripts.compare_poses \
    --starfile1 predicted_poses.star \
    --starfile2 ground_truth_poses.star \
    --symmetry C1 \
    --output_plot error_histogram.png \
    --allow_partial
```

### Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--starfile1` | str | **Required** | First STAR file with particle poses |
| `--starfile2` | str | **Required** | Second STAR file with particle poses |
| `--symmetry` | str | "C1" | Point group symmetry for error computation |
| `--output_plot` | str | None | Save error histogram plot |
| `--output_csv` | str | None | Save per-particle errors to CSV |
| `--allow_partial` | bool | False | Allow partial matching of particles |

### Features

- **Symmetry-aware errors**: Accounts for point group symmetries (C, D, T, O, I)
- **Particle matching**: Matches particles by image name across STAR files
- **Partial matching**: Option to compare only common particles
- **Statistics**: Mean, median, percentiles, and histograms of angular errors

### Output

Statistics printed to console:
```
Angular Error Statistics (degrees):
  Mean: 2.34
  Median: 1.87
  75th percentile: 3.12
  90th percentile: 4.56
  95th percentile: 5.89
  Particles matched: 10000
```

Histogram saved to `output_plot` if specified.

### Use Cases

1. **Model validation**: Compare predicted vs ground-truth poses
2. **Method comparison**: Compare poses from different algorithms
3. **Refinement assessment**: Evaluate improvement after local refinement

### See Also

- Symmetry handling: `cryoPARES/geometry/symmetry.py`
- Angular metrics: `cryoPARES/geometry/metrics_angles.py`

---

## Learning Curve Visualization

**Script:** `cryoPARES.scripts.plot_learning_curve`
**Automatic Usage:** ❌ No - Manual visualization tool only
**Manual Usage:** ✅ Yes

### Overview

Visualizes training metrics from PyTorch Lightning CSV logs to monitor training progress. Plots loss curves, geometric errors, and validation metrics with robust spike filtering.

### CLI Usage

```bash
python -m cryoPARES.scripts.plot_learning_curve \
    --csv_file /path/to/training/version_0/metrics.csv \
    --skip_steps 100 \
    --log_scale \
    --percentile_low 5 \
    --percentile_high 95
```

### Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--csv_file` | str | **Required** | Path to PyTorch Lightning metrics CSV |
| `--skip_steps` | int | 0 | Skip first N steps (for zooming into convergence) |
| `--log_scale` | bool | False | Use logarithmic y-axis scale |
| `--percentile_low` | float | 5.0 | Lower percentile for robust y-limits |
| `--percentile_high` | float | 95.0 | Upper percentile for robust y-limits |

### Metrics Plotted

**Top subplot: Geometric Errors**
- `geo_degs_epoch`: Training angular error per epoch
- `val_geo_degs`: Validation angular error
- `val_median_geo_degs`: Median validation angular error

**Bottom subplot: Loss Values**
- `loss`: Training loss
- `val_loss`: Validation loss

### Spike-Resistant Plotting

Uses percentile-based y-limits to avoid distortion from:
- Initial training spikes
- Validation outliers
- Numerical instabilities

### Finding the CSV File

PyTorch Lightning saves metrics to:
```
train_save_dir/
└── version_0/
    └── metrics.csv  <- Use this file
```

### Example Workflow

```bash
# Monitor training in real-time
tensorboard --logdir /path/to/training/version_0

# After training, create publication-quality plots
python -m cryoPARES.scripts.plot_learning_curve \
    --csv_file /path/to/training/version_0/metrics.csv \
    --skip_steps 500 \
    --log_scale
```

### See Also

- Training guide: [training_guide.md](training_guide.md)
- TensorBoard: [PyTorch Lightning Logging](https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html)

---

## STAR File Histograms

**Script:** `cryoPARES.scripts.hists_from_starfile`
**Automatic Usage:** ❌ No - Manual analysis tool only
**Manual Usage:** ✅ Yes

### Overview

Generates histograms of any metadata column(s) from RELION STAR files. Useful for quality control, data exploration, and identifying outliers.

### CLI Usage

```bash
python -m cryoPARES.scripts.hists_from_starfile \
    --input particles.star \
    --cols rlnDirectionalZScore rlnDefocusU rlnDefocusV \
    --output histograms.png \
    --clip_percentile 1 99 1 99 1 99
```

### Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--input` | str | **Required** | Path to input STAR file |
| `--cols` | List[str] | **Required** | Column names to plot |
| `--output` | str | None | Save plot image (if not provided, shows interactively) |
| `--nbins` | str/int | "auto" | Number of histogram bins |
| `--clip_percentile` | List[float] | None | Clip to percentile range (pairs of lower,upper per column) |
| `--clip_value` | List[float] | None | Clip to absolute value range (pairs of lower,upper per column) |

### Example Use Cases

**Inspect CTF parameters:**
```bash
python -m cryoPARES.scripts.hists_from_starfile \
    --input particles.star \
    --cols rlnDefocusU rlnDefocusV rlnDefocusAngle \
    --output ctf_distributions.png
```

**Analyze confidence scores:**
```bash
python -m cryoPARES.scripts.hists_from_starfile \
    --input inference_results/particles_aligned.star \
    --cols rlnDirectionalZScore \
    --clip_percentile 1 99 \
    --output confidence_scores.png
```

**Check particle coordinates:**
```bash
python -m cryoPARES.scripts.hists_from_starfile \
    --input particles.star \
    --cols rlnCoordinateX rlnCoordinateY
```

### Output Format

- Multi-panel plot with up to 3 columns
- Each subplot shows:
  - Histogram with count (left y-axis)
  - Percentage scale (right y-axis)
  - Statistics printed to console

### Common STAR File Columns

**Pose information:**
- `rlnAngleRot`, `rlnAngleTilt`, `rlnAnglePsi`: Euler angles
- `rlnOriginX`, `rlnOriginY`: Translational shifts

**CTF parameters:**
- `rlnDefocusU`, `rlnDefocusV`: Defocus values (Å)
- `rlnDefocusAngle`: Astigmatism angle
- `rlnCtfMaxResolution`: CTF fit resolution

**Quality metrics (CryoPARES-specific):**
- `rlnDirectionalZScore`: Confidence z-score
- `rlnLogLikelihood`: Log-likelihood score

### See Also

- RELION STAR format: [RELION documentation](https://relion.readthedocs.io/)
- Quality control: [troubleshooting.md](troubleshooting.md)

---

## Checkpoint Compactification

**Script:** `cryoPARES.scripts.compactify_checkpoint`
**Automatic Usage:** ❌ No - Manual packaging tool
**Manual Usage:** ✅ Yes

### Overview

Packages trained checkpoint directories into compact ZIP files for easy distribution and inference. Removes training artifacts while preserving everything needed for inference.

**Already documented in [cli.md](cli.md#compactify_checkpoint)** - see that section for complete documentation.

### Quick Reference

```bash
python -m cryoPARES.scripts.compactify_checkpoint \
    --checkpoint_dir /path/to/training/version_0 \
    --output_path my_model.zip \
    --no-reconstructions  # Optional: exclude maps to save space
```

**Size reduction:** Typically 40 GB → 10 GB (75% savings)

**Usage with inference:**
```bash
cryopares_infer \
    --checkpoint_dir my_model.zip \
    --particles_star_fname particles.star \
    --results_dir results/
```

### See Also

- Full documentation: [cli.md - compactify_checkpoint](cli.md#compactify_checkpoint)
- Inference guide: [cli.md - cryopares_infer](cli.md#cryopares_infer)

---

## Summary Table

| Script | Auto-Used | Used By | Purpose |
|--------|-----------|---------|---------|
| `gmm_hists` | ✅ Yes | Training | Confidence threshold estimation |
| `computeFsc` | ✅ Yes | Inference | Resolution assessment |
| `compare_poses` | ❌ No | - | Pose accuracy validation |
| `plot_learning_curve` | ❌ No | - | Training visualization |
| `hists_from_starfile` | ❌ No | - | STAR file QC |
| `compactify_checkpoint` | ❌ No | - | Model packaging |

---

## Integration with Main Pipeline

### Training Pipeline

```
cryopares_train
    ├── Load particles from STAR files
    ├── Train model (multiple epochs)
    ├── [AUTOMATIC] gmm_hists.compare_prob_hists()
    │   └── If --junk_particles_star_fname provided
    │       └── Estimate confidence threshold
    │       └── Save diagnostic plots
    └── Save checkpoint
```

### Inference Pipeline

```
cryopares_infer
    ├── Load model from checkpoint
    ├── Predict poses for particles
    ├── (Optional) Local refinement
    ├── Reconstruct 3D volumes
    └── [AUTOMATIC] computeFsc.compute_fsc()
        └── If half-sets available
            └── Compute FSC between half-maps
            └── Report resolution at 0.143
```

### Manual Analysis Workflow

```
1. Train model
   └── cryopares_train ...

2. Monitor training
   └── plot_learning_curve --csv_file metrics.csv

3. Run inference
   └── cryopares_infer ...

4. Validate poses
   └── compare_poses --starfile1 predicted.star --starfile2 ground_truth.star

5. Inspect results
   └── hists_from_starfile --input particles_aligned.star --cols rlnDirectionalZScore

6. Package for distribution
   └── compactify_checkpoint --checkpoint_dir version_0
```

---

## Developer Notes

All scripts use `argParseFromDoc` for automatic CLI generation. To add a new script:

1. Write function with type hints
2. Add comprehensive docstring with parameter descriptions
3. Use `parse_function_and_call()` in `if __name__ == "__main__"`

Example:
```python
def my_analysis(
    input_file: str,
    output_dir: str,
    threshold: float = 0.5
):
    """
    Brief description.

    :param input_file: Path to input file
    :param output_dir: Output directory
    :param threshold: Threshold value (default: 0.5)
    """
    # Implementation here
    pass

if __name__ == "__main__":
    from argParseFromDoc import parse_function_and_call
    parse_function_and_call(my_analysis)
```

This automatically generates a full CLI with `--help`, type validation, and default values.
