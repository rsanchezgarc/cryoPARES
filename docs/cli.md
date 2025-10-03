# Command-Line Interface

This document provides instructions for using the command-line tools included with cryoPARES.

## Table of Contents

- [`cryopares_train`](#cryopares_train) - Train a new model
- [`cryopares_infer`](#cryopares_infer) - Run inference on new particles
- [`cryopares_projmatching`](#cryopares_projmatching) - Align particles via projection matching
- [`cryopares_reconstruct`](#cryopares_reconstruct) - Reconstruct 3D volume from aligned particles
- [`compactify_checkpoint`](#compactify_checkpoint) - Package checkpoint for distribution

---

<!-- AUTO_GENERATED:train_cli:START -->
## `cryopares_train`

Train a CryoPARES model on pre-aligned particle data.

### Usage

```bash
cryopares_train [OPTIONS]
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--symmetry` | str | **Required** | Point group symmetry of the molecule (e.g., C1, D7, I, O, T) |
| `--particles_star_fname` | List[str] | **Required** | Path(s) to RELION 3.1+ format .star file(s) containing pre-aligned particles. Can accept multiple files |
| `--train_save_dir` | str | **Required** | Output directory where model checkpoints, logs, and training artifacts will be saved |
| `--particles_dir` | Optional[List[str]] | None | Root directory for particle image paths. If paths in .star file are relative, this directory is prepended (similar to RELION project directory concept) |
| `--n_epochs` | int | `100` | Number of training epochs. More epochs allow better convergence, although it does not help beyond a certain point |
| `--batch_size` | int | `64` | Number of particles per batch. Try to make it as large as possible before running out of GPU memory. We advice using batch sizes of at least 32 images |
| `--num_dataworkers` | int | `8` | Number of parallel data loading workers per GPU. Each worker is a separate CPU process. Set to 0 to load data in the main thread (useful for debugging) |
| `--image_size_px_for_nnet` | int | `160` | Target image size in pixels for neural network input. After rescaling to target sampling rate, images are cropped or padded to this size |
| `--sampling_rate_angs_for_nnet` | float | `1.5` | Target sampling rate in Angstroms/pixel for neural network input. Particle images are first rescaled to this sampling rate before processing |
| `--mask_radius_angs` | Optional[float] | None | Radius of circular mask in Angstroms applied to particle images. If not provided, defaults to half the box size |
| `--split_halfs` | bool | `True` | If True (default), trains two separate models on data half-sets for cross-validation. Use --NOT_split_halfs to train single model on all data |
| `--continue_checkpoint_dir` | Optional[str] | None | Path to checkpoint directory to resume training from a previous run |
| `--finetune_checkpoint_dir` | Optional[str] | None | Path to checkpoint directory to fine-tune a pre-trained model on new dataset |
| `--compile_model` | bool | `False` | Enable torch.compile for faster training (experimental) |
| `--val_check_interval` | Optional[float] | None | Fraction of epoch between validation checks. You generally don't want to touch it, but you can set it to smaller values (0.1-0.5) for large datasets to get quicker feedback |
| `--overfit_batches` | Optional[int] | None | Number of batches to use for overfitting test (debugging feature to verify model can memorize small dataset) |
| `--map_fname_for_simulated_pretraining` | Optional[List[str]] | None | Path(s) to reference map(s) for simulated projection warmup before training on real data. The number of maps must match number of particle star files |
| `--junk_particles_star_fname` | Optional[List[str]] | None | Optional star file(s) with junk-only particles for estimating confidence z-score thresholds |
| `--junk_particles_dir` | Optional[List[str]] | None | Root directory for junk particle image paths (analogous to particles_dir) |
<!-- AUTO_GENERATED:train_cli:END -->

- **`--map_fname_for_simulated_pretraining MAP_FNAME`**
  Reference map(s) for simulated pre-training
  Must match length of `--particles_star_fname`

### Configuration Overrides

Use `--config` as the **last** argument to override configuration parameters:

```bash
--config KEY=VALUE KEY2=VALUE2 ...
```

Common configuration overrides:

```bash
--config \
    train.learning_rate=1e-3 \
    train.weight_decay=1e-5 \
    train.accumulate_grad_batches=16 \
    models.image2sphere.lmax=8 \
    datamanager.particlesDataset.sampling_rate_angs_for_nnet=2.0 \
    datamanager.particlesDataset.image_size_px_for_nnet=128
```

### View All Config Options

```bash
python -m cryopares_train --show-config
```

### Examples

**Basic training:**
```bash
cryopares_train \
    --symmetry C1 \
    --particles_star_fname /path/to/aligned_particles.star \
    --train_save_dir /path/to/output \
    --n_epochs 20
```

**Training with custom parameters:**
```bash
cryopares_train \
    --symmetry D7 \
    --particles_star_fname /path/to/particles.star \
    --particles_dir /path/to/particles \
    --train_save_dir /path/to/output \
    --n_epochs 30 \
    --batch_size 64 \
    --compile_model \
    --config \
        train.learning_rate=5e-3 \
        models.image2sphere.lmax=10 \
        datamanager.particlesDataset.sampling_rate_angs_for_nnet=1.5
```

**Continue training from checkpoint:**
```bash
cryopares_train \
    --continue_checkpoint_dir /path/to/output/version_0 \
    --n_epochs 50
```

**Fine-tune on new data:**
```bash
cryopares_train \
    --symmetry C1 \
    --particles_star_fname /path/to/new_particles.star \
    --train_save_dir /path/to/finetuned_output \
    --finetune_checkpoint_dir /path/to/pretrained/version_0 \
    --n_epochs 10 \
    --config train.learning_rate=1e-4
```

### Important Notes

- **File descriptor limit:** Run `ulimit -n 65536` before training to avoid "too many open files" errors
- **GPU memory:** Reduce `--batch_size` or `image_size_px_for_nnet` if you encounter OOM errors
- **Monitoring:** Use TensorBoard to monitor training: `tensorboard --logdir /path/to/output/version_0`

### See Also

- [Training Guide](training_guide.md) - Comprehensive training guide
- [Configuration Guide](configuration_guide.md) - All configuration parameters

---

<!-- AUTO_GENERATED:inference_cli:START -->
## `cryopares_infer`

Run inference on new particles using a trained model.

### Usage

```bash
cryopares_infer [OPTIONS]
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--particles_star_fname` | str | **Required** | Path to input STAR file with particle metadata |
| `--checkpoint_dir` | str | **Required** | Path to training directory (or .zip file) containing half-set models with checkpoints and hyperparameters |
| `--results_dir` | str | **Required** | Output directory for inference results including predicted poses and optional reconstructions |
| `--data_halfset` | 'half1', 'half2', 'allParticles' | `allParticles` | Which particle half-set(s) to process: "half1", "half2", or "allParticles" |
| `--model_halfset` | 'half1', 'half2', 'allCombinations', 'matchingHalf' | `matchingHalf` | Model half-set selection policy: "half1", "half2", "allCombinations", or "matchingHalf" (uses matching data/model pairs) |
| `--particles_dir` | Optional[str] | None | Root directory for particle image paths. If provided, overrides paths in the .star file |
| `--batch_size` | int | `64` | Number of particles to process simultaneously per job |
| `--n_jobs` | Optional[int] | None | Number of parallel worker processes for distributed projection matching |
| `--num_dataworkers` | int | `8` | Number of parallel data loading workers per GPU. Each worker is a separate CPU process. Set to 0 to load data in the main thread (useful for debugging) |
| `--use_cuda` | bool | `True` | Enable GPU acceleration. If False, runs on CPU only |
| `--n_cpus_if_no_cuda` | int | `4` | Maximum CPU threads per worker when CUDA is disabled |
| `--compile_model` | bool | `False` | Compile model with torch.compile for faster inference (experimental, requires PyTorch 2.0+) |
| `--top_k_poses_nnet` | int | `1` | Number of top pose predictions to retrieve from neural network before local refinement |
| `--top_k_poses_localref` | int | `1` | Number of best matching poses to keep after local refinement |
| `--grid_distance_degs` | float | `6.0` | Maximum angular distance in degrees for local refinement search. Grid ranges from -grid_distance_degs to +grid_distance_degs around predicted pose |
| `--reference_map` | Optional[str] | None | Path to reference map (.mrc) for FSC computation during validation |
| `--reference_mask` | Optional[str] | None | Path to reference mask (.mrc) for masked FSC calculation |
| `--directional_zscore_thr` | Optional[float] | None | Confidence z-score threshold for filtering particles. Particles with scores below this are discarded as low-confidence |
| `--skip_localrefinement` | bool | `False` | Skip local pose refinement step and use only neural network predictions |
| `--skip_reconstruction` | bool | `False` | Skip 3D reconstruction step and output only predicted poses |
| `--subset_idxs` | Optional[List[int]] | None | List of particle indices to process (for debugging or partial processing) |
| `--n_first_particles` | Optional[int] | None | Process only the first N particles from dataset (for testing or validation) |
| `--check_interval_secs` | float | `2.0` | Polling interval in seconds for parent loop in distributed processing |
<!-- AUTO_GENERATED:inference_cli:END -->

### Configuration Overrides

```bash
--config \
    inference.directional_zscore_thr=2.0 \
    inference.top_k_poses_nnet=10 \
    inference.skip_localrefinement=False \
    inference.skip_reconstruction=False \
    projmatching.grid_distance_degs=8.0 \
    projmatching.grid_step_degs=2.0
```

### Examples

**Basic inference:**
```bash
cryopares_infer \
    --particles_star_fname /path/to/new_particles.star \
    --checkpoint_dir /path/to/training/version_0 \
    --results_dir /path/to/inference_results
```

**Inference with reference map:**
```bash
cryopares_infer \
    --particles_star_fname /path/to/particles.star \
    --checkpoint_dir /path/to/training/version_0 \
    --results_dir /path/to/results \
    --reference_map /path/to/reference.mrc \
    --config \
        inference.directional_zscore_thr=2.0 \
        projmatching.grid_distance_degs=10.0
```

**Process only half1 with matching model:**
```bash
cryopares_infer \
    --particles_star_fname /path/to/particles.star \
    --checkpoint_dir /path/to/training/version_0 \
    --results_dir /path/to/results \
    --data_halfset half1 \
    --model_halfset matchingHalf
```

**Fast inference (skip reconstruction):**
```bash
cryopares_infer \
    --particles_star_fname /path/to/particles.star \
    --checkpoint_dir /path/to/training/version_0 \
    --results_dir /path/to/results \
    --batch_size 2048 \
    --config inference.skip_reconstruction=True
```

### Output Files

The inference process creates:
- `results_dir/particles_aligned.star` - Aligned particles with predicted poses
- `results_dir/reconstruction.mrc` - 3D reconstruction (if not skipped)
- `results_dir/fsc.txt` - FSC curve (if half-sets used)
- `results_dir/inference.log` - Inference log

### See Also

- [API Reference](https://rsanchezgarc.github.io/cryoPARES/api/) - Detailed API documentation
- [Troubleshooting Guide](troubleshooting.md) - Common issues

---

<!-- AUTO_GENERATED:projmatching_cli:START -->
## `cryopares_projmatching`

Align particles to a reference volume using projection matching.

### Usage

```bash
cryopares_projmatching [OPTIONS]
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--reference_vol` | str | **Required** | Path to reference 3D volume (.mrc file) for generating projection templates |
| `--particles_star_fname` | str | **Required** | Path to input STAR file with particle metadata |
| `--out_fname` | str | **Required** | Path for output STAR file with aligned particle poses |
| `--particles_dir` | Optional[str] | **Required** | Root directory for particle image paths. If provided, overrides paths in the .star file |
| `--mask_radius_angs` | Optional[float] | None | Radius of circular mask in Angstroms applied to particle images |
| `--grid_distance_degs` | float | `8.0` | Maximum angular distance in degrees for local refinement search. Grid ranges from -grid_distance_degs to +grid_distance_degs around predicted pose |
| `--grid_step_degs` | float | `2.0` | Angular step size in degrees for grid search during local refinement |
| `--return_top_k_poses` | int | `1` | Number of top matching poses to save per particle |
| `--filter_resolution_angst` | Optional[float] | None | Low-pass filter resolution in Angstroms applied to reference volume before matching |
| `--n_jobs` | int | `1` | Number of parallel worker processes for distributed projection matching |
| `--num_dataworkers` | int | `1` | Number of CPU workers per PyTorch DataLoader for data loading |
| `--batch_size` | int | `1024` | Number of particles to process simultaneously per job |
| `--use_cuda` | bool | `True` | Enable GPU acceleration. If False, runs on CPU only |
| `--verbose` | bool | `True` | Enable progress logging and status messages |
| `--float32_matmul_precision` | 'highest', 'high', 'medium' | `high` | PyTorch float32 matrix multiplication precision mode (highest/high/medium). Higher is more accurate but slower |
| `--gpu_id` | Optional[int] | None | Specific GPU device ID to use (if multiple GPUs available) |
| `--n_first_particles` | Optional[int] | None | Process only the first N particles from dataset (for testing or validation) |
| `--correct_ctf` | bool | `True` | Apply CTF correction during processing |
| `--halfmap_subset` | Optional['1', '2' | None | Select half-map subset (1 or 2) for half-map validation |
<!-- AUTO_GENERATED:projmatching_cli:END -->

### Example

```bash
cryopares_projmatching \
    --reference_vol /path/to/your/reference.mrc \
    --particles_star_fname /path/to/your/particles.star \
    --out_fname /path/to/your/aligned_particles.star \
    --grid_distance_degs 10
```

---

<!-- AUTO_GENERATED:reconstruct_cli:START -->
## `cryopares_reconstruct`

Reconstruct a 3D volume from particles with known poses.

### Usage

```bash
cryopares_reconstruct [OPTIONS]
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--particles_star_fname` | str | **Required** | Path to input STAR file with particle metadata and poses to reconstruct |
| `--symmetry` | str | **Required** | Point group symmetry of the volume for reconstruction (e.g., C1, D2, I, O, T) |
| `--output_fname` | str | **Required** | Path for output reconstructed 3D volume (.mrc file) |
| `--particles_dir` | Optional[str] | None | Root directory for particle image paths. If provided, overrides paths in the .star file |
| `--n_jobs` | int | `1` | Number of parallel worker processes for distributed reconstruction |
| `--num_dataworkers` | int | `1` | Number of CPU workers per PyTorch DataLoader for data loading |
| `--batch_size` | int | `128` | Number of particles to backproject simultaneously per job |
| `--use_cuda` | bool | `True` | Enable GPU acceleration for reconstruction. If False, runs on CPU only |
| `--correct_ctf` | bool | `True` | Apply CTF correction during reconstruction |
| `--eps` | float | `0.001` | Regularization constant for reconstruction (ideally set to 1/SNR). Prevents division by zero and stabilizes reconstruction |
| `--min_denominator_value` | Optional[float] | None | Minimum value for denominator to prevent numerical instabilities during reconstruction |
| `--use_only_n_first_batches` | Optional[int] | None | Reconstruct using only first N batches (for testing or quick validation) |
| `--float32_matmul_precision` | Optional[str] | `high` | PyTorch float32 matrix multiplication precision mode (highest/high/medium). Higher is more accurate but slower |
| `--weight_with_confidence` | bool | `False` | Apply per-particle confidence weighting during backprojection. If True, particles with higher confidence contribute more to reconstruction |
| `--halfmap_subset` | Optional['1', '2' | None | Select half-map subset (1 or 2) for half-map reconstruction and validation |
<!-- AUTO_GENERATED:reconstruct_cli:END -->

### Example

```bash
cryopares_reconstruct \
    --particles_star_fname /path/to/your/particles.star \
    --symmetry C1 \
    --output_fname /path/to/your/reconstruction.mrc
```

---

## `compactify_checkpoint`

Package a trained checkpoint directory into a compact ZIP file for easy distribution and inference. This tool removes unnecessary files (training logs, metrics, intermediate checkpoints, etc.) and keeps only what's needed for inference.

### Usage

```bash
python -m cryoPARES.scripts.compactify_checkpoint [options]
```

### Required Arguments

- **`--checkpoint_dir CHECKPOINT_DIR`**
  Path to checkpoint directory (e.g., `/path/to/train_output/version_0`)

### Optional Arguments

- **`--output_path OUTPUT_PATH`**
  Output ZIP file path
  Default: `<checkpoint_dir_name>_compact.zip`

- **`--no-reconstructions`**
  Exclude reconstruction files to reduce size
  You'll need to provide `--reference_map` during inference if you use this option

- **`--no-compression`**
  Store files without compression (faster but larger)

- **`--quiet`**
  Suppress progress messages

### Examples

**Basic usage:**
```bash
python -m cryoPARES.scripts.compactify_checkpoint \
    --checkpoint_dir /path/to/training/version_0
```

Output: `/path/to/training/version_0_compact.zip`

**Custom output name:**
```bash
python -m cryoPARES.scripts.compactify_checkpoint \
    --checkpoint_dir /path/to/training/version_0 \
    --output_path my_model_C1.zip
```

**Exclude reconstructions (smaller size):**
```bash
python -m cryoPARES.scripts.compactify_checkpoint \
    --checkpoint_dir /path/to/training/version_0 \
    --output_path my_model_compact.zip \
    --no-reconstructions
```

### What's Included

The compactified checkpoint contains only files required for inference:

**For each half-set (half1, half2, or allParticles):**
- `checkpoints/best_script.pt` (TorchScript model, preferred)
- `checkpoints/best.ckpt` (fallback if best_script.pt doesn't exist)
- `checkpoints/best_directional_normalizer.pt` (for confidence scoring)
- `hparams.yaml` (model hyperparameters)
- `reconstructions/0.mrc` (reference map, optional)

**At root:**
- `configs_*.yml` (training configuration)

### Size Reduction

Typical size reduction: **40 GB → 10 GB** (75% reduction)

Most of the space savings come from removing:
- TensorBoard event logs
- Intermediate checkpoints (last.ckpt, etc.)
- Training metrics and validation data
- Code snapshots

Example output:
```
Original size:  42.15 GB
Compact size:   9.87 GB
Savings:        32.28 GB (76.6%)
```

### Using Compactified Checkpoints

You can use the ZIP file directly for inference:

```bash
cryopares_infer \
    --particles_star_fname /path/to/particles.star \
    --checkpoint_dir my_model_compact.zip \
    --results_dir /path/to/results
```

CryoPARES automatically detects ZIP files and reads models directly from the archive without extraction.

### See Also

- [Training Guide](training_guide.md) - How to train models
- [API Reference](https://rsanchezgarc.github.io/cryoPARES/api/) - Programmatic usage