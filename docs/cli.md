# Command-Line Interface

This document provides instructions for using the command-line tools included with cryoPARES.

## Table of Contents

- [`cryopares_train`](#cryopares_train) - Train a new model
- [`cryopares_infer`](#cryopares_infer) - Run inference on new particles
- [`cryopares_projmatching`](#cryopares_projmatching) - Align particles via projection matching
- [`cryopares_reconstruct`](#cryopares_reconstruct) - Reconstruct 3D volume from aligned particles
- [`compactify_checkpoint`](#compactify_checkpoint) - Package checkpoint for distribution

---

## `cryopares_train`

Train a new CryoPARES model on pre-aligned particle data.

### Usage

```bash
python -m cryopares_train [options] --config [CONFIG_OVERRIDES]
```

Or using the installed script:

```bash
cryopares_train [options] --config [CONFIG_OVERRIDES]
```

### Required Arguments

- **`--symmetry SYMMETRY`**
  Point group symmetry of the molecule (e.g., `C1`, `D7`, `T`, `O`, `I`)

- **`--particles_star_fname PARTICLES_STAR_FNAME`**
  Path to the RELION .star file containing pre-aligned particles
  Can be a single file or multiple files (space-separated)

- **`--train_save_dir TRAIN_SAVE_DIR`**
  Directory where model checkpoints and logs will be saved
  Creates `version_0/`, `version_1/`, etc. subdirectories

### Optional Arguments

- **`--particles_dir PARTICLES_DIR`**
  Root directory for particle files. If paths in .star file are relative, this is prepended.
  Default: Directory containing the .star file

- **`--n_epochs N_EPOCHS`**
  Number of training epochs
  Default: `10`

- **`--batch_size BATCH_SIZE`**
  Number of particles per batch
  Default: `32`

- **`--num_dataworkers NUM_DATAWORKERS`**
  Number of parallel data loading workers
  Default: `4`
  Set to `0` for single-threaded loading (useful for debugging)

- **`--split_halfs`**
  Train separate models for each half-set (recommended)
  Default: `True`

- **`--continue_checkpoint_dir CONTINUE_CHECKPOINT_DIR`**
  Path to checkpoint directory to continue training from
  Format: `/path/to/train_save_dir/version_0`
  Mutually exclusive with `--finetune_checkpoint_dir`

- **`--finetune_checkpoint_dir FINETUNE_CHECKPOINT_DIR`**
  Path to checkpoint directory to fine-tune from
  Starts from pre-trained weights but resets optimizer
  Mutually exclusive with `--continue_checkpoint_dir`

- **`--compile_model`**
  Compile model with `torch.compile` for potential speedup
  Requires PyTorch 2.0+

- **`--val_check_interval VAL_CHECK_INTERVAL`**
  Fraction of epoch between validation checks
  Default: `None` (validate once per epoch)

- **`--overfit_batches OVERFIT_BATCHES`**
  Number of batches to overfit on (for debugging/testing)

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

## `cryopares_infer`

Run inference on new particle datasets using a trained model.

### Usage

```bash
cryopares_infer [options] --config [CONFIG_OVERRIDES]
```

### Required Arguments

- **`--particles_star_fname PARTICLES_STAR_FNAME`**
  Path to .star file with particles to process
  Particles do not need to be pre-aligned

- **`--checkpoint_dir CHECKPOINT_DIR`**
  Path to trained model checkpoint directory
  Format: `/path/to/train_save_dir/version_0`

- **`--results_dir RESULTS_DIR`**
  Output directory for inference results
  Creates aligned .star file and reconstructed map

### Optional Arguments

- **`--particles_dir PARTICLES_DIR`**
  Root directory for particle files
  Default: Directory containing the .star file

- **`--data_halfset {half1,half2,allParticles}`**
  Which data half-set to process
  Default: `allParticles`

- **`--model_halfset {half1,half2,allCombinations,matchingHalf}`**
  Which model to use for inference
  Default: `matchingHalf` (recommended)

- **`--batch_size BATCH_SIZE`**
  Batch size for inference
  Default: `1024`
  Can be much larger than training batch size

- **`--num_dataworkers NUM_DATAWORKERS`**
  Number of data loading workers
  Default: `4`

- **`--reference_map REFERENCE_MAP`**
  Path to reference map (.mrc) for local refinement
  If not provided, uses half-maps from training

- **`--reference_mask REFERENCE_MASK`**
  Path to mask for FSC calculation
  Optional

- **`--compile_model`**
  Compile model for potential speedup

- **`--n_first_particles N`**
  Process only first N particles (for testing)

- **`--subset_idxs INDICES`**
  Process specific particle indices

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

## `cryopares_projmatching`

Align particles to a reference volume using projection matching. This performs local refinement by searching around existing particle orientations in the .star file.

### Usage

```
cryopares_reconstruct [options]
```

### Options

```
usage: reconstruct_starfile [-h] --particles_star_fname PARTICLES_STAR_FNAME
                            --symmetry SYMMETRY --output_fname OUTPUT_FNAME
                            [--particles_dir PARTICLES_DIR] [--n_jobs N_JOBS]
                            [--num_dataworkers NUM_DATAWORKERS]
                            [--batch_size BATCH_SIZE] [--NOT_use_cuda]
                            [--NOT_correct_ctf] [--eps EPS]
                            [--min_denominator_value MIN_DENOMINATOR_VALUE]
                            [--use_only_n_first_batches USE_ONLY_N_FIRST_BATCHES]
                            [--float32_matmul_precision FLOAT32_MATMUL_PRECISION]
                            [--weight_with_confidence]
                            [--halfmap_subset {1,2}]

options:
  -h, --help            show this help message and exit
  --particles_star_fname PARTICLES_STAR_FNAME
                        The particles to reconstruct Default=None
  --symmetry SYMMETRY   The symmetry of the volume (e.g. C1, D2, ...)
                        Default=None
  --output_fname OUTPUT_FNAME
                        The name of the output filename Default=None
  --particles_dir PARTICLES_DIR
                        The particles directory (root of the starfile fnames)
                        Default=None
  --n_jobs N_JOBS       The number of workers to split the reconstruction
                        process Default=1
  --num_dataworkers NUM_DATAWORKERS
                        Num workers for data loading Default=1
  --batch_size BATCH_SIZE
                        The number of particles to be simultaneusly
                        backprojected Default=128
  --NOT_use_cuda        if NOT, it will not use cuda devices Action:
                        store_false for variable use_cuda
  --NOT_correct_ctf     if NOT, it will not correct CTF Action: store_false
                        for variable correct_ctf
  --eps EPS             The regularization constant (ideally, this is 1/SNR)
                        Default=0.001
  --min_denominator_value MIN_DENOMINATOR_VALUE
                        Used to prevent division by 0 Default=None
  --use_only_n_first_batches USE_ONLY_N_FIRST_BATCHES
                        Use only the n first batches to reconstruct
                        Default=None
  --float32_matmul_precision FLOAT32_MATMUL_PRECISION
                        Set it to high or medium for speed up at a precision
                        cost Default=high
  --weight_with_confidence
                        If True, read and apply per-particle confidence. If
                        False (default), do NOT fetch/pass confidence (zero
                        overhead). Action: store_true for variable
                        weight_with_confidence
  --halfmap_subset {1,2}
                        The random subset of particles to use Default=None
```

### Example

```bash
cryopares_reconstruct \
    --particles_star_fname /path/to/your/particles.star \
    --symmetry C1 \
    --output_fname /path/to/your/reconstruction.mrc
```

## `cryopares_projmatching`

This tool aligns particles from a STAR file to a reference volume using projection matching.

### Usage

```
cryopares_projmatching [options]
```

### Options

```
usage: projmatching_starfile [-h] --reference_vol REFERENCE_VOL
                             --particles_star_fname PARTICLES_STAR_FNAME
                             --out_fname OUT_FNAME
                             [--particles_dir PARTICLES_DIR]
                             [--mask_radius_angs MASK_RADIUS_ANGS]
                             [--grid_distance_degs GRID_DISTANCE_DEGS]
                             [--grid_step_degs GRID_STEP_DEGS]
                             [--top_k_poses_nnet TOP_K_POSES_NNET]
                             [--filter_resolution_angst FILTER_RESOLUTION_ANGST]
                             [--n_jobs N_JOBS]
                             [--num_dataworkers NUM_DATAWORKERS]
                             [--batch_size BATCH_SIZE] [--NOT_use_cuda]
                             [--NOT_verbose]
                             [--torch_matmul_precision {highest,high,medium}]
                             [--gpu_id GPU_ID]
                             [--n_first_particles N_FIRST_PARTICLES]
                             [--NOT_correct_ctf]
                             [--halfmap_subset {1,2}]

options:
  -h, --help            show this help message and exit
  --reference_vol REFERENCE_VOL
                        Path to the reference volume file (.mrc).
  --particles_star_fname PARTICLES_STAR_FNAME
                        Input STAR file with particle metadata.
  --out_fname OUT_FNAME
                        Output STAR file with aligned particle poses.
  --particles_dir PARTICLES_DIR
                        Root directory for particle image paths.
  --mask_radius_angs MASK_RADIUS_ANGS
                        Mask radius in Angstroms.
  --grid_distance_degs GRID_DISTANCE_DEGS
                        Angular search range (degrees). Default=8.0
  --grid_step_degs GRID_STEP_DEGS
                        Angular step size (degrees). Default=2.0
  --top_k_poses_nnet TOP_K_POSES_NNET
                        Number of top poses to predict by the neural network. Default=1
  --filter_resolution_angst FILTER_RESOLUTION_ANGST
                        Low-pass filter the reference before matching.
  --n_jobs N_JOBS       Number of parallel jobs. Default=1
  --num_dataworkers NUM_DATAWORKERS
                        Number of CPU workers per DataLoader. Default=1
  --batch_size BATCH_SIZE
                        Batch size per job. Default=1024
  --NOT_use_cuda        if NOT, it will not use cuda devices Action:
                        store_false for variable use_cuda
  --NOT_verbose         if NOT, it will not log progress Action: store_false
                        for variable verbose
  --torch_matmul_precision {highest,high,medium}
                        Precision mode for matmul. Default=high
  --gpu_id GPU_ID       Specific GPU ID (if any).
  --n_first_particles N_FIRST_PARTICLES
                        Limit processing to first N particles.
  --NOT_correct_ctf     if NOT, it will not apply CTF correction Action:
                        store_false for variable correct_ctf
  --halfmap_subset {1,2}
                        Select subset '1' or '2' for half-map validation.
```

### Example

```bash
cryopares_projmatching \
    --reference_vol /path/to/your/reference.mrc \
    --particles_star_fname /path/to/your/particles.star \
    --out_fname /path/to/your/aligned_particles.star \
    --grid_distance_degs 10
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