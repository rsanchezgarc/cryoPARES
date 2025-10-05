# Configuration Guide

This guide provides a comprehensive reference for all configuration parameters in CryoPARES.

## Table of Contents

- [Configuration System Overview](#configuration-system-overview)
- [Viewing Available Options](#viewing-available-options)
- [Configuration Hierarchy](#configuration-hierarchy)
- [Training Configuration](#training-configuration)
- [Inference Configuration](#inference-configuration)
- [Data Manager Configuration](#data-manager-configuration)
- [Model Configuration](#model-configuration)
- [Projection Matching Configuration](#projection-matching-configuration)
- [Reconstruction Configuration](#reconstruction-configuration)

---

## Configuration System Overview

CryoPARES uses a hierarchical, dataclass-based configuration system that allows you to control all aspects of training and inference.

### Configuration Sources (in order of precedence)

1. **Direct command-line arguments**: `--batch_size 32`
2. **Config overrides**: `--config train.learning_rate=1e-3`
3. **YAML config files**: `--config_file my_config.yaml`
4. **Default values**: Built into the code

### Using Config Overrides

Config overrides use dot notation to specify nested parameters:

```bash
python -m cryopares_train \
    --symmetry C1 \
    --particles_star_fname data.star \
    --train_save_dir output/ \
    --config \
        train.learning_rate=1e-3 \
        train.weight_decay=1e-5 \
        models.image2sphere.lmax=8 \
        datamanager.particlesDataset.sampling_rate_angs_for_nnet=2.0
```

**Important:** The `--config` flag must be the **last** argument on the command line.

### Using YAML Config Files

Create a YAML file with your settings:

```yaml
# my_config.yaml
train:
  learning_rate: 0.001
  weight_decay: 0.00001
  n_epochs: 20
  batch_size: 32

models:
  image2sphere:
    lmax: 8
    imageencoder:
      encoderArtchitecture: "resnet"
      resnet:
        resnetName: "resnet18"

datamanager:
  particlesDataset:
    sampling_rate_angs_for_nnet: 2.0
    image_size_px_for_nnet: 128
```

Use it:

```bash
python -m cryopares_train \
    --symmetry C1 \
    --particles_star_fname data.star \
    --train_save_dir output/ \
    --config_file my_config.yaml
```

---

## Viewing Available Options

To see all available configuration options and their current values:

```bash
python -m cryopares_train --show-config
```

This displays the complete configuration tree with:
- Parameter names and paths
- Current values
- Data types
- Nested structure

Example output:
```
MainConfig:
  train:
    learning_rate: 0.001
    weight_decay: 1e-05
    n_epochs: 10
    batch_size: 32
    ...
  models:
    image2sphere:
      lmax: 8
      ...
```

---

## Configuration Hierarchy

```
MainConfig
├── train (Training parameters)
├── inference (Inference parameters)
├── datamanager (Data loading and preprocessing)
│   ├── particlesDataset (Dataset parameters)
│   └── augmentations (Data augmentation)
├── models (Model architecture)
│   ├── image2sphere (Main neural network)
│   └── directionalNormalizer (Confidence scoring)
├── projmatching (Projection matching/local refinement)
└── reconstruct (3D reconstruction)
```

---

## Training Configuration

**Path prefix:** `train.*`

### Basic Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_epochs` | int | 10 | Number of training epochs |
| `batch_size` | int | 32 | Number of particles per batch |
| `learning_rate` | float | 1e-3 | Initial learning rate for optimizer |
| `weight_decay` | float | 1e-5 | L2 regularization coefficient |
| `default_optimizer` | str | "Adam" | Optimizer class name (from torch.optim) |

### Gradient Accumulation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `accumulate_grad_batches` | int | 16 | Number of batches to accumulate gradients |

Effective batch size = `batch_size × accumulate_grad_batches`

### Learning Rate Schedule

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `warmup_n_epochs` | int | 2 | Number of warmup epochs (LR ramps from 0 to target) |
| `patient_reduce_lr_plateau_n_epochs` | int | 3 | Patience for ReduceLROnPlateau scheduler |
| `min_learning_rate_factor` | float | 0.01 | Minimum LR as fraction of initial LR |
| `monitor_metric` | str | "val_loss" | Metric to monitor for LR reduction |

### Advanced Training Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_cuda` | bool | True | Use GPU for training |
| `n_cpus_if_no_cuda` | int | 4 | Number of CPUs if GPU unavailable |
| `snr_for_simulation` | float | 0.1 | SNR for simulated pre-training |
| `n_epochs_simulation` | int | 3 | Epochs for simulated pre-training |

### Example

```bash
--config \
    train.learning_rate=5e-3 \
    train.weight_decay=1e-4 \
    train.accumulate_grad_batches=8 \
    train.warmup_n_epochs=3 \
    train.patient_reduce_lr_plateau_n_epochs=5
```

---

## Inference Configuration

**Path prefix:** `inference.*`

### Basic Inference Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | int | 1024 | Batch size for inference |
| `use_cuda` | bool | True | Use GPU |
| `n_cpus_if_no_cuda` | int | 4 | CPUs if no GPU |
| `top_k_poses_nnet` | int | 10 | Number of top pose predictions |

### Confidence Filtering

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `directional_zscore_thr` | float | 2.0 | Z-score threshold for filtering particles |

- `None`: No filtering
- `1.5`: Lenient (keeps more particles)
- `2.0`: Balanced (recommended)
- `3.0`: Strict (keeps fewer, higher-confidence particles)

### Pipeline Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `skip_localrefinement` | bool | False | Skip projection matching |
| `skip_reconstruction` | bool | False | Skip 3D reconstruction |

### Example

```bash
--config \
    inference.directional_zscore_thr=2.5 \
    inference.top_k_poses_nnet=20 \
    inference.skip_reconstruction=False
```

---

## Data Manager Configuration

**Path prefix:** `datamanager.*`

### Data Loading

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_dataworkers` | int | 8 | Number of parallel data loading workers |
| `num_augmented_copies_per_batch` | int | 1 | Augmentation multiplier |

### Particle Dataset Parameters

**Path prefix:** `datamanager.particlesDataset.*`

#### Image Preprocessing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sampling_rate_angs_for_nnet` | float | 1.5 | Target sampling rate (Å/px) |
| `image_size_px_for_nnet` | int | 160 | Target image size (pixels) |
| `mask_radius_angs` | float | None | Circular mask radius (Å) |
| `apply_mask_to_img` | bool | True | Apply mask to images |

#### Normalization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `perImg_normalization` | str | "noiseStats" | Normalization method |

Options:
- `"none"`: No normalization
- `"noiseStats"`: Normalize using noise statistics (recommended)
- `"subtractMean"`: Subtract mean only

#### CTF Correction

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ctf_correction` | str | "phase_flip" | CTF correction method |

Options:
- `"none"`: No correction
- `"phase_flip"`: Phase flip correction (recommended)
- `"ctf_multiply"`: Wiener-like correction
- `"concat_phase_flip"`: Concatenate corrected and uncorrected
- `"concat_ctf_multiply"`: Concatenate with Wiener correction

#### Memory and Performance

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_data_in_memory` | bool | False | Cache preprocessed data in RAM |
| `reduce_symmetry_in_label` | bool | True | Apply symmetry to ground truth |

### Augmentation Parameters

**Path prefix:** `datamanager.augmentations.*`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `do_shift` | bool | True | Random shifts |
| `do_rotate` | bool | True | Random in-plane rotations |
| `do_flip` | bool | False | Random flips |
| `max_shift_px` | int | 4 | Maximum shift in pixels |

### Example

```bash
--config \
    datamanager.particlesDataset.sampling_rate_angs_for_nnet=1.5 \
    datamanager.particlesDataset.image_size_px_for_nnet=160 \
    datamanager.particlesDataset.mask_radius_angs=120 \
    datamanager.particlesDataset.perImg_normalization="noiseStats" \
    datamanager.particlesDataset.ctf_correction="phase_flip" \
    datamanager.num_dataworkers=8
```

---

## Model Configuration

**Path prefix:** `models.*`

### Image2Sphere Model

**Path prefix:** `models.image2sphere.*`

#### Core Architecture

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lmax` | int | 8 | Maximum spherical harmonic degree |
| `top_k_poses_nnet` | int | 10 | Number of top pose predictions |

`lmax` controls model expressiveness:
- `lmax=6`: Fast, less expressive
- `lmax=8`: Balanced (default)
- `lmax=10`: Slow, more expressive

#### Image Encoder

**Path prefix:** `models.image2sphere.imageencoder.*`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `encoderArtchitecture` | str | "resnet" | Encoder architecture |

Options: `"resnet"`, `"unet"`, `"convmixer"`

##### ResNet Configuration

**Path prefix:** `models.image2sphere.imageencoder.resnet.*`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `resnetName` | str | "resnet18" | ResNet variant |
| `pretrained` | bool | False | Use ImageNet pre-trained weights |

Options: `"resnet18"`, `"resnet34"`, `"resnet50"`, `"resnet101"`

##### U-Net Configuration

**Path prefix:** `models.image2sphere.imageencoder.unet.*`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `depth` | int | 4 | Number of downsampling stages |
| `start_channels` | int | 64 | Number of channels in first layer |

##### ConvMixer Configuration

**Path prefix:** `models.image2sphere.imageencoder.convmixer.*`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dim` | int | 256 | Hidden dimension |
| `depth` | int | 8 | Number of ConvMixer blocks |
| `kernel_size` | int | 9 | Kernel size |

#### SO3 Components

**Path prefix:** `models.image2sphere.so3components.*`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `i2sprojector.hp_order` | int | 3 | HEALPix order for image-to-sphere projection |
| `s2conv.hp_order` | int | 3 | HEALPix order for S2 convolution |
| `so3outputgrid.hp_order` | int | 4 | HEALPix order for SO3 output grid |

Higher `hp_order` = finer angular resolution but slower and more memory

### Directional Normalizer

**Path prefix:** `models.directionalNormalizer.*`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hp_order` | int | 2 | HEALPix order for confidence scoring |

### Example

```bash
--config \
    models.image2sphere.lmax=10 \
    models.image2sphere.imageencoder.encoderArtchitecture="resnet" \
    models.image2sphere.imageencoder.resnet.resnetName="resnet50" \
    models.image2sphere.so3components.i2sprojector.hp_order=3 \
    models.image2sphere.so3components.s2conv.hp_order=3 \
    models.image2sphere.so3components.so3outputgrid.hp_order=4
```

---

## Projection Matching Configuration

**Path prefix:** `projmatching.*`

### Search Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grid_distance_degs` | float | 6.0 | Angular search range (degrees) |
| `grid_step_degs` | float | 2.0 | Angular step size (degrees) |
| `top_k_poses_localref` | int | 1 | Number of top poses to return |

**`grid_distance_degs` is the most important parameter:**
- Defines search range: `[pose - distance, pose + distance]`
- Larger values = more thorough but slower
- Typical values:
  - Neural network predictions: 5-10°
  - Random initialization: 15-30°
  - Fine refinement: 2-5°

**`grid_step_degs` controls search granularity:**
- Smaller = finer but slower
- Typical values: 1-3°

### Reference Processing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filter_resolution_angst` | float | None | Low-pass filter resolution (Å) |

### Performance

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | int | 1024 | Batch size |
| `use_cuda` | bool | True | Use GPU |
| `torch_matmul_precision` | str | "high" | Precision mode |

### Compilation Modes (Advanced)

**Path prefix:** `projmatching.*`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `compile_projectVol_mode` | str | "default" | torch.compile mode for volume projection |
| `compile_analyze_cc_mode` | str | "default" | torch.compile mode for cross-correlation analysis |
| `disable_compile_projectVol` | bool | False | Disable compilation of projection function |
| `disable_compile_analyze_cc` | bool | False | Disable compilation of CC analysis |

**Performance Optimization:** If you are **NOT** using `directional_zscore_thr` for particle filtering, you can optionally use `"max-autotune"` compilation mode for a ~10-20% speed boost:

```bash
# Optional: Faster inference when NOT using directional_zscore_thr
--config \
    projmatching.compile_projectVol_mode="max-autotune" \
    projmatching.compile_analyze_cc_mode="max-autotune"
```

**Note:** `"max-autotune"` mode is incompatible with `directional_zscore_thr` because particle filtering creates variable batch sizes, which cause CUDA graph memory errors. The default `"default"` mode works in all scenarios.

Available compilation modes:
- `"default"`: Standard compilation, works with variable batch sizes (recommended)
- `"reduce-overhead"`: Optimized for repeated calls, works with variable batch sizes
- `"max-autotune"`: Maximum optimization, **only use when NOT filtering with directional_zscore_thr**

### Example

```bash
--config \
    projmatching.grid_distance_degs=12.0 \
    projmatching.grid_step_degs=1.5 \
    projmatching.top_k_poses_localref=1 \
    projmatching.filter_resolution_angst=10.0
```

---

## Reconstruction Configuration

**Path prefix:** `reconstruct.*`

### Reconstruction Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | int | 128 | Batch size for reconstruction |
| `correct_ctf` | bool | True | Apply CTF correction |
| `eps` | float | 0.001 | Regularization constant (ideally 1/SNR) |
| `use_cuda` | bool | True | Use GPU |

### Performance

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_jobs` | int | 1 | Number of parallel reconstruction jobs |
| `num_dataworkers` | int | 1 | Data loading workers |

### CTF and Weighting

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `weight_with_confidence` | bool | False | Weight particles by confidence |
| `min_denominator_value` | float | None | Minimum denominator to prevent division by zero |

### Example

```bash
--config \
    reconstruct.eps=0.0005 \
    reconstruct.batch_size=256 \
    reconstruct.weight_with_confidence=True
```

---

## Common Configuration Recipes

### High-Quality Training

```bash
--config \
    train.n_epochs=30 \
    train.learning_rate=1e-3 \
    train.weight_decay=1e-5 \
    train.accumulate_grad_batches=16 \
    models.image2sphere.lmax=10 \
    models.image2sphere.imageencoder.resnet.resnetName="resnet50" \
    datamanager.particlesDataset.sampling_rate_angs_for_nnet=1.5 \
    datamanager.particlesDataset.image_size_px_for_nnet=160
```

### Fast Training (for testing)

```bash
--config \
    train.n_epochs=5 \
    train.batch_size=64 \
    models.image2sphere.lmax=6 \
    models.image2sphere.imageencoder.resnet.resnetName="resnet18" \
    datamanager.particlesDataset.sampling_rate_angs_for_nnet=3.0 \
    datamanager.particlesDataset.image_size_px_for_nnet=96
```

### Memory-Constrained Training

```bash
--config \
    train.batch_size=8 \
    train.accumulate_grad_batches=64 \
    datamanager.particlesDataset.image_size_px_for_nnet=96 \
    datamanager.num_dataworkers=2
```

### High-Accuracy Inference

```bash
--config \
    inference.top_k_poses_nnet=20 \
    inference.directional_zscore_thr=2.5 \
    projmatching.grid_distance_degs=10.0 \
    projmatching.grid_step_degs=1.0
```

### Fast Inference

```bash
--config \
    inference.batch_size=2048 \
    inference.top_k_poses_nnet=5 \
    projmatching.grid_distance_degs=5.0 \
    projmatching.grid_step_degs=2.0
```

---

## Saving and Loading Configurations

### Exporting Current Configuration

After training, your configuration is automatically saved to:
```
train_save_dir/version_0/configs_0.yml
```

### Reusing a Configuration

```bash
python -m cryopares_train \
    --symmetry C1 \
    --particles_star_fname new_data.star \
    --train_save_dir new_output/ \
    --config_file train_save_dir/version_0/configs_0.yml
```

### Overriding Loaded Config

You can load a config file and override specific values:

```bash
python -m cryopares_train \
    --symmetry C1 \
    --particles_star_fname new_data.star \
    --train_save_dir new_output/ \
    --config_file old_config.yml \
    --config \
        train.learning_rate=5e-4 \
        train.n_epochs=15
```

---

## See Also

- [Training Guide](training_guide.md) - Training best practices
- [API Reference](https://rsanchezgarc.github.io/cryoPARES/api/) - Type hints and function signatures
- [Troubleshooting](troubleshooting.md) - Common issues
