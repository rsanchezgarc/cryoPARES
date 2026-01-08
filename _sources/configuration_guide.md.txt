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
4. **Default values**: Built into the `_config.py` code

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

**Important:**
- The `--config` flag should be the **last** argument on the command line
- **Type matching is critical:**
  - **Float parameters** must include a decimal point: `1.0` not `1` (e.g., `train.learning_rate=1e-3` or `sampling_rate_angs_for_nnet=2.0`)
  - **Int parameters** must NOT have a decimal point: `8` not `8.0` (e.g., `models.image2sphere.lmax=8`)
  - Check parameter types with `--show-config` to see `(int)` or `(float)` annotations

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

This command displays configuration in two sections:

**1. Hierarchical view with descriptions:**
Shows the nested structure with parameter descriptions and types:
```
config:
  models:
    image2sphere:
      lmax: 12 (int)  # Maximum spherical harmonic degree for SO(3) representation
      label_smoothing: 0.05 (float)  # Label smoothing factor for loss function
      ...
  train:
    n_epochs: 100 (int)  # Number of training epochs
    learning_rate: 0.001 (float)  # Initial learning rate for optimizer
    ...
```

**2. Dot notation for easy copy-paste:**
Shows all parameters in flat format that can be directly used with `--config`:
```
  train.n_epochs=100 (int)
  train.learning_rate=0.001 (float)
  train.batch_size=64 (int)
  models.image2sphere.lmax=12 (int)
  ...

Example: --config train.n_epochs=100 train.batch_size=64
```

**Tip:** You can copy any line from the dot notation section and paste it directly after `--config` to override that parameter

---

## Configuration Hierarchy

CryoPARES configuration is organized hierarchically:

```
MainConfig
├── train (Training parameters)
├── inference (Inference parameters)
├── datamanager (Data loading and preprocessing)
│   ├── particlesDataset (Dataset parameters)
│   └── augmenter (Data augmentation)
├── models (Model architecture)
│   ├── image2sphere (Main neural network)
│   └── directionalNormalizer (Confidence scoring)
├── projmatching (Projection matching/local refinement)
└── reconstruct (3D reconstruction)
```

---

## Reading Configuration Files

All configuration parameters are defined in Python dataclass files located in `cryoPARES/configs/`. These files contain:
- **Parameter names** and their **default values**
- **Type annotations** (int, float, str, bool, etc.)
- **Inline documentation** explaining what each parameter does

### Configuration File Locations

| Config Section | File Path |
|----------------|-----------|
| **Training** | `cryoPARES/configs/train_config/train_config.py` |
| **Inference** | `cryoPARES/configs/inference_config/inference_config.py` |
| **Data Manager** | `cryoPARES/configs/datamanager_config/datamanager_config.py` |
| **Particle Dataset** | `cryoPARES/configs/datamanager_config/particles_dataset_config.py` |
| **Augmentation** | `cryoPARES/configs/datamanager_config/augmenter_config.py` |
| **Image2Sphere Model** | `cryoPARES/configs/models_config/image2sphere_config.py` |
| **Image Encoder** | `cryoPARES/configs/models_config/imageencoder_config.py` |
| **Projection Matching** | `cryoPARES/configs/projmatching_config/projmatching_config.py` |
| **Reconstruction** | `cryoPARES/configs/reconstruction_config/reconstruct_config.py` |

### How to Read Configuration Files

Each config file is a Python dataclass with this structure:

```python
@dataclass
class Train_config:
    """Training configuration parameters."""

    # Centralized parameter documentation
    PARAM_DOCS = {
        'n_epochs': 'Number of training epochs. More epochs allow better convergence...',
        'learning_rate': 'Initial learning rate for optimizer. Tune this for optimal convergence...',
        'batch_size': 'Number of particles per batch. Try to make it as large as possible...',
        ...
    }

    # Parameter definitions with defaults
    n_epochs: int = 100
    learning_rate: float = 1e-3
    batch_size: int = 32
    weight_decay: float = 1e-5
    ...
```

**Key components:**
- **`PARAM_DOCS`**: Dictionary with detailed descriptions of each parameter
- **Parameter definitions**: `name: type = default_value`
- **Nested configs**: Some configs contain sub-configurations (e.g., `imageencoder_config` contains `resnet_config`, `unet_config`)

### Example: Finding Training Parameters

1. **Open the file**: `cryoPARES/configs/train_config/train_config.py`
2. **Check PARAM_DOCS** for detailed descriptions:
   ```python
   'learning_rate': 'Initial learning rate for optimizer. Tune this for optimal convergence (typical range: 1e-4 to 1e-2)'
   ```
3. **See the default value**:
   ```python
   learning_rate: float = 1e-3
   ```
4. **Use with --config**:
   ```bash
   --config train.learning_rate=5e-3
   ```

### Example: Finding Nested Parameters

For deeply nested parameters like `models.image2sphere.imageencoder.resnet.resnetName`:

1. **Start at**: `cryoPARES/configs/models_config/image2sphere_config.py`
2. **Find nested config**:
   ```python
   imageencoder: Imageencoder_config = Imageencoder_config()
   ```
3. **Open**: `cryoPARES/configs/models_config/imageencoder_config.py`
4. **Find ResNet config**:
   ```python
   resnet: Resnet_config = Resnet_config()
   ```
5. **Check**: `cryoPARES/configs/models_config/imageencoder_config.py` for `resnetName` parameter

---

## Quick Parameter Reference

For the most commonly used parameters, see:
- **Training basics**: `cryoPARES/configs/train_config/train_config.py` → `PARAM_DOCS`
- **Data preprocessing**: `cryoPARES/configs/datamanager_config/particles_dataset_config.py` → `PARAM_DOCS`
- **Model architecture**: `cryoPARES/configs/models_config/image2sphere_config.py` → `PARAM_DOCS`
- **Local refinement**: `cryoPARES/configs/projmatching_config/projmatching_config.py` → `PARAM_DOCS`

**Remember:** The easiest way to discover parameters is:
```bash
python -m cryopares_train --show-config
```
This shows all parameters with their current values and descriptions, ready to copy-paste

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
