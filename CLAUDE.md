# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CryoPARES is a supervised deep learning system for cryo-EM pose estimation. It uses a "train once, infer many times" paradigm where a model trained on high-quality aligned particles can rapidly predict poses for new datasets of the same or similar molecules.

## Core Architecture

### Main Pipeline Components

1. **Training** (`cryoPARES/train/`): Uses pre-aligned particle datasets to train pose prediction models. Creates half-set models for cross-validation.

2. **Inference** (`cryoPARES/inference/`): Predicts poses for new particles using trained models. Supports both static mode (fixed dataset) and daemon mode (continuous processing).

3. **Projection Matching** (`cryoPARES/projmatching/`): Local refinement of poses by searching around predicted orientations.

4. **Reconstruction** (`cryoPARES/reconstruction/`): 3D volume reconstruction from particles with known poses.

### Key Modules

- **Models** (`cryoPARES/models/`):
  - `image2sphere.py`: Main network that maps 2D images to SO(3) orientation predictions
  - `directionalNormalizer/`: Confidence scoring for predictions
  - `model.py`: PyTorch Lightning wrapper (`PlModel`) with training logic

- **Data Management** (`cryoPARES/datamanager/`):
  - `particlesDataset.py`: Dataset class handling particle loading, CTF correction, augmentation, and preprocessing
  - `augmentations.py`: Image augmentation for training
  - `ctf/`: CTF correction implementations (phase flip, multiplication)

- **Configuration** (`cryoPARES/configManager/`, `cryoPARES/configs/`):
  - Uses dataclass-based hierarchical configuration system (`mainConfig.py`)
  - Supports YAML files, command-line overrides with dot notation (e.g., `models.image2sphere.lmax=6`)
  - `inject_defaults_from_config` decorator pattern for automatic parameter injection

- **Geometry** (`cryoPARES/geometry/`):
  - SO(3) operations, rotation conversions, spherical grids
  - Symmetry handling for point groups (C, D, T, O, I symmetries)

### Daemon Mode Architecture

The daemon inference system has three components:
1. **Queue Manager** (`inference/daemon/queueManager.py`): Central job queue server
2. **Spooling Filler** (`inference/daemon/spoolingFiller.py`): Watches directories for new `.star` files
3. **Daemon Inferencer** (`inference/daemon/daemonInference.py`): Worker processes that consume jobs
4. **Materialize Results** (`inference/daemon/materializePartialResults.py`): Combines partial outputs

## Commands

### Development Setup

```bash
# Install in editable mode after cloning
pip install -e .

# Important: Increase file descriptor limit before running
ulimit -n 65536
```

### Training

```bash
python -m cryopares_train \
    --symmetry C1 \
    --particles_star_fname /path/to/aligned.star \
    --particles_dir /path/to/particles \
    --output_dir /path/to/output \
    --n_epochs 10 \
    --config models.image2sphere.lmax=6
```

### Inference

```bash
cryopares_infer \
    --particles_star_fname /path/to/new.star \
    --checkpoint_dir /path/to/training/version_0 \
    --output_dir /path/to/results \
    --reference_vol /path/to/ref.mrc \
    --config projmatching.grid_distance_degs=15
```

### Projection Matching

```bash
cryopares_projmatching \
    --reference_vol /path/to/ref.mrc \
    --particles_star_fname /path/to/particles.star \
    --output_star_fname /path/to/aligned.star \
    --grid_distance_degs 10
```

### Reconstruction

```bash
cryopares_reconstruct \
    --particles_star_fname /path/to/particles.star \
    --symmetry C1 \
    --output_mrc_fname /path/to/output.mrc
```

### Testing

```bash
# Run distributed tests
pytest tests/test_inference_distrib.py
pytest tests/test_projmatching_distrib.py
pytest tests/test_reconstruct_distrib.py
pytest tests/test_daemon.py
```

### Configuration Inspection

```bash
# View all available config parameters
python -m cryopares_train --show-config
```

## Important Technical Details

### File Handling
- CryoPARES opens file handlers for each `.mrcs` file in RELION `.star` files
- **Always run `ulimit -n 65536`** before training/inference to avoid "Too many open files" errors

### Half-Set System
- Training creates two models (half1, half2) for cross-validation
- Inference supports matching data halves to model halves to prevent overfitting
- Use `--data_halfset` and `--model_halfset` flags to control behavior

### ArgParseFromDoc
- CLI are created using [argPasrseFromDoc](https://github.com/rsanchezgarc/argParseFromDoc). It requires type hints and docstrings to automatically generate them 
### Key Configuration Parameters

**Training:**
- `datamanager.particlesDataset.sampling_rate_angs_for_nnet`: Target sampling rate for neural network input
- `datamanager.particlesDataset.image_size_px_for_nnet`: Final image size after rescaling/cropping
- `datamanager.particlesDataset.mask_radius_angs`: Circular mask radius
- `train.learning_rate`: Default 1e-3
- `train.accumulate_grad_batches`: Default 16 (simulates larger batch size)

**Inference:**
- `inference.directional_zscore_thr`: Confidence threshold for filtering particles
- `projmatching.grid_distance_degs`: Angular search range for local refinement (most important)
- `projmatching.grid_step_degs`: Angular search step size

### Data Format
- Uses RELION 3.1+ `.star` files for particle metadata
- Particles stored in `.mrcs` stacks
- `--particles_dir` acts like RELION's project directory concept for relative paths

### Dependencies
- PyTorch + Lightning for training
- e3nn for SO(3)-equivariant operations
- starfile/starstack for RELION format handling
- healpy for spherical grids
- mrcfile for cryo-EM volume I/O
- you need to use the conda virtual environment "cryopares" /home/sanchezg/app/anaconda3/envs/cryopares/bin/python