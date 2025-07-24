# CryoPARES: Cryo-EM Pose Assignment for Related Experiments via Supervised deep learning

CryoPARES is a software package for assigning three-dimensional (3D) poses to two-dimensional (2D) cryo-electron microscopy (cryo-EM) particle images. It offers two main functionalities: supervised deep learning for initial pose estimation and local refinement through projection matching.

This project is built on PyTorch and PyTorch Lightning, and it is designed to be highly configurable and extensible.

## Key Features

*   **Supervised Pose Estimation:** Utilizes deep learning models to predict the 3D pose (2 rotational angles and 3 shifts) of 2D particle images.
*   **Projection Matching:** Refines particle poses by performing a local search around an initial orientation, comparing particle images against projections of a 3D reference map.
*   **Flexible Configuration:** A powerful configuration system allows for fine-grained control over all parameters. Settings can be specified in YAML files or overridden directly from the command line.
*   **RELION Compatibility:** Seamlessly reads and writes RELION `.star` files, making it compatible with existing cryo-EM data processing pipelines.
*   **Reproducibility:** Automatically saves the code, environment variables, and configuration used for each run, ensuring full reproducibility of results.

## Installation

It is recommended to use a virtual environment (e.g., conda or venv).

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/rsanchezgarc/cryoPARES.git
    cd cryoPARES
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install the package:**
    ```bash
    pip install -e .
    ```

## Configuration System

CryoPARES uses a flexible configuration system that allows you to manage settings from multiple sources.

### Configuration Methods
1.  **YAML Files:** Create a `.yaml` file with your desired parameters.
2.  **Command-Line Overrides:** Pass `KEY=VALUE` pairs to the program. Use dot notation to specify nested parameters (e.g., `models.image2sphere.lmax=6`).
3.  **Direct Arguments:** Use standard command-line flags (e.g., `--batch_size 32`).

### Precedence and Conflict Resolution
The settings are applied in the following order of precedence (highest first):
1.  **Direct Command-Line Arguments** (e.g., `--batch_size 32`)
2.  **`--config` Command-Line Overrides** (e.g., `--config train.batch_size=32`)
3.  **`--config` YAML Files** (e.g., `--config my_config.yaml`)
4.  **Default Configuration**

**Important:** If you specify the same parameter via a direct argument (e.g., `--batch_size 32`) and a `--config` override (e.g., `--config train.batch_size=64`), the program will raise a conflict error. You must provide the value from only one source or ensure they are identical.

### Viewing and Exporting Configurations
*   **Show all available options:** Run any main script with `--show-config` to print a comprehensive list of all parameters, their current values, and their paths.
    ```bash
    python -m cryoPARES.train.train --show-config
    ```
*   **Export the final configuration:** Use `--export-config` to save the final, merged configuration to a YAML file. This is useful for documenting a run.
    ```bash
    python -m cryoPARES.train.train --export-config final_config.yaml
    ```

## Core Programs

### 1. Training (`cryoPARES.train.train`)

This program trains a model to predict particle poses from a `.star` file containing aligned particles.

**Usage:**
```bash
python -m cryoPARES.train.train [ARGUMENTS] --config [CONFIG_OVERRIDES]
```

**Key Arguments:**
*   `--symmetry`: The point group symmetry of the molecule (e.g., `C1`, `D7`).
*   `--particles_star_fname`: Path to the input `.star` file with pre-aligned particles.
*   `--train_save_dir`: Directory to save model checkpoints, logs, and other outputs.
*   `--n_epochs`: Number of training epochs.
*   `--batch_size`: Number of particles per batch.
*   `--continue_checkpoint_dir`: Continue training from a previous run.
*   `--finetune_checkpoint_dir`: Fine-tune a pre-trained model on a new dataset.

**Outputs:**
*   A training directory containing:
    *   Model checkpoints (`.ckpt` files).
    *   TensorBoard logs.
    *   A copy of the source code, environment, and configuration for reproducibility.

### 2. Inference (`cryoPARES.inference.inference`)

This is the main script for using a trained model to predict poses for a new set of particles. It can optionally perform local refinement using projection matching and reconstruct a 3D map.

**Usage:**
```bash
python -m cryoPARES.inference.inference [ARGUMENTS] --config [CONFIG_OVERRIDES]
```

**Key Arguments:**
*   `--particles_star_fname`: Path to the input `.star` file of particles needing pose assignment.
*   `--checkpoint_dir`: Path to the directory containing the trained model created by `train.py`.
*   `--results_dir`: Directory where the output `.star` file and reconstruction will be saved.
*   `--perform_localrefinement`: If set, enables pose refinement using projection matching. This requires a `--reference_map`.
*   `--perform_reconstruction`: If set, reconstructs a 3D map from the final poses.
*   `--reference_map`: Path to a reference `.mrc` map, required for local refinement and reconstruction.
*   `--top_k`: Predict the top K poses for each particle.

**Key Configuration Parameters:**
*   `inference.directional_zscore_thr`: A crucial parameter for filtering particles. It is a threshold on the confidence score of the neural network's prediction. Particles with a score below this threshold can be discarded.
*   `projmatching.grid_distance_degs`: The most important parameter for local refinement. It defines the angular search range (in degrees) around the neural network's predicted orientation.
*   `projmatching.grid_step_degs`: The step size for the angular search during refinement.

**Outputs:**
*   An output directory containing:
    *   A new `.star` file (`*_nnet.star`) with the predicted poses and confidence scores.
    *   If `--perform_reconstruction` is used, reconstructed half-maps (`reconstruction_half1_nnet.mrc`, `reconstruction_half2_nnet.mrc`) are generated.

## Example Workflow

1.  **Train a model on an existing, aligned dataset:**
    ```bash
    python -m cryoPARES.train.train \
        --symmetry C1 \
        --particles_star_fname /path/to/aligned_particles.star \
        --train_save_dir /path/to/training_output \
        --n_epochs 10 \
        --config models.image2sphere.lmax=6
    ```

2.  **Run inference on a new dataset, with local refinement and reconstruction:**
    ```bash
    python -m cryoPARES.inference.inference \
        --particles_star_fname /path/to/new_particles.star \
        --checkpoint_dir /path/to/training_output/version_0 \
        --results_dir /path/to/inference_results \
        --perform_localrefinement True \
        --perform_reconstruction True \
        --reference_map /path/to/initial_model.mrc \
        --config projmatching.grid_distance_degs=15 inference.directional_zscore_thr=2.0
    ```

## Project Structure

```
cryoPARES/
├───configs/         # Default configuration files.
├───datamanager/     # Data loading, datasets, and augmentations.
├───geometry/        # Utilities for 3D geometry and rotations.
├───models/          # Deep learning model architectures.
├───inference/       # Scripts for running inference.
├───projmatching/    # Projection matching implementation.
├───reconstruction/  # 3D reconstruction tools.
├───train/           # Model training scripts.
└───utils/           # Miscellaneous utility functions.
```

## Citation

If you use CryoPARES in your research, please cite:

*(Placeholder for citation information)*

## License

This project is distributed under the terms of the [LICENSE_NAME](LICENSE) license.

## Contact

Ruben Sanchez-Garcia - ruben.sanchez-garcia@stats.ox.ac.uk

Project Link: [https://github.com/rsanchezgarc/cryoPARES](https://github.com/rsanchezgarc/cryoPARES)
