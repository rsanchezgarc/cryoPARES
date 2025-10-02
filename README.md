# CryoPARES: Cryo-EM Pose Assignment for Related Experiments via Supervised deep learning

**CryoPARES** is a software package for assigning poses to 2D cryo-electron microscopy (cryo-EM) particle images.
It uses a supervised deep learning approach to accelerate 3D reconstruction in related cryo-EM experiments. 
The key idea is to train a neural network on a high-quality reference reconstruction, and then reuse this trained model 
to rapidly estimate particle poses in other, similar datasets.

This workflow is divided into two main phases:

*   **Training:** In this phase, you use a pre-existing, high-resolution dataset (where particle poses have already been determined by
traditional methods like RELION refine) to train a CryoPARES model. This process creates a highly specialized "expert" model that 
can recognize and assign poses to particles of that specific type of macromolecule.

*   **Inference:** Once the model is trained, you can use it for inference on new datasets of the *same* or *very similar* molecules 
(e.g., the same protein with a different ligand bound). Because the model has already learned the features of the molecule, it can predict
particle poses almost instantly, bypassing the computationally expensive and time-consuming alignment steps of traditional workflows. 
This is especially powerful for applications like drug screening, where many similar datasets need to be processed quickly.

This "train once, infer many times" paradigm allows for near real-time 3D reconstruction, providing rapid feedback during data 
collection and analysis.

For a detailed explanation of the method, please refer to our paper:
[Supervised Deep Learning for Efficient Cryo-EM Image Alignment in Drug Discovery](https://www.biorxiv.org/content/10.1101/2025.03.04.641536v2)

## Documentation

- **[Training Guide](./docs/training_guide.md)** - Comprehensive guide on training models, monitoring with TensorBoard, and avoiding overfitting/underfitting
- **[API Reference](./docs/api_reference.md)** - Detailed API documentation with type hints for all major classes and functions
- **[Configuration Guide](./docs/configuration_guide.md)** - Complete reference for all configuration parameters
- **[Troubleshooting Guide](./docs/troubleshooting.md)** - Solutions to common issues
- **[CLI Reference](./docs/cli.md)** - Command-line interface documentation

## Installation

It is strongly recommended to use a virtual environment (e.g., conda) to avoid conflicts with other packages.

1.  **Create and activate a conda environment:**

    ```bash
    conda create -n cryopares python=3.12
    conda activate cryopares
    ```

### Option 1: Install from GitHub (Recommended for Users)

This is the simplest way to install cryoPARES.

```bash
pip install git+https://github.com/rsanchezgarc/cryoPARES.git
```

### Option 2: Install from a Local Clone (Recommended for Developers)

This method is recommended if you want to modify the cryoPARES source code.

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/rsanchezgarc/cryoPARES.git
    cd cryoPARES
    ```

2.  **Install the package in editable mode:**

    This allows you to make changes to the code without having to reinstall the package.

    ```bash
    pip install -e .
    ```

## Usage

**IMPORTANT:** CryoPARES keeps a file handler open for each `.mrcs` file referenced in the `.star` file. This can lead 
to a "Too many open files" error if the number of particle files is larger than the system's limit.
Before running training or inference, it is highly recommended to increase the open file limit by running 
the following command in your terminal:

```bash
ulimit -n 65536
```

CryoPARES has two main modes of operation: training and inference. Particles need to be provided as RELION 3.1+ starfile(s).

### Training

The `cryopares_train` module is used to train a new model for pose estimation. Training needs to be done first using 
a pre-aligned dataset of particles. While not mandatory, we encourage using particles alignments estimated with RELION.


**Usage:**
```bash
python -m cryopares_train [ARGUMENTS] --config [CONFIG_OVERRIDES]
```

**Key Arguments:**

*   `--symmetry`: The point group symmetry of the molecule (e.g., `C1`, `D7`).
*   `--particles_star_fname`: Path to the input `.star` file with pre-aligned particles.
*   `--particles_dir`: Optional root directory for the particle files. If the paths in the `.star` file are relative, this directory will be prepended to them. For example, if `--particles_dir` is `/path/to/particles` and a particle path in the `.star` file is `MotionCorr/job01/extract/particle_001.mrcs`, the final path will be `/path/to/particles/MotionCorr/job01/extract/particle_001.mrcs`. This is similar to RELION's project directory concept. If this argument is not provided, it is assumed that the particle files are in the same directory as the `.star` file, or that they are in a relative path with respect the current working directory.
*   `--train_save_dir`: Directory to save model checkpoints, logs, and other outputs.
*   `--n_epochs`: Number of training epochs.
*   `--batch_size`: Number of particles per batch. For performance reasons, try to make it as large as you can before running out of GPU memory.
*   `--num_dataworkers`: Number of parallel data loading workers (per GPU). Set it to 0 to read and process the data in the main thread
*   `--continue_checkpoint_dir`: Continue training from a previous run.
*   `--finetune_checkpoint_dir`: Fine-tune a pre-trained model on a new dataset.

**Important Training Parameters (in config files):**

You can override these parameters from the command line using the `--config KEY=VALUE` syntax. The `--config` needs to be
the last CLI argument provided. Several `KEY=VALUE` can be provided one after another

*   **`datamanager.particlesDataset.sampling_rate_angs_for_nnet`**: The desired sampling rate in Angstroms per pixel. Particle images are first rescaled (upsampled or downsampled) to match this value.
*   **`datamanager.particlesDataset.image_size_px_for_nnet`**: The final size of the particle images in pixels. After rescaling to the desired sampling rate, images are cropped or padded to this size. This should typically be set to the particle's box size after rescaling.
*   **`datamanager.particlesDataset.mask_radius_angs`**: The radius of the circular mask to be applied to the particle images, in Angstroms. If not provided, a mask with a radius of half the box size will be used.
*   **`train.learning_rate`**: The initial learning rate for the optimizer. (Default: `1e-3`)
*   **`train.weight_decay`**: The weight decay for the optimizer. (Default: `1e-5`)
*   **`train.accumulate_grad_batches`**: The number of batches to accumulate gradients over. This can be used to simulate a larger batch size. (Default: `16`)

For comprehensive training guidance including monitoring with TensorBoard and avoiding overfitting/underfitting, see the **[Training Guide](./docs/training_guide.md)**. For a complete list of all configuration parameters, see the **[Configuration Guide](./docs/configuration_guide.md)**.

### Inference

The `cryoPARES.inference.inference` module is used to predict poses for a new set of particles using a trained model. It can be run in two modes: static and daemon.

#### Static Mode

In static mode, the inference is run on a fixed set of particles, that again, need to be provided as RELION 3.1+ starfiles.

**Usage:**
```bash
cryopares_infer [ARGUMENTS] --config [CONFIG_OVERRIDES]
```

**Key Arguments:**

*   `--particles_star_fname`: Path to the input `.star` file of particles needing pose assignment.
*   `--particles_dir`: Optional root directory for the particle files. If the paths in the `.star` file are relative, this directory will be prepended to them. For example, if `--particles_dir` is `/path/to/particles` and a particle path in the `.star` file is `MotionCorr/job01/extract/particle_001.mrcs`, the final path will be `/path/to/particles/MotionCorr/job01/extract/particle_001.mrcs`. This is similar to RELION's project directory concept. If this argument is not provided, it is assumed that the particle files are in the same directory as the `.star` file.
*   `--checkpoint_dir`: Path to the directory containing the trained model created by `train.py`.
*   `--results_dir`: Directory where the output `.star` file and reconstruction will be saved.
*   `--batch_size`: Number of particles per batch.
*   `--num_dataworkers`: Number of parallel data loading workers. One CPU each. Set it to 0 to read and process the data in the main thread
*   `--reference_map`: Path to a reference `.mrc` used for local refinement and reconstruction. If not provided, it will use half-maps reconstructed from the training set.

**Half-Set Selection (`--data_halfset` and `--model_halfset`)**

To avoid overfitting and to ensure a fair evaluation, cryo-EM datasets are often split into two halves (half1 and half2). CryoPARES uses this concept for both the data and the model.

*   `--data_halfset`: Specifies which half of the data to use for inference.
    *   `half1`: Use only the particles from the first half of the dataset.
    *   `half2`: Use only the particles from the second half of the dataset.
    *   `allParticles`: Use all particles from the dataset. (Default)

*   `--model_halfset`: Specifies which trained model to use for inference. During training, CryoPARES creates two models, one for each half of the training data.
    *   `half1`: Use the model trained on the first half of the training data.
    *   `half2`: Use the model trained on the second half of the training data.
    *   `matchingHalf`: Use the model from the corresponding half of the data (e.g., `half1` data with `half1` model). This is the default and recommended setting.
    *   `allCombinations`: Run inference for all possible combinations of data and model halves (e.g., `half1` data with `half1` model, `half1` data with `half2` model, etc.). 

**Important Inference Parameters (in config files):**

*   **`inference.directional_zscore_thr`**: A crucial parameter for filtering particles. It is a threshold on the confidence score of the neural network's prediction. Particles with a score below this threshold can be discarded.
*   **`projmatching.grid_distance_degs`**: The most important parameter for local refinement. It defines the angular search range (in degrees) around the neural network's predicted orientation (current_pose - grid_distance_degs ... current_pose + grid_distance_degs).
*   **`projmatching.grid_step_degs`**: The step size for the angular search during refinement.

For detailed API documentation with type hints, see the **[API Reference](./docs/api_reference.md)**.

#### Daemon Mode

In daemon mode, the inference script runs continuously and watches for new particles to be added to a directory. This is useful for processing particles as they are being generated.

The daemon workflow consists of three main components:

1.  **Queue Manager:** A central server that manages a queue of jobs.
2.  **Spooling Filler:** A script that monitors a directory for new `.star` files and adds them to the queue.
3.  **Daemon Inferencer:** One or more worker processes that take jobs from the queue and perform inference.

**Workflow:**

1.  **Start the Queue Manager:**

    This script creates the central queue. It should be run once and kept running in the background.

    ```bash
    python -m cryoPARES.inference.daemon.queueManager
    ```

2.  **Start the Spooling Filler:**

    This script watches a directory for new `.star` files and adds them to the queue.

    ```bash
    python -m cryoPARES.inference.daemon.spoolingFiller --directory /path/to/watch
    ```

    You can also use other mechanisms to add jobs to the queue.

3.  **Start the Daemon Inferencer(s):**

    You can start as many inferencer workers as you want. Each worker will take jobs from the queue and process them in parallel. **Important:** Each worker must have its own results directory.

    ```bash
    # Worker 1
    python -m cryoPARES.inference.daemon.daemonInference --checkpoint_dir /path/to/checkpoint --results_dir /path/to/results_worker1 --particles_dir /path/to/particles

    # Worker 2
    python -m cryoPARES.inference.daemon.daemonInference --checkpoint_dir /path/to/checkpoint --results_dir /path/to/results_worker2 --particles_dir /path/to/particles
    ```

4.  **Materialize the Volume:**

    You can materialize the final 3D volume from the partial results at any time, even while the inferencers are still running. The script will combine all the available partial results.

    ```bash
    python -m cryoPARES.inference.daemon.materializePartialResults --partial_outputs_dirs /path/to/results_*/ --output_mrc /path/to/final_map.mrc --output_star /path/to/final_particles.star
    ```

### Projection Matching

If you have a set of particles and a reference volume, and you want to align the particles to the volume without running the full training/inference pipeline, you can use the `cryopares_projmatching` command.

This tool performs a local search around the existing particle orientations (or from an initial grid if no orientations are present) to find the best match against projections of the reference volume. It's a useful utility for refining poses or performing a quick alignment.

For detailed instructions, see the [Command-Line Interface documentation](./docs/cli.md).

### Reconstruction

If you have a set of particles with known poses (e.g., from a previous RELION run) and you simply want to reconstruct the 3D map, you can use the `cryopares_reconstruct` command.

This provides a quick way to generate a volume without going through the training or inference steps.

For detailed instructions, see the [Command-Line Interface documentation](./docs/cli.md).

### Checkpoint Compactification

After training, you can package your checkpoint into a compact ZIP file for easy distribution and storage. This reduces the checkpoint size from ~40 GB to ~10 GB by removing training logs, metrics, and intermediate files while keeping everything needed for inference.

**Compactify a checkpoint:**
```bash
python -m cryoPARES.scripts.compactify_checkpoint \
    --checkpoint_dir /path/to/training_output/version_0
```

This creates `version_0_compact.zip` containing only the essential files.

**Use the compactified checkpoint for inference:**
```bash
cryopares_infer \
    --particles_star_fname /path/to/particles.star \
    --checkpoint_dir /path/to/version_0_compact.zip \
    --results_dir /path/to/results
```

The ZIP file is used directly without extraction, making it ideal for:
- **Sharing models** with collaborators
- **Archiving** trained models efficiently
- **Deploying** to inference servers with limited storage

For more options (excluding reconstructions, custom names, etc.), see the [CLI documentation](./docs/cli.md#compactify_checkpoint).

### Configuration System

CryoPARES uses a flexible configuration system that allows you to manage settings from multiple sources.

*   **`--show-config`:** To see all available options, run any main script with the `--show-config` flag. This will print a comprehensive list of all parameters, their current values, and their paths.

    ```bash
    python -m cryopares_train --show-config
    ```

*   **YAML Files:** Create a `.yaml` file with your desired parameters.
*   **Command-Line Overrides:** Pass `KEY=VALUE` pairs to the program. Use dot notation to specify nested parameters (e.g., `models.image2sphere.lmax=6`).
*   **Direct Arguments:** Use standard command-line flags (e.g., `--batch_size 32`).

**Precedence:** Direct command-line arguments override `--config` overrides, which override YAML files, which override the default configuration.

For a complete reference of all configuration parameters, see the **[Configuration Guide](./docs/configuration_guide.md)**.

## Example Workflow

1.  **Train a model on an existing, aligned dataset:**
    ```bash
    python -m cryopares_train \
        --symmetry C1 \
        --particles_star_fname /path/to/aligned_particles.star \
        --particles_dir /path/to/particles \
        --train_save_dir /path/to/training_output \
        --n_epochs 10 \
        --config models.image2sphere.lmax=6
    ```

2.  **Run inference on a new dataset, with local refinement and reconstruction:**
    ```bash
    cryopares_infer \
        --particles_star_fname /path/to/new_particles.star \
        --particles_dir /path/to/particles \
        --checkpoint_dir /path/to/training_output/version_0 \
        --results_dir /path/to/inference_results \
        --reference_map /path/to/initial_model.mrc \
        --config projmatching.grid_distance_degs=15 inference.directional_zscore_thr=2.0
    ```

## Getting Help

If you encounter issues:

1. Check the **[Troubleshooting Guide](./docs/troubleshooting.md)** for common problems and solutions
2. Review the **[Training Guide](./docs/training_guide.md)** for training best practices
3. Consult the **[Configuration Guide](./docs/configuration_guide.md)** for parameter details
4. See the **[API Reference](./docs/api_reference.md)** for programmatic usage

For bugs or feature requests, please open an issue on [GitHub](https://github.com/rsanchezgarc/cryoPARES/issues).
