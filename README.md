# CryoPARES: Cryo-EM Pose Assignment for Related Experiments via Supervised deep learning

**CryoPARES** is a software package for assigning poses to 2D cryo-electron microscopy (cryo-EM) particle images.
It uses a supervised deep learning approach to accelerate 3D reconstruction in related cryo-EM experiments.
The key idea is to train a neural network on a high-quality reference reconstruction, and then reuse this trained model
to rapidly estimate particle poses in other, similar datasets.

This workflow is divided into two main phases:

*   **Training:** In this phase, you use a pre-existing, high-resolution dataset (where particle poses have already been determined by
traditional methods like RELION refine) to train a cryoPARES model. This process creates a model that
can recognize and assign poses to particles of that specific type of macromolecule.

*   **Inference:** Once the model is trained, you can use it for inference on new datasets of the *same* or *very similar* molecules
(e.g., the same protein with a different ligand bound). Because the model has already learned the features of the molecule, it can predict
particle poses almost instantly, bypassing the computationally expensive and time-consuming alignment steps of traditional workflows.
This is especially powerful for applications like drug screening, where many similar datasets need to be processed quickly.

This "train once, infer many times" paradigm allows for near real-time 3D reconstruction, providing rapid feedback during data
collection and analysis.

For a detailed explanation of the method, please refer to our paper:
[Supervised Deep Learning for Efficient Cryo-EM Image Alignment in Drug Discovery](https://www.biorxiv.org/content/10.1101/2025.03.04.641536)

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
    - [Static Mode](#static-mode)
    - [Daemon Mode (On-the-fly)](#daemon-mode-on-the-fly)
  - [Utility Tools](#utility-tools)
    - [Projection Matching](#projection-matching)
    - [Reconstruction](#reconstruction)
  - [Checkpoint Compactification](#checkpoint-compactification)
- [Documentation](#documentation)
  - [Configuration System](#configuration-system)
- [Example Workflow](#example-workflow)
- [Getting Help](#getting-help)
- [License and Attribution](#license-and-attribution)

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
ulimit -n 65536 #You are now able to deal with more than 30K .mrcs files. 
```
This command does not generally require sudo. If you are not allowed to increase this number, please, join the
.mrcs from different micrographs together to reduce the number of required files.

<br>
CryoPARES has two main modes of operation: training and inference. Particles need to be provided as RELION 3.1+ starfile(s).

### Training

The `cryoPARES.train.train` module is used to train a new model for pose estimation. Training needs to be done first using 
a pre-aligned dataset of particles. While not mandatory, we encourage using particles alignments estimated with RELION.


**Usage:**
```bash
cryopares_train [ARGUMENTS] [--config [CONFIG_OVERRIDES]] [--show-config]
```

**Key Arguments:**

<!-- AUTO_GENERATED:train_parameters:START -->
**Required Parameters:**

*   `--symmetry`: Point group symmetry of the molecule (e.g., C1, D7, I, O, T)

*   `--particles_star_fname`: Path(s) to RELION 3.1+ format .star file(s) containing pre-aligned particles. Can accept multiple files

*   `--train_save_dir`: Output directory where model checkpoints, logs, and training artifacts will be saved


**Optional Parameters:**

*   `--particles_dir`: Root directory for particle image paths. If paths in .star file are relative, this directory is prepended (similar to RELION project directory concept)

*   `--n_epochs`: Number of training epochs. More epochs allow better convergence, although it does not help beyond a certain point (Default: `100`)

*   `--batch_size`: Number of particles per batch. Try to make it as large as possible before running out of GPU memory. We advice using batch sizes of at least 32 images (Default: `32`)

*   `--num_dataworkers`: Number of parallel data loading workers per GPU. Each worker is a separate CPU process. Set to 0 to load data in the main thread (useful only for debugging). Try not to oversubscribe by asking more workers than CPUs (Default: `8`)

*   `--image_size_px_for_nnet`: Target image size in pixels for neural network input. After rescaling to target sampling rate, images are cropped or padded to this size. We recommend tight box-sizes (Default: `160`)

*   `--sampling_rate_angs_for_nnet`: Target sampling rate in Angstroms/pixel for neural network input. Particle images are first rescaled to this sampling rate before processing (Default: `1.5`)

*   `--mask_radius_angs`: Radius of circular mask in Angstroms applied to particle images. If not provided, defaults to half the box size

*   `--split_halves`: If True (default), trains two separate models on data half-sets for cross-validation. Use --NOT_split_halves to train single model on all data (Default: `True`)

*   `--continue_checkpoint_dir`: Path to checkpoint directory to resume training from a previous run

*   `--finetune_checkpoint_dir`: Path to checkpoint directory to fine-tune a pre-trained model on new dataset

*   `--compile_model`: Enable torch.compile for faster training (experimental) (Default: `False`)

*   `--val_check_interval`: Fraction of epoch between validation checks. You generally don't want to touch it, but you can set it to smaller values (0.1-0.5) for large datasets to get quicker feedback

*   `--overfit_batches`: Number of batches to use for overfitting test (debugging feature to verify model can memorize small dataset)

*   `--map_fname_for_simulated_pretraining`: Path(s) to reference map(s) for simulated projection warmup before training on real data. The number of maps must match number of particle star files

*   `--junk_particles_star_fname`: Optional star file(s) with junk-only particles for estimating confidence z-score thresholds

*   `--junk_particles_dir`: Root directory for junk particle image paths (analogous to particles_dir)

<!-- AUTO_GENERATED:train_parameters:END -->

**Additional relevant Parameters (via --config):**

You can override configuration parameters using `--config KEY=VALUE`. Multiple key-value pairs can be provided. The `--config` flag should be the last argument. To see all available configuration options, run `cryopares_train --show-config`.

*   **`train.learning_rate`**: Initial learning rate. (Default: `1e-3`). It needs to be tuned to get the best performance.
*   **`train.weight_decay`**: Weight decay for optimizer, that regularizes the model. (Default: `1e-5`). Make it larger if you are suffer from overfitting.
*   **`train.accumulate_grad_batches`**: Gradient accumulation batches to simulate larger batch sizes. (Default: `16`). The effecive batch size is batch_size * accumulate_grad_batches. We recommend to train with effective batches of size 512 < x < 2048. 
*   **`models.image2sphere.lmax`**: Maximum spherical harmonic degree. The larger, the more expresive the network is (Default: `12`). Reduce it if you see overfitting.
*   **`datamanager.num_augmented_copies_per_batch`**: Number of augmented copies per particle. Each copy undergoes a different data augmentation. The batch_size needs to be selected to be divisible by this number. Large batches with large num_augmented_copies_per_batch values help stabilizing training, but require a lot of GPU memory (Default: `4`)

For comprehensive training guidance including monitoring with TensorBoard and avoiding overfitting/underfitting, see the **[Training Guide](./docs/training_guide.md)**. 
For a complete list of all configuration parameters, see the **[Configuration Guide](./docs/configuration_guide.md)**.
<br>
Once the training is done, you could use the checkpoint dir contained in `--train_save_dir` to infer poses of new datasets.
The checkpoint dir is named version_0. If you run another training experiment with the same `--train_save_dir`, another
checkpoint dir names version_1 will be created.

### Inference

The `cryoPARES.inference.inference` module is used to predict poses for a new set of particles using a trained model. It can be run in two modes: static and daemon.

#### Static Mode

In static mode, the inference is run on a fixed set of particles, that again, need to be provided as RELION 3.1+ starfiles.

**Usage:**
```bash
cryopares_infer [ARGUMENTS] [--config [CONFIG_OVERRIDES]] [--show-config]
```

**Key Arguments:**

<!-- AUTO_GENERATED:inference_parameters:START -->
**Required Parameters:**

*   `--particles_star_fname`: Path to input RELION particles .star file

*   `--checkpoint_dir`: Path to training directory (or .zip file) containing half-set models with checkpoints and hyperparameters. By default they are called version_0, version_1, etc.

*   `--results_dir`: Output directory for inference results including predicted poses and optional reconstructions


**Optional Parameters:**

*   `--data_halfset`: Which particle half-set(s) to process: "half1", "half2", or "allParticles" (Default: `allParticles`)

*   `--model_halfset`: Model half-set selection policy: "half1", "half2", "allCombinations", or "matchingHalf" (uses matching data/model pairs) (Default: `matchingHalf`)

*   `--particles_dir`: Root directory for particle image paths. If provided, overrides paths in the .star file

*   `--batch_size`: Number of particles per batch for inference (Default: `64`)

*   `--n_jobs`: Number of worker processes. Defaults to number of GPUs if CUDA enabled, otherwise 1

*   `--num_dataworkers`: Number of parallel data loading workers per GPU. Each worker is a separate CPU process. Set to 0 to load data in the main thread (useful only for debugging). Try not to oversubscribe by asking more workers than CPUs (Default: `8`)

*   `--use_cuda`: Enable GPU acceleration for inference. If False, runs on CPU only (Default: `True`)

*   `--n_cpus_if_no_cuda`: Maximum CPU threads per worker when CUDA is disabled (Default: `4`)

*   `--compile_model`: Compile model with torch.compile for faster inference (experimental, requires PyTorch 2.0+) (Default: `False`)

*   `--top_k_poses_nnet`: Number of top pose predictions to retrieve from neural network before local refinement (Default: `1`)

*   `--top_k_poses_localref`: Number of best matching poses to keep after local refinement (Default: `1`)

*   `--grid_distance_degs`: Maximum angular distance in degrees for local refinement search. Grid ranges from -grid_distance_degs to +grid_distance_degs around predicted pose (Default: `6.0`)

*   `--reference_map`: Path to reference map (.mrc) for FSC computation during validation

*   `--reference_mask`: Path to reference mask (.mrc) for masked FSC calculation

*   `--directional_zscore_thr`: Confidence z-score threshold for filtering particles. Particles with scores below this are discarded as low-confidence

*   `--skip_localrefinement`: Skip local pose refinement step and use only neural network predictions (Default: `False`)

*   `--skip_reconstruction`: Skip 3D reconstruction step and output only predicted poses (Default: `False`)

*   `--subset_idxs`: List of particle indices to process (for debugging or partial processing)

*   `--n_first_particles`: Process only the first N particles from dataset (debug feature)

*   `--check_interval_secs`: Polling interval in seconds for parent loop in distributed processing (Default: `2.0`)

<!-- AUTO_GENERATED:inference_parameters:END -->

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

**Note:** Many of these parameters can also be set via `--config` (e.g., `--config projmatching.grid_step_degs=2.0`). However, using the direct CLI flags is recommended for commonly adjusted parameters. 
To see all available configuration options, run `cryopares_infer --show-config`.

For detailed API documentation, see the **[API Reference](https://rsanchezgarc.github.io/cryoPARES/api/)**.

#### Daemon Mode (On-the-fly)

In daemon mode, the inference script runs continuously and watches for new particles to be added to a directory. This is useful for processing particles as they are being generated.

The daemon workflow consists of three main components:

1.  **Queue Manager:** A central server that manages a queue of jobs.
2.  **Spooling Filler:** A script that monitors a directory for new `.star` files and adds them to the queue. You could implement other filler protocols using this module as an example.
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

**Alternative: Manually Submit Jobs**

You can also use other mechanisms to add jobs to the queue programmatically. The key is to use the
`cryoPARES.inference.daemon.queueManager.queue_connection` context manager to connect to the queue
and the `put()` method to add new `.star` files to be processed.

**Example Python script to submit jobs:**

```python
from cryoPARES.inference.daemon.queueManager import queue_connection

# Submit a single .star file
with queue_connection(ip="localhost", port=50000, authkey="shared_queue_key") as queue:
    queue.put("/path/to/particles.star")
    print("Job submitted!")

# Submit multiple .star files
with queue_connection() as queue:  # Uses default connection settings
    star_files = [
        "/path/to/particles1.star",
        "/path/to/particles2.star",
        "/path/to/particles3.star"
    ]
    for star_file in star_files:
        queue.put(star_file)
    print(f"Submitted {len(star_files)} jobs")

# Submit a .star file with a specific particle subset (advanced)
import pandas as pd
particles_subset = pd.read_csv("/path/to/particles.star", comment='#', delimiter=r'\s+')
# Filter to specific particles, e.g., particles_subset = particles_subset[particles_subset['rlnImageName'].str.contains('mic001')]

with queue_connection() as queue:
    queue.put(("/path/to/particles.star", particles_subset))  # Tuple: (star_file, DataFrame)
    print("Submitted job with particle subset")

# Send poison pill to terminate workers gracefully
with queue_connection() as queue:
    queue.put(None)  # None acts as a poison pill
    print("Sent termination signal to workers")
```

**Input formats:**
- **String:** Path to a `.star` file (all particles will be processed)
- **Tuple:** `(star_file_path, pd.DataFrame)` where the DataFrame contains a subset of particles to process
- **None:** Poison pill to signal workers to terminate gracefully


3.  **Start the Daemon Inferencer(s):**

    You can start as many inference workers as you want. Each worker will take jobs from the queue and process them in parallel. **Important:** Each worker must have its own results directory.

    ```bash
    # Worker 1
    python -m cryoPARES.inference.daemon.daemonInference --checkpoint_dir /path/to/checkpoint --results_dir /path/to/results_worker1 --particles_dir /path/to/particles

    # Worker 2
    python -m cryoPARES.inference.daemon.daemonInference --checkpoint_dir /path/to/checkpoint --results_dir /path/to/results_worker2 --particles_dir /path/to/particles
    ```
The arguments accepted by `daemonInference` are mostly the same that the ones accepted by `cryopares_infer`

4.  **Materialize the Volume:**

    You can materialize the final 3D volume from the partial results at any time, even while the inference workers are still running.
    The script will combine all the available partial results.

    ```bash
    python -m cryoPARES.inference.daemon.materializePartialResults --partial_outputs_dirs /path/to/results_worker1/ /path/to/results_worker2 \
     --output_mrc /path/to/final_map.mrc --output_star /path/to/final_particles.star
    ```

### Utility Tools

CryoPARES includes standalone utility tools for projection matching and reconstruction. 
**Note:** These tools are automatically used within the `cryopares_infer` workflow, but can also be run independently if needed.<br>

#### Projection Matching

The projection matching utility performs local pose refinement by searching around existing particle orientations to find the best match against reference volume projections. This is used automatically during inference for local refinement, but can also be run standalone.

**Usage:**
```bash
cryopares_projmatching [ARGUMENTS] [--config [CONFIG_OVERRIDES]] [--show-config]
```

**Key Arguments:**

<!-- AUTO_GENERATED:projmatching_parameters:START -->
**Required Parameters:**

*   `--reference_vol`: Path to reference 3D volume (.mrc file) for generating projection templates

*   `--particles_star_fname`: Path to input STAR file with particle metadata

*   `--out_fname`: Path for output STAR file with aligned particle poses

*   `--particles_dir`: Root directory for particle image paths. If provided, overrides paths in the .star file


**Optional Parameters:**

*   `--mask_radius_angs`: Radius of circular mask in Angstroms applied to particle images

*   `--grid_distance_degs`: Maximum angular distance in degrees for local refinement search. Grid ranges from -grid_distance_degs to +grid_distance_degs around predicted pose (Default: `6.0`)

*   `--grid_step_degs`: Angular step size in degrees for grid search during local refinement (Default: `2.0`)

*   `--return_top_k_poses`: Number of top matching poses to save per particle (Default: `1`)

*   `--filter_resolution_angst`: Low-pass filter resolution in Angstroms applied to reference volume before matching

*   `--n_jobs`: Number of parallel worker processes for distributed projection matching (Default: `1`)

*   `--num_dataworkers`: Number of CPU workers per PyTorch DataLoader for data loading (Default: `1`)

*   `--batch_size`: Number of particles to process simultaneously per job (Default: `1024`)

*   `--use_cuda`: Enable GPU acceleration. If False, runs on CPU only (Default: `True`)

*   `--verbose`: Enable verbose logging output (Default: `True`)

*   `--float32_matmul_precision`: PyTorch float32 matrix multiplication precision mode ("highest", "high", or "medium") (Default: `high`)

*   `--gpu_id`: Specific GPU device ID to use (if multiple GPUs available)

*   `--n_first_particles`: Process only the first N particles from dataset (for testing or validation)

*   `--correct_ctf`: Apply CTF correction during projection matching (Default: `True`)

*   `--halfmap_subset`: Select half-map subset (1 or 2) for half-map validation

<!-- AUTO_GENERATED:projmatching_parameters:END -->

For additional details, see the [Command-Line Interface documentation](./docs/cli.md).

#### Reconstruction

The reconstruction utility creates a 3D volume from particles with known poses using direct Fourier inversion. This is used automatically during inference to generate the final 3D map, but can also be run standalone for particles aligned by other methods (e.g., RELION).

**Usage:**
```bash
cryopares_reconstruct [--config [CONFIG_OVERRIDES]] [--show-config]
```

**Key Arguments:**

<!-- AUTO_GENERATED:reconstruct_parameters:START -->
**Required Parameters:**

*   `--particles_star_fname`: Path to input STAR file with particle metadata and poses to reconstruct

*   `--symmetry`: Point group symmetry of the volume for reconstruction (e.g., C1, D2, I, O, T)

*   `--output_fname`: Path for output reconstructed 3D volume (.mrc file)


**Optional Parameters:**

*   `--particles_dir`: Root directory for particle image paths. If provided, overrides paths in the .star file

*   `--n_jobs`: Number of parallel worker processes for distributed reconstruction (Default: `1`)

*   `--num_dataworkers`: Number of CPU workers per PyTorch DataLoader for data loading (Default: `1`)

*   `--batch_size`: Number of particles to backproject simultaneously per job (Default: `128`)

*   `--use_cuda`: Enable GPU acceleration for reconstruction. If False, runs on CPU only (Default: `True`)

*   `--correct_ctf`: Apply CTF correction during reconstruction (Default: `True`)

*   `--eps`: Regularization constant for reconstruction (ideally set to 1/SNR). Prevents division by zero and stabilizes reconstruction (Default: `0.001`)

*   `--min_denominator_value`: Minimum value for denominator to prevent numerical instabilities during reconstruction

*   `--use_only_n_first_batches`: Reconstruct using only first N batches (for testing or quick validation)

*   `--float32_matmul_precision`: PyTorch float32 matrix multiplication precision mode ("highest", "high", or "medium") (Default: `high`)

*   `--weight_with_confidence`: Apply per-particle confidence weighting during backprojection. If True, particles with higher confidence contribute more to reconstruction. It reads the confidence from the metadata label "rlnParticleFigureOfMerit" (Default: `False`)

*   `--halfmap_subset`: Select half-map subset (1 or 2) for half-map reconstruction and validation

<!-- AUTO_GENERATED:reconstruct_parameters:END -->

For additional details, see the [Command-Line Interface documentation](./docs/cli.md).

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


## Documentation

- **[Training Guide](./docs/training_guide.md)** - Comprehensive guide on training models, monitoring with TensorBoard, and avoiding overfitting/underfitting
- **[API Reference](https://rsanchezgarc.github.io/cryoPARES/api/)** - Auto-generated API documentation with type hints (hosted on GitHub Pages)
- **[Configuration Guide](./docs/configuration_guide.md)** - Complete reference for all configuration parameters
- **[Troubleshooting Guide](./docs/troubleshooting.md)** - Solutions to common issues
- **[CLI Reference](./docs/cli.md)** - Command-line interface documentation

**Building Documentation Locally:**
```bash
cd docs
pip install -r requirements.txt
make html
# Open _build/html/index.html in your browser
```


### Configuration System

CryoPARES uses a flexible configuration system that allows you to manage settings from multiple sources.

*   **`--show-config`:** To see all available options, run any main script with the `--show-config` flag. This will print a comprehensive list of all parameters, their current values, and their paths.

    ```bash
    cryopares_train --show-config
    ```

*   **YAML Files:** Create a `.yaml` file with your desired parameters.
*   **Command-Line Overrides:** Pass `KEY=VALUE` pairs to the program. Use dot notation to specify nested parameters (e.g., `models.image2sphere.lmax=6`).
*   **Direct Arguments:** Use standard command-line flags (e.g., `--batch_size 32`).

**Precedence:** Direct command-line arguments override `--config` overrides, which override YAML files, which override the default configuration.

For a complete reference of all configuration parameters, see the **[Configuration Guide](./docs/configuration_guide.md)**.

## Example Workflow

1.  **Train a model on an existing, aligned dataset:**
    ```bash
    cryopares_train \
        --symmetry C1 \
        --particles_star_fname /path/to/aligned_particles.star \
        --particles_dir /path/to/particles \
        --train_save_dir /path/to/training_output \
        --n_epochs 10 \
        --batch_size 64 \
        --image_size_px_for_nnet 160 \ #The box size seen by the neural network will be 160.
        --sampling_rate_angs_for_nnet 1.5 \ 
        --config models.image2sphere.lmax=6 #lmax=6 to have a quick, low accuracy model
    ```

2.  **Run inference on a new dataset with local refinement and reconstruction:**
    ```bash
    cryopares_infer \
        --particles_star_fname /path/to/new_particles.star \
        --particles_dir /path/to/particles \
        --checkpoint_dir /path/to/training_output/version_0 \
        --results_dir /path/to/inference_results \
        --reference_map /path/to/initial_model.mrc \
        --batch_size 64 \
        --grid_distance_degs 15 \
        --directional_zscore_thr 1.0  #Remove all particles with directional zscore <1.0
        --n_jobs 2 #To use 2 GPUs
    ```

## Getting Help

If you encounter issues:

1. Check the **[Troubleshooting Guide](./docs/troubleshooting.md)** for common problems and solutions
2. Review the **[Training Guide](./docs/training_guide.md)** for training best practices
3. Consult the **[Configuration Guide](./docs/configuration_guide.md)** for parameter details
4. See the **[API Reference](https://rsanchezgarc.github.io/cryoPARES/api/)** for programmatic usage

For bugs or feature requests, please open an issue on [GitHub](https://github.com/rsanchezgarc/cryoPARES/issues).

## License and Attribution

CryoPARES is licensed under the **GNU General Public License v3.0** (GPL-3.0). See the [LICENSE](LICENSE) file for details.

### Third-Party Code

This project incorporates code derived from the following open-source projects:

- **[torch-fourier-slice](https://github.com/teamtomo/torch-fourier-slice)** (Copyright © 2023 Alister Burt, BSD 3-Clause License)
  - Used in: `cryoPARES/reconstruction/insert_central_slices_rfft_3d.py`
  - Used in: `cryoPARES/projmatching/projmatchingUtils/extract_central_slices_as_real.py`

See [THIRD-PARTY-LICENSES](THIRD-PARTY-LICENSES) for complete license texts and attribution details.
