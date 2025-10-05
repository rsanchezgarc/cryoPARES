# Training Guide

This guide provides detailed instructions for training CryoPARES models, including best practices for avoiding overfitting/underfitting and monitoring training progress.

## Table of Contents

- [Quick Start](#quick-start)
- [Training Parameters](#training-parameters)
- [Monitoring Training with TensorBoard](#monitoring-training-with-tensorboard)
- [Overfitting and Underfitting](#overfitting-and-underfitting)
- [Data Preprocessing](#data-preprocessing)
- [Advanced Training Options](#advanced-training-options)
- [Troubleshooting Training Issues](#troubleshooting-training-issues)

## Quick Start

Basic training command:

```bash
python -m cryopares_train \
    --symmetry C1 \
    --particles_star_fname /path/to/aligned_particles.star \
    --particles_dir /path/to/particles \
    --train_save_dir /path/to/output \
    --n_epochs 10 \
    --batch_size 32
```

**Important:** Before training, increase the file descriptor limit:
```bash
ulimit -n 65536
```
If you cannot increase it, make sure that you keep all your images into a small number of stack files.

## Training Parameters

### Essential Parameters

| Parameter | Type | Default  | Description |
|-----------|------|----------|-------------|
| `--symmetry` | str | Required | Point group symmetry (C1, D7, T, O, I, etc.) |
| `--particles_star_fname` | str | Required | Path to pre-aligned RELION .star file |
| `--train_save_dir` | str | Required | Output directory for checkpoints and logs |
| `--n_epochs` | int | 100      | Number of training epochs |
| `--batch_size` | int | 64       | Number of particles per batch |

### Data Configuration Parameters

Override via `--config` flag with dot notation:

```bash
--config datamanager.particlesDataset.sampling_rate_angs_for_nnet=2.0 \
         datamanager.particlesDataset.image_size_px_for_nnet=128
```

**Key data parameters:**

- **`datamanager.particlesDataset.sampling_rate_angs_for_nnet`** (float)
  - Target sampling rate in Ångströms/pixel
  - Images are rescaled to this value before training
  - Lower values = higher resolution but more memory/compute
  - Downsampling helps to reduce noise.
- **`datamanager.particlesDataset.image_size_px_for_nnet`** (int)
  - Final image size in pixels after rescaling
  - Images are cropped/padded to this size
  - Must be large enough to contain the particle after rescaling
    - But we recomend using tight boxes.

- **`datamanager.particlesDataset.mask_radius_angs`** (float, optional)
  - Radius of circular mask in Ångströms
  - If not set, uses half the box size
  - Helps the network focus on the particle
  - Notice that image_size_px_for_nnet is not the same as mask_radius_angs, but that they should be closelly related.

### Optimizer Configuration

```bash
--config train.learning_rate=1e-3 \
         train.weight_decay=1e-5 \
         train.accumulate_grad_batches=16
```

**Key optimizer parameters:**

- **`train.learning_rate`** (float)
  - Initial learning rate for the Adam optimizer
  - Automatically reduced on plateau (see below)

- **`train.weight_decay`** (float)
  - L2 regularization coefficient
  - Helps prevent overfitting

- **`train.accumulate_grad_batches`** (int)
  - Number of batches to accumulate gradients over
  - Simulates larger batch sizes: effective_batch_size = batch_size × accumulate_grad_batches
  - Useful when GPU memory is limited

- **`train.patient_reduce_lr_plateau_n_epochs`** (int, default: 3)
  - Patience for ReduceLROnPlateau scheduler
  - LR reduced by 0.5 if val_loss doesn't improve for this many epochs

### Model Architecture Parameters

Example:
```bash
--config models.image2sphere.lmax=8
```

**Key architecture parameters:**

- **`models.image2sphere.lmax`** (int, default: 12)
  - Maximum spherical harmonic degree for SO(3) representation
  - Higher values = more expressive model but slower training and more memory usage
  - Typical values: 8, 10, 12

- **`models.image2sphere.label_smoothing`** (float, default: 0.05)
  - Label smoothing factor for loss function to prevent overconfidence
  - Range: 0.0 (no smoothing) to 0.2 (strong smoothing)
  - Helps with generalization

- **`models.image2sphere.so3components.i2sprojector.sphere_fdim`** (int, default: 512)
  - Feature dimension for spherical representation in the image-to-sphere projector
  - Higher values = more capacity but slower training
  - Typical values: 256, 512, 1024

- **`models.image2sphere.so3components.i2sprojector.rand_fraction_points_to_project`** (float, default: 0.5)
  - Fraction of points to randomly sample for projection (reduces computation)
  - Range: 0.1 to 1.0 (1.0 = use all points)
  - Lower values = faster but potentially less accurate

- **`models.image2sphere.so3components.s2conv.f_out`** (int, default: 64)
  - Number of output features from S2 (sphere) convolution
  - Higher values = more capacity but slower training
  - Typical values: 32, 64, 128

- **`models.image2sphere.so3components.i2sprojector.hp_order`** (int, default: 3)
  - HEALPix order for image-to-sphere projector grid resolution
  - Higher values = finer resolution but more computation
  - Each increment roughly doubles resolution. Going beyond 4 is not advisable

- **`models.image2sphere.so3components.s2conv.hp_order`** (int, default: 4)
  - HEALPix order for S2 convolution grid resolution
  - Controls the resolution of spherical convolution

- **`models.image2sphere.so3components.so3outputgrid.hp_order`** (int, default: 4)
  - HEALPix order for SO(3) output grid resolution
  - Affects the final orientation prediction granularity

### Data Augmentation Parameters

Data augmentation is **enabled by default** and helps the model generalize by creating variations of training images. CryoPARES applies multiple augmentation operations randomly to each particle image.

**View current augmentation settings:**
```bash
cryopares_train --show-config | grep -A 20 "augmenter:"
```

**Key augmentation parameters:**

- **`datamanager.augment_train`** (bool, default: True)
  - Enable/disable data augmentation for training
  - Keep enabled for better generalization

- **`datamanager.num_augmented_copies_per_batch`** (int, default: 4)
  - Number of augmented copies per particle in each batch
  - Each copy undergoes different random augmentations
  - Batch size must be divisible by this value
  - Higher values improve robustness but increase computation

- **`datamanager.augmenter.prob_augment_each_image`** (float, default: 0.95)
  - Probability of applying augmentation to each image
  - Range: 0.0 (no augmentation) to 1.0 (always augment)

- **`datamanager.augmenter.min_n_augm_per_img`** (int, default: 1)
  - Minimum number of augmentation operations to apply per image

- **`datamanager.augmenter.max_n_augm_per_img`** (int, default: 8)
  - Maximum number of augmentation operations to apply per image

**Available augmentation operations (with default probabilities):**

- **Gaussian noise** (`operations.randomGaussNoise.p=0.1`)
  - Adds random Gaussian noise to simulate imaging variations

- **Uniform noise** (`operations.randomUnifNoise.p=0.2`)
  - Adds uniform random noise

- **Gaussian blur** (`operations.gaussianBlur.p=0.2`)
  - Blurs the image to simulate defocus variations

- **Size perturbation** (`operations.sizePerturbation.p=0.2`)
  - Slightly scales the particle (simulates magnification errors)

- **Random erasing** (`operations.erasing.p=0.1`)
  - Randomly erases rectangular regions (simulates occlusions)

- **Elastic deformation** (`operations.randomElastic.p=0.1`)
  - Applies elastic distortions to the image

- **In-plane rotations (90°)** (`operations.inPlaneRotations90.p=1.0`)
  - Rotates by 90°, 180°, or 270° (always applied for SO(3) symmetry)

- **In-plane rotations (small)** (`operations.inPlaneRotations.p=0.5`)
  - Random rotations up to ±20° (simulates alignment errors)

- **In-plane shifts** (`operations.inPlaneShifts.p=0.5`)
  - Random shifts up to 5% of image size (simulates centering errors)

**Example: Adjusting augmentation strength**

To reduce augmentation (if overfitting is not an issue):
```bash
--config datamanager.augmenter.prob_augment_each_image=0.5 \
         datamanager.augmenter.max_n_augm_per_img=4
```

To increase augmentation (to combat overfitting):
```bash
--config datamanager.num_augmented_copies_per_batch=8 \
         datamanager.augmenter.prob_augment_each_image=0.98 \
         datamanager.augmenter.operations.gaussianBlur.p=0.3
```

To disable specific augmentations:
```bash
--config datamanager.augmenter.operations.randomElastic.p=0.0 \
         datamanager.augmenter.operations.erasing.p=0.0
```

## Monitoring Training with TensorBoard

CryoPARES uses PyTorch Lightning, which automatically logs metrics to TensorBoard.

### Launching TensorBoard

```bash
tensorboard --logdir /path/to/train_save_dir/version_0
```

Then open your browser to `http://localhost:6006`

### Key Metrics to Monitor

#### 1. Training Loss (`loss`)

- Should decrease steadily during training
- Measures how well the model predicts rotations on training data
- If it plateaus early, try:
  - Increasing learning rate
  - Checking if data augmentation is too aggressive
  - Reducing weight decay

#### 2. Validation Loss (`val_loss`)

- Most important metric for model quality
- Should track training loss but slightly higher
- **Warning signs:**
  - Val loss much higher than train loss → overfitting
  - Val loss not decreasing → underfitting or learning rate too low

#### 3. Angular Error (`geo_degs`, `val_geo_degs`)

- Average angular error in degrees
- **Goal:** As low as possible (typically < 15° for good models)
- `val_geo_degs` is the key metric for final model quality

#### 4. Median Angular Error (`val_median_geo_degs`)

- More robust than mean to outliers
- Should also decrease during training
- **Goal:** As low as possible (typically < 8° for good models)
- `val_geo_degs` and `val_median_geo_degs` represent the same property, but aggregated in a different manner.
They should follow the same trends.

#### 5. Learning Rate (`lr`)

- Displayed in optimizer logs
- Watch for automatic reductions via ReduceLROnPlateau
- If LR drops too early, increase `train.patient_reduce_lr_plateau_n_epochs`

#### 6. Visualization: Rotation Matrices

- TensorBoard shows predicted vs. ground truth rotation matrices
- Available under "Images" tab
- Visual confirmation that the model is learning meaningful rotations

### Example TensorBoard Monitoring Session

```bash
# Start training
python -m cryopares_train \
    --symmetry C1 \
    --particles_star_fname data/particles.star \
    --train_save_dir experiments/run_001 \
    --n_epochs 20

# In another terminal, launch TensorBoard
tensorboard --logdir experiments/run_001/version_0

# Monitor these curves:
# 1. loss (should decrease smoothly)
# 2. val_loss (should track loss)
# 3. val_geo_degs (target: < 5 degrees)
# 4. Learning rate (should stay constant, then drop on plateau)
```

## Overfitting and Underfitting

### What is Overfitting?

Overfitting occurs when the model learns to memorize training data instead of generalizing to new data.

**Symptoms:**
- Training loss continues to decrease while validation loss increases or plateaus
- Large gap between `loss` and `val_loss`
- `val_geo_degs` stops improving or gets worse

**Solutions:**

1. **Increase regularization:**
   ```bash
   --config train.weight_decay=1e-4  models.image2sphere.label_smoothing=0.1 # Increase from default 1e-5, and 0.05 respectively
   ```

2. **Reduce model complexity:**
   ```bash
   --config models.image2sphere.lmax=10 # Decrease from default 12 models.image2sphere.lmax=10
   ```

3. **Add more training data:**
   - Use more particles in your training .star file

4. **Increase data augmentation:**
   Data augmentation is enabled by default. Check current settings:
   ```bash
   python -m cryopares_train --show-config | grep augmentation
   ```

5. **Early stopping:**
   Training automatically saves the best checkpoint based on `val_loss`
   - Stop training if val_loss hasn't improved in 5+ epochs

### What is Underfitting?

Underfitting occurs when the model is too simple to capture patterns in the data.

**Symptoms:**
- Both training and validation loss remain high
- `val_geo_degs` > 10 degrees even after many epochs
- Loss curves plateau early

**Solutions:**

1. **Increase model complexity:**
   ```bash
   --config models.image2sphere.lmax=10  # Increase from default 8
   ```

2. **Train longer:**
   ```bash
   --n_epochs 30  # Increase from default 10
   ```

3. **Increase learning rate:**
   ```bash
   --config train.learning_rate=5e-3  # Increase from default 1e-3
   ```

4. **Reduce regularization:**
   ```bash
   --config train.weight_decay=1e-6  # Decrease from default 1e-5
   ```

5. **Check data preprocessing:**
   - Ensure `sampling_rate_angs_for_nnet` matches your data resolution
   - Verify particle images are properly centered and normalized

6. **Use a better encoder:**
   ```bash
   --config models.image2sphere.imageencoder.resnet.resnetName="resnet50"
   ```
   (Default is resnet18; resnet50 is more powerful)

### The Sweet Spot

**Ideal training behavior:**
- Both `loss` and `val_loss` decrease together
- Small gap between train and validation metrics
- `val_geo_degs` reaches < 3-5 degrees
- Validation metrics improve for at least 30-35 epochs

**Example of good training:**
```
TODO: Copy here from bgal
```

## Data Preprocessing

### Image Size and Sampling Rate

The neural network operates on rescaled particle images. Understanding this is crucial:

1. **Original images:** Read from `.star` file with original pixel size and sampling rate
2. **Rescaling:** Images are rescaled to `sampling_rate_angs_for_nnet`
3. **Crop/Pad:** Images are cropped or padded to `image_size_px_for_nnet`

**Example:**
```
Original: 256×256 px at 1.0 Å/px (256 Å box)
Target: 128×128 px at 2.0 Å/px (256 Å box)

Result: Image is downsampled by 2×, then center-cropped to 128×128
```

**Guidelines:**
- `sampling_rate_angs_for_nnet`: 1.5-3.0 Å/px works well for most proteins
- `image_size_px_for_nnet`: Should contain entire particle after rescaling
- Rule of thumb: `image_size_px_for_nnet × sampling_rate_angs_for_nnet ≥ particle_diameter + padding`

### Masking

```bash
--config datamanager.particlesDataset.mask_radius_angs=100
```

- Applies a soft circular mask to focus on the particle
- Set to slightly larger than particle radius
- Too small → cuts off particle features
- Too large → includes too much noise

### CTF Correction

CTF correction is applied automatically during training:
- Phase flip by default
- Ensures the model sees properly corrected images

## Advanced Training Options

### Continue Training

Resume from a previous checkpoint:

```bash
python -m cryopares_train \
    --continue_checkpoint_dir /path/to/train_save_dir/version_0 \
    --n_epochs 30  # Train for 30 total epochs
```

### Fine-tuning

Start from a pre-trained model and adapt to new data:

```bash
python -m cryopares_train \
    --symmetry C1 \
    --particles_star_fname /path/to/new_particles.star \
    --train_save_dir /path/to/finetuned_model \
    --finetune_checkpoint_dir /path/to/pretrained_model/version_0 \
    --n_epochs 5 \
    --config train.learning_rate=1e-4  # Lower LR for fine-tuning
```

**When to fine-tune:**
- Training on similar protein with different ligand
- Limited new training data
- Want to adapt pre-trained model to slightly different conditions

### Half-Set Training

By default, CryoPARES trains two models (half1 and half2) for cross-validation:

```bash
--split_halfs True  # Default
```

This creates:
- `version_0/half1/` - Model trained on particles with RandomSubset=1
- `version_0/half2/` - Model trained on particles with RandomSubset=2

**Benefits:**
- Prevents overfitting during inference
- Enables gold-standard FSC calculations
- Recommended for production use

To train on all data (single model):
```bash
--split_halfs False
```

### Simulated Pre-training

Pre-train on simulated data before training on real particles:

```bash
python -m cryopares_train \
    --symmetry C1 \
    --particles_star_fname /path/to/real_particles.star \
    --train_save_dir /path/to/output \
    --map_fname_for_simulated_pretraining /path/to/reference_map.mrc \
    --n_epochs 40 \
    --config train.n_epochs_simulation=5 #We will train the model for 3 epochs using simulated data
```

This first trains on simulated projections of the reference map, then fine-tunes on real data. 

### Model Compilation

Speed up training with PyTorch compilation (requires PyTorch 2.0+):

```bash
--compile_model
```

**Note:** Compilation adds overhead at startup but can speed up training by 10-30%.

### Debugging: Overfitting on Small Batches

Test your setup quickly by overfitting on a few batches:

```bash
python -m cryopares_train \
    --symmetry C1 \
    --particles_star_fname /path/to/particles.star \
    --train_save_dir /tmp/overfit_test \
    --n_epochs 100 \
    --overfit_batches 10
```

## Troubleshooting Training Issues

### Training is very slow

**Solutions:**
- Enable model compilation: `--compile_model`
- Reduce image size: `--config datamanager.particlesDataset.image_size_px_for_nnet=96`
- Increase batch size: `--batch_size 64`
- Use multiple GPUs (automatically detected)
- Reduce model complexity, like `lmax`: `--config models.image2sphere.lmax=10`

### Out of memory errors

**Solutions:**
- Reduce batch size: `--batch_size 16`
- Reduce image size: `--config datamanager.particlesDataset.image_size_px_for_nnet=96`
- Increase gradient accumulation: `--config train.accumulate_grad_batches=32`

### "Too many open files" error

```bash
ulimit -n 65536
```

Run this before every training session, or add to `.bashrc`:

```bash
echo "ulimit -n 65536" >> ~/.bashrc
```

### Loss is NaN

**Causes:**
- Learning rate too high
- Numerical instability

**Solutions:**
```bash
--config train.learning_rate=1e-4  # Reduce LR
```

### Model not improving

**Checklist:**
1. Verify data is properly aligned in the input .star file
2. Check that particles are centered
3. Ensure sufficient training data (>100000 particles recommended)
4. Verify symmetry is correct
5. Try increasing learning rate: `--config train.learning_rate=5e-3`
6. Increase model capacity: `--config models.image2sphere.lmax=12 models.image2sphere.so3components.s2conv.f_out=128`

### Validation loss jumps around

**Causes:**
- Validation set too small
- Batch size too small

**Solutions:**
- Ensure >10000 particles in validation set
- Increase batch size: `--batch_size 64`

## See Also

- [Configuration Guide](configuration_guide.md) - Detailed parameter reference
- [API Reference](https://rsanchezgarc.github.io/cryoPARES/api/) - Type hints and function signatures
- [Troubleshooting](troubleshooting.md) - Common issues and solutions
