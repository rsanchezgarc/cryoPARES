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

## Training Parameters

### Essential Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--symmetry` | str | Required | Point group symmetry (C1, D7, T, O, I, etc.) |
| `--particles_star_fname` | str | Required | Path to pre-aligned RELION .star file |
| `--train_save_dir` | str | Required | Output directory for checkpoints and logs |
| `--n_epochs` | int | 10 | Number of training epochs |
| `--batch_size` | int | 32 | Number of particles per batch |

### Data Configuration Parameters

Access via `--config` flag with dot notation:

```bash
--config datamanager.particlesDataset.sampling_rate_angs_for_nnet=2.0 \
         datamanager.particlesDataset.image_size_px_for_nnet=128
```

**Key data parameters:**

- **`datamanager.particlesDataset.sampling_rate_angs_for_nnet`** (float, default: 1.5)
  - Target sampling rate in Ångströms/pixel
  - Images are rescaled to this value before training
  - Lower values = higher resolution but more memory/compute

- **`datamanager.particlesDataset.image_size_px_for_nnet`** (int, default: 160)
  - Final image size in pixels after rescaling
  - Images are cropped/padded to this size
  - Must be large enough to contain the particle after rescaling

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

- **`train.learning_rate`** (float, default: 1e-3)
  - Initial learning rate for the Adam optimizer
  - Automatically reduced on plateau (see below)

- **`train.weight_decay`** (float, default: 1e-5)
  - L2 regularization coefficient
  - Helps prevent overfitting

- **`train.accumulate_grad_batches`** (int, default: 16)
  - Number of batches to accumulate gradients over
  - Simulates larger batch sizes: effective_batch_size = batch_size × accumulate_grad_batches
  - Useful when GPU memory is limited

- **`train.warmup_n_epochs`** (int, default: 2)
  - Number of epochs for learning rate warmup
  - LR linearly increases from 0 to target value

- **`train.patient_reduce_lr_plateau_n_epochs`** (int, default: 3)
  - Patience for ReduceLROnPlateau scheduler
  - LR reduced by 0.5 if val_loss doesn't improve for this many epochs

### Model Architecture Parameters

```bash
--config models.image2sphere.lmax=8
```

**Key architecture parameters:**

- **`models.image2sphere.lmax`** (int, default: 12)
  - Maximum spherical harmonic degree
  - Higher values = more expressive model but slower training
  - Typical range: 8,10,12


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
- **Goal:** As low as possible (typically < 5° for good models)
- `val_geo_degs` is the key metric for final model quality

#### 4. Median Angular Error (`val_median_geo_degs`)

- More robust than mean to outliers
- Should also decrease during training

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
   --config train.weight_decay=1e-4  # Increase from default 1e-5
   ```

2. **Reduce model complexity:**
   ```bash
   --config models.image2sphere.lmax=10 # Decrease from default 12
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
Epoch 1:  loss=0.150, val_loss=0.160, val_geo_degs=8.5
Epoch 5:  loss=0.080, val_loss=0.085, val_geo_degs=4.2
Epoch 10: loss=0.055, val_loss=0.058, val_geo_degs=2.8
Epoch 15: loss=0.048, val_loss=0.050, val_geo_degs=2.5
Epoch 20: loss=0.045, val_loss=0.048, val_geo_degs=2.4
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

### Simulated Pre-training (Experimental)

Pre-train on simulated data before training on real particles:

```bash
python -m cryopares_train \
    --symmetry C1 \
    --particles_star_fname /path/to/real_particles.star \
    --train_save_dir /path/to/output \
    --map_fname_for_simulated_pretraining /path/to/reference_map.mrc \
    --n_epochs 10 \
    --config train.n_epochs_simulation=3
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

If the model can't overfit on 10 batches, something is wrong with the setup.

## Troubleshooting Training Issues

### Training is very slow

**Solutions:**
- Enable model compilation: `--compile_model`
- Reduce image size: `--config datamanager.particlesDataset.image_size_px_for_nnet=96`
- Increase batch size: `--batch_size 64`
- Use multiple GPUs (automatically detected)
- Reduce `lmax`: `--config models.image2sphere.lmax=6`

### Out of memory errors

**Solutions:**
- Reduce batch size: `--batch_size 16`
- Reduce image size: `--config datamanager.particlesDataset.image_size_px_for_nnet=96`
- Increase gradient accumulation: `--config train.accumulate_grad_batches=32`
- Use mixed precision (automatic in PyTorch Lightning)

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
3. Ensure sufficient training data (>5000 particles recommended)
4. Verify symmetry is correct
5. Try increasing learning rate: `--config train.learning_rate=5e-3`
6. Increase model capacity: `--config models.image2sphere.lmax=10`

### Validation loss jumps around

**Causes:**
- Validation set too small
- Batch size too small

**Solutions:**
- Ensure >1000 particles in validation set
- Increase batch size: `--batch_size 64`

## See Also

- [Configuration Guide](configuration_guide.md) - Detailed parameter reference
- [API Reference](https://rsanchezgarc.github.io/cryoPARES/api/) - Type hints and function signatures
- [Troubleshooting](troubleshooting.md) - Common issues and solutions
