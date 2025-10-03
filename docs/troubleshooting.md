# Troubleshooting Guide

This guide covers common issues and their solutions when using CryoPARES.

## Table of Contents

- [Installation Issues](#installation-issues)
- [File System Issues](#file-system-issues)
- [Memory Issues](#memory-issues)
- [Training Issues](#training-issues)
- [Inference Issues](#inference-issues)
- [Data Issues](#data-issues)
- [Performance Issues](#performance-issues)
- [CUDA/GPU Issues](#cudagpu-issues)
- [Output Quality Issues](#output-quality-issues)

---

## Installation Issues

### pip install fails with dependency conflicts

**Symptoms:**
```
ERROR: pip's dependency resolver does not currently take into account all the packages
```

**Solution:**

1. Create a fresh conda environment:
```bash
conda create -n cryopares_fresh python=3.12
conda activate cryopares_fresh
```

2. Install in order:
```bash
# Install PyTorch first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Then install cryoPARES
pip install git+https://github.com/rsanchezgarc/cryoPARES.git
```

### ImportError: No module named 'cryoPARES'

**Symptoms:**
```python
ImportError: No module named 'cryoPARES'
```

**Solutions:**

1. Verify installation:
```bash
pip list | grep cryoPARES
```

2. If using development install, check you're in the right directory:
```bash
cd /path/to/cryoPARES
pip install -e .
```

3. Check Python environment:
```bash
which python
# Should point to your conda environment
```

### CUDA version mismatch

**Symptoms:**
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**Solution:**

Reinstall PyTorch with correct CUDA version:

```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch (example for CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## File System Issues

### "Too many open files" error

**Symptoms:**
```
OSError: [Errno 24] Too many open files
```

**Root cause:** CryoPARES opens file handlers for each `.mrcs` file in the .star file.

**Solutions:**

1. **Immediate fix:** Increase file descriptor limit:
```bash
ulimit -n 65536
```

2. **Permanent fix:** Add to `.bashrc` or `.bash_profile`:
```bash
echo "ulimit -n 65536" >> ~/.bashrc
source ~/.bashrc
```

3. **System-wide fix** (requires root):
```bash
# Edit /etc/security/limits.conf
sudo nano /etc/security/limits.conf

# Add these lines:
* soft nofile 65536
* hard nofile 65536
```

4. **Verify limit:**
```bash
ulimit -n
# Should show 65536 or higher
```

### Permission denied when writing outputs

**Symptoms:**
```
PermissionError: [Errno 13] Permission denied: '/path/to/output'
```

**Solutions:**

1. Check directory permissions:
```bash
ls -ld /path/to/output
```

2. Create directory with correct permissions:
```bash
mkdir -p /path/to/output
chmod 755 /path/to/output
```

3. Use a different output directory:
```bash
--train_save_dir ~/cryopares_outputs/
```

### Particle files not found

**Symptoms:**
```
FileNotFoundError: Particle file not found: /path/to/particles/...
```

**Solutions:**

1. **Check `--particles_dir` argument:**
```bash
# If .star file has relative paths like:
# MotionCorr/job01/particles_001.mrcs

# Use:
--particles_dir /path/to/relion/project/
```

2. **Verify .star file paths:**
```bash
head -20 /path/to/particles.star
# Check the _rlnImageName column
```

3. **Make paths absolute:**
```python
import starfile
df = starfile.read('/path/to/particles.star')
df['rlnImageName'] = df['rlnImageName'].apply(
    lambda x: f'/absolute/path/{x}'
)
starfile.write(df, 'particles_absolute.star')
```

---

## Memory Issues

### Out of memory (OOM) during training

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions:**

1. **Reduce batch size:**
```bash
--batch_size 16
```

2. **Reduce image size:**
```bash
--config datamanager.particlesDataset.image_size_px_for_nnet=96
```

3. **Increase gradient accumulation:**
```bash
--config train.accumulate_grad_batches=32
# Maintains effective batch size while reducing memory
```

4. **Reduce model complexity:**
```bash
--config models.image2sphere.lmax=6
```

5. **Use smaller encoder:**
```bash
--config models.image2sphere.imageencoder.resnet.resnetName="resnet18"
```

6. **Enable gradient checkpointing** (if implemented):
```bash
--config models.use_gradient_checkpointing=True
```

7. **Clear cache periodically:**
```python
import torch
torch.cuda.empty_cache()
```

### Out of memory during inference

**Solutions:**

1. **Reduce inference batch size:**
```bash
--batch_size 512
```

2. **Use CPU for very large datasets:**
```bash
--config inference.use_cuda=False
```

3. **Process in chunks:**
```bash
# Process first 10,000 particles
--n_first_particles 10000
```

### RAM exhausted when loading data

**Symptoms:**
```
MemoryError: Unable to allocate array
```

**Solutions:**

1. **Disable in-memory caching:**
```bash
--config datamanager.particlesDataset.store_data_in_memory=False
```

2. **Reduce number of workers:**
```bash
--num_dataworkers 2
```

3. **Process smaller subsets:**
Use `--subset_idxs` or `--n_first_particles`

---

## Training Issues

### Loss becomes NaN

**Symptoms:**
```
loss: nan, geo_degs: nan
```

**Causes and solutions:**

1. **Learning rate too high:**
```bash
--config train.learning_rate=1e-4
```

2. **Numerical instability:**
```bash
--config train.weight_decay=1e-6  # Reduce regularization
```

3. **Bad data:**
Check for corrupted particles or extreme values in .star file

4. **Mixed precision issues:**
```bash
--config float32_matmul_precision="highest"
```

### Training doesn't improve

**Symptoms:**
- Loss plateaus immediately
- `val_geo_degs` > 10° after many epochs

**Diagnostic steps:**

1. **Verify data is pre-aligned:**
```bash
# Check that .star file contains orientation columns:
# _rlnAngleRot, _rlnAngleTilt, _rlnAnglePsi
grep "rlnAngle" /path/to/particles.star
```

2. **Test overfitting capability:**
```bash
python -m cryopares_train \
    --symmetry C1 \
    --particles_star_fname data.star \
    --train_save_dir /tmp/overfit_test \
    --n_epochs 100 \
    --overfit_batches 10
```
If model can't overfit 10 batches, there's a bug.

3. **Check symmetry:**
Wrong symmetry can make training impossible.

4. **Increase learning rate:**
```bash
--config train.learning_rate=5e-3
```

5. **Increase model capacity:**
```bash
--config models.image2sphere.lmax=10
```

### Validation loss higher than training loss

**Normal:** Small gap (< 20%) is expected and healthy.

**Concerning:** Large gap (> 50%) indicates overfitting.

**Solutions for overfitting:**

1. **Increase regularization:**
```bash
--config train.weight_decay=1e-4
```

2. **Reduce model complexity:**
```bash
--config models.image2sphere.lmax=6
```

3. **More training data:**
Use more particles in .star file

4. **Check for data leakage:**
Ensure train/val split is correct

### Training crashes with "Killed"

**Symptoms:**
Process killed with no error message.

**Cause:** System OOM (out of RAM, not GPU memory).

**Solutions:**

1. **Reduce workers:**
```bash
--num_dataworkers 2
```

2. **Reduce batch size:**
```bash
--batch_size 16
```

3. **Monitor memory:**
```bash
watch -n 1 free -h
```

### Checkpoints not saving

**Symptoms:**
No `.ckpt` files in `train_save_dir/version_0/half1/checkpoints/`

**Solutions:**

1. **Check disk space:**
```bash
df -h /path/to/train_save_dir
```

2. **Check write permissions:**
```bash
ls -ld /path/to/train_save_dir
```

3. **Verify training completes at least one epoch:**
Check logs for validation step completion

---

## Inference Issues

### Predicted poses are random

**Symptoms:**
- Angular error > 90°
- Reconstruction looks like noise

**Causes and solutions:**

1. **Wrong checkpoint directory:**
```bash
# Should point to version_0, not to half1 or half2
--checkpoint_dir /path/to/training/version_0
# NOT: /path/to/training/version_0/half1
```

2. **Model not trained:**
Check training logs to verify training completed

3. **Different molecule:**
Model trained on different protein than inference data

4. **Wrong symmetry:**
Verify symmetry matches training

### Reconstruction is blurry

**Causes and solutions:**

1. **Insufficient local refinement:**
```bash
--config projmatching.grid_distance_degs=15.0
```

2. **Too strict confidence filtering:**
```bash
--config inference.directional_zscore_thr=1.5
```

3. **Not enough particles passing filter:**
Check output .star file size

4. **Wrong reference map:**
Provide better reference:
```bash
--reference_map /path/to/good_reference.mrc
```

### "No particles passed confidence threshold"

**Symptoms:**
```
Warning: No particles passed directional_zscore_thr=2.0
```

**Solutions:**

1. **Lower threshold:**
```bash
--config inference.directional_zscore_thr=1.0
```

2. **Disable filtering:**
```bash
--config inference.directional_zscore_thr=None
```

3. **Check if model and data match:**
Verify you're using the correct trained model

### Inference slower than expected

**Solutions:**

1. **Increase batch size:**
```bash
--batch_size 2048
```

2. **Reduce top_k predictions:**
```bash
--config inference.top_k_poses_nnet=5
```

3. **Skip reconstruction if not needed:**
```bash
--config inference.skip_reconstruction=True
```

4. **Use GPU:**
```bash
--config inference.use_cuda=True
```

5. **Compile model:**
```bash
--compile_model
```

---

## Data Issues

### CTF parameters missing

**Symptoms:**
```
KeyError: '_rlnDefocusU'
```

**Solution:**

CryoPARES requires CTF parameters in .star file. Ensure your .star file was generated after CTF estimation (e.g., CTFFIND or Gctf in RELION).

Required CTF columns:
- `_rlnDefocusU`
- `_rlnDefocusV`
- `_rlnDefocusAngle`
- `_rlnVoltage`
- `_rlnSphericalAberration`
- `_rlnAmplitudeContrast`

### Orientation columns missing

**Symptoms:**
```
KeyError: '_rlnAngleRot'
```

**Solution:**

For **training**, you need pre-aligned particles with:
- `_rlnAngleRot`
- `_rlnAngleTilt`
- `_rlnAnglePsi`
- `_rlnOriginXAngst` (or `_rlnOriginX`)
- `_rlnOriginYAngst` (or `_rlnOriginY`)

For **inference**, orientations are optional (will be predicted).

### Particles are poorly centered

**Symptoms:**
- High angular errors
- Poor reconstruction quality

**Solution:**

Re-center particles in RELION:
1. Extract particles with larger box size
2. Run 2D or 3D classification
3. Re-extract centered particles

Or use CryoPARES projection matching from random initialization:
```bash
cryopares_projmatching \
    --reference_vol /path/to/initial_model.mrc \
    --particles_star_fname unaligned.star \
    --out_fname aligned.star \
    --grid_distance_degs 30  # Large range for initial alignment
```

### Different sampling rate in data vs. reference

**Symptoms:**
```
RuntimeError: Size mismatch between particles and reference
```

**Solution:**

CryoPARES handles rescaling automatically, but verify:

1. **Sampling rates are specified in .star file:**
Check for `_rlnImagePixelSize` or `_rlnDetectorPixelSize`

2. **Reference map has correct pixel size:**
```bash
# Check with mrcfile
python -c "import mrcfile; print(mrcfile.open('ref.mrc').voxel_size)"
```

---

## Performance Issues

### Training is very slow

**Diagnostic:**

Check if GPU is being used:
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.device_count())   # Should be > 0
```

**Solutions:**

1. **Enable model compilation:**
```bash
--compile_model
```

2. **Increase batch size:**
```bash
--batch_size 64
```

3. **Reduce image size:**
```bash
--config datamanager.particlesDataset.image_size_px_for_nnet=96
```

4. **Use multiple GPUs:**
CryoPARES automatically uses all available GPUs

5. **Reduce model complexity:**
```bash
--config models.image2sphere.lmax=6
```

6. **Check data loading is not bottleneck:**
```bash
--num_dataworkers 8
```

7. **Use faster precision:**
```bash
--config float32_matmul_precision="medium"
```

### Inference is very slow

**Solutions:**

1. **Increase batch size:**
```bash
--batch_size 4096
```

2. **Reduce angular search range:**
```bash
--config projmatching.grid_distance_degs=5.0
```

3. **Skip local refinement (if acceptable):**
```bash
--config inference.skip_localrefinement=True
```

4. **Use coarser search:**
```bash
--config projmatching.grid_step_degs=3.0
```

### Data loading is bottleneck

**Symptoms:**
- GPU utilization < 50%
- High CPU usage from data workers

**Solutions:**

1. **Increase workers:**
```bash
--num_dataworkers 8
```

2. **Enable in-memory caching (if enough RAM):**
```bash
--config datamanager.particlesDataset.store_data_in_memory=True
```

3. **Use faster storage:**
Move data to local SSD instead of network drive

4. **Reduce preprocessing:**
Ensure images don't require heavy rescaling

---

## CUDA/GPU Issues

### "CUDA out of memory" but GPU seems empty

**Cause:** Memory fragmentation.

**Solutions:**

1. **Restart training:**
```bash
# Clear GPU memory
nvidia-smi
# Kill any zombie processes
```

2. **Clear cache in code:**
```python
import torch
torch.cuda.empty_cache()
```

3. **Reduce batch size:**
Even if GPU shows free memory, fragmentation can prevent allocation

### Multiple GPUs not being used

**Diagnostic:**
```bash
nvidia-smi
# Only GPU 0 shows activity
```

**Solution:**

CryoPARES uses PyTorch Lightning's automatic multi-GPU training. To force specific GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m cryopares_train ...
```

### GPU slower than expected

**Solutions:**

1. **Check GPU is not throttling:**
```bash
nvidia-smi --query-gpu=temperature.gpu,clocks.current.graphics --format=csv
```

2. **Ensure not using integrated graphics:**
```bash
nvidia-smi --query-gpu=name --format=csv
```

3. **Update NVIDIA drivers:**
```bash
nvidia-smi
# Check driver version, update if old
```

---

## Output Quality Issues

### Reconstruction has artifacts

**Causes and solutions:**

1. **Overfitting to noise:**
   - Use `directional_zscore_thr` to filter low-confidence particles
   - Use matching half-sets (default behavior)

2. **CTF correction issues:**
   - Verify CTF parameters are correct
   - Try different CTF correction: `--config datamanager.particlesDataset.ctf_correction="phase_flip"`

3. **Mask too tight:**
```bash
--config datamanager.particlesDataset.mask_radius_angs=150  # Increase
```

4. **Insufficient particles:**
Check how many particles passed filtering

### FSC is lower than expected

**Solutions:**

1. **Use independent half-sets:**
```bash
# For half1:
--data_halfset half1 --model_halfset half1

# For half2:
--data_halfset half2 --model_halfset half2
```

2. **More thorough local refinement:**
```bash
--config projmatching.grid_distance_degs=10.0 \
        projmatching.grid_step_degs=1.0
```

3. **Filter particles more aggressively:**
```bash
--config inference.directional_zscore_thr=2.5
```

### Reconstructed map has wrong hand

**Solution:**

The model learns the hand from training data. If training data had wrong hand, flip reference map:

```python
import mrcfile
vol = mrcfile.open('volume.mrc').data
vol_flipped = vol[:, :, ::-1]  # Flip along z
mrcfile.write('volume_flipped.mrc', vol_flipped)
```

---

## Getting More Help

### Enable Debug Mode

For more verbose output:

```bash
python -m cryopares_train \
    --show_debug_stats \
    ... other args ...
```

### Check Logs

Training logs are saved to:
```
train_save_dir/version_0/logs/
```

Inference logs:
```
results_dir/inference.log
```

### Report Issues

If you encounter a bug:

1. **Check existing issues:** https://github.com/rsanchezgarc/cryoPARES/issues

2. **Provide:**
   - Full command used
   - Error message and stack trace
   - CryoPARES version: `pip show cryopares`
   - PyTorch version: `python -c "import torch; print(torch.__version__)"`
   - CUDA version: `nvidia-smi`

3. **Minimal reproducible example:**
Create smallest possible example that shows the bug

---

## See Also

- [Training Guide](training_guide.md) - Training best practices
- [Configuration Guide](configuration_guide.md) - All configuration parameters
- [API Reference](https://rsanchezgarc.github.io/cryoPARES/api/) - Detailed API documentation
