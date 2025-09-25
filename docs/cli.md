# Command-Line Interface

This document provides instructions for using the command-line tools included with cryoPARES.

## `cryopares_reconstruct`

This tool performs a 3D reconstruction from a set of 2D particle images and their corresponding orientation information.

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