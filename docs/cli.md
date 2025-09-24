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
