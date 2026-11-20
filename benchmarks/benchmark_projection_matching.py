"""Time complete real-particle refinement, including preprocessing and STAR output.

Run each checkout sequentially on an idle GPU with identical input arguments.
The first run includes lazy initialization/compilation (disk compiler caches may
already exist). Subsequent runs include a fresh matcher, reference loading/FFT,
particle loading, CTF processing, search, subpixel refinement, and STAR writing.
This deliberately measures the application rather than only its projector.
Python interpreter startup and imports are outside the timed region. Each run
uses the default compiled channel-based projector, CTF correction, noise-PSD
whitening, and subpixel shift/SO(3) interpolation. GPU work is synchronized.

Example (use a new output directory for every invocation)::

    python benchmarks/benchmark_projection_matching.py --source /tmp/baseline \
        --reference /data/reference.mrc --particles /data/particles.star \
        --particles-dir /data --output-dir /tmp/pm-functional --batch-size 16
"""

import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--particles', required=True)
    parser.add_argument('--particles-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error('repeats and batch size must be positive; workers must be nonnegative')
    source = Path(args.source).resolve()
    sys.path.insert(0, str(source))
    import cryoPARES
    assert Path(cryoPARES.__file__).resolve().is_relative_to(source)
    import numpy as np
    import starfile
    import torch
    from cryoPARES.configs.mainConfig import main_config
    from cryoPARES.projmatching.projMatcher import ProjectionMatcher

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    main_config.cachedir = output / 'cache'
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('high')
    assert torch.cuda.is_available()
    columns = ['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi',
               'rlnOriginXAngst', 'rlnOriginYAngst']
    expected_rows = len(starfile.read(args.particles)['particles'])
    result = dict(source=str(source), gpu=torch.cuda.get_device_name(0),
                  torch_version=torch.__version__, particles=expected_rows,
                  reference=args.reference, input=args.particles,
                  batch_size=args.batch_size, workers=args.workers, grid_distance_degs=4,
                  grid_step_degs=1, resolution_angstrom=12,
                  compiled_projector=not main_config.projmatching.disable_compile_projectVol,
                  noise_psd_whitening=main_config.projmatching.noise_psd_whitening,
                  subpixel_shifts=main_config.projmatching.use_subpixel_shifts,
                  so3_interpolation=main_config.projmatching.use_so3_interpolation,
                  runs=[])
    reference_output = None
    for run in range(args.repeats + 1):
        gc.collect()
        torch.cuda.synchronize()
        started = time.perf_counter()
        matcher = ProjectionMatcher(reference_vol=args.reference,
                                    grid_distance_degs=4, grid_step_degs=1,
                                    max_resolution_A=12, top_k_poses_localref=1,
                                    correct_ctf=True, verbose=False)
        matcher.align_star(args.particles, str(output / f'run_{run}.star'),
                           data_rootdir=args.particles_dir, batch_size=args.batch_size,
                           device='cuda:0', n_cpus=args.workers)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        values = starfile.read(output / f'run_{run}.star')['particles'][columns].to_numpy()
        assert values.shape == (expected_rows, 5) and np.isfinite(values).all()
        if reference_output is None:
            reference_output = values
        else:
            np.testing.assert_array_equal(values, reference_output)
        result['runs'].append(dict(first_run=run == 0, seconds=seconds))
        print(f'run={run} seconds={seconds:.4f}', flush=True)
        del matcher
    result['median_warmed_seconds'] = statistics.median(
        row['seconds'] for row in result['runs'][1:])
    result['particles_per_second'] = expected_rows / result['median_warmed_seconds']
    np.save(output / 'poses_and_shifts.npy', reference_output)
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
