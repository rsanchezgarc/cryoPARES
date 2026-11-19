"""Compare warmed Fourier operators against a source checkout or exported baseline.

Run the variants sequentially on the same idle GPU, for example::

    python benchmarks/benchmark_fourier_operators.py --source /tmp/baseline \
        --variant functional --output /tmp/comparison/functional.json
    python benchmarks/benchmark_fourier_operators.py --source . \
        --variant objects --output /tmp/comparison/objects.json

The fixed workload uses a 96-cubed volume, 500 projection rotations, eight
backprojection images, and 32 Fourier shifts. Five warmup calls exclude geometry
setup and compilation; each of five timing batches averages 20 calls. Times
include Python dispatch and synchronized GPU completion. Peak memory is the
additional live allocation for one warmed call, excluding persistent inputs and
operator buffers. NPZ outputs support numerical comparisons between variants.
"""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--source', required=True)
parser.add_argument('--variant', choices=['functional', 'objects'], required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
source = Path(args.source).resolve()
sys.path.insert(0, str(source))
import cryoPARES
assert Path(cryoPARES.__file__).resolve().is_relative_to(source), cryoPARES.__file__
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from cryoPARES.utils.fourier_shift import fourier_shift_dft_2d
from cryoPARES.utils.fourier_slice import extract_central_slices_rfft_3d
from cryoPARES.projmatching.projmatchingUtils.extract_central_slices_as_real import (
    extract_central_slices_rfft_3d_multichannel,
    compiled_extract_central_slices_rfft_3d_multichannel,
)
from cryoPARES.reconstruction.insert_central_slices_rfft_3d import (
    insert_central_slices_rfft_3d_multichannel,
    compiled_insert_central_slices_rfft_3d_multichannel,
)
if args.variant == 'objects':
    from cryoPARES.utils.fourier_operators import FourierProjector, FourierBackprojector, FourierShift2D

torch.set_num_threads(4)
torch.set_float32_matmul_precision('highest')
assert torch.cuda.is_available()
generator = torch.Generator().manual_seed(91)
shape, cutoff = (96, 96, 96), 0.35
volume = torch.randn(96, 96, 49, dtype=torch.complex64, generator=generator).cuda()
channels = torch.stack((volume.real, volume.imag))
rotations = torch.tensor(Rotation.random(500, random_state=91).as_matrix(), dtype=torch.float32,
                         device='cuda').reshape(4, 125, 3, 3)
slices = torch.randn(8, 3, 96, 49, generator=generator).cuda()
insertion_rotations = torch.tensor(Rotation.random(8, random_state=92).as_matrix(), dtype=torch.float32, device='cuda')
dft = torch.randn(32, 96, 49, dtype=torch.complex64, generator=generator).cuda()
shifts = torch.randn(32, 2, generator=generator).cuda()


def build(name):
    if name == 'shift':
        if args.variant == 'objects':
            op = FourierShift2D((96, 96), fftshifted=True, device='cuda')
            return lambda: op(dft, shifts)
        return lambda: fourier_shift_dft_2d(dft, (96, 96), shifts, True, True)
    if name == 'complex_projection':
        if args.variant == 'objects':
            op = FourierProjector(shape, cutoff, device='cuda')
            return lambda: op(volume, rotations)
        return lambda: extract_central_slices_rfft_3d(volume, rotations, image_shape=shape, fftfreq_max=cutoff)
    compiled = name.endswith('_compiled')
    if name.startswith('projection'):
        if args.variant == 'objects':
            op = FourierProjector(shape, cutoff, representation='channels', compiled=compiled, device='cuda')
            return lambda: op(channels, rotations)
        fn = compiled_extract_central_slices_rfft_3d_multichannel if compiled else extract_central_slices_rfft_3d_multichannel
        return lambda: fn(channels, shape, rotations, fftfreq_max=cutoff)
    if args.variant == 'objects':
        op = FourierBackprojector(shape, cutoff, compiled=compiled, device='cuda')
        return lambda: op(slices, insertion_rotations)
    fn = compiled_insert_central_slices_rfft_3d_multichannel if compiled else insert_central_slices_rfft_3d_multichannel
    return lambda: fn(slices, shape, insertion_rotations, fftfreq_max=cutoff)


result = {'variant': args.variant, 'source': str(source), 'gpu': torch.cuda.get_device_name(0), 'torch_version': torch.__version__,
          'shape': shape, 'cutoff': cutoff, 'projection_rotations': 500, 'cases': {}}
with torch.inference_mode():
    for name in ['shift', 'complex_projection', 'projection_eager', 'projection_compiled', 'backprojection_eager', 'backprojection_compiled']:
        torch._dynamo.reset()
        fn = build(name)
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        timings = []
        for _ in range(5):
            started = time.perf_counter()
            for _ in range(20):
                fn()
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - started) * 1000 / 20)
        torch.cuda.reset_peak_memory_stats()
        initial = torch.cuda.memory_allocated()
        output = fn()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - initial
        values = output if isinstance(output, tuple) else (output,)
        np.savez(Path(args.output).with_name(args.variant + '_' + name + '.npz'),
                 **{str(i): value.cpu().numpy() for i, value in enumerate(values)})
        result['cases'][name] = {'median_ms': statistics.median(timings), 'samples_ms': timings,
                                 'temporary_peak_bytes': peak}
        print(name, result['cases'][name], flush=True)
        del output, values, fn
Path(args.output).write_text(json.dumps(result, indent=2))
