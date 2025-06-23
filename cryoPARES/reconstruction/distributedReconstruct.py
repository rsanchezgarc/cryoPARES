#!/usr/bin/env python
"""
Optimized distributed reconstruction with diagnostics and performance fixes
"""

import os

os.environ['DASK_DISTRIBUTED__WORKER__DAEMON'] = 'False'
# Reduce Dask memory limits to prevent memory issues
os.environ['DASK_DISTRIBUTED__WORKER__MEMORY__TARGET'] = '0.8'  # Start spilling at 80%
os.environ['DASK_DISTRIBUTED__WORKER__MEMORY__SPILL'] = '0.85'  # Spill to disk at 85%
os.environ['DASK_DISTRIBUTED__WORKER__MEMORY__PAUSE'] = '0.9'  # Pause at 90%

import sys
import time
import gc
from typing import List, Dict, Optional
import numpy as np
import torch
import mrcfile
import tqdm
from dask import delayed
from dask.distributed import Client, LocalCluster

from torch.utils.data import DataLoader, Dataset
from starstack import ParticlesStarSet
from starstack.constants import RELION_ANGLES_NAMES, RELION_SHIFTS_NAMES, RELION_EULER_CONVENTION
from torch_fourier_shift import fourier_shift_dft_2d
from torch_fourier_slice import insert_central_slices_rfft_3d, insert_central_slices_rfft_3d_multichannel
from torch_grid_utils import fftfreq_grid

from cryoPARES.datamanager.ctf.rfft_ctf import compute_ctf_rfft
from cryoPARES.geometry.convert_angles import euler_angles_to_matrix
from cryoPARES.utils.paths import FnameType


class Reconstructor:
    """Minimal reconstructor class"""

    def __init__(self, symmetry: str, device: str,
                 correct_ctf: bool = True, eps=1e-3, min_denominator_value=1e-4):
        self.symmetry = symmetry.upper()
        self.device = device
        self.correct_ctf = correct_ctf
        self.eps = eps
        self.min_denominator_value = min_denominator_value
        self.box_size = None
        self.sampling_rate = None

    def init_volumes(self, box_size, sampling_rate):
        self.box_size = box_size
        self.sampling_rate = sampling_rate
        self.particle_shape = (box_size, box_size)
        nky, nkx = self.box_size, self.box_size // 2 + 1
        self.f_num = torch.zeros((self.box_size, self.box_size, nkx), dtype=torch.complex64, device=self.device)
        self.weights = torch.zeros_like(self.f_num, dtype=torch.float32)
        self.ctfs = torch.zeros_like(self.f_num, dtype=torch.float32)

    def get_partial_volumes(self):
        return {
            'f_num': self.f_num.cpu(),
            'weights': self.weights.cpu(),
            'ctfs': self.ctfs.cpu(),
            'box_size': self.box_size,
            'sampling_rate': self.sampling_rate
        }

    def generate_volume(self, fname, overwrite_fname=True):
        dft = torch.zeros_like(self.f_num)
        mask = self.weights > self.min_denominator_value

        if self.correct_ctf:
            denominator = self.ctfs[mask] + self.eps * self.weights[mask]
            denominator[denominator.abs() < self.min_denominator_value] = self.min_denominator_value
            dft[mask] = self.f_num[mask] / denominator
        else:
            dft[mask] = self.f_num[mask] / self.weights[mask]

        dft = torch.fft.ifftshift(dft, dim=(-3, -2))
        dft = torch.fft.irfftn(dft, dim=(-3, -2, -1))
        dft = torch.fft.ifftshift(dft, dim=(-3, -2, -1))

        grid = fftfreq_grid(image_shape=dft.shape, rfft=False, fftshift=True, norm=True, device=dft.device)
        vol = dft / torch.sinc(grid) ** 2
        vol = vol.cpu()

        mrcfile.write(fname, data=vol.detach().numpy(), overwrite=overwrite_fname,
                      voxel_size=self.sampling_rate)


def reconstruct_chunk_optimized(chunk_params: Dict) -> Dict:
    """Optimized worker function with diagnostics"""

    chunk_id = chunk_params['chunk_id']
    star_file = chunk_params['star_file']
    particles_dir = chunk_params['particles_dir']
    indices = chunk_params['indices']
    device = chunk_params['device']
    batch_size = chunk_params['batch_size']
    use_dataloader_workers = chunk_params['use_dataloader_workers']

    print(f"\n[Worker {chunk_id}] Starting on {device}")
    print(f"[Worker {chunk_id}] Processing {len(indices)} particles")

    # Timing diagnostics
    t0 = time.time()

    # Load particles metadata once
    particles = ParticlesStarSet(starFname=star_file, particlesDir=particles_dir)
    box_size = particles.particle_shape[-1]
    sampling_rate = particles.sampling_rate

    # Pre-extract metadata for our indices to avoid repeated access
    print(f"[Worker {chunk_id}] Pre-loading metadata...")
    metadata_list = []
    for idx in indices:
        _, md = particles[idx]
        metadata_list.append(md)

    t1 = time.time()
    print(f"[Worker {chunk_id}] Metadata loaded in {t1 - t0:.1f}s")

    # Create reconstructor
    reconstructor = Reconstructor(
        symmetry=chunk_params['symmetry'],
        device=device,
        correct_ctf=chunk_params['correct_ctf'],
        eps=chunk_params['eps'],
        min_denominator_value=chunk_params['min_denominator_value']
    )
    reconstructor.init_volumes(box_size, sampling_rate)

    # Process in batches WITHOUT DataLoader to avoid overhead
    print(f"[Worker {chunk_id}] Processing particles...")

    # Process timing
    batch_times = []

    for batch_start in tqdm.tqdm(range(0, len(indices), batch_size),
                                 desc=f"Worker {chunk_id}",
                                 position=chunk_id):
        batch_t0 = time.time()

        # Get batch indices
        batch_end = min(batch_start + batch_size, len(indices))
        batch_indices = indices[batch_start:batch_end]

        # Load batch data
        batch_imgs = []
        batch_ctfs = []
        batch_rotMats = []
        batch_shifts = []

        for i, idx in enumerate(batch_indices):
            img, _ = particles[idx]
            md = metadata_list[batch_start + i]

            # Convert to tensors
            img = torch.FloatTensor(img)
            degEuler = torch.FloatTensor([md[name] for name in RELION_ANGLES_NAMES])
            rotMat = euler_angles_to_matrix(torch.deg2rad(degEuler), convention=RELION_EULER_CONVENTION)
            hwShiftAngs = torch.FloatTensor([md[name] for name in RELION_SHIFTS_NAMES[::-1]])

            batch_imgs.append(img)
            batch_rotMats.append(rotMat)
            batch_shifts.append(hwShiftAngs)

            if chunk_params['correct_ctf']:
                ctf = compute_ctf_rfft(
                    img.shape[-2], sampling_rate,
                    md["rlnDefocusU"], md["rlnDefocusV"], md["rlnDefocusAngle"],
                    float(particles.optics_md["rlnVoltage"][0]),
                    float(particles.optics_md["rlnSphericalAberration"][0]),
                    float(particles.optics_md["rlnAmplitudeContrast"][0]),
                    phase_shift=0, bfactor=None, fftshift=True, device="cpu"
                )
                batch_ctfs.append(ctf)

        # Stack batch
        imgs = torch.stack(batch_imgs).to(device, non_blocking=True)
        rotMats = torch.stack(batch_rotMats).to(device, non_blocking=True)
        hwShiftAngs = torch.stack(batch_shifts).to(device, non_blocking=True)
        if chunk_params['correct_ctf']:
            ctf = torch.stack(batch_ctfs).to(device, non_blocking=True)

        # Process batch
        imgs = torch.fft.fftshift(imgs, dim=(-2, -1))
        imgs = torch.fft.rfftn(imgs, dim=(-2, -1))
        imgs = torch.fft.fftshift(imgs, dim=(-2,))

        imgs = fourier_shift_dft_2d(
            dft=imgs,
            image_shape=(box_size, box_size),
            shifts=hwShiftAngs / sampling_rate,
            rfft=True,
            fftshifted=True,
        )

        if chunk_params['correct_ctf']:
            imgs *= ctf
            imgs = torch.stack([imgs.real, imgs.imag, ctf ** 2], dim=1)

            dft_ctf, weights = insert_central_slices_rfft_3d_multichannel(
                image_rfft=imgs,
                volume_shape=(box_size,) * 3,
                rotation_matrices=rotMats,
                fftfreq_max=None,
                zyx_matrices=False,
            )

            reconstructor.f_num += torch.view_as_complex(
                dft_ctf[:2, ...].permute(1, 2, 3, 0).contiguous()
            )
            reconstructor.ctfs += dft_ctf[-1, ...]
            reconstructor.weights += weights
        else:
            dft_3d, weights = insert_central_slices_rfft_3d(
                image_rfft=imgs,
                volume_shape=(box_size,) * 3,
                rotation_matrices=rotMats,
                fftfreq_max=None,
                zyx_matrices=False,
            )
            reconstructor.f_num += dft_3d
            reconstructor.weights += weights

        # Clean up GPU memory
        if device.startswith('cuda'):
            torch.cuda.empty_cache()

        batch_time = time.time() - batch_t0
        batch_times.append(batch_time)

    # Clean up
    del particles
    gc.collect()

    # Report timing
    total_time = time.time() - t0
    avg_batch_time = np.mean(batch_times)
    print(f"[Worker {chunk_id}] Complete in {total_time:.1f}s")
    print(f"[Worker {chunk_id}] Avg batch time: {avg_batch_time:.2f}s")

    return reconstructor.get_partial_volumes()


def test_distributed_reconstruction_optimized():
    """Optimized test function"""

    # Configuration
    config = {
        'particles_star_fname': "/home/sanchezg/cryo/data/preAlignedParticles/EMPIAR-10166/data/allparticles.star",
        'symmetry': "C1",
        'output_fname': "/tmp/distributed_test.mrc",
        'particles_dir': "/home/sanchezg/cryo/data/preAlignedParticles/EMPIAR-10166/data/",
        'n_workers': 4,
        'batch_size': 64,  # Smaller batch size might help
        'use_dataloader_workers': False,  # Disable DataLoader multiprocessing
        'correct_ctf': True,
        'eps': 1e-3,
        'min_denominator_value': 1e-4,
        'memory_limit': '6GB'  # Limit memory per worker
    }

    print("=" * 70)
    print("EMPIAR-10166 Distributed Reconstruction - Optimized")
    print("=" * 70)

    start_time = time.time()

    # Create cluster with memory limits
    cluster = LocalCluster(
        n_workers=config['n_workers'],
        threads_per_worker=1,
        processes=True,
        memory_limit=config['memory_limit'],
        dashboard_address=':8787'
    )
    client = Client(cluster)
    print(f"Dashboard: {client.dashboard_link}")

    try:
        # Count particles
        print("\nCounting particles...")
        particles = ParticlesStarSet(
            starFname=config['particles_star_fname'],
            particlesDir=config['particles_dir']
        )
        total_particles = len(particles)
        print(f"Total particles: {total_particles}")

        # Get metadata for comparison
        box_size = particles.particle_shape[-1]
        sampling_rate = particles.sampling_rate
        print(f"Box size: {box_size}")
        print(f"Sampling rate: {sampling_rate}")

        del particles
        gc.collect()

        # Distribute work
        particles_per_worker = total_particles // config['n_workers']

        # GPU assignment
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            print(f"\nGPUs: {n_gpus}")
            devices = [f"cuda:{i % n_gpus}" for i in range(config['n_workers'])]
        else:
            devices = ["cpu"] * config['n_workers']

        # Create tasks
        tasks = []
        for i in range(config['n_workers']):
            start_idx = i * particles_per_worker
            end_idx = start_idx + particles_per_worker if i < config['n_workers'] - 1 else total_particles

            chunk_params = {
                'chunk_id': i,
                'star_file': config['particles_star_fname'],
                'particles_dir': config['particles_dir'],
                'indices': list(range(start_idx, end_idx)),
                'device': devices[i],
                'batch_size': config['batch_size'],
                'use_dataloader_workers': config['use_dataloader_workers'],
                'symmetry': config['symmetry'],
                'correct_ctf': config['correct_ctf'],
                'eps': config['eps'],
                'min_denominator_value': config['min_denominator_value']
            }

            task = delayed(reconstruct_chunk_optimized)(chunk_params)
            tasks.append(task)

        # Execute
        print(f"\nProcessing {config['n_workers']} chunks...")
        print(f"Batch size: {config['batch_size']}")
        print(f"Memory limit per worker: {config['memory_limit']}")

        results = client.compute(tasks, sync=True)

        # Combine
        print("\nCombining results...")
        combined = results[0].copy()
        for r in results[1:]:
            combined['f_num'] += r['f_num']
            combined['weights'] += r['weights']
            combined['ctfs'] += r['ctfs']

        # Generate volume
        print("Generating final volume...")
        final_reconstructor = Reconstructor(
            symmetry=config['symmetry'],
            device='cpu',
            correct_ctf=config['correct_ctf'],
            eps=config['eps'],
            min_denominator_value=config['min_denominator_value']
        )

        final_reconstructor.box_size = combined['box_size']
        final_reconstructor.sampling_rate = combined['sampling_rate']
        final_reconstructor.particle_shape = (combined['box_size'], combined['box_size'])
        final_reconstructor.f_num = combined['f_num']
        final_reconstructor.weights = combined['weights']
        final_reconstructor.ctfs = combined['ctfs']

        final_reconstructor.generate_volume(config['output_fname'])

        # Report
        elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print("✓ Complete!")
        print(f"Total time: {elapsed:.1f}s ({elapsed / 60:.1f} minutes)")
        print(f"Particles/second: {total_particles / elapsed:.1f}")

        # Compare with expected single-threaded performance
        expected_single_thread_time = total_particles * 0.1  # ~0.1s per particle is typical
        speedup = expected_single_thread_time / elapsed
        print(f"Estimated speedup: {speedup:.1f}x")

        # Check output
        with mrcfile.open(config['output_fname']) as mrc:
            print(f"\nOutput: {config['output_fname']}")
            print(f"Shape: {mrc.data.shape}")
            print(f"Mean: {mrc.data.mean():.6f}, Std: {mrc.data.std():.6f}")

    finally:
        client.close()
        cluster.close()


def benchmark_single_worker():
    """Benchmark single worker performance for comparison"""
    print("\nRunning single worker benchmark...")

    # Process just 1000 particles
    params = {
        'chunk_id': 0,
        'star_file': "/home/sanchezg/cryo/data/preAlignedParticles/EMPIAR-10166/data/allparticles.star",
        'particles_dir': "/home/sanchezg/cryo/data/preAlignedParticles/EMPIAR-10166/data/",
        'indices': list(range(1000)),
        'device': 'cpu', #'cuda:0' if torch.cuda.is_available() else 'cpu',
        'batch_size': 64,
        'use_dataloader_workers': False,
        'symmetry': 'C1',
        'correct_ctf': True,
        'eps': 1e-3,
        'min_denominator_value': 1e-4
    }

    start = time.time()
    result = reconstruct_chunk_optimized(params)
    elapsed = time.time() - start

    print(f"Single worker: 1000 particles in {elapsed:.1f}s")
    print(f"Rate: {1000 / elapsed:.1f} particles/second")

    return 1000 / elapsed


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method('spawn')

    # Optional: Run benchmark first
    # rate = benchmark_single_worker()
    # print(f"\nExpected distributed rate: ~{rate * 4:.1f} particles/second with 4 workers\n")

    # Run distributed test
    test_distributed_reconstruction_optimized()