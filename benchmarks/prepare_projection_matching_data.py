"""Prepare a seeded, balanced 1,000-particle EMPIAR-10166 regression workload.

Original files are read only. Fourier downsampling preserves the field of view;
a small centered cubic regridding sets exactly 1.5 Angstrom/pixel in a 284 box
(426 Angstrom versus the original 426.72 Angstrom). Both images and reference
receive the same transformation. Preparation is outside benchmark timings.
"""

import argparse
import json
from pathlib import Path

import mrcfile
import numpy as np
from scipy.ndimage import affine_transform
from scipy.signal import resample
from scipy.spatial.transform import Rotation
import starfile


def resample_image(image, original_pixel, box=284, target_pixel=1.5):
    old_box = image.shape[-1]
    for axis in range(image.ndim):
        image = resample(image, box, axis=axis)
    fourier_pixel = original_pixel * old_box / box
    scale = target_pixel / fourier_pixel
    return affine_transform(image, np.eye(image.ndim) * scale,
                            offset=np.full(image.ndim, box / 2 * (1 - scale)),
                            order=3, mode='wrap').astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    source, output = Path(args.source_dir), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    data = starfile.read(source / 'allparticles.star')
    generator = np.random.default_rng(20261120)
    rows = np.concatenate([generator.choice(np.flatnonzero(
        data['particles']['rlnRandomSubset'].to_numpy() == half), 500, replace=False)
        for half in [1, 2]])
    generator.shuffle(rows)
    particles = data['particles'].iloc[rows].copy().reset_index(drop=True)
    original_names = particles['rlnImageName'].tolist()
    assert len(set(original_names)) == 1000
    original_pixel = float(data['optics']['rlnImagePixelSize'].iloc[0])
    with mrcfile.mmap(source / 'frealign_stack.mrcs', mode='r') as src:
        with mrcfile.new_mmap(output / 'particles.mrcs', shape=(1000, 284, 284), mrc_mode=2) as dst:
            for index, name in enumerate(original_names):
                row, stack = name.split('@')
                assert stack == 'frealign_stack.mrcs'
                dst.data[index] = resample_image(src.data[int(row) - 1], original_pixel)
                if index % 100 == 0:
                    print(f'resampled {index}/1000', flush=True)
            dst.voxel_size = 1.5
            dst.update_header_stats()
    with mrcfile.open(source / 'allparticles_reconstruct.mrc') as src:
        reference = resample_image(src.data.copy(), float(src.voxel_size.x))
    with mrcfile.new(output / 'reference.mrc') as dst:
        dst.set_data(reference)
        dst.voxel_size = 1.5
    particles['rlnImageName'] = [f'{i + 1:06d}@particles.mrcs' for i in range(1000)]
    optics = data['optics'].copy()
    optics['rlnImagePixelSize'] = 1.5
    optics['rlnImageSize'] = 284
    starfile.write(dict(optics=optics, particles=particles), output / 'original.star')
    angles = ['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']
    shifts = ['rlnOriginXAngst', 'rlnOriginYAngst']
    axes = generator.normal(size=(1000, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    perturbed = Rotation.from_rotvec(axes * np.deg2rad(3)) * Rotation.from_euler(
        'ZYZ', particles[angles].to_numpy(), degrees=True)
    particles[angles] = perturbed.as_euler('ZYZ', degrees=True)
    directions = generator.uniform(0, 2 * np.pi, size=1000)
    particles[shifts] += 1.5 * np.stack([np.cos(directions), np.sin(directions)], axis=1)
    starfile.write(dict(optics=optics, particles=particles), output / 'perturbed.star')
    manifest = dict(seed=20261120, particles=1000, unique_particles=1000,
                    halfset_counts={str(k): int(v) for k, v in particles['rlnRandomSubset'].value_counts().items()},
                    pixel_size_angstrom=1.5, box_size=284, source_dir=str(source),
                    source_rows=rows.tolist(), original_image_names=original_names,
                    initial_angular_perturbation_degrees=3, initial_shift_perturbation_angstrom=1.5,
                    resampling=resample_image.__doc__ or __doc__)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Prepared', output, flush=True)


if __name__ == '__main__':
    main()
