"""Ensure the real-data regression gate detects changes without Euler false alarms."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    'pm_comparison', Path(__file__).resolve().parents[1] / 'benchmarks/compare_projection_matching.py')
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)
preparation_spec = importlib.util.spec_from_file_location(
    'pm_preparation', Path(__file__).resolve().parents[1] / 'benchmarks/prepare_projection_matching_data.py')
preparation = importlib.util.module_from_spec(preparation_spec)
preparation_spec.loader.exec_module(preparation)


@pytest.fixture
def particles():
    return pd.DataFrame(dict(rlnImageName=['000001@particles.mrcs', '000002@particles.mrcs'],
                             rlnAngleRot=[20., 60.], rlnAngleTilt=[45., 0.],
                             rlnAnglePsi=[30., 10.], rlnOriginXAngst=[1., 2.],
                             rlnOriginYAngst=[3., 4.], rlnParticleFigureOfMerit=[0.5, 0.9]))


def compare(a, b):
    return comparison.compare_particles(a, b, expected_rows=2)


def test_identical_results(particles):
    result = compare(particles, particles.copy())
    assert result['passed'] and result['bit_identical']


def test_equivalent_euler_angles(particles):
    candidate = particles.copy()
    candidate.loc[0, 'rlnAngleRot'] += 360
    # At zero tilt, only the sum of the outer Euler angles determines rotation.
    candidate.loc[1, ['rlnAngleRot', 'rlnAnglePsi']] = [10, 60]
    result = compare(particles, candidate)
    assert result['passed'] and not result['bit_identical']


@pytest.mark.parametrize('column,delta', [('rlnAngleTilt', .02), ('rlnOriginXAngst', .02),
                                         ('rlnParticleFigureOfMerit', .001)])
def test_real_regressions_fail(particles, column, delta):
    candidate = particles.copy()
    candidate.loc[0, column] += delta
    assert not compare(particles, candidate)['passed']


def test_shift_tolerance_is_vector_distance(particles):
    candidate = particles.copy()
    candidate.loc[0, ['rlnOriginXAngst', 'rlnOriginYAngst']] += .008
    assert not compare(particles, candidate)['passed']


def test_small_rounding_differences_pass(particles):
    candidate = particles.copy()
    candidate.loc[0, 'rlnAngleTilt'] += .001
    candidate.loc[0, 'rlnOriginXAngst'] += .001
    candidate.loc[0, 'rlnParticleFigureOfMerit'] += .00001
    assert compare(particles, candidate)['passed']


@pytest.mark.parametrize('invalid', ['missing', 'reordered', 'wrong_id', 'nan', 'infinite', 'bad_confidence'])
def test_invalid_comparisons_are_rejected(particles, invalid):
    candidate = particles.copy()
    if invalid == 'missing':
        candidate = candidate.iloc[:1]
    elif invalid == 'reordered':
        candidate = candidate.iloc[::-1]
    elif invalid == 'wrong_id':
        candidate.loc[0, 'rlnImageName'] = '000003@particles.mrcs'
    elif invalid == 'nan':
        candidate.loc[0, 'rlnAngleRot'] = np.nan
    elif invalid == 'infinite':
        candidate.loc[0, 'rlnOriginXAngst'] = np.inf
    else:
        candidate.loc[0, 'rlnParticleFigureOfMerit'] = 1.1
    with pytest.raises(ValueError):
        compare(particles, candidate)


@pytest.mark.parametrize('ndim', [2, 3])
def test_resampling_preserves_constants(ndim):
    image = np.full((100,) * ndim, 3.25, dtype=np.float32)
    result = preparation.resample_image(image, 1.27, box=84)
    assert result.shape == (84,) * ndim and result.dtype == np.float32
    np.testing.assert_allclose(result, 3.25, atol=2e-6)


@pytest.mark.parametrize('ndim', [2, 3])
def test_resampling_uses_physical_pixel_size_and_fourier_center(ndim):
    # A periodic physical signal has a known phase at each target coordinate.
    wavelength = 127 / 4
    coordinates = (np.arange(100) - 50) * 1.27
    signal = np.cos(2 * np.pi * coordinates / wavelength)
    image = np.broadcast_to(signal, (100,) * ndim).astype(np.float32)
    result = preparation.resample_image(image, 1.27, box=84)
    target_coordinates = (np.arange(84) - 42) * 1.5
    expected = np.cos(2 * np.pi * target_coordinates / wavelength)
    line = result[(42,) * (ndim - 1)]
    # Cubic regridding approximates a band-limited sinusoid; this bound allows
    # its interpolation error while resolving a pixel-scale/centering mistake.
    np.testing.assert_allclose(line[4:-4], expected[4:-4], atol=1e-4)
    assert np.argmax(line) == 42
