"""Characterization and independent checks for circular projection masks."""

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from cryoPARES.utils.torch_grid_utils_compat import circle


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]
SHAPES = [(1, 1), (1, 7), (4, 6), (5, 7)]


def _reference_circle(radius, shape, center=None, smoothing=0):
    """Brute-force nearest foreground pixel, without SciPy or TeamTomo.

    Upstream smoothing is based on the rasterized disk, not its ideal perimeter.
    Its empty-disk case inherits an implicit SciPy site at (-1, 0); keep that
    historical behavior explicit rather than hiding it in another EDT call.
    """
    if isinstance(shape, int):
        shape = (shape, shape)
    if center is None:
        center = tuple(n // 2 for n in shape)
    coordinates = np.moveaxis(np.indices(shape, dtype=np.float32), 0, -1)
    radial = np.sqrt(np.sum((coordinates - np.asarray(center, dtype=np.float32)) ** 2, axis=-1))
    hard = radial < radius
    if smoothing == 0:
        return hard.astype(np.float32)
    sites = np.argwhere(hard)
    if not len(sites):
        sites = np.array([[-1, 0]])
    result = hard.astype(np.float32)
    for y, x in np.ndindex(shape):
        distance = np.float32(min(math.hypot(y - sy, x - sx) for sy, sx in sites))
        if 0 < distance <= np.float32(smoothing):
            result[y, x] = math.cos(math.pi / 2 * float(distance / np.float32(smoothing)))
    return result


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("center", [None, (0., 0.), (1.25, 2.5)])
@pytest.mark.parametrize("radius", [-1., 0., 0.25, 1., 2.5, 100.])
@pytest.mark.parametrize("smoothing", [0., 0.5, 1., 2.75, 4.])
def test_circle_matches_independent_reference(shape, center, radius, smoothing, device):
    result = circle(radius, shape, center=center, smoothing_radius=smoothing, device=device)
    assert result.shape == shape
    assert result.dtype == torch.float32
    assert result.device.type == device
    np.testing.assert_allclose(result.cpu().numpy(), _reference_circle(radius, shape, center, smoothing),
                               atol=2e-7, rtol=1e-6)


@pytest.mark.parametrize("shape", [(1, 7), (4, 6), (5, 7)])
@pytest.mark.parametrize("center", [None, (1.25, -0.5)])
@pytest.mark.parametrize("radius", [0., 0.25, 2.5, 100.])
@pytest.mark.parametrize("smoothing", [0., 0.5, 3.])
def test_optional_circle_matches_upstream(shape, center, radius, smoothing):
    upstream = pytest.importorskip("torch_grid_utils")
    actual = circle(radius, shape, center=center, smoothing_radius=smoothing)
    expected = upstream.circle(radius, shape, center=center, smoothing_radius=smoothing)
    torch.testing.assert_close(actual, expected)


def test_hard_circle_excludes_exact_boundary():
    mask = circle(5, (11, 11))
    # Integer Pythagorean triple exercises diagonal as well as axial boundaries.
    assert mask[5, 10] == mask[8, 9] == mask[9, 8] == 0
    assert mask[5, 9] == mask[8, 8] == mask[5, 5] == 1


@pytest.mark.parametrize("shape", [(4, 6), (5, 7), (6, 5)])
def test_default_center_is_fft_center(shape):
    mask = circle(0.1, shape)
    assert mask.sum() == 1
    assert mask[shape[0] // 2, shape[1] // 2] == 1


@pytest.mark.parametrize("smoothing", [0., 0.25, 2., 3.75])
def test_integer_shape_is_square_shorthand(smoothing):
    torch.testing.assert_close(circle(2.5, 7, smoothing_radius=smoothing),
                               circle(2.5, (7, 7), smoothing_radius=smoothing))


def test_soft_edge_uses_rasterized_foreground_distance():
    mask = circle(1, (7, 7), smoothing_radius=2)
    assert mask[3, 3] == 1
    assert mask[3, 4] == pytest.approx(math.cos(math.pi / 4), abs=1e-7)
    assert mask[4, 4] == pytest.approx(math.cos(math.pi * math.sqrt(2) / 4), abs=1e-7)
    assert mask[3, 5] == pytest.approx(0, abs=1e-7)
    assert mask[3, 6] == 0


def test_smoothed_empty_mask_preserves_legacy_distance_transform_behavior():
    # Compatibility fixture: upstream produces a nonzero corner halo even when
    # the hard disk is empty. A future behavior fix must be an explicit change.
    mask = circle(0, (3, 5), smoothing_radius=2)
    assert mask[0, 0] == pytest.approx(math.sqrt(0.5), abs=1e-7)
    assert mask[0, 1] == pytest.approx(math.cos(math.pi * math.sqrt(2) / 4), abs=1e-7)
    assert mask[2, 4] == 0


@pytest.mark.parametrize("smoothing", [0., 1., 100.])
def test_full_mask_stays_full(smoothing):
    torch.testing.assert_close(circle(100, (5, 7), smoothing_radius=smoothing), torch.ones(5, 7))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("smoothing", [0., 2.75])
def test_mask_is_float32_regardless_of_default_dtype(dtype, smoothing):
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        result = circle(2.5, (5, 7), smoothing_radius=smoothing)
        assert result.dtype == torch.float32
        np.testing.assert_allclose(result.numpy(), _reference_circle(2.5, (5, 7), smoothing=smoothing),
                                   atol=2e-7, rtol=1e-6)
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("smoothing", [0., 2.75])
def test_default_device_is_respected(device, smoothing):
    previous = torch.get_default_device()
    try:
        torch.set_default_device(device)
        assert circle(2, (5, 7), smoothing_radius=smoothing).device.type == device
    finally:
        torch.set_default_device(previous)


@pytest.mark.parametrize("center", [(-10., -10.), (12., 1.), (2.5, -1.)])
def test_center_can_be_outside_image(center):
    actual = circle(3.5, (5, 7), center=center, smoothing_radius=3)
    np.testing.assert_allclose(actual.numpy(), _reference_circle(3.5, (5, 7), center, 3),
                               atol=2e-7, rtol=1e-6)


def test_inputs_and_previous_outputs_are_not_modified():
    shape, center = [5, 7], [1.25, 2.5]
    first = circle(2, shape, center=center, smoothing_radius=3)
    expected = first.clone()
    first.fill_(100)
    torch.testing.assert_close(circle(2, shape, center=center, smoothing_radius=3), expected)
    assert shape == [5, 7]
    assert center == [1.25, 2.5]


@pytest.mark.parametrize("radius", [1., 2.5, 4.])
@pytest.mark.parametrize("smoothing", [0., 1., 3.])
def test_centered_odd_mask_is_symmetric(radius, smoothing):
    mask = circle(radius, (11, 11), smoothing_radius=smoothing)
    torch.testing.assert_close(mask, mask.flip(0))
    torch.testing.assert_close(mask, mask.flip(1))
    torch.testing.assert_close(mask, mask.T)


@pytest.mark.parametrize("smoothing", [0., 0.5, 2.75])
def test_larger_radius_does_not_decrease_nonempty_mask(smoothing):
    masks = [circle(r, (9, 11), smoothing_radius=smoothing) for r in [0.25, 1.5, 3., 5.]]
    for smaller, larger in zip(masks, masks[1:]):
        assert torch.all(larger >= smaller - 1e-7)


@pytest.mark.parametrize("shape", [(8, 8), (32, 32), (64, 64)])
@pytest.mark.parametrize("radius", [1., 7.25, 16.])
def test_mask_has_finite_bounded_values_at_production_sizes(shape, radius):
    mask = circle(radius, shape, smoothing_radius=3)
    assert torch.isfinite(mask).all()
    assert mask.min() >= -1e-7  # float32 cosine at pi/2 rounds slightly below zero
    assert mask.max() <= 1


@pytest.mark.parametrize("size", [8, 12])
@pytest.mark.parametrize("rfft", [False, True])
@pytest.mark.parametrize("inner_radius", [None, 0., 2.5])
@pytest.mark.parametrize("outer_radius", [None, 3.5])
def test_projection_matching_band_mask(size, rfft, inner_radius, outer_radius):
    from cryoPARES.projmatching.projmatchingUtils.fourierOperations import _mask_for_dft_2d

    expected = _reference_circle(size // 2 if outer_radius is None else outer_radius,
                                 (size, size), smoothing=3)
    if inner_radius:
        expected *= 1 - _reference_circle(inner_radius, (size, size), smoothing=2)
    if rfft:
        # Positive frequencies 0..Nyquist map into the centered full-FFT mask.
        columns = (np.arange(size // 2 + 1) + size // 2) % size
        expected = expected[:, columns]
    actual = _mask_for_dft_2d((size, size), outer_radius, inner_radius, rfft, True, torch.device("cpu"))
    assert actual.shape == (1, *expected.shape)
    np.testing.assert_allclose(actual[0].numpy(), expected, atol=3e-7, rtol=1e-6)


@pytest.mark.parametrize("seed", range(10))
def test_optional_random_fractional_masks_match_upstream(seed):
    upstream = pytest.importorskip("torch_grid_utils")
    rng = np.random.default_rng(seed)
    shape = tuple(int(x) for x in rng.integers(2, 40, size=2))
    center = tuple(float(x) for x in rng.uniform(-2, min(shape) + 2, size=2))
    radius, smoothing = (float(x) for x in rng.uniform(0, 15, size=2))
    torch.testing.assert_close(circle(radius, shape, center, smoothing),
                               upstream.circle(radius, shape, center, smoothing))


def test_compatibility_export_is_local():
    from cryoPARES.utils.masks import circle as local_circle

    assert circle is local_circle


def test_projection_preprocessing_runs_without_teamtomo_packages():
    script = """
import sys
from importlib.abc import MetaPathFinder
import torch

class BlockTeamTomo(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'torch_grid_utils', 'torch_image_interpolation', 'torch_fourier_slice',
            'torch_fourier_shift', 'torch_ctf', 'torch_fourier_shell_correlation',
        }:
            raise AssertionError('Unexpected upstream import: ' + fullname)

sys.meta_path.insert(0, BlockTeamTomo())
from cryoPARES.projmatching.projmatchingUtils.fourierOperations import (
    compute_dft_3d, _mask_for_dft_2d,
)
dft, shape, padding = compute_dft_3d(torch.ones(8, 8, 8), pad=False)
assert dft.shape == (8, 8, 5)
assert torch.isfinite(dft).all()
mask = _mask_for_dft_2d((8, 8), 3, 1, True, True, torch.device('cpu'))
assert mask.shape == (1, 8, 5)
assert torch.isfinite(mask).all()
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parents[1], timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
