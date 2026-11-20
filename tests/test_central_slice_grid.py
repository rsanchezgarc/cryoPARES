"""Contracts for a central XY Fourier plane in array (ZYX) coordinates."""

import itertools
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from cryoPARES.utils.fourier_grid import central_slice_fftfreq_grid as central_slice_grid


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]
SHAPES = [(1, 1, 1), (1, 1, 6), (3, 7, 1), (4, 6, 8), (5, 7, 9),
          (6, 5, 8), (3, 4, 7), (9, 1, 5), (2, 3, 4)]


@pytest.fixture(params=[torch.float32, torch.float64])
def dtype(request):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(request.param)
    try:
        yield request.param
    finally:
        torch.set_default_dtype(previous)


def _numpy_plane(shape, rfft, shifted):
    y = np.fft.fftfreq(shape[-2])
    x = (np.fft.rfftfreq if rfft else np.fft.fftfreq)(shape[-1])
    if shifted:
        y = np.fft.fftshift(y)
        if not rfft:
            x = np.fft.fftshift(x)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    return np.stack([np.zeros_like(yy), yy, xx], axis=-1)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("rfft,shifted", itertools.product([False, True], repeat=2))
def test_plane_against_numpy(shape, rfft, shifted, dtype, device):
    actual = central_slice_grid(shape, rfft=rfft, fftshift=shifted, device=device)
    expected = _numpy_plane(shape, rfft, shifted)
    assert actual.shape == expected.shape
    assert actual.device.type == device
    assert actual.dtype == dtype
    assert torch.count_nonzero(actual[..., 0]) == 0
    np.testing.assert_allclose(actual.cpu().numpy(), expected,
                               atol=1e-7 if dtype == torch.float32 else 1e-15, rtol=1e-7)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("rfft,shifted", itertools.product([False, True], repeat=2))
def test_plane_matches_zero_z_section_of_full_volume_grid(shape, rfft, shifted):
    from cryoPARES.utils.fourier_grid import fftfreq_grid

    volume_grid = fftfreq_grid(shape, rfft=rfft, fftshift=shifted)
    z_dc = shape[0] // 2 if shifted else 0
    actual = central_slice_grid(shape, rfft=rfft, fftshift=shifted)
    torch.testing.assert_close(actual, volume_grid[z_dc])


@pytest.mark.parametrize("rfft,shifted", itertools.product([False, True], repeat=2))
def test_volume_depth_does_not_change_in_plane_frequencies(rfft, shifted):
    expected = central_slice_grid((1, 5, 8), rfft=rfft, fftshift=shifted)
    for depth in [2, 7, 16, 257]:
        torch.testing.assert_close(central_slice_grid((depth, 5, 8), rfft=rfft, fftshift=shifted), expected)


@pytest.mark.parametrize("shape", [(3, 6, 8), (4, 5, 7)])
@pytest.mark.parametrize("rfft,shifted", itertools.product([False, True], repeat=2))
def test_dc_axis_order_and_nyquist(shape, rfft, shifted):
    _, height, width = shape
    grid = central_slice_grid(shape, rfft=rfft, fftshift=shifted)
    row = height // 2 if shifted else 0
    column = width // 2 if shifted and not rfft else 0
    torch.testing.assert_close(grid[row, column], torch.zeros(3))
    # Moving one row/column from DC changes exactly the Y/X component.
    torch.testing.assert_close(grid[(row + 1) % height, column], torch.tensor([0., 1 / height, 0.]))
    torch.testing.assert_close(grid[row, (column + 1) % grid.shape[1]], torch.tensor([0., 0., 1 / width]))
    if rfft:
        assert grid[..., 2].min() == 0
        assert grid[..., 2].max() == pytest.approx((width // 2) / width)
    elif width % 2 == 0:
        assert grid[..., 2].min() == -0.5


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rfft", [False, True])
def test_default_device_is_respected(device, rfft):
    previous = torch.get_default_device()
    try:
        torch.set_default_device(device)
        assert central_slice_grid((3, 5, 8), rfft=rfft).device.type == device
    finally:
        torch.set_default_device(previous)


def test_each_grid_owns_its_storage_and_does_not_modify_shape():
    shape = [3, 5, 8]
    first = central_slice_grid(shape, rfft=True)
    expected = first.clone()
    first.fill_(123)
    torch.testing.assert_close(central_slice_grid(shape, rfft=True), expected)
    assert shape == [3, 5, 8]


@pytest.mark.parametrize("shape_type", [tuple, list, torch.Size])
def test_shape_containers(shape_type):
    actual = central_slice_grid(shape_type((3, 5, 8)), rfft=True)
    torch.testing.assert_close(actual, central_slice_grid((3, 5, 8), rfft=True))


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("rfft,shifted", itertools.product([False, True], repeat=2))
def test_optional_grid_matches_upstream(shape, rfft, shifted, dtype):
    upstream = pytest.importorskip("torch_fourier_slice._grids")
    actual = central_slice_grid(shape, rfft=rfft, fftshift=shifted)
    expected = upstream._central_slice_fftfreq_grid(shape, rfft=rfft, fftshift=shifted)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(), (3, 4), (2, 3, 4, 5)])
def test_rejects_non_volume_shape(shape):
    with pytest.raises(ValueError, match="volume_shape"):
        central_slice_grid(shape, rfft=True)


@pytest.mark.parametrize("rfft,shifted", itertools.product([False, True], repeat=2))
def test_compilation_capture(rfft, shifted):
    compiled = torch.compile(central_slice_grid, backend="eager", fullgraph=True)
    for shape in [(4, 6, 8), (3, 5, 7)]:
        torch.testing.assert_close(compiled(shape, rfft, shifted), central_slice_grid(shape, rfft, shifted))


@pytest.mark.parametrize("shape", [(4, 6, 8), (5, 7, 9), (1, 4, 5)])
@pytest.mark.parametrize("cutoff", [None, -0.1, 0., 0.25, 0.5, 0.75])
def test_projection_grid_precomputation_cutoffs_and_coordinate_order(shape, cutoff):
    from cryoPARES.projmatching.projmatchingUtils.extract_central_slices_as_real import (
        _get_freq_grid_valid_coords_and_freq_grid_mask,
    )

    expected = _numpy_plane(shape, True, True).astype(np.float32)
    mask = np.ones(expected.shape[:2], dtype=bool) if cutoff is None else np.linalg.norm(expected, axis=-1) <= cutoff
    rfft_shape, coordinates, indices = _get_freq_grid_valid_coords_and_freq_grid_mask(
        shape, torch.device("cpu"), cutoff,
    )
    assert rfft_shape == expected.shape[:2]
    assert coordinates.shape == (int(mask.sum()), 3, 1)
    np.testing.assert_allclose(coordinates[..., 0].numpy(), expected[mask], atol=1e-7)
    expected_indices = np.nonzero(mask)
    for actual, wanted in zip(indices, expected_indices):
        np.testing.assert_array_equal(actual.numpy(), wanted)


@pytest.mark.parametrize("shape", [(4, 6, 8), (5, 7, 9)])
@pytest.mark.parametrize("rfft,shifted", itertools.product([False, True], repeat=2))
@pytest.mark.parametrize("cutoff", [None, -0.1, 0., 0.25, 0.5])
def test_backprojection_grid_precomputation(shape, rfft, shifted, cutoff):
    from cryoPARES.reconstruction.insert_central_slices_rfft_3d import _initialize_insertion

    expected = _numpy_plane(shape, rfft, shifted).astype(np.float32)
    mask = np.ones(expected.shape[:2], dtype=bool) if cutoff is None else np.linalg.norm(expected, axis=-1) <= cutoff
    coordinates, indices = _initialize_insertion(
        shape, torch.device("cpu"), cutoff, torch.Size(expected.shape[:2]), rfft=rfft, fftshift=shifted,
    )
    np.testing.assert_allclose(coordinates[..., 0].numpy(), expected[mask], atol=1e-7)
    np.testing.assert_array_equal(indices.numpy(), np.flatnonzero(mask))


def test_legacy_alias_and_local_consumer_imports():
    from cryoPARES.utils.torch_fourier_slice_compat import _central_slice_fftfreq_grid
    from cryoPARES.projmatching.projmatchingUtils import extract_central_slices_as_real
    from cryoPARES.reconstruction import insert_central_slices_rfft_3d

    assert _central_slice_fftfreq_grid is central_slice_grid
    assert extract_central_slices_as_real._central_slice_fftfreq_grid is central_slice_grid
    assert insert_central_slices_rfft_3d._central_slice_fftfreq_grid is central_slice_grid


def test_local_projection_and_backprojection_run_without_teamtomo_packages():
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
from cryoPARES.utils.fourier_grid import central_slice_fftfreq_grid
from cryoPARES.projmatching.projmatchingUtils.extract_central_slices_as_real import (
    extract_central_slices_rfft_3d_multichannel,
)
from cryoPARES.reconstruction.insert_central_slices_rfft_3d import (
    insert_central_slices_rfft_3d_multichannel,
)
shape = (6, 6, 6)
assert central_slice_fftfreq_grid(shape, rfft=True, fftshift=True).shape == (6, 4, 3)
volume = torch.zeros(2, 6, 6, 4)
volume[0] = 1
rotations = torch.eye(3)[None]
projections = extract_central_slices_rfft_3d_multichannel(volume, shape, rotations)
assert projections.shape == (1, 2, 6, 4)
assert torch.isfinite(projections).all() and projections.abs().max() > 0
inserted, weights = insert_central_slices_rfft_3d_multichannel(projections, shape, rotations)
assert inserted.shape == volume.shape and weights.shape == (6, 6, 4)
assert torch.isfinite(inserted).all() and torch.isfinite(weights).all()
assert weights.max() > 0
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parents[1], timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
