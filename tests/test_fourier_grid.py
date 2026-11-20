"""Independent numerical contracts for the first local Fourier primitives.

The mathematical contracts were checked against the compatibility imports before
the migration. Core tests need no upstream package; optional comparisons below
also check the installed legacy implementations when available.
"""

import itertools
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from cryoPARES.utils.fourier_grid import array_to_grid_sample, fftfreq_grid


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]
SHAPES = [(1, 1), (1, 6), (7, 1), (4, 6), (5, 7), (4, 7),
          (1, 1, 1), (1, 4, 5), (3, 1, 6), (4, 5, 1),
          (4, 6, 8), (3, 5, 7), (4, 5, 6)]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("rfft,shifted,norm", itertools.product([False, True], repeat=3))
@pytest.mark.parametrize("anisotropic", [False, True])
def test_frequency_grid_against_numpy(shape, rfft, shifted, norm, anisotropic, device):
    spacing = tuple(0.75 + i * 0.5 for i in range(len(shape))) if anisotropic else 1.5
    per_axis = spacing if anisotropic else [spacing] * len(shape)
    axes = []
    for i, (size, step) in enumerate(zip(shape, per_axis)):
        transform = np.fft.rfftfreq if rfft and i == len(shape) - 1 else np.fft.fftfreq
        values = transform(size, d=step)
        if shifted and not (rfft and i == len(shape) - 1):
            values = np.fft.fftshift(values)
        axes.append(values)
    expected = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    if norm:
        expected = np.linalg.norm(expected, axis=-1)
    actual = fftfreq_grid(shape, rfft=rfft, fftshift=shifted, spacing=spacing,
                          norm=norm, device=device)
    assert actual.device.type == device
    assert actual.dtype == torch.get_default_dtype()
    np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize("shape", [(8, 10), (7, 9), (6, 7, 8), (5, 7, 9)])
@pytest.mark.parametrize("rfft", [False, True])
def test_shifted_grid_dc_and_nyquist(shape, rfft):
    grid = fftfreq_grid(shape, rfft=rfft, fftshift=True)
    center = tuple(n // 2 for n in shape[:-1]) + ((0,) if rfft else (shape[-1] // 2,))
    torch.testing.assert_close(grid[center], torch.zeros(len(shape)))
    if rfft:
        assert grid[..., -1].min() == 0
        assert grid[..., -1].max() == pytest.approx((shape[-1] // 2) / shape[-1])
    for axis, n in enumerate(shape):
        if n % 2 == 0 and not (rfft and axis == len(shape) - 1):
            assert grid[..., axis].min() == -0.5


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_frequency_grid_respects_default_dtype(dtype):
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        assert fftfreq_grid((5, 8), rfft=True).dtype == dtype
    finally:
        torch.set_default_dtype(previous)


def test_frequency_grid_returns_fresh_storage():
    first = fftfreq_grid((4, 6), rfft=True)
    expected = first.clone()
    first.fill_(123)
    torch.testing.assert_close(fftfreq_grid((4, 6), rfft=True), expected)


@pytest.mark.parametrize("shape", [(7,), (2, 3, 4, 5)])
def test_frequency_grid_rejects_unsupported_dimensions(shape):
    with pytest.raises(NotImplementedError):
        fftfreq_grid(shape, rfft=False)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(3, 5), (4, 6), (3, 4, 5), (4, 5, 6)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("leading", [(), (0,), (7,), (2, 3)])
def test_array_coordinates_against_scalar_formula(shape, dtype, leading, device):
    generator = torch.Generator().manual_seed(137)
    coords = torch.rand((*leading, len(shape)), generator=generator, dtype=dtype).to(device)
    coords = coords * coords.new_tensor(shape) * 2 - 1  # includes points outside the image
    before = coords.clone()
    result = array_to_grid_sample(coords, shape)
    expected = torch.stack([
        coords[..., axis] / ((shape[axis] - 1) / 2) - 1
        for axis in reversed(range(len(shape)))
    ], dim=-1)
    assert result.dtype == dtype
    assert result.device == coords.device
    assert result.shape == coords.shape
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(coords, before)


@pytest.mark.parametrize("shape", [(3, 6), (3, 4, 7)])
def test_coordinate_corners_and_center(shape):
    corners = torch.tensor(list(itertools.product(*[(0, n - 1) for n in shape])), dtype=torch.float64)
    expected = corners.sign().flip(-1) * 2 - 1
    torch.testing.assert_close(array_to_grid_sample(corners, shape), expected)
    center = torch.tensor([(n - 1) / 2 for n in shape], dtype=torch.float64)
    torch.testing.assert_close(array_to_grid_sample(center, shape), torch.zeros_like(center))


@pytest.mark.parametrize("shape", [(3, 6), (3, 4, 7)])
def test_coordinate_conversion_samples_every_voxel(shape):
    image = torch.arange(np.prod(shape), dtype=torch.float64).reshape(1, 1, *shape)
    coords = torch.stack(torch.meshgrid(*[
        torch.arange(n, dtype=torch.float64) for n in shape
    ], indexing="ij"), dim=-1)
    grid = array_to_grid_sample(coords, shape).unsqueeze(0)
    sampled = F.grid_sample(image, grid, align_corners=True, mode="bilinear")
    torch.testing.assert_close(sampled, image)


@pytest.mark.parametrize("shape", [(4, 7), (3, 5, 8)])
def test_coordinate_conversion_interpolates_linear_field(shape):
    axes = torch.meshgrid(*[torch.arange(n, dtype=torch.float64) for n in shape], indexing="ij")
    coefficients = torch.arange(1, len(shape) + 1, dtype=torch.float64)
    image = sum(c * a for c, a in zip(coefficients, axes))[None, None]
    point = torch.tensor([0.35 * (n - 1) for n in shape], dtype=torch.float64)
    grid = array_to_grid_sample(point, shape).reshape(1, *([1] * len(shape)), len(shape))
    sampled = F.grid_sample(image, grid, align_corners=True)
    torch.testing.assert_close(sampled.squeeze(), (point * coefficients).sum())


@pytest.mark.parametrize("shape", [(4, 7), (3, 5, 8)])
def test_coordinate_conversion_gradients(shape):
    coords = torch.randn(2, len(shape), dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: array_to_grid_sample(x, shape), (coords,))
    array_to_grid_sample(coords, shape).sum().backward()
    expected = torch.tensor([2 / (n - 1) for n in shape], dtype=coords.dtype).expand_as(coords)
    torch.testing.assert_close(coords.grad, expected)


def test_coordinate_conversion_noncontiguous_input():
    coords = torch.arange(24, dtype=torch.float64).reshape(3, 8).T
    assert not coords.is_contiguous()
    torch.testing.assert_close(array_to_grid_sample(coords, (4, 5, 6)),
                               array_to_grid_sample(coords.contiguous(), (4, 5, 6)))


def test_coordinate_conversion_does_not_clip_outside_points():
    coords = torch.tensor([[-1., 8.], [6., -2.]])
    expected = torch.tensor([[3., -1.5], [-2., 2.]])
    torch.testing.assert_close(array_to_grid_sample(coords, (5, 5)), expected)


def test_coordinate_conversion_preserves_nonfinite_values():
    coords = torch.tensor([[float("nan"), float("inf")], [-float("inf"), 0.]])
    result = array_to_grid_sample(coords, (3, 5))
    assert torch.isnan(result[0, 1])
    assert torch.isposinf(result[0, 0])
    assert torch.isneginf(result[1, 1])


def test_coordinate_conversion_singleton_axis_matches_existing_contract():
    # Existing upstream normalization divides by N-1. Preserve its undefined
    # singleton-axis results in this compatibility migration; do not silently
    # introduce different sampling semantics at the same time.
    result = array_to_grid_sample(torch.tensor([[0., 0.], [1., 2.]]), (1, 3))
    assert torch.isnan(result[0, 1])
    assert torch.isposinf(result[1, 1])


@pytest.mark.parametrize("shape", [(5, 8), (4, 5, 8)])
@pytest.mark.parametrize("spacing_kind", [tuple, list])
def test_frequency_grid_spacing_sequence_and_input_unchanged(shape, spacing_kind):
    spacing = spacing_kind([1.25] * len(shape))
    before = spacing.copy() if isinstance(spacing, list) else spacing
    torch.testing.assert_close(fftfreq_grid(shape, rfft=True, spacing=spacing),
                               fftfreq_grid(shape, rfft=True, spacing=1.25))
    assert spacing == before


@pytest.mark.parametrize("shape", [(4, 6), (3, 5, 7)])
@pytest.mark.parametrize("spacing", [(), (1,), (1, 2, 3, 4)])
def test_frequency_grid_rejects_spacing_length_mismatch(shape, spacing):
    with pytest.raises(ValueError, match="spacing"):
        fftfreq_grid(shape, rfft=True, spacing=spacing)


@pytest.mark.parametrize("shape", [(5, 8), (6, 7), (3, 5, 8), (4, 6, 7)])
@pytest.mark.parametrize("rfft,shifted", itertools.product([False, True], repeat=2))
def test_grid_identifies_fft_peak_of_known_plane_wave(shape, rfft, shifted):
    # A real cosine has peaks at +/-k. Axis-specific physical spacings let this
    # detect component swaps and wrong units as well as FFT layout mistakes.
    spacing = tuple(1.25 + i for i in range(len(shape)))
    axes = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    signal = np.cos(2 * np.pi * sum(x / n for x, n in zip(axes, shape)))
    spectrum = np.fft.rfftn(signal) if rfft else np.fft.fftn(signal)
    if shifted:
        spectrum = np.fft.fftshift(spectrum, axes=tuple(range(len(shape) - int(rfft))))
    peak = np.unravel_index(np.argmax(np.abs(spectrum)), spectrum.shape)
    observed = fftfreq_grid(shape, rfft=rfft, fftshift=shifted, spacing=spacing)[peak]
    expected = torch.tensor([1 / (n * d) for n, d in zip(shape, spacing)])
    if observed[-1] < 0:
        expected = -expected
    torch.testing.assert_close(observed, expected)


@pytest.mark.parametrize("shape", [(4, 7), (3, 5, 8)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sampling_coordinate_round_trip(shape, dtype):
    coords = torch.linspace(-2, 10, 12 * len(shape), dtype=dtype).reshape(3, 4, len(shape))
    normalized = array_to_grid_sample(coords, shape)
    recovered = (normalized.flip(-1) + 1) * (coords.new_tensor(shape) - 1) / 2
    torch.testing.assert_close(recovered, coords)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(4, 7), (3, 5, 8)])
def test_sampling_linear_field_gradients_through_grid_sample(shape, device):
    axes = torch.meshgrid(*[torch.arange(n, dtype=torch.float64, device=device) for n in shape], indexing="ij")
    coefficients = torch.arange(1, len(shape) + 1, dtype=torch.float64, device=device)
    image = sum(c * a for c, a in zip(coefficients, axes))[None, None]
    point = torch.tensor([0.35 * (n - 1) for n in shape], dtype=torch.float64,
                         device=device, requires_grad=True)
    grid = array_to_grid_sample(point, shape).reshape(1, *([1] * len(shape)), len(shape))
    F.grid_sample(image, grid, align_corners=True).sum().backward()
    torch.testing.assert_close(point.grad, coefficients)


@pytest.mark.parametrize("shape", [(3, 5), (3, 4, 7)])
def test_coordinate_conversion_second_derivatives(shape):
    coords = torch.randn(2, len(shape), dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradgradcheck(lambda x: array_to_grid_sample(x, shape), (coords,))


@pytest.mark.parametrize("shape", [(4, 7), (3, 5, 8)])
@pytest.mark.parametrize("padding", ["zeros", "border", "reflection"])
def test_outside_sampling_obeys_caller_padding_policy(shape, padding):
    image = torch.ones(1, 1, *shape, dtype=torch.float64)
    point = torch.tensor([-2.] * len(shape), dtype=torch.float64)
    grid = array_to_grid_sample(point, shape).reshape(1, *([1] * len(shape)), len(shape))
    actual = F.grid_sample(image, grid, padding_mode=padding, align_corners=True)
    torch.testing.assert_close(actual, torch.full_like(actual, 0 if padding == "zeros" else 1))


@pytest.mark.parametrize("shape", [(4, 7), (3, 5, 8)])
def test_coordinate_conversion_compiles_fullgraph(shape):
    # Eager backend checks Dynamo graph capture without requiring a compiler or
    # GPU. Downstream projection kernels compile this helper into their graph.
    compiled = torch.compile(array_to_grid_sample, backend="eager", fullgraph=True)
    for batch in [2, 5]:
        coords = torch.arange(batch * len(shape), dtype=torch.float32).reshape(batch, -1)
        torch.testing.assert_close(compiled(coords, shape), array_to_grid_sample(coords, shape))


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("rfft,shifted,norm", itertools.product([False, True], repeat=3))
def test_optional_frequency_grid_matches_upstream(shape, rfft, shifted, norm):
    upstream = pytest.importorskip("torch_grid_utils")
    spacing = tuple(1.25 + i for i in range(len(shape)))
    actual = fftfreq_grid(shape, rfft=rfft, fftshift=shifted, norm=norm, spacing=spacing)
    expected = upstream.fftfreq_grid(shape, rfft=rfft, fftshift=shifted, norm=norm, spacing=spacing)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("shape", [(3, 5), (4, 6), (3, 4, 7), (1, 3), (3, 1, 4)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_optional_sampling_coordinates_match_upstream(shape, dtype):
    upstream = pytest.importorskip("torch_image_interpolation.grid_sample_utils")
    generator = torch.Generator().manual_seed(137)
    coords = torch.randn(2, 7, len(shape), dtype=dtype, generator=generator)
    torch.testing.assert_close(array_to_grid_sample(coords, shape),
                               upstream.array_to_grid_sample(coords, shape), equal_nan=True)


def test_compatibility_modules_export_local_implementations():
    from cryoPARES.utils import torch_grid_utils_compat, torch_image_interpolation_compat

    assert torch_grid_utils_compat.fftfreq_grid is fftfreq_grid
    assert torch_image_interpolation_compat.array_to_grid_sample is array_to_grid_sample


def test_new_helpers_import_and_run_without_teamtomo_packages():
    # Fresh interpreter prevents already-imported packages from hiding a runtime
    # dependency, including imports through the compatibility modules.
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
from cryoPARES.utils.torch_grid_utils_compat import circle, fftfreq_grid
from cryoPARES.utils.torch_image_interpolation_compat import array_to_grid_sample
assert fftfreq_grid((5, 8), rfft=True, fftshift=True).shape == (5, 5, 2)
assert circle(1, (5, 7), smoothing_radius=2).shape == (5, 7)
result = array_to_grid_sample(torch.tensor([[0., 0.], [4., 7.]]), (5, 8))
torch.testing.assert_close(result, torch.tensor([[-1., -1.], [1., 1.]]))
"""
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True,
                               text=True, cwd=Path(__file__).resolve().parents[1], timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_existing_multichannel_sampler_uses_new_coordinate_conversion(device, dtype):
    from cryoPARES.projmatching.projmatchingUtils.extract_central_slices_as_real import sample_image_3d

    shape = (3, 5, 7)
    z, y, x = torch.meshgrid(*[
        torch.arange(n, dtype=dtype, device=device) for n in shape
    ], indexing="ij")
    image = torch.stack([z + 2 * y + 3 * x, 5 - 2 * z + y - x])
    coords = torch.tensor([[[0., 0., 0.], [2., 4., 6.], [0.5, 1.25, 2.75]],
                           [[-0.1, 1., 1.], [1., 4.1, 1.], [1., 1., 6.1]]],
                          dtype=dtype, device=device)
    actual = sample_image_3d(image, coords)
    expected = torch.stack([coords[..., 0] + 2 * coords[..., 1] + 3 * coords[..., 2],
                            5 - 2 * coords[..., 0] + coords[..., 1] - coords[..., 2]], dim=-1)
    # This consumer zeroes any out-of-bounds point, independently of grid_sample's
    # border policy. Both channel layout and its own boundary contract must survive.
    expected[1] = 0
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(5, 8), (3, 5, 8)])
def test_frequency_grid_default_device(shape, device):
    previous = torch.get_default_device()
    try:
        torch.set_default_device(device)
        assert fftfreq_grid(shape, rfft=True).device.type == device
    finally:
        torch.set_default_device(previous)


@pytest.mark.parametrize("shape", [(5, 8), (3, 5, 8)])
@pytest.mark.parametrize("shape_type", [list, tuple, torch.Size])
def test_sampling_coordinate_shape_containers(shape, shape_type):
    coords = torch.tensor([[n - 1 for n in shape]], dtype=torch.float64)
    torch.testing.assert_close(array_to_grid_sample(coords, shape_type(shape)), torch.ones_like(coords))
