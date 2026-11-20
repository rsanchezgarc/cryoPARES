"""Fourier translations checked against independent references and real images."""

import itertools
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from cryoPARES.utils.torch_fourier_shift_compat import (
    fourier_shift_dft_2d, fourier_shift_image_2d,
)


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]
SHAPES = [(1, 1), (1, 6), (7, 1), (4, 6), (5, 7), (4, 7), (5, 6)]
SHIFTS = [(0., 0.), (1., -2.), (0.25, -1.5), (-7.25, 12.5)]


@pytest.fixture(params=[torch.float32, torch.float64])
def dtype(request):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(request.param)
    try:
        yield request.param
    finally:
        torch.set_default_dtype(previous)


def _tolerances(dtype):
    return dict(atol=3e-5, rtol=2e-5) if dtype == torch.float32 else dict(atol=2e-12, rtol=2e-12)


def _numpy_translate_dft(dft, shape, shifts, rfft, shifted):
    # Build signed frequency indices directly, independently of the local grids.
    height, width = shape
    ky = np.arange(height)
    ky = np.where(ky < (height + 1) // 2, ky, ky - height) / height
    kx = np.arange(width // 2 + 1) / width if rfft else np.arange(width)
    if not rfft:
        kx = np.where(kx < (width + 1) // 2, kx, kx - width) / width
    if shifted:
        ky = np.roll(ky, height // 2)
        if not rfft:
            kx = np.roll(kx, width // 2)
    shifts = np.asarray(shifts)
    phase = ky[:, None] * shifts[..., 0, None, None] + kx[None, :] * shifts[..., 1, None, None]
    return dft * np.exp(-2j * np.pi * phase)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("rfft,shifted", itertools.product([False, True], repeat=2))
@pytest.mark.parametrize("shift", SHIFTS)
def test_dft_translation_against_independent_phase_formula(shape, rfft, shifted, shift, dtype, device):
    generator = torch.Generator().manual_seed(103)
    image = torch.randn(shape, dtype=dtype, generator=generator).to(device)
    dft = torch.fft.rfft2(image) if rfft else torch.fft.fft2(image)
    if shifted:
        dft = torch.fft.fftshift(dft, dim=(-2,) if rfft else (-2, -1))
    shifts = torch.tensor(shift, dtype=dtype, device=device)
    expected = _numpy_translate_dft(dft.cpu().numpy(), shape, shift, rfft, shifted)
    actual = fourier_shift_dft_2d(dft, shape, shifts, rfft=rfft, fftshifted=shifted)
    assert actual.dtype == dft.dtype
    assert actual.device == dft.device
    assert actual.shape == dft.shape
    np.testing.assert_allclose(actual.cpu().numpy(), expected, **_tolerances(dtype))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(4, 6), (5, 6), (1, 2)])
@pytest.mark.parametrize("shift", [(0, 0), (1, 0), (0, -2), (-3, 4), (12, -18)])
def test_integer_image_shift_matches_roll(shape, shift, dtype, device):
    image = torch.randn(shape, dtype=dtype, device=device)
    actual = fourier_shift_image_2d(image, torch.tensor(shift, dtype=dtype, device=device))
    torch.testing.assert_close(actual, torch.roll(image, shift, (-2, -1)), **_tolerances(dtype))


@pytest.mark.parametrize("shape", [(5, 6), (8, 10)])
def test_fractional_translation_of_analytic_periodic_wave(shape, dtype):
    y, x = torch.meshgrid(torch.arange(shape[0]), torch.arange(shape[1]), indexing="ij")
    y, x = y.to(dtype), x.to(dtype)
    dy, dx = 0.35, -1.2
    image = torch.cos(2 * torch.pi * (y / shape[0] + x / shape[1]))
    expected = torch.cos(2 * torch.pi * ((y - dy) / shape[0] + (x - dx) / shape[1]))
    torch.testing.assert_close(fourier_shift_image_2d(image, torch.tensor([dy, dx])),
                               expected, **_tolerances(dtype))


@pytest.mark.parametrize("image_batch,shift_batch", [
    ((), (3,)), ((3,), ()), ((3,), (3,)), ((2, 1), (1, 3)), ((2, 3), (2, 3)),
])
@pytest.mark.parametrize("rfft,shifted", itertools.product([False, True], repeat=2))
def test_dft_batch_broadcasting(image_batch, shift_batch, rfft, shifted, dtype):
    shape = (5, 8)
    image = torch.randn(*image_batch, *shape, dtype=dtype)
    shifts = torch.randn(*shift_batch, 2, dtype=dtype)
    dft = torch.fft.rfft2(image) if rfft else torch.fft.fft2(image)
    if shifted:
        dft = torch.fft.fftshift(dft, dim=(-2,) if rfft else (-2, -1))
    expected = _numpy_translate_dft(dft.numpy(), shape, shifts.numpy(), rfft, shifted)
    actual = fourier_shift_dft_2d(dft, shape, shifts, rfft, shifted)
    assert actual.shape[:-2] == torch.broadcast_shapes(image_batch, shift_batch)
    np.testing.assert_allclose(actual.numpy(), expected, **_tolerances(dtype))


@pytest.mark.parametrize("rfft,shifted", itertools.product([False, True], repeat=2))
def test_dft_composition_inverse_and_magnitude(rfft, shifted, dtype):
    shape = (5, 8)
    image = torch.randn(2, *shape, dtype=dtype)
    dft = torch.fft.rfft2(image) if rfft else torch.fft.fft2(image)
    if shifted:
        dft = torch.fft.fftshift(dft, dim=(-2,) if rfft else (-2, -1))
    first, second = torch.tensor([0.25, -1.5]), torch.tensor([-0.5, 0.75])
    actual = fourier_shift_dft_2d(dft, shape, first, rfft, shifted)
    composed = fourier_shift_dft_2d(actual, shape, second, rfft, shifted)
    combined = fourier_shift_dft_2d(dft, shape, first + second, rfft, shifted)
    restored = fourier_shift_dft_2d(actual, shape, -first, rfft, shifted)
    torch.testing.assert_close(composed, combined, **_tolerances(dtype))
    torch.testing.assert_close(restored, dft, **_tolerances(dtype))
    torch.testing.assert_close(actual.abs(), dft.abs(), **_tolerances(dtype))


@pytest.mark.parametrize("shift", SHIFTS)
def test_image_translation_preserves_mean(shift, dtype):
    image = torch.randn(3, 5, 8, dtype=dtype)
    result = fourier_shift_image_2d(image, torch.tensor(shift))
    torch.testing.assert_close(result.mean((-2, -1)), image.mean((-2, -1)), **_tolerances(dtype))


def test_fractional_nyquist_mode_real_projection(dtype):
    # An even-width Nyquist cosine cannot retain an arbitrary complex phase in
    # a real image. Its expected amplitude is cos(pi*dx), rather than unit norm.
    image = torch.tensor([1., -1.] * 4, dtype=dtype).repeat(4, 1)
    shift = torch.tensor([0., 0.25])
    actual = fourier_shift_image_2d(image, shift)
    torch.testing.assert_close(actual, image * 2 ** -0.5, **_tolerances(dtype))


@pytest.mark.parametrize("cache", [False, True])
def test_inputs_are_unchanged_and_cache_hint_does_not_reuse_shift_values(cache):
    image = torch.randn(2, 4, 6)
    shifts = torch.tensor([[0.25, -1.5], [1.5, 0.25]])
    dft = torch.fft.rfft2(image)
    image_before, shifts_before, dft_before = image.clone(), shifts.clone(), dft.clone()
    first = fourier_shift_image_2d(image, shifts, cache_intermediates=cache)
    fourier_shift_dft_2d(dft, (4, 6), shifts, True, False, cache_intermediates=cache)
    torch.testing.assert_close(image, image_before)
    torch.testing.assert_close(shifts, shifts_before)
    torch.testing.assert_close(dft, dft_before)
    shifts.zero_()
    restored = fourier_shift_image_2d(image, shifts, cache_intermediates=cache)
    torch.testing.assert_close(restored, image)
    assert not torch.allclose(first, restored)


def test_image_translation_shift_and_image_gradients():
    # Double default is important: the legacy helper builds grids using the
    # default dtype rather than the input dtype.
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        image = torch.randn(3, 4, requires_grad=True)
        shifts = torch.tensor([0.35, -0.7], requires_grad=True)
        assert torch.autograd.gradcheck(fourier_shift_image_2d, (image, shifts))
        assert torch.autograd.gradgradcheck(fourier_shift_image_2d, (image, shifts))
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("shape", [(1, 1), (7, 1), (1, 7), (5, 7), (4, 7)])
@pytest.mark.parametrize("shift", [(0, 0), (1, -2), (-3, 4)])
def test_odd_and_singleton_widths_preserve_original_shape(shape, shift, dtype):
    image = torch.randn(shape, dtype=dtype)
    result = fourier_shift_image_2d(image, torch.tensor(shift))
    assert result.shape == image.shape
    torch.testing.assert_close(result, torch.roll(image, shift, (-2, -1)), **_tolerances(dtype))


@pytest.mark.parametrize("shape", [(3, 5), (5, 8), (4, 7)])
@pytest.mark.parametrize("shift", [(0.25, -1.5), (-2.2, 3.75)])
def test_fractional_image_shift_against_numpy(shape, shift, dtype):
    image = torch.randn(2, *shape, dtype=dtype)
    spectrum = np.fft.rfft2(image.numpy())
    shifted = _numpy_translate_dft(spectrum, shape, shift, True, False)
    expected = np.fft.irfft2(shifted, s=shape, axes=(-2, -1))
    actual = fourier_shift_image_2d(image, torch.tensor(shift))
    np.testing.assert_allclose(actual.numpy(), expected, **_tolerances(dtype))


def test_fractional_odd_image_shift_preserves_energy_and_inverts(dtype):
    image = torch.randn(2, 5, 7, dtype=dtype)
    shifts = torch.tensor([[0.25, -1.5], [1.2, 0.35]])
    shifted = fourier_shift_image_2d(image, shifts)
    restored = fourier_shift_image_2d(shifted, -shifts)
    torch.testing.assert_close(restored, image, **_tolerances(dtype))
    torch.testing.assert_close(shifted.square().sum(), image.square().sum(), **_tolerances(dtype))


@pytest.mark.parametrize("image_batch,shift_batch", [
    ((), (3,)), ((3,), ()), ((3,), (3,)), ((2, 1), (1, 3)), ((2, 3), (2, 3)),
])
def test_image_batch_broadcasting(image_batch, shift_batch, dtype):
    shape = (5, 7)
    image = torch.randn(*image_batch, *shape, dtype=dtype)
    shifts = torch.randn(*shift_batch, 2, dtype=dtype)
    expected_shape = (*torch.broadcast_shapes(image_batch, shift_batch), *shape)
    images = image.expand(expected_shape)
    offsets = shifts.expand(*expected_shape[:-2], 2)
    actual = fourier_shift_image_2d(image, shifts)
    expected = torch.stack([
        fourier_shift_image_2d(img, shift)
        for img, shift in zip(images.reshape(-1, *shape), offsets.reshape(-1, 2))
    ]).reshape(expected_shape)
    assert actual.shape == expected_shape
    torch.testing.assert_close(actual, expected, **_tolerances(dtype))


@pytest.mark.parametrize("view", ["transpose", "strided", "expanded"])
def test_noncontiguous_image_inputs(view, dtype):
    base = torch.randn(2, 8, 10, dtype=dtype)
    image = base.transpose(-1, -2) if view == "transpose" else base[..., ::2, ::2] if view == "strided" else base[:1].expand(3, -1, -1)
    assert not image.is_contiguous()
    shift = torch.tensor([1., -2.])
    torch.testing.assert_close(fourier_shift_image_2d(image, shift),
                               torch.roll(image, (1, -2), (-2, -1)), **_tolerances(dtype))


def test_noncontiguous_dft_and_shifts(dtype):
    dft = torch.fft.fft2(torch.randn(2, 5, 8, dtype=dtype)).transpose(-1, -2)
    shifts = torch.randn(2, 4, dtype=dtype)[:, ::2]
    assert not dft.is_contiguous() and not shifts.is_contiguous()
    actual = fourier_shift_dft_2d(dft, (8, 5), shifts, False, True)
    expected = fourier_shift_dft_2d(dft.contiguous(), (8, 5), shifts.contiguous(), False, True)
    torch.testing.assert_close(actual, expected, **_tolerances(dtype))


@pytest.mark.parametrize("rfft,shifted", itertools.product([False, True], repeat=2))
def test_dft_gradients(rfft, shifted):
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        shape = (3, 5)
        dft = torch.randn(3, 3 if rfft else 5, dtype=torch.complex128, requires_grad=True)
        shifts = torch.tensor([0.35, -0.7], requires_grad=True)
        fn = lambda values, offsets: fourier_shift_dft_2d(values, shape, offsets, rfft, shifted)
        assert torch.autograd.gradcheck(fn, (dft, shifts))
        assert torch.autograd.gradgradcheck(fn, (dft, shifts))
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("cache", [False, True])
def test_cache_hint_does_not_retain_autograd_graph(cache):
    image = torch.randn(2, 5, 7)
    for _ in range(2):
        shifts = torch.tensor([[0.25, -0.75], [0.5, 1.25]], requires_grad=True)
        result = fourier_shift_image_2d(image, shifts, cache_intermediates=cache)
        (result * image).sum().backward()
        assert shifts.grad is not None and torch.isfinite(shifts.grad).all()
        assert shifts.grad.abs().max() > 0


@pytest.mark.parametrize("rfft,shifted", itertools.product([False, True], repeat=2))
def test_compiled_dft_shift_matches_eager(rfft, shifted):
    compiled = torch.compile(fourier_shift_dft_2d, backend="eager", fullgraph=True)
    shape = (5, 7)
    for batch in [2, 3]:
        dft = torch.randn(batch, 5, 4 if rfft else 7, dtype=torch.complex64)
        shifts = torch.randn(batch, 2)
        torch.testing.assert_close(compiled(dft, shape, shifts, rfft, shifted),
                                   fourier_shift_dft_2d(dft, shape, shifts, rfft, shifted))


@pytest.mark.parametrize("shape", [(4, 6), (5, 7), (7, 1)])
def test_compiled_image_shift_matches_eager(shape):
    compiled = torch.compile(fourier_shift_image_2d, backend="eager", fullgraph=True)
    image = torch.randn(2, *shape)
    shifts = torch.randn(2, 2)
    torch.testing.assert_close(compiled(image, shifts), fourier_shift_image_2d(image, shifts))


@pytest.mark.parametrize("shape", [(1, 1), (5, 7), (4, 6)])
@pytest.mark.parametrize("rfft,shifted", itertools.product([False, True], repeat=2))
@pytest.mark.parametrize("shift_dtype", [torch.float32, torch.float64, torch.int64])
def test_optional_dft_shift_matches_upstream(shape, rfft, shifted, shift_dtype):
    upstream = pytest.importorskip("torch_fourier_shift")
    dft = torch.fft.rfft2(torch.randn(shape)) if rfft else torch.fft.fft2(torch.randn(shape))
    shifts = torch.tensor([1, -2], dtype=shift_dtype)
    actual = fourier_shift_dft_2d(dft, shape, shifts, rfft, shifted)
    expected = upstream.fourier_shift_dft_2d(dft, shape, shifts, rfft, shifted)
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("image_dtype,shift_dtype", itertools.product([torch.float32, torch.float64], repeat=2))
@pytest.mark.parametrize("shape", [(4, 6), (5, 8)])
def test_optional_even_width_image_matches_upstream(shape, image_dtype, shift_dtype):
    upstream = pytest.importorskip("torch_fourier_shift")
    image = torch.randn(3, *shape, dtype=image_dtype)
    shifts = torch.tensor([[0.25, -1.5], [1.2, 0.35], [-4.5, 5.75]], dtype=shift_dtype)
    actual = fourier_shift_image_2d(image, shifts)
    expected = upstream.fourier_shift_image_2d(image, shifts)
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("bad_shifts", [torch.tensor(1.), torch.zeros(1), torch.zeros(3), torch.zeros(2, 1)])
def test_rejects_invalid_shift_component_count(bad_shifts):
    with pytest.raises(ValueError, match="shifts"):
        fourier_shift_dft_2d(torch.ones(4, 4, dtype=torch.complex64), (4, 6), bad_shifts, True, False)


@pytest.mark.parametrize("dft_shape,image_shape,rfft", [
    ((4, 4), (4, 7), False), ((4, 7), (4, 7), True), ((3, 4), (4, 7), True),
    ((4,), (4, 6), True),
])
def test_rejects_inconsistent_dft_shape(dft_shape, image_shape, rfft):
    with pytest.raises(ValueError, match="DFT"):
        fourier_shift_dft_2d(torch.ones(dft_shape, dtype=torch.complex64), image_shape,
                              torch.zeros(2), rfft, False)


@pytest.mark.parametrize("shape", [(7,), (3, 4, 5)])
def test_rejects_invalid_real_space_dimensions(shape):
    with pytest.raises(ValueError, match="image_shape"):
        fourier_shift_dft_2d(torch.ones(4, 4, dtype=torch.complex64), shape, torch.zeros(2), True, False)


def test_rejects_incompatible_batch_shapes():
    with pytest.raises(RuntimeError):
        fourier_shift_image_2d(torch.randn(3, 5, 7), torch.randn(2, 2))


@pytest.mark.gpu
def test_rejects_mixed_input_devices():
    with pytest.raises(RuntimeError):
        fourier_shift_image_2d(torch.randn(5, 7, device="cuda"), torch.zeros(2))


def test_compatibility_exports_are_local():
    from cryoPARES.utils import fourier_shift

    assert fourier_shift_image_2d is fourier_shift.fourier_shift_image_2d
    assert fourier_shift_dft_2d is fourier_shift.fourier_shift_dft_2d


def test_shifts_run_without_teamtomo_packages():
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
from cryoPARES.utils.torch_fourier_shift_compat import fourier_shift_image_2d, fourier_shift_dft_2d
image = torch.arange(35, dtype=torch.float32).reshape(5, 7)
shift = torch.tensor([1., -2.])
torch.testing.assert_close(fourier_shift_image_2d(image, shift), torch.roll(image, (1, -2), (-2, -1)), atol=2e-5, rtol=1e-5)
dft = torch.fft.rfft2(image)
assert fourier_shift_dft_2d(dft, (5, 7), shift, True, False).shape == dft.shape
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parents[1], timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
