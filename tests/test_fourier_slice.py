"""Independent geometry and interpolation tests for Fourier central slices."""

import itertools
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from cryoPARES.utils.torch_fourier_slice_compat import extract_central_slices_rfft_3d as extract


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]


@pytest.fixture
def fresh_compilation_cache():
    """Keep shape/device specialization tests independent of later pipelines.

    The compiled production kernels are shared module globals. Exercising CUDA
    variants here can otherwise exhaust Dynamo's per-function recompile limit
    when subsequent tests specialize the same kernels for CPU reconstruction.
    """
    torch._dynamo.reset()
    try:
        yield
    finally:
        torch._dynamo.reset()


def _rotations(batch=(), dtype=torch.float32, device="cpu"):
    count = int(np.prod(batch)) if batch else 1
    values = Rotation.random(count, random_state=17).as_matrix().reshape(*batch, 3, 3)
    return torch.tensor(values, dtype=dtype, device=device)


def _reference_slice(volume, shape, matrices, cutoff=None, zyx=False):
    """Direct scalar trilinear interpolation, without grid_sample or TeamTomo."""
    stack = matrices.shape[:-2]
    result = np.zeros((*stack, shape[1], shape[2] // 2 + 1), dtype=volume.dtype)
    for batch in np.ndindex(stack):
        matrix = matrices[batch] if zyx else matrices[batch][::-1, ::-1]
        for row, fy in enumerate(np.fft.fftshift(np.fft.fftfreq(shape[1]))):
            for col, fx in enumerate(np.fft.rfftfreq(shape[2])):
                if cutoff is not None and np.hypot(fy, fx) > cutoff:
                    continue
                frequency = matrix @ np.array([0., fy, fx])
                conjugate = frequency[2] < 0
                if conjugate:
                    frequency = -frequency
                coordinate = frequency * shape + np.array([shape[0] // 2, shape[1] // 2, 0])
                if np.any(coordinate < 0) or np.any(coordinate > np.array(volume.shape) - 1):
                    continue
                lower = np.floor(coordinate).astype(int)
                fractions = coordinate - lower
                value = 0j
                for offset in itertools.product([0, 1], repeat=3):
                    index = np.minimum(lower + offset, np.array(volume.shape) - 1)
                    weight = np.prod(np.where(offset, fractions, 1 - fractions))
                    value += volume[tuple(index)] * weight
                result[(*batch, row, col)] = value.conjugate() if conjugate else value
    return result


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(4, 4, 4), (4, 8, 8), (8, 8, 4), (3, 5, 7), (4, 5, 8), (1, 5, 7), (3, 1, 1)])
@pytest.mark.parametrize("batch", [(), (1,), (2, 3)])
@pytest.mark.parametrize("cutoff", [None, -0.1, 0., 0.25, 0.5])
@pytest.mark.parametrize("zyx", [False, True])
def test_extract_against_scalar_trilinear_reference(shape, batch, cutoff, zyx, device):
    generator = torch.Generator().manual_seed(42)
    volume = torch.randn(shape[0], shape[1], shape[2] // 2 + 1,
                         dtype=torch.complex64, generator=generator).to(device)
    matrices = _rotations(batch, device=device)
    actual = extract(volume, matrices, image_shape=shape, fftfreq_max=cutoff, zyx_matrices=zyx)
    expected = _reference_slice(volume.cpu().numpy(), shape, matrices.cpu().numpy(), cutoff, zyx)
    assert actual.dtype == volume.dtype and actual.device == volume.device
    assert actual.shape == (*batch, shape[1], shape[2] // 2 + 1)
    np.testing.assert_allclose(actual.cpu().numpy(), expected, atol=4e-6, rtol=3e-5)


@pytest.mark.parametrize("shape", [(4, 4, 4), (4, 8, 8), (8, 8, 4), (3, 5, 7), (4, 5, 8), (1, 5, 7), (3, 1, 1)])
def test_identity_slice_equals_transform_of_direct_volume_sum(shape):
    generator = torch.Generator().manual_seed(41)
    volume = torch.randn(shape, generator=generator)
    spectrum = torch.fft.fftshift(torch.fft.rfftn(volume), dim=(-3, -2))
    expected = torch.fft.fftshift(torch.fft.rfft2(volume.sum(0)), dim=(-2,))
    actual = extract(spectrum, torch.eye(3), image_shape=shape)
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=2e-5)


@pytest.mark.parametrize("axis", ["x", "y"])
def test_quarter_turn_equals_transform_of_axis_projection(axis):
    size = 8
    generator = torch.Generator().manual_seed(43)
    volume = torch.randn(size, size, size, generator=generator)
    spectrum = torch.fft.fftshift(torch.fft.rfftn(volume), dim=(-3, -2))
    if axis == "x":
        matrix = torch.tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        projection = volume.sum(1)
    else:
        matrix = torch.tensor([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
        reverse = (-torch.arange(size)) % size
        projection = volume.sum(2).index_select(0, reverse).T
    expected = torch.fft.fftshift(torch.fft.rfft2(projection), dim=(-2,))
    actual = extract(spectrum, matrix, image_shape=(size,) * 3)
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=2e-5)


@pytest.mark.parametrize("batch", [(), (2,), (2, 3)])
def test_xyz_and_zyx_rotation_conventions_agree(batch):
    volume = torch.randn(8, 8, 5, dtype=torch.complex64)
    xyz = _rotations(batch)
    actual = extract(volume, xyz, image_shape=(8, 8, 8))
    expected = extract(volume, xyz.flip((-2, -1)), image_shape=(8, 8, 8), zyx_matrices=True)
    torch.testing.assert_close(actual, expected)


def test_hermitian_fold_conjugates_negative_x_samples():
    volume = torch.full((8, 8, 5), 2 + 3j, dtype=torch.complex64)
    matrix = torch.diag(torch.tensor([-1., -1., 1.]))
    actual = extract(volume, matrix, image_shape=(8, 8, 8), fftfreq_max=0.3)
    # Row 4 is ky=0, column 1 is kx=1/8. A half-turn sends it to -kx.
    torch.testing.assert_close(actual[4, 1], torch.tensor(2 - 3j))
    torch.testing.assert_close(actual[4, 0], torch.tensor(2 + 3j))


def test_inputs_are_not_modified():
    volume = torch.randn(8, 8, 5, dtype=torch.complex64)
    matrices = _rotations((2, 3))
    before_volume, before_matrices = volume.clone(), matrices.clone()
    extract(volume, matrices, image_shape=(8, 8, 8), fftfreq_max=0.4)
    torch.testing.assert_close(volume, before_volume)
    torch.testing.assert_close(matrices, before_matrices)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(3, 5, 7), (4, 6, 8), (1, 5, 7)])
@pytest.mark.parametrize("complex_input", [False, True])
def test_double_precision_against_reference(shape, complex_input, device):
    dtype = torch.complex128 if complex_input else torch.float64
    volume = torch.randn(*shape[:2], shape[-1] // 2 + 1, dtype=dtype, device=device)
    matrices = _rotations((2,), dtype=torch.float64, device=device)
    # The scalar reference is complex-valued internally; use it for real input
    # by treating real Fourier fields as complex fields with zero imaginary part.
    values = volume.cpu().numpy().astype(np.complex128)
    expected = _reference_slice(values, shape, matrices.cpu().numpy(), cutoff=0.45)
    actual = extract(volume, matrices, image_shape=shape, fftfreq_max=0.45)
    assert actual.dtype == dtype
    np.testing.assert_allclose(actual.cpu().numpy(), expected if complex_input else expected.real,
                               atol=1e-13, rtol=1e-12)


@pytest.mark.parametrize("shape", [(3, 5, 7), (4, 6, 8)])
def test_double_precision_identity_projection(shape):
    volume = torch.randn(shape, dtype=torch.float64)
    spectrum = torch.fft.fftshift(torch.fft.rfftn(volume), dim=(-3, -2))
    actual = extract(spectrum, torch.eye(3, dtype=torch.float64), image_shape=shape)
    expected = torch.fft.fftshift(torch.fft.rfft2(volume.sum(0)), dim=(-2,))
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("shape", [(4, 6, 8), (3, 5, 8), (3, 5, 1)])
def test_shape_inference_matches_explicit_even_or_singleton_width(shape):
    volume = torch.randn(*shape[:2], shape[-1] // 2 + 1, dtype=torch.complex64)
    rotations = _rotations((2,))
    torch.testing.assert_close(extract(volume, rotations), extract(volume, rotations, image_shape=shape))


@pytest.mark.parametrize("cutoff", [-1., 0., 0.25, 0.5, 1.])
def test_cutoff_is_inclusive_and_excluded_pixels_are_exact_zeros(cutoff):
    shape = (8, 8, 8)
    volume = torch.ones(8, 8, 5, dtype=torch.complex64)
    actual = extract(volume, torch.eye(3), image_shape=shape, fftfreq_max=cutoff)
    fy, fx = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(8)), np.fft.rfftfreq(8), indexing="ij")
    expected = (np.hypot(fy, fx) <= cutoff).astype(np.complex64)
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-6)
    assert torch.count_nonzero(actual[torch.from_numpy(np.hypot(fy, fx) > cutoff)]) == 0


def test_outside_full_axis_range_is_zero_instead_of_border_replication():
    volume = torch.ones(4, 4, 3, dtype=torch.complex64)
    # XYZ +90° about Z maps the stored +X Nyquist frequency to +Y Nyquist,
    # which lies beyond this fftshifted array's last positive-frequency row.
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    result = extract(volume, rotation, image_shape=(4, 4, 4))
    assert result[2, 2] == 0
    assert result[2, 1] == 1


@pytest.mark.parametrize("batch", [(0,), (2, 0), (2, 1, 3)])
def test_arbitrary_and_empty_rotation_batches(batch):
    volume = torch.ones(4, 4, 3, dtype=torch.complex64)
    rotations = torch.eye(3).expand(*batch, 3, 3)
    result = extract(volume, rotations, image_shape=(4, 4, 4))
    assert result.shape == (*batch, 4, 3)
    torch.testing.assert_close(result, torch.ones_like(result))


@pytest.mark.parametrize("complex_input", [False, True])
def test_volume_and_rotation_gradients(complex_input):
    dtype = torch.complex128 if complex_input else torch.float64
    volume = torch.randn(3, 5, 4, dtype=dtype, requires_grad=True)
    matrices = _rotations(dtype=torch.float64).requires_grad_()
    fn = lambda values, rotations: extract(values, rotations, image_shape=(3, 5, 7), fftfreq_max=0.3)
    assert torch.autograd.gradcheck(fn, (volume, matrices))


@pytest.mark.parametrize("device", DEVICES)
def test_backward_has_finite_nonzero_volume_and_rotation_gradients(device):
    volume = torch.randn(4, 6, 5, dtype=torch.complex64, device=device, requires_grad=True)
    rotations = _rotations((2,), device=device).requires_grad_()
    result = extract(volume, rotations, image_shape=(4, 6, 8), fftfreq_max=0.35)
    result.abs().square().sum().backward()
    for gradient in [volume.grad, rotations.grad]:
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().max() > 0


def test_noncontiguous_volume_and_rotations():
    volume = torch.randn(6, 4, 5, dtype=torch.complex64).transpose(0, 1)
    rotations = _rotations((2, 3)).transpose(0, 1)
    assert not volume.is_contiguous() and not rotations.is_contiguous()
    actual = extract(volume, rotations, image_shape=(4, 6, 8), zyx_matrices=True)
    expected = extract(volume.contiguous(), rotations.contiguous(), image_shape=(4, 6, 8), zyx_matrices=True)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("cutoff", [None, 0.3])
def test_compilation_capture(cutoff):
    compiled = torch.compile(extract, backend="eager", fullgraph=True)
    volume = torch.randn(4, 6, 5, dtype=torch.complex64)
    for batch in [1, 3]:
        rotations = _rotations((batch,))
        torch.testing.assert_close(compiled(volume, rotations, image_shape=(4, 6, 8), fftfreq_max=cutoff),
                                   extract(volume, rotations, image_shape=(4, 6, 8), fftfreq_max=cutoff))


@pytest.mark.parametrize("shape", [(4, 4, 4), (4, 8, 8), (8, 6, 10)])
@pytest.mark.parametrize("cutoff", [None, 0., 0.3, 0.5])
@pytest.mark.parametrize("zyx", [False, True])
def test_optional_even_size_comparison_with_upstream(shape, cutoff, zyx):
    upstream = pytest.importorskip("torch_fourier_slice.slice_extraction")
    volume = torch.randn(*shape[:2], shape[-1] // 2 + 1, dtype=torch.complex64)
    rotations = _rotations((2, 3))
    actual = extract(volume, rotations, image_shape=shape, fftfreq_max=cutoff, zyx_matrices=zyx)
    expected = upstream.extract_central_slices_rfft_3d(volume, image_shape=shape,
        rotation_matrices=rotations, fftfreq_max=cutoff, zyx_matrices=zyx)
    torch.testing.assert_close(actual, expected, atol=6e-6, rtol=5e-5)


@pytest.mark.parametrize("shape", [(3, 5, 7), (4, 6, 8)])
def test_existing_multichannel_kernel_agrees_with_complex_extraction(shape):
    from cryoPARES.projmatching.projmatchingUtils.extract_central_slices_as_real import (
        extract_central_slices_rfft_3d_multichannel,
    )

    volume = torch.randn(*shape[:2], shape[-1] // 2 + 1, dtype=torch.complex64)
    rotations = _rotations((2, 3))
    channels = torch.stack((volume.real, volume.imag))
    sampled = extract_central_slices_rfft_3d_multichannel(channels, shape, rotations, fftfreq_max=0.35)
    expected = torch.complex(sampled[..., 0, :, :], sampled[..., 1, :, :])
    torch.testing.assert_close(extract(volume, rotations, image_shape=shape, fftfreq_max=0.35),
                               expected, atol=4e-6, rtol=3e-5)


@pytest.mark.parametrize("volume,rotations,shape,message", [
    (torch.ones(4, 4), torch.eye(3), None, "volume_rfft"),
    (torch.ones(4, 4, 3), torch.ones(2), None, "rotation_matrices"),
    (torch.ones(4, 4, 3), torch.ones(2, 3), None, "rotation_matrices"),
    (torch.ones(4, 4, 3), torch.eye(3), (4, 4), "image_shape"),
    (torch.ones(4, 4, 3), torch.eye(3), (4, 0, 4), "image_shape"),
    (torch.ones(4, 4, 3), torch.eye(3), (4, 4, 8), "volume_rfft"),
])
def test_invalid_shapes_are_rejected(volume, rotations, shape, message):
    with pytest.raises(ValueError, match=message):
        extract(volume, rotations, image_shape=shape)


def test_integer_volume_is_rejected():
    with pytest.raises(TypeError, match="dtype"):
        extract(torch.ones(4, 4, 3, dtype=torch.int64), torch.eye(3))


@pytest.mark.gpu
def test_mixed_devices_are_rejected():
    with pytest.raises(RuntimeError):
        extract(torch.ones(4, 4, 3, dtype=torch.complex64, device="cuda"), torch.eye(3))


@pytest.mark.gpu
def test_cuda_extraction_agrees_with_cpu():
    volume = torch.randn(6, 8, 6, dtype=torch.complex64)
    rotations = _rotations((2, 3))
    expected = extract(volume, rotations, image_shape=(6, 8, 10), fftfreq_max=0.35)
    actual = extract(volume.cuda(), rotations.cuda(), image_shape=(6, 8, 10), fftfreq_max=0.35)
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-5, rtol=5e-5)


@pytest.mark.gpu
@pytest.mark.parametrize("shape", [(6, 6, 6), (5, 7, 9)])
def test_production_compiled_cuda_projection_agrees_with_eager(shape, fresh_compilation_cache, monkeypatch):
    from cryoPARES.configs.mainConfig import main_config
    from cryoPARES.projmatching.projmatchingUtils.extract_central_slices_as_real import (
        compiled_extract_central_slices_rfft_3d_multichannel,
        extract_central_slices_rfft_3d_multichannel,
    )

    monkeypatch.setattr(main_config.projmatching, 'disable_compile_projectVol', False)
    volume = torch.randn(2, *shape[:2], shape[-1] // 2 + 1, device="cuda")
    for batch in [1, 3]:
        rotations = _rotations((batch,), device="cuda")
        expected = extract_central_slices_rfft_3d_multichannel(volume, shape, rotations, fftfreq_max=0.35)
        actual = compiled_extract_central_slices_rfft_3d_multichannel(volume, shape, rotations, fftfreq_max=0.35)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=5e-5)


@pytest.mark.parametrize('shape', [(6, 6, 6), (5, 7, 9)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('cutoff', [None, 0.35])
@pytest.mark.parametrize('zyx', [False, True])
def test_projection_wrapper_respects_runtime_compile_config(shape, dtype, cutoff, zyx, monkeypatch):
    import importlib
    module = importlib.import_module(
        'cryoPARES.projmatching.projmatchingUtils.extract_central_slices_as_real')
    from cryoPARES.configs.mainConfig import main_config

    calls = []

    def compiled_kernel(*args, **kwargs):
        calls.append(True)
        return module._extract_central_slices_rfft_3d_multichannel_precomputed(*args, **kwargs)

    # Exercise real operator dispatch without invoking the compiler in CPU tests.
    monkeypatch.setattr(module, '_compiled_extract_central_slices_rfft_3d_multichannel', compiled_kernel)
    volume = torch.randn(2, *shape[:2], shape[-1] // 2 + 1, dtype=dtype)
    rotations = _rotations((2,), dtype=dtype)
    expected_complex = extract(torch.complex(volume[0], volume[1]), rotations,
                               image_shape=shape, fftfreq_max=cutoff, zyx_matrices=zyx)
    expected = torch.stack((expected_complex.real, expected_complex.imag), dim=-3)
    for disabled in [True, False, True, False]:
        monkeypatch.setattr(main_config.projmatching, 'disable_compile_projectVol', disabled)
        previous_calls = len(calls)
        actual = module.compiled_extract_central_slices_rfft_3d_multichannel(
            volume, shape, rotations, fftfreq_max=cutoff, zyx_matrices=zyx)
        assert len(calls) == previous_calls + int(not disabled)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=5e-5)


@pytest.mark.gpu
@pytest.mark.parametrize("shape", [(6, 6, 6), (5, 7, 9)])
def test_production_compiled_cuda_backprojection_agrees_with_eager(shape, fresh_compilation_cache):
    from cryoPARES.reconstruction.insert_central_slices_rfft_3d import (
        compiled_insert_central_slices_rfft_3d_multichannel,
        insert_central_slices_rfft_3d_multichannel,
    )

    for batch in [1, 3]:
        # Channels carry real, imaginary and CTF-squared contributions.
        slices = torch.randn(batch, 3, shape[1], shape[2] // 2 + 1, device="cuda")
        rotations = _rotations((batch,), device="cuda")
        expected = insert_central_slices_rfft_3d_multichannel(slices, shape, rotations, fftfreq_max=0.35)
        actual = compiled_insert_central_slices_rfft_3d_multichannel(slices, shape, rotations, fftfreq_max=0.35)
        for compiled, eager in zip(actual, expected):
            torch.testing.assert_close(compiled, eager, atol=5e-5, rtol=1e-4)


@pytest.mark.parametrize("shape", [(5, 7), (6, 8), (1, 7)])
def test_projection_inverse_fft_preserves_original_shape(shape):
    from cryoPARES.projmatching.projmatchingUtils.fourierOperations import (
        _real_to_fourier_2d, _fourier_proj_to_real_2d,
    )

    image = torch.randn(2, *shape)
    spectrum = _real_to_fourier_2d(image)
    torch.testing.assert_close(_fourier_proj_to_real_2d(spectrum, None, image_shape=shape), image)


def test_simulation_and_reconstruction_work_without_external_numerical_packages():
    script = """
import os
import sys
import tempfile
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
from tests.test_simulate_particles import _make_volume, _make_star
from cryoPARES.simulation.simulateParticlesHelper import run_simulation
from cryoPARES.simulation.simulateParticles import _write_output_star
from cryoPARES.reconstruction.reconstructor import Reconstructor
from cryoPARES.projmatching.projMatcher import ProjectionMatcher
from cryoPARES.configs.mainConfig import main_config
main_config.reconstruct.disable_compile_insert_central_slices_rfft_3d_multichannel = True
with tempfile.TemporaryDirectory() as directory:
    volume_path = os.path.join(directory, 'volume.mrc')
    input_star = os.path.join(directory, 'input.star')
    out = os.path.join(directory, 'out')
    _make_volume(volume_path, box=8)
    _make_star(input_star, n=3, box=8)
    stacks = run_simulation(volume_path, input_star, out, 'stack', images_per_file=4,
        batch_size=2, num_workers=0, apply_ctf=True, device='cpu', disable_tqdm=True)
    star = _write_output_star(stacks, out, 'stack', input_star, 4)
    reconstruction = Reconstructor(symmetry='C1', correct_ctf=True, eps=1e-3)
    reconstruction.backproject_particles(particles_star_fname=star, particles_dir=out,
        batch_size=2, num_dataworkers=0)
    result = reconstruction.generate_volume(os.path.join(directory, 'reconstruction.mrc'))
    assert result.shape == (8, 8, 8) and torch.isfinite(result).all()
    assert result.abs().max() > 0
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parents[1], timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
