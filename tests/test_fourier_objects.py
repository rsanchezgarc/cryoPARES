"""Lifecycle and numerical checks for reusable Fourier operators."""

import itertools
import weakref

import pytest
import torch

from cryoPARES.utils.fourier_geometry import FourierGeometry, fftfreq_to_dft_coordinates
from cryoPARES.utils.fourier_grid import central_slice_fftfreq_grid
from cryoPARES.utils.fourier_operators import FourierProjector, FourierBackprojector, FourierShift2D
from cryoPARES.utils.fourier_slice import extract_central_slices_rfft_3d


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(1, 1, 1), (3, 5, 7), (4, 6, 8)])
@pytest.mark.parametrize("cutoff", [None, -1., 0., 0.3, 0.5])
@pytest.mark.parametrize("rfft,shifted", itertools.product([False, True], repeat=2))
def test_geometry_buffers_match_functional_grid(shape, cutoff, rfft, shifted, device):
    geometry = FourierGeometry(shape, cutoff, rfft=rfft, fftshift=shifted, device=device)
    plane = central_slice_fftfreq_grid(shape, rfft, shifted, device=device)
    mask = torch.ones(plane.shape[:2], dtype=torch.bool, device=device) if cutoff is None else plane.square().sum(-1).sqrt() <= cutoff
    torch.testing.assert_close(geometry.plane, plane)
    torch.testing.assert_close(geometry.frequency_mask, mask)
    torch.testing.assert_close(geometry.valid_coords[..., 0], plane[mask])
    torch.testing.assert_close(geometry.valid_indices, mask.reshape(-1).nonzero().squeeze(-1))
    assert geometry.plane_shape == tuple(plane.shape[:2])
    torch.testing.assert_close(geometry.row_indices, mask.nonzero()[:, 0])
    torch.testing.assert_close(geometry.column_indices, mask.nonzero()[:, 1])
    assert not any(t.requires_grad for t in geometry.buffers())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_geometry_coordinate_constants_match_functional_mapping(dtype):
    geometry = FourierGeometry((3, 5, 7), dtype=dtype)
    frequencies = torch.randn(2, 4, 3, dtype=dtype, requires_grad=True)
    torch.testing.assert_close(geometry.to_dft_coordinates(frequencies),
                               fftfreq_to_dft_coordinates(frequencies, (3, 5, 7), True))
    assert torch.autograd.gradcheck(geometry.double().to_dft_coordinates,
                                   (frequencies.detach().double().requires_grad_(),))


def test_geometry_double_rebuilds_native_precision_and_preserves_index_types():
    geometry = FourierGeometry((3, 5, 7), 0.35).double()
    expected = FourierGeometry((3, 5, 7), 0.35, dtype=torch.float64)
    for name, value in geometry.named_buffers():
        torch.testing.assert_close(value, dict(expected.named_buffers())[name], rtol=0, atol=0)
    assert geometry.valid_indices.dtype == torch.long
    assert geometry.frequency_mask.dtype == torch.bool
    assert geometry.state_dict() == {}


@pytest.mark.gpu
def test_geometry_moves_with_parent_module():
    parent = torch.nn.Module()
    parent.geometry = FourierGeometry((3, 5, 7), 0.35)
    parent.cuda()
    assert all(t.device.type == 'cuda' for t in parent.geometry.buffers())
    parent.cpu()
    assert all(t.device.type == 'cpu' for t in parent.geometry.buffers())


def test_geometry_owns_its_storage_and_releases_it():
    first, second = FourierGeometry((3, 5, 7)), FourierGeometry((3, 5, 7))
    assert first.plane.data_ptr() != second.plane.data_ptr()
    pointers = [t.data_ptr() for t in first.buffers()]
    for _ in range(3):
        first.to_dft_coordinates(torch.zeros(2, 3))
    assert pointers == [t.data_ptr() for t in first.buffers()]
    reference = weakref.ref(first.plane)
    del first
    assert reference() is None


@pytest.mark.parametrize("shape", [(3, 4), (0, 3, 4), (3.5, 4, 6), (True, 3, 4)])
def test_geometry_rejects_invalid_shapes(shape):
    with pytest.raises(ValueError):
        FourierGeometry(shape)


def test_unsupported_dtype_move_is_rejected_without_changing_geometry():
    geometry = FourierGeometry((3, 5, 7))
    plane = geometry.plane
    with pytest.raises(TypeError):
        geometry.half()
    assert geometry.plane is plane


def test_geometry_shape_and_cutoff_are_readonly():
    geometry = FourierGeometry((3, 5, 7), 0.35)
    with pytest.raises(AttributeError):
        geometry.volume_shape = (4, 6, 8)
    with pytest.raises(AttributeError):
        geometry.fftfreq_max = 0.5


def rotations(batch=2, device='cpu', dtype=torch.float32):
    from scipy.spatial.transform import Rotation
    return torch.tensor(Rotation.random(batch, random_state=13).as_matrix(), device=device, dtype=dtype)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('shape', [(3, 5, 7), (4, 6, 8), (1, 5, 1)])
@pytest.mark.parametrize('cutoff', [None, -1., 0., 0.35])
@pytest.mark.parametrize('representation', ['complex', 'channels'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_projector_matches_independent_functional_path(shape, cutoff, representation, dtype, device):
    complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    volume = torch.randn(*shape[:2], shape[-1] // 2 + 1, device=device, dtype=complex_dtype)
    matrix = rotations(device=device, dtype=dtype)
    projector = FourierProjector(shape, cutoff, representation=representation, device=device, dtype=dtype)
    value = volume if representation == 'complex' else torch.stack((volume.real, volume.imag))
    actual = projector(value, matrix)
    if representation == 'channels':
        actual = torch.complex(actual[..., 0, :, :], actual[..., 1, :, :])
    expected = extract_central_slices_rfft_3d(volume, matrix, image_shape=shape, fftfreq_max=cutoff)
    torch.testing.assert_close(actual, expected, atol=4e-6 if dtype == torch.float32 else 1e-13,
                               rtol=3e-5 if dtype == torch.float32 else 1e-12)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('channels', [2, 3, 4])
def test_backprojector_identity_inserts_exact_plane(device, dtype, channels):
    operator = FourierBackprojector((4, 6, 8), device=device, dtype=dtype)
    slices = torch.randn(2, channels, 6, 5, device=device, dtype=dtype)
    matrices = torch.eye(3, device=device, dtype=dtype).expand(2, -1, -1)
    actual, weights = operator(slices, matrices)
    expected = slices.new_zeros(channels, 4, 6, 5)
    expected[:, 2] = slices.sum(0)
    expected_weights = slices.new_zeros(4, 6, 5)
    expected_weights[2] = 2
    # Existing insertion policy requires all eight trilinear corners in bounds,
    # including zero-weight corners at exact edges. Preserve that convention.
    expected[:, 2, -1, :] = 0
    expected[:, 2, :, -1] = 0
    expected_weights[2, -1, :] = 0
    expected_weights[2, :, -1] = 0
    torch.testing.assert_close(actual, expected, atol=2e-6 if dtype == torch.float32 else 1e-14, rtol=0)
    torch.testing.assert_close(weights, expected_weights, atol=2e-6 if dtype == torch.float32 else 1e-14, rtol=0)


@pytest.mark.parametrize('representation', ['complex', 'channels'])
def test_projector_reuses_only_constants_across_backward_passes(representation, monkeypatch):
    import cryoPARES.utils.fourier_geometry as geometry_module
    original = geometry_module.central_slice_fftfreq_grid
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(geometry_module, 'central_slice_fftfreq_grid', counted)
    operator = FourierProjector((3, 5, 7), 0.35, representation=representation, dtype=torch.float64)
    pointers = [t.data_ptr() for t in operator.buffers()]
    for _ in range(3):
        volume = torch.randn(3, 5, 4, dtype=torch.complex128, requires_grad=True)
        matrix = rotations(dtype=torch.float64).requires_grad_()
        value = volume if representation == 'complex' else torch.stack((volume.real, volume.imag))
        result = operator(value, matrix)
        result.abs().square().sum().backward()
        assert volume.grad is not None and matrix.grad is not None
        assert torch.isfinite(volume.grad).all() and torch.isfinite(matrix.grad).all()
    assert len(calls) == 1
    assert pointers == [t.data_ptr() for t in operator.buffers()]
    assert all(t.grad_fn is None and not t.requires_grad for t in operator.buffers())


def test_backprojector_gradients_and_repeated_calls():
    operator = FourierBackprojector((3, 5, 7), 0.3, dtype=torch.float64)
    slices = torch.randn(2, 3, 5, 4, dtype=torch.float64, requires_grad=True)
    matrices = rotations(dtype=torch.float64).requires_grad_()
    assert torch.autograd.gradcheck(operator, (slices, matrices))
    for _ in range(2):
        contribution, weights = operator(slices, matrices)
        (contribution.square().sum() + weights.square().sum()).backward()
    assert torch.isfinite(slices.grad).all() and torch.isfinite(matrices.grad).all()
    assert all(t.grad_fn is None for t in operator.buffers())


def test_projector_reads_current_volume_and_does_not_retain_inputs():
    operator = FourierProjector((4, 4, 4))
    volume = torch.ones(4, 4, 3, dtype=torch.complex64)
    matrix = torch.eye(3)
    first = operator(volume, matrix)
    volume.mul_(2)
    torch.testing.assert_close(operator(volume, matrix), first * 2)
    reference = weakref.ref(volume)
    del volume
    assert reference() is None


@pytest.mark.parametrize('rfft,shifted', itertools.product([False, True], repeat=2))
@pytest.mark.parametrize('device', DEVICES)
def test_shifter_matches_functional_dft_path(rfft, shifted, device):
    from cryoPARES.utils.fourier_shift import fourier_shift_dft_2d
    operator = FourierShift2D((5, 7), rfft=rfft, fftshifted=shifted, device=device)
    spectrum = torch.randn(2, 5, 4 if rfft else 7, dtype=torch.complex64, device=device)
    shifts = torch.tensor([[0.25, -1.5], [1., 2.]], device=device)
    torch.testing.assert_close(operator(spectrum, shifts), fourier_shift_dft_2d(spectrum, (5, 7), shifts, rfft, shifted))
    pointers = [t.data_ptr() for t in operator.buffers()]
    operator(spectrum, shifts + 1)
    assert pointers == [t.data_ptr() for t in operator.buffers()]


@pytest.mark.parametrize('shape', [(1, 1), (5, 7), (4, 6)])
def test_shifter_image_translation_and_repeated_gradients(shape):
    operator = FourierShift2D(shape, dtype=torch.float64)
    image = torch.randn(2, *shape, dtype=torch.float64, requires_grad=True)
    shifts = torch.tensor([[1., -2.], [-1., 1.]], dtype=torch.float64, requires_grad=True)
    actual = operator.shift_image(image, shifts)
    expected = torch.stack([torch.roll(image[0], (1, -2), (-2, -1)), torch.roll(image[1], (-1, 1), (-2, -1))])
    torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-12)
    for _ in range(2):
        operator.shift_image(image, shifts).square().sum().backward()
    assert torch.isfinite(image.grad).all() and torch.isfinite(shifts.grad).all()


def test_operator_dtype_moves_have_native_precision():
    projector = FourierProjector((3, 5, 7)).double()
    volume = torch.randn(3, 5, 4, dtype=torch.complex128)
    matrix = rotations(dtype=torch.float64)
    torch.testing.assert_close(projector(volume, matrix), extract_central_slices_rfft_3d(volume, matrix, image_shape=(3, 5, 7)),
                               atol=1e-13, rtol=1e-12)
    shifter = FourierShift2D((5, 7)).double()
    expected = FourierShift2D((5, 7), dtype=torch.float64)
    torch.testing.assert_close(shifter.radians, expected.radians, atol=0, rtol=0)
    assert projector.state_dict() == {} and shifter.state_dict() == {}


@pytest.mark.parametrize('representation', ['complex', 'channels'])
def test_projector_object_captures_fullgraph(representation):
    operator = FourierProjector((3, 5, 7), 0.35, representation=representation)
    compiled = torch.compile(operator, backend='eager', fullgraph=True)
    volume = torch.randn(3, 5, 4, dtype=torch.complex64)
    if representation == 'channels':
        volume = torch.stack((volume.real, volume.imag))
    for batch in [1, 3]:
        matrix = rotations(batch)
        torch.testing.assert_close(compiled(volume, matrix), operator(volume, matrix))


def test_operators_reject_shape_and_dtype_mismatches():
    projector = FourierProjector((3, 5, 7))
    with pytest.raises(ValueError, match='shape'):
        projector(torch.ones(4, 5, 4, dtype=torch.complex64), torch.eye(3))
    with pytest.raises(TypeError, match='dtype'):
        projector(torch.ones(3, 5, 4, dtype=torch.complex128), torch.eye(3))
    with pytest.raises(ValueError, match='rotation_matrices'):
        projector(torch.ones(3, 5, 4), torch.ones(2))
    with pytest.raises(ValueError, match='batch'):
        FourierBackprojector((3, 5, 7))(torch.ones(2, 3, 5, 4), torch.eye(3))
    with pytest.raises(ValueError, match='image shape'):
        FourierShift2D((5, 7)).shift_image(torch.ones(5, 6), torch.zeros(2))


@pytest.mark.gpu
def test_operator_children_move_with_pipeline_owners():
    from cryoPARES.reconstruction.reconstructor import Reconstructor
    parent = torch.nn.Module()
    parent.projector = FourierProjector((3, 5, 7))
    parent.shifter = FourierShift2D((5, 7))
    parent.cuda()
    assert all(t.device.type == 'cuda' for t in parent.buffers())
    reconstructor = Reconstructor(symmetry='C1')
    assert reconstructor.backprojector is None and reconstructor.fourier_shift is None


@pytest.mark.gpu
@pytest.mark.parametrize('kind', ['project', 'backproject'])
def test_compiled_objects_match_eager_on_cuda(kind):
    torch._dynamo.reset()
    try:
        shape = (5, 7, 9)
        matrices = rotations(2, device='cuda')
        if kind == 'project':
            values = torch.randn(2, 5, 7, 5, device='cuda')
            eager = FourierProjector(shape, 0.35, representation='channels', device='cuda')
            compiled = FourierProjector(shape, 0.35, representation='channels', device='cuda', compiled=True)
            torch.testing.assert_close(compiled(values, matrices), eager(values, matrices), atol=2e-5, rtol=5e-5)
        else:
            values = torch.randn(2, 3, 7, 5, device='cuda')
            eager = FourierBackprojector(shape, 0.35, device='cuda')
            compiled = FourierBackprojector(shape, 0.35, device='cuda', compiled=True)
            for actual, expected in zip(compiled(values, matrices), eager(values, matrices)):
                torch.testing.assert_close(actual, expected, atol=5e-5, rtol=1e-4)
    finally:
        torch._dynamo.reset()
