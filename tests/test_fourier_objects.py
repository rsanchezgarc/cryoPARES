"""Lifecycle and numerical checks for reusable Fourier operators."""

import itertools
import weakref

import pytest
import torch

from cryoPARES.utils.fourier_geometry import FourierGeometry, fftfreq_to_dft_coordinates
from cryoPARES.utils.fourier_grid import central_slice_fftfreq_grid


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
