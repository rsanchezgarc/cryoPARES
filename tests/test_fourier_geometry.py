"""Lattice-coordinate checks, including odd and singleton Fourier axes."""

import pytest
import torch

from cryoPARES.utils.fourier_geometry import fftfreq_to_dft_coordinates
from cryoPARES.utils.fourier_grid import fftfreq_grid, central_slice_fftfreq_grid


@pytest.mark.parametrize("shape", [(1, 1), (4, 6), (5, 7), (4, 7), (1, 3, 5), (4, 6, 8), (3, 5, 7)])
@pytest.mark.parametrize("rfft", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_frequency_samples_map_to_their_exact_array_indices(shape, rfft, dtype):
    grid = fftfreq_grid(shape, rfft=rfft, fftshift=True, dtype=dtype)
    actual = fftfreq_to_dft_coordinates(grid, shape, rfft)
    sizes = (*shape[:-1], shape[-1] // 2 + 1) if rfft else shape
    expected = torch.stack(torch.meshgrid(*[
        torch.arange(n, dtype=dtype) for n in sizes
    ], indexing="ij"), dim=-1)
    torch.testing.assert_close(actual, expected, atol=1e-6 if dtype == torch.float32 else 1e-14, rtol=0)


@pytest.mark.parametrize("shape", [(3, 5, 7), (4, 6, 8), (1, 1, 1)])
@pytest.mark.parametrize("rfft", [False, True])
def test_dc_and_fractional_index_conversion(shape, rfft):
    dc = torch.zeros(3, dtype=torch.float64)
    centers = [n // 2 for n in shape]
    if rfft:
        centers[-1] = 0
    torch.testing.assert_close(fftfreq_to_dft_coordinates(dc, shape, rfft), dc.new_tensor(centers))
    coordinates = torch.tensor([[0.25, 1.5, 2.75], [-1., 3.25, 4.5]], dtype=torch.float64)
    frequencies = (coordinates - coordinates.new_tensor(centers)) / coordinates.new_tensor(shape)
    torch.testing.assert_close(fftfreq_to_dft_coordinates(frequencies, shape, rfft), coordinates)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_explicit_grid_dtype_overrides_default(dtype):
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64 if dtype == torch.float32 else torch.float32)
        assert fftfreq_grid((5, 7), True, dtype=dtype).dtype == dtype
        assert central_slice_fftfreq_grid((3, 5, 7), True, dtype=dtype).dtype == dtype
    finally:
        torch.set_default_dtype(previous)


def test_conversion_gradients_and_no_mutation():
    frequencies = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    before = frequencies.detach().clone()
    assert torch.autograd.gradcheck(lambda f: fftfreq_to_dft_coordinates(f, (3, 5, 7), True), (frequencies,))
    torch.testing.assert_close(frequencies.detach(), before)
    fftfreq_to_dft_coordinates(frequencies, (3, 5, 7), True).sum().backward()
    torch.testing.assert_close(frequencies.grad, frequencies.new_tensor([3, 5, 7]).expand_as(frequencies))


@pytest.mark.parametrize("frequencies,shape", [(torch.zeros(3), (5, 7)), (torch.zeros(()), (3,))])
def test_rejects_mismatched_dimension_count(frequencies, shape):
    with pytest.raises(ValueError, match="dimension"):
        fftfreq_to_dft_coordinates(frequencies, shape, True)


def test_odd_rfft_roundoff_does_not_put_final_sample_outside_array():
    frequencies = torch.tensor([[0., 0., torch.fft.rfftfreq(7)[-1]]])
    coordinates = fftfreq_to_dft_coordinates(frequencies, (3, 5, 7), True)
    assert coordinates[0, 2] == 3


def test_true_outside_coordinates_are_not_clamped():
    frequencies = torch.tensor([[-1., 1., 1.]])
    torch.testing.assert_close(fftfreq_to_dft_coordinates(frequencies, (3, 5, 7), True),
                               torch.tensor([[-2., 7., 7.]]))
