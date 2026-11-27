"""Fused correlation must preserve the eager spectrum, shifts, and gradients."""

import itertools

import pytest
import torch

from cryoPARES.projmatching.projmatchingUtils.fourierOperations import (
    _correlate_dft_2d_real, _real_to_fourier_2d, build_ccorr_sign_grid, correlate_dft_2d,
)


def inputs(dtype, channels, noncontiguous, device='cpu', batch=2, candidates=3):
    cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    parts = torch.randn(batch, 1, 6, 5, dtype=cdtype, device=device)
    projs = torch.randn(batch, candidates, 6, 10 if noncontiguous else 5, dtype=cdtype, device=device)
    if noncontiguous:
        projs = projs[..., ::2]
        assert not projs.is_contiguous()
    if channels:
        parts, projs = torch.view_as_real(parts), torch.view_as_real(projs)
    return parts, projs


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('channels,noncontiguous,whitening,sign', itertools.product([False, True], repeat=4))
def test_real_correlation_matches_eager(dtype, channels, noncontiguous, whitening, sign):
    parts, projs = inputs(dtype, channels, noncontiguous)
    filt = torch.rand(2, 1, 6, 5, dtype=dtype) if whitening else None
    phase = build_ccorr_sign_grid(6, 8, 'cpu').to(dtype) if sign else None
    expected = correlate_dft_2d(parts, projs, filt, phase)
    actual = _correlate_dft_2d_real(parts, projs, filt, phase)
    torch.testing.assert_close(actual, expected)


def test_real_correlation_accepts_conjugate_views():
    parts, projs = inputs(torch.float64, False, True)
    actual = _correlate_dft_2d_real(parts.conj(), projs.conj())
    torch.testing.assert_close(actual, correlate_dft_2d(parts.conj(), projs.conj()))


@pytest.mark.parametrize('box', [8, 10, 12, 14])
@pytest.mark.parametrize('sign', [False, True])
def test_autocorrelation_peak_stays_centered_for_both_even_box_parities(box, sign):
    images = torch.randn(2, box, box, dtype=torch.float64)
    spectrum = _real_to_fourier_2d(images).unsqueeze(1)
    phase = build_ccorr_sign_grid(box, box, 'cpu').double() if sign else None
    actual = _correlate_dft_2d_real(spectrum, spectrum, ccorr_sign_grid=phase)
    reference = correlate_dft_2d(spectrum, spectrum)
    torch.testing.assert_close(actual, reference)
    peak = actual.flatten(-2).argmax(-1)
    assert (peak == (box // 2) * box + box // 2).all()


@pytest.mark.parametrize('channels', [False, True])
def test_real_correlation_preserves_gradients(channels):
    parts, projs = inputs(torch.float64, channels, True, batch=1)
    parts.requires_grad_(); projs.requires_grad_()
    filt = torch.rand(1, 1, 6, 5, dtype=torch.float64, requires_grad=True)
    phase = build_ccorr_sign_grid(6, 8, 'cpu').double()
    args = (parts, projs, filt)
    reference = correlate_dft_2d(*args, phase)
    actual = _correlate_dft_2d_real(*args, phase)
    expected_grads = torch.autograd.grad(reference.square().sum(), args)
    actual_grads = torch.autograd.grad(actual.square().sum(), args)
    for found, expected in zip(actual_grads, expected_grads):
        torch.testing.assert_close(found, expected, atol=1e-12, rtol=1e-12)


def test_real_correlation_rejects_invalid_channels():
    with pytest.raises(ValueError, match='channels'):
        _correlate_dft_2d_real(torch.ones(6, 5, 3), torch.ones(6, 5, 2))


@pytest.mark.parametrize('disabled', [False, True])
def test_matcher_correlation_backend_follows_config(disabled, monkeypatch):
    from cryoPARES.configs.mainConfig import main_config
    from cryoPARES.projmatching.projMatcher import ProjectionMatcher
    config = main_config.projmatching
    for key, value in dict(disable_compile_projectVol=True, disable_compile_analyze_cc=True,
                           disable_compile_correlate_dft_2d=disabled, grid_distance_degs=1.,
                           grid_step_degs=1., max_resolution_A=None).items():
        monkeypatch.setattr(config, key, value)
    original_compile = torch.compile
    selected = []

    def compile_spy(function, *args, **kwargs):
        if function is _correlate_dft_2d_real:
            selected.append(function)
            return function
        return original_compile(function, *args, **kwargs)

    monkeypatch.setattr(torch, 'compile', compile_spy)
    matcher = ProjectionMatcher(torch.zeros(8, 8, 8), pixel_size=1.5)
    assert bool(selected) is not disabled
    assert matcher.correlate_dft_2d is (correlate_dft_2d if disabled else _correlate_dft_2d_real)


@pytest.mark.gpu
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('channels', [False, True])
def test_compiled_real_correlation_matches_eager_cuda(dtype, channels):
    torch._dynamo.reset()
    try:
        compiled = torch.compile(_correlate_dft_2d_real, fullgraph=True)
        for batch, candidates in [(1, 3), (3, 3), (2, 5)]:
            parts, projs = inputs(dtype, channels, True, device='cuda', batch=batch, candidates=candidates)
            filt = torch.rand(batch, 1, 6, 5, dtype=dtype, device='cuda')
            phase = build_ccorr_sign_grid(6, 8, 'cuda').to(dtype)
            actual = compiled(parts, projs, filt, phase)
            torch.testing.assert_close(actual, correlate_dft_2d(parts, projs, filt, phase))
    finally:
        torch._dynamo.reset()
