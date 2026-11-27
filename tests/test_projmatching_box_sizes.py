"""Unsupported odd boxes must fail before matching geometry is constructed."""

import importlib

import mrcfile
import pytest
import torch


@pytest.mark.parametrize('box', [5, 9, 33])
@pytest.mark.parametrize('source', ['tensor', 'file'])
@pytest.mark.parametrize('resolution', [None, 6.])
def test_odd_reference_box_is_rejected_before_projection_fft(box, source, resolution, tmp_path, monkeypatch):
    module = importlib.import_module('cryoPARES.projmatching.projMatcher')
    from cryoPARES.configs.mainConfig import main_config

    # The constructor's decorator writes explicit arguments back to the shared
    # config before odd-box validation raises. Scope test settings with
    # monkeypatch so later, lazily imported inference classes see the originals.
    for key, value in dict(grid_distance_degs=1.0, grid_step_degs=1.0,
                           max_resolution_A=resolution, verbose=False).items():
        monkeypatch.setattr(main_config.projmatching, key, value)
    volume = torch.zeros(box, box, box)
    reference = volume
    if source == 'file':
        reference = str(tmp_path / 'odd.mrc')
        mrcfile.write(reference, volume.numpy(), voxel_size=1.5)

    def forbidden(*args, **kwargs):
        pytest.fail('An odd box reached Fourier projection preparation')

    monkeypatch.setattr(module, 'compute_dft_3d', forbidden)
    with pytest.raises(ValueError, match='requires even box sizes'):
        module.ProjectionMatcher(reference, pixel_size=1.5)
