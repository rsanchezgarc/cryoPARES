from functools import lru_cache
from typing import Optional, Sequence, Union

import numpy as np
import torch

from libtilt.shapes import circle

from libtilt.projection.project_fourier import _compute_dft
from ..configs.mainConfig import main_config

compute_dft = _compute_dft


@lru_cache(1)
def _mask_for_dft_2d(img_shape, max_freq_pixels, rfft, fftshifted, device):

    img_size = img_shape[-2]
    if max_freq_pixels is None:
        max_freq_pixels = img_size // 2
    mask = circle(radius=max_freq_pixels,
        smoothing_radius=3,
        image_shape=(img_size, img_size),
        device="cpu" #TODO: At the moment device is broken, thus we set it to "cpu" and then we use .to(img)
    ).unsqueeze(0).to(device)

    # filer_digital_freq = max_freq_pixels/img_size
    # delta = 4/img_size
    # raised_cosine_filter = _raised_cosine_filter([img_size, img_size], freq_or_res=filer_digital_freq, delta=delta, sampling_rate=None)
    # raised_cosine_filter = raised_cosine_filter.unsqueeze(0).to(device)
    if rfft:
        centre = img_size // 2
        last_freq = mask[..., 0:1]
        begining = mask[..., centre:]
        mask = torch.concatenate([begining, last_freq], dim=-1)


    # import matplotlib.pyplot as plt
    # f, axes = plt.subplots(1, 2)
    # axes[0].imshow(mask.cpu().abs().numpy()[0,...])
    # # axes[1].imshow(raised_cosine_filter.cpu().abs().numpy()[0,...])
    # plt.show()

    if not fftshifted:
        raise NotImplementedError()

    return mask

def correlate_dft_2d(
    a: torch.Tensor,
    b: torch.Tensor,
    max_freq_pixels: Optional[Union[int, Sequence[int]]] = None
) -> torch.Tensor:
    """Correlate fftshifted rfft discrete Fourier transforms of images"""
    #TODO: add Alister newcode to limit the resolution range


    result = a * torch.conj(b)
    if max_freq_pixels is not None:  # TODO: use alister trick to project only the desired frequencies
        mask = _mask_for_dft_2d(result.shape, max_freq_pixels, rfft=True, fftshifted=True,
                            device=result.device)
        result *= mask
    result = torch.fft.ifftshift(result, dim=(-2,))
    result = torch.fft.irfftn(result, dim=(-2, -1))
    result = torch.fft.ifftshift(result, dim=(-2, -1))
    return torch.real(result)


if not main_config.projmatching.disable_compile_projectVol:
    correlate_dft_2d = torch.compile(correlate_dft_2d, mode=main_config.projmatching.compile_correlate_dft_2d_mode)
