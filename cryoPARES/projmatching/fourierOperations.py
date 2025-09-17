from functools import lru_cache
from typing import Optional, Sequence, Union, Tuple

import torch
import torch.nn.functional as F
import numpy as np

from torch_grid_utils import fftfreq_grid
from torch_grid_utils.shapes_2d import circle

from ..configs.mainConfig import main_config


def compute_dft_3d(
    volume: torch.Tensor,
    pad: bool = True,
    pad_length: int | None = None
) -> Tuple[torch.Tensor, Tuple[int,int,int], int | None]:
    """Computes the DFT of a volume. Intended to be used as a preprocessing before using extract_central_slices_rfft.

    Parameters
    ----------
    volume: torch.Tensor
        `(d, d, d)` volume.
    pad: bool
        Whether to pad the volume with zeros to increase sampling in the DFT.
    pad_length: int | None
        The length used for padding each side of each dimension. If pad_length=None, and pad=True then volume.shape[-1] // 2 is used instead

    Returns
    -------
    projections: Tuple[torch.Tensor, torch.Tensor, int]
        `(..., d, d, d)` dft of the volume. fftshifted rfft
        Tuple[int,int,int] the shape of the volume after padding
        int with the padding length
    """
    # padding
    if pad is True:
        if pad_length is None:
            pad_length = volume.shape[-1] // 2
        volume = F.pad(volume, pad=[pad_length] * 6, mode='constant', value=0)

    vol_shape = tuple(volume.shape)
    # premultiply by sinc2
    grid = fftfreq_grid(
        image_shape=vol_shape,
        rfft=False,
        fftshift=True,
        norm=True,
        device=volume.device
    )
    volume = volume * torch.sinc(grid) ** 2

    # calculate DFT
    dft = torch.fft.fftshift(volume, dim=(-3, -2, -1))  # volume center to array origin
    dft = torch.fft.rfftn(dft, dim=(-3, -2, -1))
    dft = torch.fft.fftshift(dft, dim=(-3, -2,))  # actual fftshift of rfft

    return dft, vol_shape, pad_length


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
        beginning = mask[..., centre:]
        mask = torch.concatenate([beginning, last_freq], dim=-1)


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


def _compute_one_batch_fft(imgs): #TODO: Is it worth it to compile it?
    imgs = torch.fft.fftshift(imgs, dim=(-2, -1))
    imgs = torch.fft.rfftn(imgs, dim=(-2, -1))
    imgs = torch.fft.fftshift(imgs, dim=(-2,))
    # imgs = imgs / imgs.abs().sum() #Normalization
    return imgs

if not main_config.projmatching.disable_compile_projectVol:
    correlate_dft_2d = torch.compile(correlate_dft_2d, mode=main_config.projmatching.compile_correlate_dft_2d_mode)
