import warnings
from functools import lru_cache

import torch
from torch_grid_utils import circle


def _fourier_proj_to_real_2d(projections, pad_length):
    if not projections.is_complex():
        projections = torch.view_as_complex(projections.contiguous())
    projections = torch.fft.ifftshift(projections, dim=(-2,))  # ifftshift of rfft
    projections = torch.fft.irfftn(projections, dim=(-2, -1))
    projections = torch.fft.ifftshift(projections, dim=(-2, -1))  # recenter real space

    if pad_length is not None:
        projections = projections[..., pad_length: -pad_length, pad_length: -pad_length]
    return projections


def _real_to_fourier_2d(imgs, as_real_img=False, radius_px=None):

    mask = _get_real_mask(imgs.shape, radius_px=radius_px, device=imgs.device)
    imgs *= mask
    imgs = torch.fft.fftshift(imgs, dim=(-2, -1))
    imgs = torch.fft.rfftn(imgs, dim=(-2, -1))
    imgs = torch.fft.fftshift(imgs, dim=(-2,))
    if as_real_img:
        imgs = torch.view_as_real(imgs)
    return imgs


@lru_cache(1)
def _mask_for_dft_2d(img_shape, max_freq_pixels, rfft, fftshifted, device):

    img_size = img_shape[-2]
    if max_freq_pixels is None:
        max_freq_pixels = img_size // 2
    mask = circle(radius=max_freq_pixels,
        smoothing_radius=3,
        image_shape=(img_size, img_size),
        device=device
    ).unsqueeze(0)

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

@lru_cache(1)
def _get_real_mask(particle_shape, radius_px=None, device=None):
    if radius_px is None:
        radius_px = particle_shape[-2] // 2
    return circle(radius_px, image_shape=particle_shape, smoothing_radius=radius_px * .05).to(device)


if __name__ == "__main__":

    _mask_for_dft_2d((256, 256), max_freq_pixels=27, rfft=True, fftshifted=True, device="cpu")