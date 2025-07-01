import warnings

import torch


def _fourier_proj_to_real_2d(projections, pad_length):
    if not projections.is_complex():
        projections = torch.view_as_complex(projections.contiguous())
    projections = torch.fft.ifftshift(projections, dim=(-2,))  # ifftshift of rfft
    projections = torch.fft.irfftn(projections, dim=(-2, -1))
    projections = torch.fft.ifftshift(projections, dim=(-2, -1))  # recenter real space

    if pad_length is not None:
        projections = projections[..., pad_length: -pad_length, pad_length: -pad_length]
    return projections


def _real_to_fourier_2d(imgs, as_real_img=False):
    imgs = torch.fft.fftshift(imgs, dim=(-2, -1))
    imgs = torch.fft.rfftn(imgs, dim=(-2, -1))
    imgs = torch.fft.fftshift(imgs, dim=(-2,))
    if as_real_img:
        imgs = torch.view_as_real(imgs)
    return imgs