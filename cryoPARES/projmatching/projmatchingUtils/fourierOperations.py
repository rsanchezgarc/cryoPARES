from functools import lru_cache
from typing import Optional, Sequence, Union, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from sympy.abc import alpha

from cryoPARES.utils.torch_grid_utils_compat import fftfreq_grid, circle

from cryoPARES.configs.mainConfig import main_config

def _tri_kernel_1d(u: torch.Tensor, nyquist: float = 0.5) -> torch.Tensor:
    """
    Triangle kernel in frequency with support [-nyquist, +nyquist].
    Peak 1 at 0; goes to 0 at ±nyquist. 'u' is in cycles/pixel.
    """
    return torch.clamp(1.0 - (u.abs() / nyquist), min=0.0)

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
    # grid = fftfreq_grid(
    #     image_shape=vol_shape,
    #     rfft=False,
    #     fftshift=True,
    #     norm=True,
    #     device=volume.device
    # )
    # volume = volume * torch.sinc(grid) ** 2

    # calculate DFT
    dft = torch.fft.fftshift(volume, dim=(-3, -2, -1))  # volume center to array origin
    dft = torch.fft.rfftn(dft, dim=(-3, -2, -1))
    dft = torch.fft.fftshift(dft, dim=(-3, -2,))  # actual fftshift of rfft

    grid = fftfreq_grid(
        image_shape=vol_shape,  # pre-FFT real-space shape
        rfft=True,  # last axis is rFFT (0..Nyquist)
        fftshift=True,  # z,y are shifted (as above)
        norm=False,
        device=volume.device,
    )
    kz, ky, kx = grid[..., 0], grid[..., 1], grid[..., 2]  # cycles/pixel

    # Separable triangle kernel per axis (models trilinear in k)
    eps = 1e-3
    alpha = 0.5
    Kz = _tri_kernel_1d(kz)
    Ky = _tri_kernel_1d(ky)
    Kx = _tri_kernel_1d(kx)
    K = (Kz * Ky * Kx).clamp_min(eps)

    # Gentle inverse: divide by K^alpha
    dft = dft / K.pow(alpha)

    return dft, vol_shape, pad_length


@lru_cache(maxsize=4)
def _mask_for_dft_2d(img_shape, max_freq_pixels, min_freq_pixels, rfft, fftshifted, device):
    """Build a 2D Fourier band-pass mask in fftshifted rfft layout.

    Parameters
    ----------
    max_freq_pixels:
        Outer (low-pass) radius in pixels. None → use full half-box.
    min_freq_pixels:
        Inner (high-pass) radius in pixels. 0 or None → no high-pass.
    rfft:
        If True, return only the non-redundant rfft half (W//2+1 columns).
    fftshifted:
        Must be True (non-shifted layout not implemented).
    """
    img_size = img_shape[-2]
    if max_freq_pixels is None:
        max_freq_pixels = img_size // 2
    mask = circle(radius=max_freq_pixels,
        smoothing_radius=3,
        image_shape=(img_size, img_size),
        device=device,
    ).unsqueeze(0).to(device)

    if min_freq_pixels is not None and min_freq_pixels > 0:
        inner = circle(radius=min_freq_pixels,
                       smoothing_radius=2,
                       image_shape=(img_size, img_size),
                       device=device,
                       ).unsqueeze(0).to(device)
        mask = mask * (1 - inner)

    if rfft:
        centre = img_size // 2
        last_freq = mask[..., 0:1]
        beginning = mask[..., centre:]
        mask = torch.concatenate([beginning, last_freq], dim=-1)

    if not fftshifted:
        raise NotImplementedError()

    return mask

def correlate_dft_2d(
    parts: torch.Tensor,
    projs: torch.Tensor,
    zero_dc: bool = False,
    whitening_filter: torch.Tensor | None = None,
) -> torch.Tensor:
    """Correlate fftshifted rfft discrete Fourier transforms of images.

    Parameters
    ----------
    parts:
        Particle DFTs, fftshifted rfft, shape (..., H, W//2+1) complex or (..., H, W//2+1, 2).
    projs:
        Projection DFTs, same layout as parts.
    zero_dc:
        If True, zero the DC bin before computing the cross-product to remove low-frequency bias.
        DC is at row H//2, col 0 in the fftshifted rfft layout.
    whitening_filter:
        Optional real-valued tensor broadcastable to parts/projs shape. Applied as a multiply
        before the cross-product (spectral whitening). Pre-computed in ProjectionMatcher.__init__.
    """
    if not parts.is_complex():
        parts = torch.view_as_complex(parts.contiguous())
    if not projs.is_complex():
        projs = torch.view_as_complex(projs.contiguous())

    if zero_dc:
        # DC bin: row H//2, col 0 in fftshifted rfft layout
        dc_row = parts.shape[-2] // 2
        parts = parts.clone()
        projs = projs.clone()
        parts[..., dc_row, 0] = 0
        projs[..., dc_row, 0] = 0

    if whitening_filter is not None:
        # Apply whitening only to projections (templates), not to particle data.
        # Whitening both sides gives 1/amp² weighting, which can amplify particle noise.
        # Whitening only projections gives 1/amp weighting and is more robust.
        projs = projs * whitening_filter

    result = parts * torch.conj(projs)
    result = torch.fft.ifftshift(result, dim=(-2,))
    result = torch.fft.irfftn(result, dim=(-2, -1))
    result = torch.fft.ifftshift(result, dim=(-2, -1))
    return torch.real(result)

def _build_whitening_map_2d(
    vol_rfft: torch.Tensor,
    img_size: int,
    fftfreq_max: float = 0.5,
) -> torch.Tensor:
    """Build a 2D spectral whitening filter from a 3D rfft volume.

    Computes the radial amplitude spectrum of vol_rfft, then evaluates
    1/sqrt(spectrum + eps) at each 2D frequency to produce a whitening map
    for fftshifted rfft images.

    Parameters
    ----------
    vol_rfft:
        Complex 3D rfft of the reference volume, shape (D, D, D//2+1). May also
        be the 2-float real view in (2, D, D, D//2+1) layout — handled automatically.
    img_size:
        Output image size (= volume box size). Result is shape (1, img_size, img_size//2+1).

    Returns
    -------
    whitening_map: torch.Tensor
        Float32 tensor of shape (1, img_size, img_size//2+1).
    """
    import math

    # Normalise to complex if stored as two float32 channels
    if not vol_rfft.is_complex():
        # shape (2, D, D, D//2+1) → (D, D, D//2+1) complex
        vol_rfft = torch.view_as_complex(vol_rfft.permute(1, 2, 3, 0).contiguous())

    # --- radial amplitude spectrum from 3D volume ---
    amp = vol_rfft.abs()  # (D, D, D//2+1)
    D = vol_rfft.shape[0]
    n_bins = D // 2

    # Build 3D frequency radii (in pixels)
    kz = torch.fft.fftshift(torch.fft.fftfreq(D, device=amp.device)) * D   # (D,)
    ky = kz.clone()
    kx = torch.arange(D // 2 + 1, device=amp.device, dtype=torch.float32)  # 0..D//2
    KZ, KY, KX = torch.meshgrid(kz, ky, kx, indexing='ij')
    radii_3d = torch.sqrt(KZ**2 + KY**2 + KX**2)  # (D, D, D//2+1)

    radii_int = radii_3d.long().clamp(0, n_bins - 1)
    spectrum = torch.zeros(n_bins, device=amp.device, dtype=torch.float32)
    counts = torch.zeros(n_bins, device=amp.device, dtype=torch.float32)
    spectrum.scatter_add_(0, radii_int.reshape(-1), amp.reshape(-1).float())
    counts.scatter_add_(0, radii_int.reshape(-1), torch.ones_like(amp.reshape(-1)))
    counts.clamp_(min=1)
    spectrum = spectrum / counts  # mean amplitude per shell

    # --- build 2D whitening map (fftshifted rfft layout) ---
    ky2d = torch.fft.fftshift(torch.fft.fftfreq(img_size, device=amp.device)) * img_size
    kx2d = torch.arange(img_size // 2 + 1, device=amp.device, dtype=torch.float32)
    KY2D, KX2D = torch.meshgrid(ky2d, kx2d, indexing='ij')  # (img_size, img_size//2+1)
    radii_2d = torch.sqrt(KY2D**2 + KX2D**2).long().clamp(0, n_bins - 1)

    s = spectrum[radii_2d]  # (img_size, img_size//2+1)
    eps = s.mean() * 1e-3 + 1e-8
    whitening_map = 1.0 / torch.sqrt(s + eps)

    # Zero out whitening beyond the resolution cutoff to avoid boosting high-frequency noise
    if fftfreq_max < 0.5:
        max_px = fftfreq_max * img_size
        KY2D_f = torch.fft.fftshift(torch.fft.fftfreq(img_size, device=amp.device)) * img_size
        KX2D_f = torch.arange(img_size // 2 + 1, device=amp.device, dtype=torch.float32)
        KY2D_f, KX2D_f = torch.meshgrid(KY2D_f, KX2D_f, indexing='ij')
        radii_2d_f = torch.sqrt(KY2D_f**2 + KX2D_f**2)
        within_band = (radii_2d_f <= max_px).float()
        whitening_map = whitening_map * within_band

    return whitening_map.unsqueeze(0)  # (1, img_size, img_size//2+1)


def _real_to_fourier_2d(imgs, view_complex_as_two_float32=False):
    imgs = torch.fft.fftshift(imgs, dim=(-2, -1))
    imgs = torch.fft.rfftn(imgs, dim=(-2, -1))
    imgs = torch.fft.fftshift(imgs, dim=(-2,))
    # imgs = imgs / imgs.abs().sum() #Normalization
    if view_complex_as_two_float32:
        imgs = torch.view_as_real(imgs)
    return imgs

def _fourier_proj_to_real_2d(projections, pad_length):
    if not projections.is_complex():
        projections = torch.view_as_complex(projections.contiguous())
    projections = torch.fft.ifftshift(projections, dim=(-2,))  # ifftshift of rfft
    projections = torch.fft.irfftn(projections, dim=(-2, -1))
    projections = torch.fft.ifftshift(projections, dim=(-2, -1))  # recenter real space

    if pad_length is not None:
        projections = projections[..., pad_length: -pad_length, pad_length: -pad_length]
    return projections