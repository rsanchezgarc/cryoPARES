import functools

import einops
import torch

from .common import _compute_ctf


@functools.lru_cache(1)
def _get2DFreqsRFFT(imageSize, sampling_rate, device=None):
    """RFFT frequency computation"""
    freqs_y = torch.fft.fftshift(torch.fft.fftfreq(imageSize))
    freqs_x = torch.fft.fftfreq(imageSize)[:imageSize // 2 + 1]

    y, x = torch.meshgrid(freqs_y, freqs_x, indexing='ij')
    freqs = torch.stack([x, y], dim=-1) / sampling_rate
    freqs = freqs.reshape(-1, 2)

    if device is not None:
        freqs = freqs.to(device)
    return freqs


def compute_ctf(image_size, sampling_rate, dfu, dfv, dfang, volt, cs, w, phase_shift, bfactor, device):
    """Compute CTF using RFFT frequency grid"""
    freqs = _get2DFreqsRFFT(image_size, sampling_rate, device=device)
    ctf = _compute_ctf(freqs, dfu, dfv, dfang, volt, cs, w, phase_shift, bfactor)
    ctf = einops.rearrange(ctf, "... (s0 s1) -> ... s0 s1", s0=image_size, s1=image_size // 2 + 1)
    return ctf


def correct_ctf(image, sampling_rate, dfu, dfv, dfang, volt, cs, w, phase_shift=0, bfactor=None, mode='phase_flip',
                wiener_parameter=0.15):
    '''
    Apply the 2D CTF through phase flip or wigner filter using RFFT

    Input parameters same as FFT version
    '''

    ctf = compute_ctf(image.shape[-1], sampling_rate, dfu, dfv, dfang, volt, cs, w,
                           phase_shift, bfactor, device=image.device)

    # Apply 1D fftshift only on the first dimension (rows)
    fimage = torch.fft.rfft2(image)
    fimage = torch.fft.fftshift(fimage, dim=-2)

    if mode == 'phase_flip':
        fimage_corrected = fimage * torch.sign(ctf)
    elif mode == 'wiener':
        fimage_corrected = fimage / (ctf + torch.sign(ctf) * wiener_parameter)
    else:
        raise ValueError("Only phase_flip and wiener are valid")

    # Undo the 1D fftshift before inverse transform
    fimage_corrected = torch.fft.ifftshift(fimage_corrected, dim=-2)
    image_corrected = torch.fft.irfft2(fimage_corrected, s=image.shape[-2:])

    return ctf, image_corrected


def corrupt_with_ctf(image, sampling_rate, dfu, dfv, dfang, volt, cs, w, phase_shift=0, bfactor=None):
    '''
    Corrupt an image applying CTF using RFFT
    '''
    ctf = compute_ctf(image.shape[-1], sampling_rate, dfu, dfv, dfang, volt, cs, w,
                           phase_shift, bfactor, device=image.device)

    # Apply 1D fftshift only on the first dimension
    fimage = torch.fft.rfft2(image)
    fimage = torch.fft.fftshift(fimage, dim=-2)

    image_corrupted = fimage * ctf

    # Undo the 1D fftshift before inverse transform
    image_corrupted = torch.fft.ifftshift(image_corrupted, dim=-2)
    image_corrupted = torch.fft.irfft2(image_corrupted, s=image.shape[-2:])

    return ctf, image_corrupted
