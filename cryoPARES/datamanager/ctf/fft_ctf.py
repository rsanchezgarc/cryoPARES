import functools

import einops
import torch

from .common import _compute_ctf


@functools.lru_cache(1)
def _get2DFreqs(imageSize, sampling_rate, device=None):
    freqs1d = torch.fft.fftshift(torch.fft.fftfreq(imageSize))
    freqs = torch.stack(torch.meshgrid(freqs1d, freqs1d, indexing='ij'), -1) / sampling_rate
    freqs = freqs.reshape(-1, 2)
    if device is not None:
        freqs = freqs.to(device)
    return freqs


def compute_ctf(image_size, sampling_rate, dfu, dfv, dfang, volt, cs, w, phase_shift, bfactor, device):
    freqs = _get2DFreqs(image_size, sampling_rate, device=device)
    ctf = _compute_ctf(freqs, dfu, dfv, dfang, volt, cs, w, phase_shift, bfactor)
    ctf = einops.rearrange(ctf, "... (s0 s1) -> ... s0 s1", s0=image_size, s1=image_size)
    return ctf


def correct_ctf(image, sampling_rate, dfu, dfv, dfang, volt, cs, w, phase_shift=0, bfactor=None, mode='phase_flip',
                wiener_parameter=0.15):
    '''
    Apply the 2D CTF through phase flip or wigner filter using FFT

    Input:
        image (Tensor) the DxD image in real space
        sampling_rate: in A/pixel
        dfu (float or Bx1 tensor): DefocusU (Angstrom). Positive for underfocus
        dfv (float or Bx1 tensor): DefocusV (Angstrom). Positive for underfocus
        dfang (float or Bx1 tensor): DefocusAngle (degrees)
        volt (float or Bx1 tensor): accelerating voltage (kV)
        cs (float or Bx1 tensor): spherical aberration (mm)
        w (float or Bx1 tensor): amplitude contrast ratio
        phase_shift (float or Bx1 tensor): degrees
        bfactor (float or Bx1 tensor): envelope fcn B-factor (Angstrom^2)
        mode (string): CTF correction: 'phase_flip' or 'wiener'
        wiener_parameter (float): wiener parameter for not dividing by zero
    '''
    ctf = compute_ctf(image.shape[-1], sampling_rate, dfu, dfv, dfang, volt, cs, w,
                      phase_shift, bfactor, device=image.device)

    fimage = torch.fft.fftshift(torch.fft.fft2(image), dim=(-2, -1))

    if mode == 'phase_flip':
        fimage_corrected = fimage * torch.sign(ctf)
    elif mode == 'wiener':
        fimage_corrected = fimage / (ctf + torch.sign(ctf) * wiener_parameter)
    else:
        raise ValueError("Only phase_flip and wiener are valid")

    image_corrected = torch.real(torch.fft.ifft2(torch.fft.ifftshift(fimage_corrected, dim=(-2, -1))))
    return ctf, image_corrected


def corrupt_with_ctf(image, sampling_rate, dfu, dfv, dfang, volt, cs, w, phase_shift=0, bfactor=None):
    '''
    Corrupt an image applying CTF using FFT
    '''
    ctf = compute_ctf(image.shape[-1], sampling_rate, dfu, dfv, dfang, volt, cs, w,
                      phase_shift, bfactor, device=image.device)

    fimage = torch.fft.fftshift(torch.fft.fft2(image), dim=(-2, -1))
    image_corrupted = fimage * ctf
    image_corrupted = torch.real(torch.fft.ifft2(torch.fft.ifftshift(image_corrupted, dim=(-2, -1))))

    return ctf, image_corrupted
