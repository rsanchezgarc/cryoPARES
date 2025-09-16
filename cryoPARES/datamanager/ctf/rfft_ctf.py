import functools
import sys

import einops
import torch

from .common import _compute_ctf


@functools.lru_cache(1)
def _get2DFreqsRFFT(imageSize, sampling_rate, fftshift: bool, device=None):


    """RFFT frequency computation"""
    # freqs_y = torch.fft.fftshift(torch.fft.fftfreq(imageSize,d=sampling_rate))
    freqs_y = torch.fft.fftfreq(imageSize, d=sampling_rate)
    freqs_x = torch.fft.rfftfreq(imageSize, d=sampling_rate)

    if fftshift:
        freqs_y = torch.fft.fftshift(freqs_y)


    if imageSize % 2 == 0 and freqs_x.numel() == imageSize // 2 + 1:
        freqs_x = freqs_x.clone()
        freqs_x[-1] = -freqs_x[-1]

    y, x = torch.meshgrid(freqs_y, freqs_x, indexing='ij')
    freqs = torch.stack([y, x], dim=-1)

    # from matplotlib import pyplot as plt
    # f, axes = plt.subplots(nrows=1, ncols=2)
    # axes[0].imshow(freqs[...,0])
    # axes[1].imshow(freqs[...,1])
    # plt.show()

    freqs = freqs.reshape(-1, 2)

    if device is not None:
        freqs = freqs.to(device)
    return freqs


def compute_ctf_rfft(image_size, sampling_rate, dfu, dfv, dfang, volt, cs, w,
                     phase_shift, bfactor, fftshift: bool, device: torch.device):

    """Compute CTF using RFFT frequency grid
        Input:
        image_size: the side of the image
        sampling_rate: In A/pixel
        dfu (float or Bx1 tensor): DefocusU (Angstrom)
        dfv (float or Bx1 tensor): DefocusV (Angstrom)
        dfang (float or Bx1 tensor): DefocusAngle (degrees)
        volt (float or Bx1 tensor): accelerating voltage (kV)
        cs (float or Bx1 tensor): spherical aberration (mm)
        w (float or Bx1 tensor): amplitude contrast ratio
        phase_shift (float or Bx1 tensor): degrees
        bfactor (float or Bx1 tensor): envelope fcn B-factor (Angstrom^2)
        fftshift: if true, fftshift the ctf
    """
    freqs = _get2DFreqsRFFT(image_size, sampling_rate, fftshift=fftshift, device=device)
    ctf = _compute_ctf(freqs, dfu, dfv, dfang, volt, cs, w, phase_shift, bfactor)
    s1 = image_size // 2 + 1
    ctf = einops.rearrange(ctf, "... (s0 s1) -> ... s0 s1", s0=image_size, s1=s1)

    # import matplotlib.pyplot as plt
    # plt.imshow(ctf)
    # plt.show()
    return ctf


def correct_ctf(image, sampling_rate, dfu, dfv, dfang, volt, cs, w, phase_shift=0, bfactor=None, mode='phase_flip',
                fftshift=True, wiener_parameter=0.15):
    '''
    Apply the 2D CTF through phase flip or wigner filter using RFFT

    Input:
        image: a real space image
        sampling rate: Angstrom/pixel
        dfu (float or Bx1 tensor): DefocusU (Angstrom)
        dfv (float or Bx1 tensor): DefocusV (Angstrom)
        dfang (float or Bx1 tensor): DefocusAngle (degrees)
        volt (float or Bx1 tensor): accelerating voltage (kV)
        cs (float or Bx1 tensor): spherical aberration (mm)
        w (float or Bx1 tensor): amplitude contrast ratio
        phase_shift (float or Bx1 tensor): degrees
        bfactor (float or Bx1 tensor): envelope fcn B-factor (Angstrom^2)
        mode (Choice["phase_flip", "wiener"]): how to correct the ctf
        fftshift (bool): If true, fftshift will be applied (and the returned ctf will be also fftshifted)
        wiener_parameter (float):
    Output
        ctf, corrected_image: ctf is rfft, can be fftshifted or not. corrected_image is a real space image
    '''

    ctf = compute_ctf_rfft(image.shape[-2], sampling_rate, dfu, dfv, dfang, volt, cs, w, phase_shift, bfactor,
                           fftshift=fftshift, device=image.device)

    # Apply 1D fftshift only on the first dimension (rows)
    fimage = torch.fft.rfft2(image)
    if fftshift:
        fimage = torch.fft.fftshift(fimage, dim=-2)

    if mode == 'phase_flip':
        fimage_corrected = fimage * torch.sign(ctf)
    elif mode == 'wiener':
        fimage_corrected = fimage / (ctf + torch.sign(ctf) * wiener_parameter)
    else:
        raise ValueError("Only phase_flip and wiener are valid")

    # Undo the 1D fftshift before inverse transform
    if fftshift:
        fimage_corrected = torch.fft.ifftshift(fimage_corrected, dim=-2)
    image_corrected = torch.fft.irfft2(fimage_corrected, s=image.shape[-2:])

    return ctf, image_corrected


def corrupt_with_ctf(image, sampling_rate, dfu, dfv, dfang, volt, cs, w, phase_shift=0, bfactor=None, fftshift=True):
    '''
    Corrupt an image applying CTF using RFFT
    Output
        ctf, image_corrupted: ctf is rfft, can be fftshifted or not. corrected_image is a real space image
    '''
    ctf = compute_ctf_rfft(image.shape[-1], sampling_rate, dfu, dfv, dfang, volt, cs, w, phase_shift, bfactor,
                           fftshift=fftshift, device=image.device)

    # Apply 1D fftshift only on the first dimension
    fimage = torch.fft.rfft2(image)
    if fftshift:
        fimage = torch.fft.fftshift(fimage, dim=-2)

    image_corrupted = fimage * ctf

    # Undo the 1D fftshift before inverse transform
    if fftshift:
        image_corrupted = torch.fft.ifftshift(image_corrupted, dim=-2)
    image_corrupted = torch.fft.irfft2(image_corrupted, s=image.shape[-2:])

    return ctf, image_corrupted


def _test_vs_litbilt():
    img_size = 64
    fftshift = True

    ctf1 = compute_ctf_rfft(image_size=img_size, sampling_rate=1.5, dfu=4000, dfv=4000, dfang=0,
                            volt=300, cs=2.7, w=0.1,
                            phase_shift=0, bfactor=0, device="cpu", fftshift=fftshift)



    sys.path.append("/home/sanchezg/cryo/tools/libtilt/src")
    from libtilt.ctf.ctf_2d import calculate_ctf


    if fftshift:
        _ctf2 = calculate_ctf(
            defocus=0.4, astigmatism=0, astigmatism_angle=0,
            voltage=300, spherical_aberration=2.7,
            amplitude_contrast=torch.FloatTensor([0.1]), b_factor=0, phase_shift=0,
            pixel_size=1.5, image_shape=(img_size, img_size),
            rfft=False, fftshift=True, device=None,
        )

        centre = _ctf2.shape[-2] // 2
        ctf2 = torch.zeros(1, img_size, img_size//2+1)
        ctf2[..., :-1] = _ctf2[..., centre:]
        ctf2[..., -1] = _ctf2[..., 0]
    else:
        ctf2 = calculate_ctf(
            defocus=0.4, astigmatism=0, astigmatism_angle=0,
            voltage=300, spherical_aberration=2.7,
            amplitude_contrast=torch.FloatTensor([0.1]), b_factor=0, phase_shift=0,
            pixel_size=1.5, image_shape=(img_size, img_size),
            rfft=True, fftshift=False, device=None,
        )
    print(ctf1.allclose(ctf2[0,...], atol=1e-5))

    import matplotlib.pyplot as plt
    f, axes = plt.subplots(1, 3, squeeze=False)
    axes[0,0].imshow(ctf1)
    axes[0, 1].imshow(ctf2[0,...])
    axes[0, 2].imshow(ctf1-ctf2[0,...])
    plt.show()

def _test_correct():
    img_size = 64

    # ctf1 = compute_ctf_rfft(image_size=img_size, sampling_rate=1.5, dfu=4000, dfv=4000, dfang=0,
    #                         volt=300, cs=2.7, w=0.1,
    #                         phase_shift=0, bfactor=0, fftshift=True, device="cpu")

    from starstack import ParticlesStarSet
    parts = ParticlesStarSet(starFname="/home/sanchezg/cryo/data/preAlignedParticles/EMPIAR-10166/data/projections/1000proj_with_ctf.star",
                             particlesDir="/home/sanchezg/cryo/data/preAlignedParticles/EMPIAR-10166/data/")
    img, md = parts[0]
    ctf, img_corrected = correct_ctf(torch.FloatTensor(img), parts.sampling_rate, dfu=md["rlnDefocusU"],
                                dfv=md["rlnDefocusV"], dfang=md["rlnDefocusAngle"],
                                volt=parts.optics_md["rlnVoltage"][0], cs=parts.optics_md["rlnSphericalAberration"][0],
                                w=parts.optics_md["rlnAmplitudeContrast"][0], phase_shift=0, bfactor=None,
                                mode='wiener', fftshift=True, wiener_parameter=0.15)

    import matplotlib.pyplot as plt
    f, axes = plt.subplots(1, 2, squeeze=False)
    axes[0, 0].imshow(img, cmap="gray")
    axes[0, 1].imshow(img_corrected, cmap="gray")
    plt.show()

if __name__ == "__main__":
    # _get2DFreqsRFFT(imageSize=8, sampling_rate=1.5)
    _test_vs_litbilt()
    # _test_correct()