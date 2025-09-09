import torch
from typing import Literal, get_args

from cryoPARES.utils.reconstructionUtils import get_vol, write_vol


def _raised_cosine_filter(in_shape, freq_or_res, delta=0.01, sampling_rate=None):
    """
    A raised cosine filter with the input shape with.

    :param in_shape: The shape of the input (can be 2D, 3D, etc.).
    :param freq_or_res: The cutoff frequency in normalized units (0.5 corresponds to Nyquist) if sampling_rate=None or
                        the cutoff resolution in Å otherwise.
    :param delta: The extension of the filter in normalized units if sampling_rate=None or in Å otherwise.
    :param sampling_rate: The sampling rate for freq and delta.
    :return: The filter.
    """
    if sampling_rate is not None:  # Then we have been provided a resolution value
        assert freq_or_res >= 2 * sampling_rate, "Error, filtering resolution has to be larger than Nyquist"
        freq = sampling_rate / freq_or_res
        delta = sampling_rate / (freq_or_res - delta) - freq
    else:
        freq = freq_or_res
        assert 0 < freq < 0.5

    fcutoff = freq + delta
    grid = torch.stack(torch.meshgrid([torch.fft.fftshift(torch.fft.fftfreq(size)) for size in in_shape], indexing="ij"), dim=-1)
    # grid = torch.stack(torch.meshgrid([torch.linspace(-0.5, 0.5, size) for size in in_shape], indexing="ij"), dim=-1)
    r = torch.sqrt((grid ** 2).sum(-1))
    filt = (r - freq)
    filt = (torch.cos(2 * torch.pi * filt / (delta * 2)) + 1) / 2

    filt[r < freq] = 1
    filt[r > fcutoff] = 0

    return filt


def _apply_filter(img, filt):
    """
    Apply the filter to the image.

    :param img: The image to be filtered. Can be ndimensional.
    :param filt: The filter to be applied.
    :return: The filtered image.
    """
    assert (img.shape == filt.shape)
    fImg = torch.fft.fftshift(torch.fft.fftn(img))
    im2 = fImg * filt
    return torch.fft.ifftn(torch.fft.ifftshift(im2)).real

def low_pass_filter(img, resolution, sampling_rate, mode: Literal["raised_cos"] = "raised_cos",
                    delta_resolution: float = 0.01):
    """
    Apply a low-pass filter to the image.

    :param img: The input image.
    :param resolution: The desired resolution in Å.
    :param sampling_rate: The sampling rate in Å.
    :param mode: The filtering mode.
    :param delta_resolution: Only applies to mode="raised_cos". The last resolution with signal will be
                             resolution+delta_resolution. In Å
    :return: The filtered image.
    """
    if mode == "raised_cos":
        filt = _raised_cosine_filter(img.shape, freq_or_res=resolution, delta=delta_resolution,
                                     sampling_rate=sampling_rate)
        imgF = _apply_filter(img, filt)
    else:
        raise ValueError(f"Error, no valid filtering mode {mode}")
    return imgF


delta_resolution_mode_typeHint = Literal["Angstroms", "digital", "n_shells"]
def low_pass_filter_fname(vol_fname, resolution, out_fname: str, mode: Literal["raised_cos"] = "raised_cos",
                    delta_resolution: float = 2.,
                    delta_resolution_mode: delta_resolution_mode_typeHint = "n_shells"):
    """

    :param vol_fname:
    :param resolution:
    :param out_fname:
    :param mode:
    :param delta_resolution:
    :param delta_resolution_mode:
    :return:
    """

    vol, pixel_size = get_vol(vol_fname, pixel_size=None)

    #We delta_resolution resolution expressed in Angstroms
    if delta_resolution_mode == "digital":
        assert delta_resolution < 0.5, "Error, delta_resolution is beyond nyquist"
        freq = pixel_size / resolution
        delta_resolution = resolution - pixel_size / ((freq + delta_resolution))
    elif delta_resolution_mode == "n_shells":
        assert delta_resolution >= 1
        per_pixel_freq = delta_resolution / (2 * vol.shape[-1])
        # freq = np.interp([resolution], (pixel_size / torch.fft.rfftfreq(vol.shape[-1])).flip(-1), torch.fft.rfftfreq(vol.shape[-1]).flip(-1))
        freq = pixel_size / resolution
        delta_resolution = resolution - pixel_size / (freq+delta_resolution*per_pixel_freq)
    elif delta_resolution_mode != "Angstroms":
        raise NotImplementedError(f"Error, no valid delta_resolution_mode {delta_resolution_mode}. Only allowed: {get_args(delta_resolution_mode_typeHint)}")
    vol = low_pass_filter(vol, resolution, sampling_rate=vol.pixel_size, mode=mode, delta_resolution=delta_resolution)
    vol.pixel_size = pixel_size
    write_vol(vol, out_fname, pixel_size=pixel_size)
    return vol, (resolution - delta_resolution)


def _test():
    for resolution in [4,5,6,7,8]:
        low_pass_filter_fname(vol_fname="/home/sanchezg/tmp/emd_0520.map", resolution=resolution,
                              out_fname=f"/tmp/filtered_map{resolution}.mrc")

if __name__ == "__main__":
    _test()