import math
import time
from functools import lru_cache, cached_property
from typing import Tuple, Optional, Union, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_fourier_slice import extract_central_slices_rfft_3d
from torch_grid_utils import fftfreq_grid
from torch_grid_utils.shapes_2d import circle #TODO we can use other things to limit frequencies

from cryoPARES.geometry.convert_angles import euler_angles_to_matrix
from cryoPARES.geometry.grids import so3_near_identity_grid_cartesianprod
from cryoPARES.utils.paths import MAP_AS_ARRAY_OR_FNAME_TYPE
from cryoPARES.utils.reconstructionUtils import get_vol
from cryoPARES.constants import RELION_EULER_CONVENTION, BATCH_PARTICLES_NAME, BATCH_POSE_NAME


class ProjectionMatcher(nn.Module):
    def __init__(self, reference_vol: MAP_AS_ARRAY_OR_FNAME_TYPE,
                 grid_distance_degs: float,
                 grid_step_degs: float,
                 pixel_size: Optional[float] = None,
                 filter_resolution_angst: Optional[float] = None,
                 max_shift_fraction: float = 0.2, #TODO: Add this to CONFIG,
                 return_top_k: int = 2
                 ): #TODO: Add downsampling of the volume

        super().__init__()

        self.grid_distance_degs = grid_distance_degs
        self.grid_step_degs = grid_step_degs
        self.filter_resolution_angst = filter_resolution_angst
        self.max_shift_fraction = max_shift_fraction
        self.return_top_k = return_top_k

        self.ori_vol_shape = None
        self.ori_image_shape = None
        self.vol_voxel_size = None


        self._store_reference_vol(reference_vol, pixel_size, filter_resolution_angst)
        self._store_so3_grid_rotmats()

        self._correlateF = self._correlateCrossCorrelation


    def _store_reference_vol(self, reference_vol: MAP_AS_ARRAY_OR_FNAME_TYPE,
                             pixel_size: Optional[float] = None,
                             filter_resolution_angst: Optional[float] = None):
        """

        :param reference_vol: A volume fname or torch tensor or numpy array representing a cryoEM volume
        :param pixel_size: The sampling rate in Å/px

        stores the reference_vol as a tensor and sets the following attributes: self.ori_vol_shape,
        and self.ori_image_shape, that contain information about the original volume; and  self.vol_shape  and
        self.image_shape that contain information about the volume after padding or other transformation

        """

        reference_vol, pixel_size = get_vol(reference_vol, pixel_size)
        assert pixel_size
        # reference_vol = reference_vol - reference_vol.mean()

        self.ori_vol_shape = tuple(reference_vol.shape)

        assert len(self.ori_vol_shape) == 3
        assert len(set(self.ori_vol_shape)) == 1, "Only cubic volumes are allowed"
        assert set(self.ori_vol_shape).pop() % 2 == 0, "Only even boxsizes are allowed"

        self.ori_image_shape = tuple(self.ori_vol_shape[-2:])
        self.half_particle_size = 0.5 * self.ori_image_shape[-2]
        self.vol_voxel_size = pixel_size
        reference_vol = torch.as_tensor(reference_vol.numpy(), device=reference_vol.device, dtype=reference_vol.dtype)

        reference_vol, vol_shape, pad_length = compute_dft(reference_vol, pad=False) # reference_vol is computed with rfft=True, fftshift=True,
        self.pad_length = pad_length

        #TODO: apply raised cosine filter to limit the resolution. OR PERHAPS IT IS NOT NEEDED TO THE NEW API of the fourier projector

        self.register_buffer("reference_vol", reference_vol)

        # This two properties will need to be updated if the volume is resized/padded
        self.vol_shape = vol_shape
        self.image_shape = vol_shape[-2:]

        self.fftfreq_max = None
        if self.filter_resolution_angst:
            self.fftfreq_max = self.vol_voxel_size/self.filter_resolution_angst

    def _store_so3_grid_rotmats(self):
        n_angles = math.ceil(self.grid_distance_degs / (self.grid_step_degs))
        n_angles = n_angles if n_angles % 2 == 1 else n_angles + 1 #We always want an odd number
        so3_local_grid = so3_near_identity_grid_cartesianprod(self.grid_distance_degs/2, n_angles, transposed=False)
        so3_local_grid = torch.deg2rad(so3_local_grid)
        grid_rotmats = euler_angles_to_matrix(so3_local_grid, convention=RELION_EULER_CONVENTION)
        self.register_buffer("grid_rotmats", grid_rotmats)


    @property
    def device(self):
        return self.reference_vol.device

    def _fourier_proj_to_real(self, projections):
        projections = torch.fft.ifftshift(projections, dim=(-2,))  # ifftshift of rfft
        projections = torch.fft.irfftn(projections, dim=(-2, -1))
        projections = torch.fft.ifftshift(projections, dim=(-2, -1))  # recenter real space

        if self.pad_length is not None:
            projections = projections[..., self.pad_length: -self.pad_length, self.pad_length: -self.pad_length]
        return projections

    def _real_to_fourier(self, imgs):
        imgs = torch.fft.fftshift(imgs, dim=(-2, -1))
        imgs = torch.fft.rfftn(imgs, dim=(-2, -1))
        imgs = torch.fft.fftshift(imgs, dim=(-2,))
        return imgs

    def _apply_ctfF(self, projs, ctf):
        proj_ctfed = projs * ctf
        return proj_ctfed

    def _projectF(self, rotmats):
        projs = extract_central_slices_rfft_3d(self.reference_vol, self.ori_vol_shape, rotation_matrices=rotmats,
                                               fftfreq_max=self.fftfreq_max)
        return projs

    def _correlateCrossCorrelation(self, parts: torch.Tensor, projs: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:

        # #We assume the particles have been already normalized
        # parts = self._normalize_fourier_img(parts)
        # projs = self._normalize_fourier_img(projs)

        corrs = correlate_dft_2d(parts, projs)

        # corrs = torch.fft.ifftshift(_correlate_dft_2d(parts, projs, rfft=True, fftshifted=True), dim=(-2,-1))
        # from matplotlib import pyplot as plt
        # f, axes = plt.subplots(2,1)
        # # axes[0].imshow(corr_dft2d[0].abs().log().cpu(), cmap="gray")
        # axes[1].imshow(corrs[0].abs().log().cpu(), cmap="gray"); plt.show()

        perImgCorr, pixelShiftsXY = self._extract_ccor_max(corrs)

        # pixel_shift_frac = pixelShiftsXY/parts.shape[1]
        # shifted_proj = self._shiftF(projs, pixel_shift_frac)
        # # logprob = RelionProb(projs.shape[-2:], rfft=True, fftshift=True).to(projs.device).get_logprob(parts, shifted_proj)
        #
        # from matplotlib import pyplot as plt
        # f, axes = plt.subplots(1,3)
        # _shifted_proj_real = _fourier_proj_to_real(shifted_proj[0]).cpu()
        # _proj_real = _fourier_proj_to_real(projs[0]).cpu()
        # _part_real = _fourier_proj_to_real(parts[0]).cpu()
        # from scipy.stats import pearsonr
        # pcc = pearsonr(_shifted_proj_real.flatten(), _part_real.flatten())
        # print(pcc)
        # axes[0].imshow(_shifted_proj_real, cmap="gray")
        # axes[1].imshow(_proj_real, cmap="gray")
        # axes[2].imshow(_part_real, cmap="gray")
        # plt.show()
        # breakpoint()
        #TODO: Once we have estimated the shifts, we can compute a more accurate error function
        # perImgCorr = perImgCorr / (parts - projs).abs().mean((-2,-1)) #np.prod(parts.shape[1:]) #This is the temperature for the softmax

        return perImgCorr, pixelShiftsXY

    def _extract_ccor_max_unconstrained(self, corrs):
        pixelShiftsXY = torch.empty(corrs.shape[0], 2, device=corrs.device, dtype=torch.int64)
        maxCorrsJ, maxIndxJ = corrs.max(-1)
        perImgCorr, maxIndxI = maxCorrsJ.max(-1)
        pixelShiftsXY[:, 1] = maxIndxI
        pixelShiftsXY[:, 0] = torch.gather(maxIndxJ, 1, maxIndxI.unsqueeze(1)).squeeze(1) # maxIndxJ[torch.arange(maxIndxI.shape[0]), maxIndxI]
        return perImgCorr, pixelShiftsXY

    def _extract_ccor_max(self, corrs):

        pixelShiftsXY = torch.empty(corrs.shape[:-2]+(2,), device=corrs.device, dtype=torch.int64)

        h0, h1, w0, w1 = self._get_begin_end_from_max_shift(corrs.shape[-2:], self.max_shift_fraction)
        corrs = corrs[..., h0:h1, w0:w1]
        maxCorrsJ, maxIndxJ = corrs.max(-1)
        perImgCorr, maxIndxI = maxCorrsJ.max(-1)
        pixelShiftsXY[..., 1] = h0 + maxIndxI
        pixelShiftsXY[..., 0] = w0 + torch.gather(maxIndxJ, -1, maxIndxI.unsqueeze(1)).squeeze(1)
        return perImgCorr, pixelShiftsXY

    @classmethod
    @lru_cache(1)
    def _get_begin_end_from_max_shift(cls, image_shape, max_shift):
        h, w = image_shape
        one_minux_max_shift = 1 - max_shift
        delta_h = math.ceil((h * one_minux_max_shift) / 2)
        h0 = delta_h
        h1 = h - delta_h

        delta_w = math.ceil((w * one_minux_max_shift) / 2)
        w0 = delta_w
        w1 = w - delta_w

        return h0, h1, w0, w1

    def align_particles(self, fimg, ctf, rotmats):

        expanded_rotmats = torch.einsum("gjk, btij -> btgij", self.grid_rotmats, rotmats)
        projs = self._projectF(expanded_rotmats)

        # from matplotlib import pyplot as plt
        # f, axes = plt.subplots(2, 2)
        # axes[0, 0].imshow(self._fourier_proj_to_real(projs)[0, 0, 0, ...])
        # axes[0, 1].imshow(self._fourier_proj_to_real(projs)[0, 1, 0, ...])
        # axes[1, 0].imshow(self._fourier_proj_to_real(projs)[1, 0, 0, ...])
        # axes[1, 1].imshow(self._fourier_proj_to_real(projs)[1, 1, 0, ...])
        # plt.show()

        projs *= ctf[:, None, None, ...]
        perImgCorr, pixelShiftsXY = self._correlateCrossCorrelation(fimg[:, None, None, ...], projs)
        b, topk, nrots = perImgCorr.shape[:3]
        reshaped_perImgCorr = perImgCorr.reshape(b, -1)
        maxCorrs, maxIdxs = reshaped_perImgCorr.topk(self.return_top_k, largest=True, sorted=True)
        bestPixelShiftsXY = pixelShiftsXY.reshape(b, -1)[torch.arange(b).unsqueeze(1), maxIdxs]
        mean_corr = torch.mean(reshaped_perImgCorr, dim=-1, keepdim=True)
        std_corr = torch.std(reshaped_perImgCorr, dim=-1, keepdim=True)
        comparedWeight = torch.distributions.Normal(mean_corr,std_corr).cdf(maxCorrs) #1-P(I_i > All_images)

        predShiftsAngs = -(bestPixelShiftsXY - self.half_particle_size) * self.vol_voxel_size
        predRotMatsIdxs = maxIdxs%nrots
        predRotMats = self.grid_rotmats[predRotMatsIdxs]

        return maxCorrs, predRotMats, predShiftsAngs, comparedWeight


def compute_dft(
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




def correlate_dft_2d(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Correlate fftshifted rfft discrete Fourier transforms of images"""

    #TODO: limit the comparison to the mask that alister uses
    """
        normed_grid = (einops.reduce(freq_grid**2, "h w zyx -> h w", reduction="sum") ** 0.5)
        freq_grid_mask = normed_grid <= fftfreq_max
        valid_coords = freq_grid[freq_grid_mask, ...]  # (b, zyx)
    """

    result = a * torch.conj(b)

    result = torch.fft.ifftshift(result, dim=(-2,))
    result = torch.fft.irfftn(result, dim=(-2, -1))
    result = torch.fft.ifftshift(result, dim=(-2, -1))
    return torch.real(result)


def _test0():
    device = "cuda"
    batch_size = 16
    input_topk_mats = 2

    pj = ProjectionMatcher(reference_vol="/home/sanchezg/tmp/cak_11799_usedTraining_ligErased.mrc",
                           grid_distance_degs=12,
                           grid_step_degs=2,
                           pixel_size=None,
                           filter_resolution_angst=5,
                           max_shift_fraction=0.2,
                           return_top_k=1)
    pj.to(device)
    fakefimage = torch.rand(batch_size, *pj.reference_vol.shape[-2:], dtype=torch.complex64, device=device)
    fakeCtf = torch.rand(batch_size, *pj.reference_vol.shape[-2:], dtype=torch.float32, device=device)
    from scipy.spatial.transform import Rotation
    rotmats = torch.as_tensor(Rotation.random(batch_size, random_state=1).as_matrix(), dtype=torch.float32, device=device
                              ).unsqueeze(1).repeat(1, input_topk_mats, 1, 1)
    print("align", flush=True)
    t=time.time()
    pj.align_particles(fakefimage, fakeCtf, rotmats)
    print("DONE", time.time()-t)

def _test1():
    device = "cuda"
    batch_size = 4

    pj = ProjectionMatcher(reference_vol="/home/sanchezg/cryo/data/preAlignedParticles/EMPIAR-10166/data/allparticles_reconstruct.mrc",
                           grid_distance_degs=12,
                           grid_step_degs=2,
                           pixel_size=None,
                           filter_resolution_angst=5,
                           max_shift_fraction=0.2,
                           return_top_k=1)

    from cryoPARES.configs.mainConfig import main_config
    main_config.datamanager.particlesdataset.desired_image_size_px=360
    main_config.datamanager.particlesdataset.desired_sampling_rate_angs=1.27
    from cryoPARES.datamanager.datamanager import DataManager

    datamanager = DataManager(star_fnames=["/home/sanchezg/cryo/data/preAlignedParticles/EMPIAR-10166/data/allparticles.star"],
                              symmetry="C1",
                              batch_size=batch_size,
                              particles_dir=None,
                              halfset=None,
                              save_train_val_partition_dir=None,
                              is_global_zero=True)

    with torch.inference_mode():
        for batch in datamanager._create_dataloader():
            imgs = batch[BATCH_PARTICLES_NAME]
            rotmats = batch[BATCH_POSE_NAME][0].unsqueeze(1)
            fimages = pj._real_to_fourier(imgs)
            pj.align_particles(fimages, ctfs, rotmats)

if __name__ == "__main__":
    _test1()
