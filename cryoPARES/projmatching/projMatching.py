import math
import os.path
import sys
import time
from functools import lru_cache, cached_property
from typing import Tuple, Optional, Union, Sequence

import numpy as np
import pandas as pd
import starfile
import torch
from scipy.spatial.transform import Rotation
from torch import nn
from torch.nn import functional as F

from cryoPARES.configs.mainConfig import main_config
from cryoPARES.projmatching.extract_central_slices_as_real import extract_central_slices_rfft_3d_multichannel
from torch_grid_utils import fftfreq_grid
from torch_grid_utils.shapes_2d import circle #TODO we can use other things to limit frequencies

from cryoPARES.geometry.convert_angles import euler_angles_to_matrix, matrix_to_euler_angles
from cryoPARES.geometry.grids import so3_near_identity_grid_cartesianprod
from cryoPARES.projmatching.fftOps import _real_to_fourier_2d, _fourier_proj_to_real_2d
from cryoPARES.utils.paths import MAP_AS_ARRAY_OR_FNAME_TYPE
from cryoPARES.utils.reconstructionUtils import get_vol
from cryoPARES.constants import (RELION_EULER_CONVENTION, BATCH_PARTICLES_NAME, BATCH_POSE_NAME,
                                 BATCH_ORI_IMAGE_NAME, BATCH_ORI_CTF_NAME, BATCH_IDS_NAME, BATCH_MD_NAME,
                                 RELION_ANGLES_NAMES, PROJECTION_MATCHING_SCORE, RELION_SHIFTS_NAMES)


class ProjectionMatcher(nn.Module):
    def __init__(self, reference_vol: MAP_AS_ARRAY_OR_FNAME_TYPE,
                 grid_distance_degs: float, #TODO: Add this to CONFIG
                 grid_step_degs: float,     #TODO: Add this to CONFIG
                 pixel_size: Optional[float] = None, #Only used if reference_vol is a np.array/torch.tensor
                 filter_resolution_angst: Optional[float] = None, #TODO: Add this to CONFIG
                 max_shift_fraction: Optional[float] = 0.2, #TODO: Add this to CONFIG
                 return_top_k: int = 1,
                 correct_ctf: bool = True
                 ): #TODO: Add downsampling of the volume particles for speed
        """

        :param reference_vol:
        :param grid_distance_degs: ~Cone width
        :param grid_step_degs:
        :param pixel_size:
        :param filter_resolution_angst:
        :param max_shift_fraction:
        :param return_top_k:
        :param correct_ctf:
        """

        super().__init__()

        self.grid_distance_degs = grid_distance_degs
        self.grid_step_degs = grid_step_degs
        self.filter_resolution_angst = filter_resolution_angst
        self.max_shift_fraction = max_shift_fraction
        self.return_top_k = return_top_k
        self.correct_ctf = correct_ctf

        self.vol_shape = None
        self.ori_image_shape = None
        self.vol_voxel_size = None


        self._store_reference_vol(reference_vol, pixel_size)
        self._store_so3_grid_rotmats()

        self._correlateF = self._correlateCrossCorrelation
        self.background_stream = torch.cuda.Stream()

    def _store_reference_vol(self, reference_vol: MAP_AS_ARRAY_OR_FNAME_TYPE,
                             pixel_size: Optional[float] = None):
        """

        :param reference_vol: A volume fname or torch tensor or numpy array representing a cryoEM volume
        :param pixel_size: The sampling rate in Å/px

        stores the reference_vol as a tensor and sets the following attributes: self.vol_shape,
        and self.ori_image_shape, that contain information about the original volume; and  self.vol_shape  and
        self.image_shape that contain information about the volume after padding or other transformation

        """

        reference_vol, pixel_size = get_vol(reference_vol, pixel_size)
        assert pixel_size
        # reference_vol = reference_vol - reference_vol.mean()

        self.vol_shape = tuple(reference_vol.shape)

        assert len(self.vol_shape) == 3
        assert len(set(self.vol_shape)) == 1, "Only cubic volumes are allowed"
        assert set(self.vol_shape).pop() % 2 == 0, "Only even boxsizes are allowed"

        self.ori_image_shape = tuple(self.vol_shape[-2:])
        self.half_particle_size = 0.5 * self.ori_image_shape[-2]
        self.vol_voxel_size = pixel_size
        reference_vol = torch.as_tensor(reference_vol.numpy(), device=reference_vol.device, dtype=reference_vol.dtype)

        reference_vol, vol_shape, pad_length = compute_dft_3d(reference_vol,
                                                              pad=False)  # reference_vol is computed with rfft=True, fftshift=True,
        self.pad_length = pad_length


        reference_vol = torch.view_as_real(reference_vol).permute([-1, 0, 1, 2]).contiguous()

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
        so3_local_grid = so3_near_identity_grid_cartesianprod(self.grid_distance_degs/2, n_angles, transposed=False, degrees=True)
        grid_rotmats = euler_angles_to_matrix( torch.deg2rad(so3_local_grid), convention=RELION_EULER_CONVENTION)
        self.register_buffer("grid_rotmats", grid_rotmats)


    @property
    def device(self):
        return self.reference_vol.device

    def _fourier_proj_to_real(self, projections):
        return _fourier_proj_to_real_2d(projections, self.pad_length)

    def _real_to_fourier(self, imgs):#TODO. I should be able to get the fimages from the rfft_ctf.correct_ctf. I would need to apply a phase shift to reproduce the real space fftshift
        return _real_to_fourier_2d(imgs, as_real_img=True)

    def _apply_ctfF(self, projs, ctf):
        projs.mul_(ctf)
        return projs

    def _projectF(self, rotmats):
        projs = extract_central_slices_rfft_3d_multichannel(self.reference_vol, self.vol_shape,
                                                            rotation_matrices=rotmats, fftfreq_max= self.fftfreq_max,
                                                            zyx_matrices=False)
        with torch.cuda.stream(self.background_stream):
            projs = projs.permute([0, 1, 2, 4, 5, 3]).contiguous()
        return projs

    def _correlateCrossCorrelation(self, parts: torch.Tensor, projs: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:

        # #We assume the particles have been already normalized
        # parts = self._normalize_fourier_img(parts)
        # projs = self._normalize_fourier_img(projs)

        corrs = correlate_dft_2d(parts, projs)

        # from matplotlib import pyplot as plt
        # f, axes = plt.subplots(3,1)
        # axes[0].imshow(self._fourier_proj_to_real(parts[0, 0, 0]).cpu(), cmap="gray")
        # axes[1].imshow(self._fourier_proj_to_real(projs[0, 0, 5]).cpu(), cmap="gray")
        # # # axes[2].imshow(self._fourier_proj_to_real(projs[0, 0, 5]).cpu().flip(-2).flip(-1), cmap="gray")
        # axes[2].imshow(self._fourier_proj_to_real(projs[0, 0, 5].cpu()) - self._fourier_proj_to_real(parts[0, 0, 0].cpu()), cmap="gray")
        # plt.show()
        # from skimage.feature import match_template
        # match_template(self._fourier_proj_to_real(projs[0, 0, 5].cpu()).numpy(), self._fourier_proj_to_real(parts[0, 0, 0].cpu()).numpy())

        # axes[0].imshow(corr_dft2d[0].abs().log().cpu(), cmap="gray")
        # axes[2].imshow(corrs[0, 0, 5], cmap="gray"); plt.show()

        perImgCorr, pixelShiftsXY = _extract_ccor_max(corrs, self.max_shift_fraction)

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

        return perImgCorr, pixelShiftsXY

    def align_particles(self, fimgs, ctf, rotmats):

        # expanded_rotmats = torch.einsum("gij, btjk -> btgik", self.grid_rotmats, rotmats)
        expanded_rotmats = torch.matmul(rotmats.unsqueeze(2), self.grid_rotmats.unsqueeze(0).unsqueeze(0))
        del rotmats
        projs = self._projectF(expanded_rotmats)

        # from matplotlib import pyplot as plt
        # f, axes = plt.subplots(2, 2)
        # cpu_projs = projs.to("cpu", copy=True)
        # mat_idxs = [5, 6]
        # axes[0, 0].imshow(self._fourier_proj_to_real(cpu_projs)[0, 0, mat_idxs[0], ...])
        # axes[0, 1].imshow(self._fourier_proj_to_real(cpu_projs)[0, 0, mat_idxs[1], ...])
        # axes[1, 0].imshow(self._fourier_proj_to_real(cpu_projs)[1, 0, mat_idxs[0], ...])
        # axes[1, 1].imshow(self._fourier_proj_to_real(cpu_projs)[1, 0, mat_idxs[1], ...])
        # plt.show()
        self.background_stream.synchronize()
        if self.correct_ctf:
            projs *= ctf[:, None, None, ..., None] #TODO. Multiply ctf in the particles
        del ctf
        perImgCorr, pixelShiftsXY = self._correlateCrossCorrelation(fimgs[:, None, None, ...], projs)

        maxCorrs, predRotMats, predShiftsAngs, comparedWeight = _analyze_cross_correlation(perImgCorr, pixelShiftsXY,
                                                                           grid_rotmats=expanded_rotmats,
                                                                           return_top_k=self.return_top_k,
                                                                           half_particle_size=self.half_particle_size,
                                                                           vol_voxel_size=self.vol_voxel_size)
        #TODO: PreRotMats comes from multyping the (grid_rotmat @ rotmat)
        return maxCorrs, predRotMats, predShiftsAngs, comparedWeight

    def forward(self, imgs, ctfs, rotmats):
        with torch.cuda.stream(self.background_stream):
            fimages = self._real_to_fourier(imgs)
        maxCorrs, predRotMats, predShiftsAngs, comparedWeight = self.align_particles(fimages, ctfs, rotmats)
        return maxCorrs, predRotMats, predShiftsAngs, comparedWeight

#TODO: we should define a _analyze_cross_correlation_FACTORY to use main_config.projmatching properly
@torch.compile(fullgraph=True, disable=main_config.projmatching.disable_compile_analyze_cc,
               mode=main_config.projmatching.compile_analyze_cc_mode)
def _analyze_cross_correlation(perImgCorr:torch.Tensor, pixelShiftsXY:torch.Tensor, grid_rotmats:torch.Tensor,
                               return_top_k:int, half_particle_size:float,
                               vol_voxel_size:float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    b, topk, nrots = perImgCorr.shape
    batch_arange= torch.arange(b, device=perImgCorr.device).unsqueeze(1)
    topk_arange= torch.arange(topk, device=perImgCorr.device).unsqueeze(1)
    reshaped_perImgCorr = perImgCorr.reshape(b, -1)
    maxCorrs, maxIdxs = reshaped_perImgCorr.topk(return_top_k, largest=True, sorted=True)
    bestPixelShiftsXY = pixelShiftsXY.reshape(b, -1, 2)[batch_arange, maxIdxs]
    mean_corr = torch.mean(reshaped_perImgCorr, dim=-1, keepdim=True)
    std_corr = torch.std(reshaped_perImgCorr, dim=-1, keepdim=True)
    comparedWeight = torch.distributions.Normal(mean_corr, std_corr).cdf(maxCorrs)  # 1-P(I_i > All_images)
    predShiftsAngs = -(bestPixelShiftsXY - half_particle_size) * vol_voxel_size
    predRotMatsIdxs = maxIdxs % nrots
    predRotMats = grid_rotmats[batch_arange, topk_arange, predRotMatsIdxs]

    return maxCorrs, predRotMats, predShiftsAngs, comparedWeight

@lru_cache(1)
def _get_begin_end_from_max_shift(image_shape, max_shift):
    h, w = image_shape
    one_minus_max_shift = 1 - max_shift
    delta_h = math.ceil((h * one_minus_max_shift) / 2)
    h0 = delta_h
    h1 = h - delta_h

    delta_w = math.ceil((w * one_minus_max_shift) / 2)
    w0 = delta_w
    w1 = w - delta_w

    return h0, h1, w0, w1

#TODO: we should define a _extract_ccor_maxFACTORY to use main_config.projmatching properly
@torch.compile(fullgraph=True, disable=main_config.projmatching.disable_compile_analyze_cc,
               mode=main_config.projmatching.compile_analyze_cc_mode)
def _extract_ccor_max(corrs, max_shift_fraction):
    pixelShiftsXY = torch.empty(corrs.shape[:-2] + (2,), device=corrs.device, dtype=torch.int64)

    if max_shift_fraction is not None:
        h0, h1, w0, w1 = _get_begin_end_from_max_shift(corrs.shape[-2:], max_shift_fraction)
        corrs = corrs[..., h0:h1, w0:w1]
    else:
        h0, w0 = 0, 0
    maxCorrsJ, maxIndxJ = corrs.max(-1)
    perImgCorr, maxIndxI = maxCorrsJ.max(-1)
    pixelShiftsXY[..., 1] = h0 + maxIndxI
    pixelShiftsXY[..., 0] = w0 + torch.gather(maxIndxJ, -1, maxIndxI.unsqueeze(-1)).squeeze(-1)

    return perImgCorr, pixelShiftsXY

#TODO: we should define a correlate_dft_2d_FACTORY to use main_config.projmatching properly
@torch.compile(fullgraph=True, disable=main_config.projmatching.disable_compile_correlate_dft_2d,
               mode=main_config.projmatching.compile_correlate_dft_2d_mode)
def correlate_dft_2d(
    parts: torch.Tensor,
    projs: torch.Tensor,
) -> torch.Tensor:
    """Correlate fftshifted rfft discrete Fourier transforms of images"""

    #TODO: try to limit the comparison to the mask that Alister uses
    """
        normed_grid = (einops.reduce(freq_grid**2, "h w zyx -> h w", reduction="sum") ** 0.5)
        freq_grid_mask = normed_grid <= fftfreq_max
        valid_coords = freq_grid[freq_grid_mask, ...]  # (b, zyx)
    """

    if not parts.is_complex():
        parts = torch.view_as_complex(parts)
    if not projs.is_complex():
        projs = torch.view_as_complex(projs)
    result = parts * torch.conj(projs)

    result = torch.fft.ifftshift(result, dim=(-2,),)
    result = torch.fft.irfftn(result, dim=(-2, -1))
    result = torch.fft.ifftshift(result, dim=(-2, -1)).real

    # from matplotlib import pyplot as plt
    # plt.imshow(result)
    # plt.show()
    return result


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
    batch_size = 32

    torch.set_float32_matmul_precision('high')
    pj = ProjectionMatcher(reference_vol=os.path.expanduser("~/cryo/data/preAlignedParticles/EMPIAR-10166/data/allparticles_reconstruct.mrc"),
                           grid_distance_degs=10, #12
                           grid_step_degs=5,      #2
                           pixel_size=None,
                           filter_resolution_angst=6,
                           max_shift_fraction=None,
                           return_top_k=1,
                           correct_ctf=False)

    from cryoPARES.datamanager.datamanager import DataManager
    from tqdm import tqdm

    from cryoPARES.configs.mainConfig import main_config
    # main_config.datamanager.particlesdataset.desired_image_size_px=360
    # main_config.datamanager.particlesdataset.desired_sampling_rate_angs=1.27

    datamanager = DataManager(
          # star_fnames=[os.path.expanduser("~/cryo/data/preAlignedParticles/EMPIAR-10166/data/allparticles.star")],
          star_fnames=[os.path.expanduser("~/cryo/data/preAlignedParticles/EMPIAR-10166/data/projections/proj_noCTF_no_shifts.star")],
          symmetry="C1",
          batch_size=batch_size,
          particles_dir=[os.path.expanduser("~/cryo/data/preAlignedParticles/EMPIAR-10166/data/")],
          halfset=None,
          save_train_val_partition_dir=None,
          is_global_zero=True,
          return_ori_imagen=True,
          num_augmented_copies_per_batch=1,
          num_data_workers=2)
    pj.to(device)
    with torch.inference_mode():
        pd_list = []
        dl = datamanager._create_dataloader()
        for bix, batch in enumerate(dl): #WARMUP
            # print(batch[BATCH_IDS_NAME])
            imgs = batch[BATCH_ORI_IMAGE_NAME].to(device, non_blocking=True)
            ctfs = batch[BATCH_ORI_CTF_NAME].to(device, non_blocking=True)
            rotmats = batch[BATCH_POSE_NAME][0].unsqueeze(1).to(device, non_blocking=True)
            md = batch[BATCH_MD_NAME]
            # torch.compiler.cudagraph_mark_step_begin()  #<- This is needed for compilation.  #!!!!!!!!!!!!!!!!!!
            fimages = pj._real_to_fourier(imgs) #TODO. I should be able to get the fimages from the rfft_ctf.correct_ctf. I would need to apply a phase shift to reproduce the real space fftshift
            maxCorrs, predRotMats, predShiftsAngs, comparedWeight = pj.align_particles(fimages, ctfs, rotmats)
            break
        for bix, batch in enumerate(tqdm(dl)):
            # print(batch[BATCH_IDS_NAME])
            imgs = batch[BATCH_ORI_IMAGE_NAME].to(device, non_blocking=True)
            ctfs = batch[BATCH_ORI_CTF_NAME].to(device, non_blocking=True)
            rotmats = batch[BATCH_POSE_NAME][0].unsqueeze(1).to(device, non_blocking=True)
            md = batch[BATCH_MD_NAME]
            with torch.cuda.stream(pj.background_stream):
                fimages = pj._real_to_fourier(imgs) #TODO. I should be able to get the fimages from the rfft_ctf.correct_ctf. I would need to apply a phase shift to reproduce the real space fftshift
            maxCorrs, predRotMats, predShiftsAngs, comparedWeight = pj.align_particles(fimages, ctfs, rotmats)
        #     print("GT Rots\n", torch.rad2deg(matrix_to_euler_angles(rotmats, RELION_EULER_CONVENTION)).squeeze(1)) #TODO: I don't trust matrix_to_euler
        #     print("Pred Rots\n",torch.rad2deg(matrix_to_euler_angles(predRotMats, RELION_EULER_CONVENTION)).squeeze(1))
        #
        #     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        #     print("GT shifts\n", batch[BATCH_POSE_NAME][1])
        #     print("pred shifts\n", predShiftsAngs.squeeze(1))
        #     print("---------------------------------------------------")
        #     print()
        #
        #     for topk in range(predRotMats.shape[1]):
        #         suffix = "" if topk == 0 else f"_top{topk}"
        #         eulers = Rotation.from_matrix(predRotMats.cpu().numpy()[:,topk,...]).as_euler(RELION_EULER_CONVENTION, degrees=True)
        #         for i, angName in enumerate(RELION_ANGLES_NAMES):
        #             md[angName+suffix] = eulers[:,i]
        #         for i, shiftName in enumerate(RELION_SHIFTS_NAMES):
        #             md[shiftName+suffix] = predShiftsAngs[:,topk,i].cpu().numpy()
        #         md[PROJECTION_MATCHING_SCORE] = comparedWeight[:,topk,...].cpu().numpy()
        #     pd_list.append(pd.DataFrame(md))
        #     del predRotMats, eulers, predShiftsAngs, comparedWeight
        # optics = getattr(dl.dataset, "particles", dl.dataset.datasets[0].particles).optics_md
        # parts = pd.concat(pd_list)
        # starfile.write(dict(optics=optics, particles=parts), filename="/tmp/particles.star")

if __name__ == "__main__":
    _test1()
