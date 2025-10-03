import gc
import os
import tempfile
import math
from functools import cached_property, lru_cache
from typing import Tuple, Optional, Literal

from starstack import ParticlesStarSet
from torch import nn

import numpy as np
import starfile
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader
from torch_grid_utils import circle

from cryoPARES.configManager.inject_defaults import inject_defaults_from_config, CONFIG_PARAM
from cryoPARES.constants import (RELION_EULER_CONVENTION, BATCH_POSE_NAME, RELION_PRED_POSE_CONFIDENCE_NAME,
                                 BATCH_ORI_IMAGE_NAME, BATCH_ORI_CTF_NAME, RELION_ANGLES_NAMES, RELION_SHIFTS_NAMES,
                                 BATCH_IDS_NAME)
from torch_fourier_slice.slice_extraction import extract_central_slices_rfft_3d
from cryoPARES.utils.paths import MAP_AS_ARRAY_OR_FNAME_TYPE, FNAME_TYPE

from cryoPARES.configs.mainConfig import main_config
from cryoPARES.geometry.convert_angles import euler_angles_to_matrix, matrix_to_euler_angles
from cryoPARES.geometry.grids import so3_near_identity_grid_cartesianprod
from cryoPARES.geometry.metrics_angles import rotation_error_with_sym
from cryoPARES.utils.reconstructionUtils import get_vol
from cryoPARES.projmatching.projmatchingUtils.loggers import getWorkerLogger
from cryoPARES.projmatching.projmatchingUtils.myProgressBar import myTqdm as tqdm

from cryoPARES.projmatching.projmatchingUtils.filterToResolution import low_pass_filter_fname
from cryoPARES.projmatching.projmatchingUtils.fourierOperations import correlate_dft_2d, compute_dft_3d, _real_to_fourier_2d

REPORT_ALIGNMENT_DISPLACEMENT = True
USE_TWO_FLOAT32_FOR_COMPLEX = True


def get_rotmat(degAngles, convention: str = RELION_EULER_CONVENTION, device=None):
    if device is None:
        device = degAngles.device
    return euler_angles_to_matrix(torch.deg2rad(degAngles), convention=convention).to(device)


def get_eulers(rotmats, convention: str = RELION_EULER_CONVENTION, device=None):
    if device is None:
        device = rotmats.device
    ori_shape = rotmats.shape
    eulerDegs = torch.rad2deg(matrix_to_euler_angles(rotmats.reshape(-1, 3, 3), convention=convention)).to(device)
    eulerDegs = eulerDegs.view(*ori_shape[:-2], 3)
    return eulerDegs


class ProjectionMatcher(nn.Module):
    """
    Single concrete aligner (Fourier pipeline).
    This class folds together the previous abstract base + Fourier subclass.
    """

    @inject_defaults_from_config(default_config=main_config.projmatching, update_config_with_args=True)
    def __init__(
            self,
            reference_vol: MAP_AS_ARRAY_OR_FNAME_TYPE,
            pixel_size: float | None = None,  #Only needed if reference_vol is not a filename
            grid_distance_degs: float | Tuple[float, float, float] = CONFIG_PARAM(),
            grid_step_degs: float | Tuple[float, float, float] = CONFIG_PARAM(),
            max_resolution_A: Optional[float] = CONFIG_PARAM(),
            top_k_poses_localref: int = CONFIG_PARAM(),
            verbose: bool = CONFIG_PARAM(),
            correct_ctf: bool = CONFIG_PARAM(),
            mask_radius_angs: Optional[float] = CONFIG_PARAM(config=main_config.datamanager.particlesdataset)
    ):
        super().__init__()
        self.grid_distance_degs = grid_distance_degs
        self.grid_step_degs = grid_step_degs
        self.max_resolution_A = max_resolution_A
        self.padding_factor = 0  # We do not support padding yet
        self.top_k_poses_localref = top_k_poses_localref
        self.max_shift_fraction = main_config.projmatching.max_shift_fraction
        self.mask_radius_angs = mask_radius_angs

        self._store_reference_vol(reference_vol, pixel_size)
        self.verbose = verbose
        self.correct_ctf = correct_ctf
        self.mainLogger = getWorkerLogger(self.verbose)
        if USE_TWO_FLOAT32_FOR_COMPLEX:
            if main_config.projmatching.disable_compile_projectVol:
                from cryoPARES.projmatching.projmatchingUtils.extract_central_slices_as_real import \
                                extract_central_slices_rfft_3d_multichannel
                self.extract_central_slices_rfft_3d_multichannel = extract_central_slices_rfft_3d_multichannel
            else:
                from cryoPARES.projmatching.projmatchingUtils.extract_central_slices_as_real import \
                                compiled_extract_central_slices_rfft_3d_multichannel
                print("Compiling extract_central_slices_rfft_3d_multichannel")
                self.extract_central_slices_rfft_3d_multichannel = compiled_extract_central_slices_rfft_3d_multichannel
            self.projectF = self._projectF_USE_TWO_FLOAT32_FOR_COMPLEX

        else:
            self.projectF = self._projectF

        if not main_config.projmatching.disable_compile_analyze_cc:
            self._extract_ccor_max = torch.compile(_extract_ccor_max, fullgraph=True,
                                                            mode=main_config.projmatching.compile_analyze_cc_mode,
                                                            dynamic=True)
        else:
            self._extract_ccor_max = _extract_ccor_max

        if main_config.projmatching.disable_compile_correlate_dft_2d:
            self.correlate_dft_2d = correlate_dft_2d
        else:
            print("Compiling correlate_dft_2d")
            self.correlate_dft_2d = torch.compile(correlate_dft_2d, fullgraph=True,
                                                  mode=main_config.projmatching.compile_correlate_dft_2d_mode
                                                  )
    # ----------------------- Basic props/helpers -----------------------

    @property
    def device(self):
        return self.reference_vol.device

    def _get_so3_delta(self, device: torch.device, as_rotmats=False):
        """
        Build a cached SO(3) delta grid (degrees) around (0,0,0) using
        self.grid_distance_degs and self.grid_step_degs.
        """
        if (not hasattr(self, "_so3_delta") or self._so3_delta.device != device or
                getattr(self, "_so3_delta_is_rotmat", None) != as_rotmats):

            n_angles = math.ceil(self.grid_distance_degs * 2 / self.grid_step_degs)
            n_angles = n_angles if n_angles % 2 == 1 else n_angles + 1  # We always want an odd number
            self._so3_delta = so3_near_identity_grid_cartesianprod(self.grid_distance_degs, n_angles,
                                                                   transposed=False, degrees=True,
                                                                   remove_duplicates=False).to(device)
            #TODO: remove_duplicates=True makes the whole things less accurate. Probably due to euler angles singularities
            if as_rotmats:
                # Using rotmats seems a bad idea as well. Not giving good results. Is it perhaps because we
                # convert the grid from euler to rots, and near identity the numerical errors are important?
                self._so3_delta_is_rotmat = True
                self._so3_delta = get_rotmat(self._so3_delta)
            else:
                self._so3_delta_is_rotmat = False

        return self._so3_delta

    # ----------------------- Fourier-specific core -----------------------

    def _store_reference_vol(
            self, reference_vol: MAP_AS_ARRAY_OR_FNAME_TYPE, pixel_size: float | None = None
    ):
        """
        Load volume, move to Fourier domain (rfft & shift), register buffers and
        set up correlation choice.
        """

        if self.max_resolution_A is not None:
            reference_vol , pixel_size = low_pass_filter_fname(reference_vol, resolution=self.max_resolution_A,
                                                               out_fname=None)[:2]
        else:
            reference_vol, pixel_size = get_vol(reference_vol, pixel_size=pixel_size)

        self.ori_vol_shape = reference_vol.shape
        self.ori_image_shape = self.ori_vol_shape[-2:]
        self.vol_voxel_size = pixel_size
        reference_vol = torch.as_tensor(
            reference_vol.numpy(),
            device=reference_vol.device,
            dtype=reference_vol.dtype,
        )

        pad_length = int(self.padding_factor * reference_vol.shape[-1] // 2)
        reference_vol, vol_shape, pad_length = compute_dft_3d(
            reference_vol, pad_length=pad_length
        )
        self.pad_length = pad_length

        if USE_TWO_FLOAT32_FOR_COMPLEX:
            #We are storing the volumes as real and imaginary float32
            reference_vol = torch.view_as_real(reference_vol).permute([-1, 0, 1, 2]).contiguous()

        self.register_buffer("reference_vol", reference_vol)

        self.vol_shape = vol_shape
        self.image_shape = vol_shape[-2:]
        assert len(self.vol_shape) == 3
        assert len(set(self.vol_shape)) == 1, "Only cubic volumes are allowed"
        assert self.image_shape[-2] % 2 == 0, "Only even boxsizes are allowed"
        self.half_particle_size = 0.5 * self.image_shape[-2]

        # Prepare mask
        radius_px = self.image_shape[-2] // 2
        if self.mask_radius_angs is not None:
            assert 1 < self.mask_radius_angs <= radius_px, \
                "Error, the mask redius is larger than the box size"
            radius_px = self.mask_radius_angs
        rmask = circle(radius_px, image_shape=self.image_shape, smoothing_radius=radius_px * .05)
        self.register_buffer("rmask", rmask)
        if self.max_resolution_A is not None:
            self.fftfreq_max = min(self.vol_voxel_size / self.max_resolution_A, 0.5)
        else:
            self.fftfreq_max = 0.5

    def _projectF(self, rotMats: torch.Tensor) -> torch.Tensor:

        return extract_central_slices_rfft_3d(
            self.reference_vol,
            image_shape=self.vol_shape,
            rotation_matrices=rotMats,
            fftfreq_max=self.fftfreq_max,
            zyx_matrices=False, )

    def _projectF_USE_TWO_FLOAT32_FOR_COMPLEX(self, rotMats: torch.Tensor) -> torch.Tensor:
        projs = self.extract_central_slices_rfft_3d_multichannel(self.reference_vol,
                                                                 self.vol_shape,
                                                                 rotation_matrices=rotMats,
                                                                 fftfreq_max=self.fftfreq_max,
                                                                 zyx_matrices=False)
        projs = projs.permute([0, 2, 3, 1]).contiguous()
        return projs

    def correlateF(self, parts: torch.Tensor, projs: torch.Tensor):
        return self._correlateCrossCorrelation(parts, projs)

    def _apply_ctfF(self, projs, ctf):
        return projs * ctf

    @cached_property
    def max_resoluton_freq_pixels(self):
        pixel_size = self.vol_voxel_size  # A/pixel
        n_pixels = self.vol_shape[-1]
        if self.max_resolution_A is not None:
            assert self.max_resolution_A >= 2 * pixel_size
            radius = n_pixels * pixel_size / self.max_resolution_A
            return radius
        else:
            return None

    def _correlateCrossCorrelation(self, parts: torch.Tensor, projs: torch.Tensor):
        corrs = self.correlate_dft_2d(parts, projs)
        b, options, l0, l1 = corrs.shape
        h0, h1, w0, w1 = _get_begin_end_from_max_shift(corrs.shape[-2:], self.max_shift_fraction)
        maxcorr, maxcorrIdxs = self._extract_ccor_max(corrs.reshape(-1, *corrs.shape[-2:]),
                                                      h0, h1, w0, w1)

        del corrs
        return maxcorr.reshape(b, options), maxcorrIdxs.reshape(b, options, 2)

    # ----------------------- Shared helpers originally in base -----------------------

    def _compute_projections_from_euleres(self, so3_degs_grid: torch.Tensor):
        so3_degs_grid = so3_degs_grid.to(self.reference_vol.device)
        rotMats = get_rotmat(so3_degs_grid, device=so3_degs_grid.device)
        projs = self.projectF(rotMats)
        return projs, so3_degs_grid

    def _compute_projections_from_rotmats(self, rotMats: torch.Tensor):
        projs = self.projectF(rotMats)
        return projs

    def preprocess_particles(
            self,
            particles: FNAME_TYPE,
            data_rootdir,
            batch_size,
            n_cpus,
            halfset=None,
            subset_idxs=None
    ):

        from cryoPARES.datamanager.datamanager import DataManager
        dm = DataManager(particles,
                         symmetry="C1",
                         particles_dir=data_rootdir,
                         halfset=halfset,
                         batch_size=batch_size,
                         save_train_val_partition_dir=None,
                         is_global_zero=True,
                         num_augmented_copies_per_batch=1,
                         num_dataworkers=n_cpus,
                         return_ori_imagen=True,
                         subset_idxs=subset_idxs
                         )

        ds = dm.create_dataset(None)
        return ds

    def forward(self, imgs, ctfs, rotmats):

        fparts = _real_to_fourier_2d(imgs * self.rmask,
                                     view_complex_as_two_float32=USE_TWO_FLOAT32_FOR_COMPLEX)

        eulerDegs = get_eulers(rotmats)
        expanded_eulerDegs = (self._get_so3_delta(eulerDegs.device).unsqueeze(0).unsqueeze(0) +
                              eulerDegs.unsqueeze(2)) #BxTopKxRotsx3
        bsize = expanded_eulerDegs.size(0)
        n_input_angles = expanded_eulerDegs.size(2)
        expanded_eulerDegs = expanded_eulerDegs.reshape(bsize, -1, 3)  # unrolling all the input angles into one dim
        # TODO: Remove duplicate angles if n_input_angles > 1. Duplicates may happen if several topK poses were provided as input
        # Also, for large batches, there could be redundant angles (to a certain precision)
        projs = self._compute_projections_from_euleres(expanded_eulerDegs.reshape(-1, 3))[0]

        #TODO: I need another way of generating the rotmats grid to avoid singularities
        # expanded_rotmats = (rotmats.unsqueeze(1) @ self._get_so3_delta(rotmats.device, as_rotmats=True).unsqueeze(0))
        # bsize = expanded_rotmats.size(0)
        # expanded_rotmats = expanded_rotmats.reshape(bsize, -1, 3, 3)  # unrolling all the input angles into one
        # projs = self._compute_projections_from_rotmats(expanded_rotmats.reshape(-1, 3, 3))

        if self.correct_ctf:
            ctfs = ctfs.unsqueeze(1)
            if USE_TWO_FLOAT32_FOR_COMPLEX:
                _shapeIdx = -3
                ctfs = ctfs.unsqueeze(-1)
            else:
                _shapeIdx = -2
            projs = self._apply_ctfF(projs.reshape(bsize, -1, *projs.shape[_shapeIdx:]), ctfs)
        del ctfs
        perImgCorr, pixelShiftsXY = self.correlateF(fparts.unsqueeze(1), projs)
        maxCorrs, maxCorrsIdxs = perImgCorr.topk(self.top_k_poses_localref, sorted=True, largest=True, dim=-1)

        batch_idxs_range = torch.arange(pixelShiftsXY.size(0)).unsqueeze(1)
        pixelShiftsXY = pixelShiftsXY[batch_idxs_range, maxCorrsIdxs]
        predEulers = expanded_eulerDegs[
            batch_idxs_range, maxCorrsIdxs]  #TODO: Try to minimize euler->rotmat conversions
        predRotMats = get_rotmat(predEulers)
        mean_corr = perImgCorr.mean(-1, keepdims=True)
        std_corr = perImgCorr.std(-1, keepdims=True)
        comparedWeight = torch.distributions.Normal(mean_corr, std_corr + 1e-6).cdf(maxCorrs)  # 1-P(I_i > All_images)
        predShiftsAngsXY = -(pixelShiftsXY.float() - self.half_particle_size) * self.vol_voxel_size
        return maxCorrs, predRotMats, predShiftsAngsXY, comparedWeight

    @torch.inference_mode()
    def align_star(
            self,
            particles: FNAME_TYPE,
            starFnameOut: FNAME_TYPE,
            data_rootdir: str | None = None,
            batch_size=256,
            device="cuda",
            n_cpus=1,
            halfset: Optional[Literal[1,2]] = None,
    ) -> ParticlesStarSet:
        """
        Align particles (input STAR file) to the reference.
        Writes a STAR with predicted poses/shifts if starFnameOut provided.
        """
        if starFnameOut is not None:
            assert not os.path.isfile(
                starFnameOut
            ), f"Error, the starFnameOut {starFnameOut} already exists"

        n_cpus = n_cpus if n_cpus > 0 else 1
        particlesDataSet = self.preprocess_particles(
            particles,
            data_rootdir,
            batch_size,
            n_cpus,
            halfset=halfset
        )
        assert len(particlesDataSet) > 0, "Error, the starfile contains no particles"
        try:
            pixel_size = particlesDataSet.sampling_rate
            particle_size = particlesDataSet[0].original_image_size()
        except AttributeError:
            pixel_size = particlesDataSet.datasets[0].original_sampling_rate()
            particle_size = particlesDataSet.datasets[0].original_image_size()

        assert np.isclose(pixel_size, self.vol_voxel_size, atol=1e-2), ("Error, the pixel size of the particle images "
                                                                        f"{pixel_size} "
                                                                        "do not match the voxel size of the reference "
                                                                        "{self.vol_voxel_size} ")

        assert np.allclose(particle_size, self.ori_image_shape, atol=1e-1), (f"Error, the size of the particle images "
                                                                            f"{particle_size} "
                                                                            f"do not match the size of the reference "
                                                                            f"{self.ori_image_shape}")
        half_particle_size = self.image_shape[-1] / 2
        assert half_particle_size == self.half_particle_size
        n_particles = len(particlesDataSet)

        results_corr_matrix = torch.zeros(n_particles, self.top_k_poses_localref)
        results_shiftsAngs_matrix = torch.zeros(n_particles, self.top_k_poses_localref, 2)
        results_eulerDegs_matrix = torch.zeros(n_particles, self.top_k_poses_localref, 3)
        stats_corr_matrix = torch.zeros(n_particles, self.top_k_poses_localref)
        idds_list = [None] * n_particles

        self.to(device)
        self.mainLogger.info(f"Total number of particles: {n_particles}")

        try:
            particlesStar = particlesDataSet.particles.copy()
            confidence = particlesDataSet.dataDict["confidences"].sum(-1)
        except AttributeError:
            particlesStar = particlesDataSet.datasets[0].particles.copy()
            try:
                confidence = torch.tensor(particlesStar.particles_md.loc[:, RELION_PRED_POSE_CONFIDENCE_NAME].values,
                                          dtype=torch.float32)
            except KeyError:
                confidence = torch.ones(len(particlesStar.particles_md))
        if "rlnImageId" in particlesStar.particles_md.columns:
            particlesStar.particles_md.drop("rlnImageId", axis=1, inplace=True)

        dl = DataLoader(
            particlesDataSet,
            batch_size=batch_size,
            num_workers=n_cpus, shuffle=False, pin_memory=True,
            multiprocessing_context='spawn' if n_cpus > 0 else None
        )  #get_context('loky')
        non_blocking = True
        _partIdx = 0
        for batch in tqdm(dl, desc="Aligning particles", disable=not self.verbose):

            n_items = len(batch[BATCH_ORI_IMAGE_NAME])
            partIdx = torch.arange(_partIdx, _partIdx + n_items);
            _partIdx += n_items
            idds = batch[BATCH_IDS_NAME]
            rotmats = batch[BATCH_POSE_NAME][0].to(device, non_blocking=non_blocking)
            parts = batch[BATCH_ORI_IMAGE_NAME].to(device, non_blocking=non_blocking)
            ctfs = batch[BATCH_ORI_CTF_NAME].to(device, non_blocking=non_blocking)

            rotmats = rotmats.unsqueeze(1) #This is used becase the code expect k poses per particle
            maxCorrs, predRotMats, predShiftsAngsXY, comparedWeight = self.forward(parts, ctfs, rotmats)
            results_corr_matrix[partIdx, :] = maxCorrs.detach().cpu()
            # results_shifts_matrix[partIdx, :] = pixelShiftsXY.detach().cpu()
            results_shiftsAngs_matrix[partIdx, :] = predShiftsAngsXY.detach().cpu()
            predEulerDegs = get_eulers(predRotMats)
            results_eulerDegs_matrix[partIdx, :] = predEulerDegs.detach().cpu()
            stats_corr_matrix[partIdx, :] = comparedWeight.detach().cpu()
            for _i, i_partIdx in enumerate(partIdx.tolist()): idds_list[i_partIdx] = idds[_i]

        predEulerDegs = results_eulerDegs_matrix
        # predShiftsAngs = -(results_shifts_matrix.float() - half_particle_size) * pixel_size
        predShiftsAngs = results_shiftsAngs_matrix
        prob_x_y = stats_corr_matrix
        n_topK = predEulerDegs.shape[1]

        finalParticlesStar = particlesStar
        particles_md = particlesStar.particles_md
        for k in range(n_topK):
            suffix = "" if k == 0 else f"_top{k}"
            angles_names = [x + suffix for x in RELION_ANGLES_NAMES]
            shiftsXYangs_names = [x + suffix for x in RELION_SHIFTS_NAMES]
            confide_name = RELION_PRED_POSE_CONFIDENCE_NAME + suffix
            for col in angles_names + shiftsXYangs_names + [confide_name]:
                if col not in particles_md.columns:
                    particles_md[col] = 0.0

            eulerdegs = predEulerDegs[:, k, :].numpy()
            shiftsXYangs = predShiftsAngs[:, k, :].numpy()
            if REPORT_ALIGNMENT_DISPLACEMENT:
                try:
                    _ref_angles_df = particles_md.loc[:, RELION_ANGLES_NAMES].copy()
                    _ref_shifts_df = particles_md.loc[:, RELION_SHIFTS_NAMES].copy()
                except KeyError:
                    _ref_angles_df = None
                    _ref_shifts_df = None

                base_angles = _ref_angles_df.loc[idds_list, RELION_ANGLES_NAMES].values
                base_shifts = _ref_shifts_df.loc[idds_list, RELION_SHIFTS_NAMES].values

                r1 = torch.from_numpy(
                    Rotation.from_euler(RELION_EULER_CONVENTION, eulerdegs, degrees=True).as_matrix()
                ).float()
                r2 = torch.from_numpy(
                    Rotation.from_euler(RELION_EULER_CONVENTION, base_angles, degrees=True).as_matrix()
                ).float()

                ang_err = torch.rad2deg(rotation_error_with_sym(r1, r2, symmetry="C1"))
                shift_error = np.sqrt(((shiftsXYangs - base_shifts) ** 2).sum(-1))

                print(f"Median Ang   Error degs (top-{k + 1}):", np.median(ang_err.numpy()))
                print(f"Median Shift Error Angs (top-{k + 1}):", np.median(shift_error))
                ######## END of Debug code

            particles_md.loc[idds_list, angles_names] = eulerdegs
            particles_md.loc[idds_list, shiftsXYangs_names] = shiftsXYangs
            _confidence = confidence * prob_x_y[:, k]
            particles_md.loc[idds_list, confide_name] = _confidence.numpy()

        if starFnameOut is not None:
            finalParticlesStar.save(starFname=starFnameOut)
            self.mainLogger.info(f"particles were saved at {starFnameOut}")

        del particlesDataSet
        gc.collect()
        torch.cuda.empty_cache()
        return finalParticlesStar


@lru_cache(1)
def _get_begin_end_from_max_shift(image_shape, max_shift):
    if max_shift is None:
        return 0, -1, 0, -1
    h, w = image_shape
    one_minus_max_shift = 1 - max_shift
    delta_h = math.ceil((h * one_minus_max_shift) / 2)
    h0 = delta_h
    h1 = h - delta_h

    delta_w = math.ceil((w * one_minus_max_shift) / 2)
    w0 = delta_w
    w1 = w - delta_w

    return h0, h1, w0, w1

def _extract_ccor_max(corrs, h0, h1, w0, w1):
    pixelShiftsXY = torch.empty(corrs.shape[:-2] + (2,), device=corrs.device, dtype=torch.int64)
    if h0 > 0:
        corrs = corrs[..., h0:h1, w0:w1]
    maxCorrsJ, maxIndxJ = corrs.max(-1)
    perImgCorr, maxIndxI = maxCorrsJ.max(-1)
    pixelShiftsXY[..., 1] = h0 + maxIndxI
    pixelShiftsXY[..., 0] = w0 + torch.gather(maxIndxJ, -1, maxIndxI.unsqueeze(-1)).squeeze(-1)

    return perImgCorr, pixelShiftsXY

def extract_ccor_max(corrs, max_shift_fraction):
    h0, h1, w0, w1 = _get_begin_end_from_max_shift(corrs.shape[-2:], max_shift_fraction)
    return _extract_ccor_max(corrs, h0, h1, w0, w1)



def align_star(
        reference_vol: str,
        particles_star_fname: str,
        out_fname: str,
        particles_dir: Optional[str],
        mask_radius_angs: Optional[float] = None,
        grid_distance_degs: float = 8.0,
        grid_step_degs: float = 2.0,
        return_top_k_poses: int = 1,
        filter_resolution_angst: Optional[float] = None,
        num_dataworkers: int = 1,
        batch_size: int = 1024,
        use_cuda: bool = True,
        verbose: bool = True,
        float32_matmul_precision: Literal["highest", "high", "medium"] = "high",
        gpu_id: Optional[int] = None,
        n_first_particles: Optional[int] = None,
        correct_ctf: bool = True,
        halfmap_subset: Optional[Literal["1", "2"]] = None,
):
    """

    :param reference_vol: Path to the reference volume file (e.g., .mrc).
    :param particles_star_fname: Input STAR file containing particle metadata.
    :param out_fname: Path for the output STAR file with aligned particle poses.
    :param particles_dir: Root directory for particle image paths if they are relative in the STAR file.
    :param mask_radius_angs: Radius of the circular mask to apply to particles, in Angstroms.
    :param grid_distance_degs: Angular search range around the initial orientation, in degrees.
    :param grid_step_degs: Step size for the angular search grid, in degrees.
    :param return_top_k_poses: Number of top-scoring poses to save for each particle.
    :param filter_resolution_angst: Low-pass filter the reference volume to this resolution (Angstroms) before matching.
    :param num_dataworkers: Number of CPU workers for data loading
    :param batch_size: Number of particles to process in each batch on each job.
    :param use_cuda: If True, use a CUDA-enabled GPU for processing.
    :param verbose: If True, print progress and informational messages.
    :param float32_matmul_precision: Precision for torch.set_float32_matmul_precision ('highest', 'high', 'medium').
    :param gpu_id: Specific GPU ID to use when use_cuda is True.
    :param n_first_particles: Process only the first N particles from the input STAR file.
    :param correct_ctf: If True, apply CTF correction during matching.
    :param halfmap_subset: Process only a specific random subset ('1' or '2') of particles for half-map validation.
    :return: particles: ParticlesStarSet
    """

    import torch.multiprocessing as mp

    # Torch setup
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    torch.set_float32_matmul_precision(float32_matmul_precision)
    _num_dataworkers = 1 if num_dataworkers == 0 else num_dataworkers
    torch.set_num_interop_threads(_num_dataworkers)
    torch.set_num_threads(_num_dataworkers)

    if halfmap_subset is not None:
        halfmap_subset = int(halfmap_subset)
    # Paths
    reference_vol = os.path.expanduser(reference_vol)
    particles_star_fname = os.path.expanduser(particles_star_fname)
    out_fname = os.path.expanduser(out_fname)
    data_rootdir = os.path.expanduser(particles_dir) if particles_dir else None

    with tempfile.TemporaryDirectory() as tmpdir:
        # Optional limit subset
        if n_first_particles is not None:
            star_data = starfile.read(particles_star_fname)
            particles_df = star_data["particles"]
            optics_df = star_data["optics"]
            particles_df = particles_df[:n_first_particles]
            star_in_limited = os.path.join(tmpdir, f"input_particles_{os.path.basename(particles_star_fname)}")
            starfile.write({"optics": optics_df, "particles": particles_df}, star_in_limited)
            particles_star_fname = star_in_limited


        # Build aligner and run
        aligner = ProjectionMatcher(
            reference_vol=reference_vol,
            grid_distance_degs=grid_distance_degs,
            grid_step_degs=grid_step_degs,
            top_k_poses_localref=return_top_k_poses,
            max_resolution_A=filter_resolution_angst,
            verbose=verbose,
            correct_ctf=correct_ctf,
            mask_radius_angs=mask_radius_angs
        )

        device = "cuda" if use_cuda else "cpu"
        if gpu_id is not None and use_cuda:
            device = f"cuda:{gpu_id}"

        particles = aligner.align_star(
                particles_star_fname,
                out_fname,
                data_rootdir=data_rootdir,
                batch_size=batch_size,
                device=device,
                halfset=halfmap_subset
            )
        return particles

# CLI entry
if __name__ == "__main__":
    import sys, shlex

    print(' '.join(shlex.quote(arg) for arg in sys.argv[1:]))
    from argParseFromDoc import parse_function_and_call

    parse_function_and_call(align_star)

"""

python -m cryoPARES.projmatching.projMatcher \
--reference_vol ~/cryo/data/preAlignedParticles/EMPIAR-10166/data/donwsampled/output_volume.mrc \
--particles_star_fname ~/cryo/data/preAlignedParticles/EMPIAR-10166/data/donwsampled/down1000particles.star \
--particles_dir ~/cryo/data/preAlignedParticles/EMPIAR-10166/data/donwsampled/ \
--out_fname /tmp/pruebaCryoparesProjMatch.star \
--n_first_particles 100 \
--grid_distance_degs 15 \
--grid_step_degs 5 \
--filter_resolution_angst 9 \
--batch_size 2

# --grid_distance_degs 15 --grid_step_degs 5 ==> Grid goes from -15 to + 15
"""
