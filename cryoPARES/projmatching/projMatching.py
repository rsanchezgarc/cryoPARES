import gc
import os
import tempfile
import threading
import math
from functools import cached_property, lru_cache
from typing import Tuple, Iterable, Optional, Literal

import numpy as np
import starfile
import torch
from cryoPARES.constants import RELION_EULER_CONVENTION
from torch import nn
from joblib import Parallel, delayed
from more_itertools import chunked
from torch_scatter import scatter_max, scatter_sum

from cryoPARES.configs.mainConfig import main_config
from cryoPARES.geometry.convert_angles import euler_angles_to_matrix
from cryoPARES.utils.reconstructionUtils import get_vol
from .loggers import getWorkerLogger
from .myProgressBar import myTqdm as tqdm

from .tensorCache.lookAheadCache import LookAheadTensorCache
from .dataUtils.dataCaches import create_particles_cache
from .dataUtils.filterToResolution import low_pass_filter_fname

from .dataUtils.dataTypes import IMAGEFNAME_MRCIMAGE, FNAME
from starstack import ParticlesStarSet
from .dataUtils.idxsToBatches import IdxsBatcher
from .metrics import euler_degs_diff, shifst_angs_diff
from .numbaUtils import (
    init_int_to_intList_dict,
    typed_dict_add_keyList_valList,
)
from .so3grid import SO3_discretizer
from .torchUtils import OnlineStatsRecorder

# Fourier-side ops and dataset
from .fourierOperations import (
    correlate_dft_2d,
    extract_central_slices_rfft,
    phase_shift_dft_2d,
    compute_dft,
)
from .preprocessParticles import ParticlesFourierDataset

REPORT_ALIGNMENT_DISPLACEMENT = True
def get_rotmat(degAngles, convention:str=RELION_EULER_CONVENTION, device="cpu"):
    return euler_angles_to_matrix(torch.deg2rad(degAngles), convention=convention).to(device)

class Aligner(nn.Module):
    """
    Single concrete aligner (Fourier pipeline).
    This class folds together the previous abstract base + Fourier subclass.
    """

    def __init__(
        self,
        reference_vol: IMAGEFNAME_MRCIMAGE,
        pixel_size: float | None = None,
        grid_distance_degs: float | Tuple[float, float, float] = 8.0,
        grid_step_degs: float | Tuple[float, float, float] = 2.0,
        max_resolution_A: Optional[float] = None,
        padding_factor: Optional[float] = 0.0,
        keep_top_k_values: int = 2,
        n_cpus: int = 1,
        verbose: bool = True,
        correct_ctf: bool = True,
    ):
        super().__init__()
        self.grid_distance_degs = grid_distance_degs
        self.grid_step_degs = grid_step_degs
        self.max_resolution_A = max_resolution_A
        self.padding_factor = padding_factor
        self.keep_top_k_values = keep_top_k_values
        self.max_shift_fraction = main_config.projmatching.max_shift_fraction

        self._store_reference_vol(reference_vol, pixel_size)
        self.so3_discretizer = SO3_discretizer(
            SO3_discretizer.pick_hp_order(self.grid_step_degs), verbose=verbose
        )
        self.n_cpus = n_cpus if n_cpus > 0 else 1
        self.verbose = verbose
        self.correct_ctf = correct_ctf
        self.mainLogger = getWorkerLogger(self.verbose)

    # ----------------------- Basic props/helpers -----------------------

    @property
    def device(self):
        return self.reference_vol.device

    def _get_so3_delta(self, device: torch.device):
        """
        Build a cached SO(3) delta grid (degrees) around (0,0,0) using
        self.grid_distance_degs and self.grid_step_degs.
        """
        if not hasattr(self, "_so3_delta"):
            if not isinstance(self.grid_distance_degs, Iterable):
                grid_distance_degs = (
                    self.grid_distance_degs,
                    self.grid_distance_degs,
                    self.grid_distance_degs,
                )
            else:
                grid_distance_degs = self.grid_distance_degs

            if not isinstance(self.grid_step_degs, Iterable):
                grid_step_degs = (
                    self.grid_step_degs,
                    self.grid_step_degs,
                    self.grid_step_degs,
                )
            else:
                grid_step_degs = self.grid_step_degs

            get_grid = lambda dist, resol: torch.linspace(
                -dist, dist, int(1 + 2 * (dist / resol)), device=device
            )
            a = get_grid(grid_distance_degs[0], grid_step_degs[0])
            b = get_grid(grid_distance_degs[1], grid_step_degs[1])
            c = get_grid(grid_distance_degs[2], grid_step_degs[2])
            self._so3_delta = torch.cartesian_prod(a, b, c).T
        return self._so3_delta

    def get_so3_grid(self, eulers_degs: torch.Tensor) -> torch.Tensor:
        """
        Expand a center Euler (Bx3 degs) to a grid (BxKx3) over SO(3).
        """
        delta = self._get_so3_delta(eulers_degs.device)
        return (eulers_degs[..., None] + delta[None, ...]).permute(0, 2, 1)

    # ----------------------- Fourier-specific core -----------------------

    def _store_reference_vol(
        self, reference_vol: IMAGEFNAME_MRCIMAGE, pixel_size: float | None = None
    ):
        """
        Load volume, move to Fourier domain (rfft & shift), register buffers and
        set up correlation choice.
        """
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
        reference_vol, vol_shape, pad_length = compute_dft(
            reference_vol, pad_length=pad_length
        )
        self.pad_length = pad_length

        self.register_buffer("reference_vol", reference_vol)

        self.vol_shape = vol_shape
        self.image_shape = vol_shape[-2:]


        self._correlateF = self._correlateCrossCorrelation


    def projectF(self, rotMats: torch.Tensor) -> torch.Tensor:
        return extract_central_slices_rfft(
            self.reference_vol,
            image_shape=self.vol_shape,
            rotation_matrices=rotMats,
            rotation_matrix_zyx=False,
        )

    def _shiftF(self, projs: torch.Tensor, shift_fraction: torch.Tensor) -> torch.Tensor:
        return phase_shift_dft_2d(
            projs, image_shape=self.image_shape, shifts=shift_fraction, rfft=True, fftshifted=True
        )

    def correlateF(self, parts: torch.Tensor, projs: torch.Tensor):
        return self._correlateF(parts, projs)

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
        corrs = correlate_dft_2d(
            parts, projs, max_freq_pixels=self.max_resoluton_freq_pixels
        )
        return self._extract_ccor_max(corrs, max_shift_fraction=self.max_shift_fraction)

    def _correlateFRC(self, parts: torch.Tensor, projs: torch.Tensor):
        shift_fractions = torch.tensor([-0.1, 0.0, 0.1], device=projs.device)
        shift_fractions = torch.cartesian_prod(shift_fractions, shift_fractions)
        shifted_projs = self._shiftF(projs, shift_fractions)
        parts_repeat = parts.unsqueeze(1).expand(-1, shifted_projs.shape[1], -1, -1)
        corrs = self.correlator(parts_repeat, shifted_projs).sum(-1)
        shifts_idxs = corrs.argmax(-1)
        fracitonShiftsXY = shift_fractions[shifts_idxs]
        pixelShiftsXY = (fracitonShiftsXY * parts.shape[-2]).round().long()
        perImgCorr = corrs.amax(-1)
        return perImgCorr, pixelShiftsXY

    # ----------------------- Shared helpers originally in base -----------------------

    def _compute_projections(self, so3_degs_grid: torch.Tensor):
        so3_degs_grid = so3_degs_grid.to(self.reference_vol.device, non_blocking=True)
        rotMats = get_rotmat(so3_degs_grid, device=so3_degs_grid.device)
        projs = self.projectF(rotMats)
        return projs, so3_degs_grid

    def preprocess_particles(
        self,
        particles: FNAME | ParticlesStarSet,
        data_rootdir,
        particle_radius_angs,
        batch_size,
        device,
    ):
        # Load particle metadata
        if not isinstance(particles, ParticlesStarSet):
            particlesSet = ParticlesStarSet(particles, data_rootdir)
        else:
            particlesSet = particles

        sampling_rate = particlesSet.sampling_rate
        assert np.isclose(
            sampling_rate, self.vol_voxel_size, atol=1e-2
        ), (
            "Error, particles and volume have different pixel_size "
            f"{sampling_rate} {self.vol_voxel_size}"
        )
        try:
            particle_shape = particlesSet.particle_shape
        except FileNotFoundError:
            self.mainLogger.error(
                ">> particles mrcs file not found. Did you forget including --particles_dir"
            )
            raise

        assert np.isclose(particle_shape, self.ori_image_shape).all(), (
            "Error, particles and volume have different number of pixels"
        )

        # NOTE: Directly use the Fourier dataset here; no class attribute indirection.
        particlesDataset = ParticlesFourierDataset(
            particlesSet,
            mmap_dirname=None,
            particle_radius_angs=particle_radius_angs,
            pad_length=self.pad_length,
            device=device,
            batch_size=3 * batch_size,
            n_jobs=self.n_cpus,
            verbose=self.verbose,
        )

        dataDict = particlesDataset.dataDict
        pose_to_particleIdx = init_int_to_intList_dict()

        threadLock = threading.Lock()

        def preprocess_angles_batch(idxs):
            batch_size = len(idxs)
            eulerDegs = dataDict[idxs]["eulerDegs"].view(-1, 3).cpu()
            eulerDegs_grid = self.get_so3_grid(eulerDegs)
            poseIdxs = self.so3_discretizer.eulerDegs_to_idx(eulerDegs_grid).view(
                batch_size, -1
            )
            poseIdxs = [poseIdxs[i, ...].unique(sorted=True) for i in range(poseIdxs.shape[0])]
            _parts_idxs = [
                np.ones(pIdxs.shape[-1], dtype=np.int64) * idxs[i]
                for i, pIdxs in enumerate(poseIdxs)
            ]
            poseIdxs = np.concatenate(poseIdxs)
            with threadLock:
                typed_dict_add_keyList_valList(
                    pose_to_particleIdx, poseIdxs, np.concatenate(_parts_idxs)
                )
            return poseIdxs.shape[0]

        # Parallel angle pre-discretization/grouping
        part_orientation_batch = min(
            1 + (8 * batch_size // self.n_cpus), particlesDataset.n_partics
        )
        n_jobs = self.n_cpus
        angles_idxs = Parallel(n_jobs=n_jobs, backend="threading", return_as="generator")(
            delayed(preprocess_angles_batch)(batchIdxs)
            for batchIdxs in chunked(
                range(particlesDataset.n_partics), n=part_orientation_batch
            )
        )
        n_items = 0
        for n_i in tqdm(
            angles_idxs,
            total=particlesDataset.n_partics // part_orientation_batch
            + bool(particlesDataset.n_partics % part_orientation_batch),
            desc="Discretizing and grouping orientations",
            disable=not self.verbose,
        ):
            n_items += n_i

        return particlesDataset, pose_to_particleIdx, n_items

    @torch.inference_mode()
    def align_star(
        self,
        particles: FNAME | ParticlesStarSet,
        starFnameOut: FNAME,
        data_rootdir: str | None = None,
        particle_radius_angs=None,
        batch_size=256,
        device="cuda",
        particles_gpu_cache_size=1024,
        fft_in_device=False,
    ) -> ParticlesStarSet:
        """
        Align particles (input STAR file or ParticlesStarSet) to the reference.
        Writes a STAR with predicted poses/shifts if starFnameOut provided.
        """
        if starFnameOut is not None:
            assert not os.path.isfile(
                starFnameOut
            ), f"Error, the starFnameOut {starFnameOut} already exists"

        particlesDataSet, pose_to_particleIdx, n_items = self.preprocess_particles(
            particles,
            data_rootdir,
            particle_radius_angs,
            batch_size,
            device if fft_in_device else "cpu",
        )
        pixel_size = particlesDataSet.sampling_rate
        half_particle_size = self.image_shape[-1] / 2
        n_particles = len(particlesDataSet)

        results_corr_matrix = torch.zeros(n_particles, self.keep_top_k_values)
        # Storing poseIdx, shiftXpx, shiftYpx (int) for top-k
        results_info_matrix = torch.zeros(
            n_particles, self.keep_top_k_values, 3, dtype=torch.int64
        )
        softmax_denominator = torch.zeros(n_particles, dtype=torch.float64)

        particlesDataSet.device = "cpu"
        data_cache = create_particles_cache(
            particlesDataSet,
            particles_gpu_cache_size,
            10 * particles_gpu_cache_size,
            l1_device=device,
        )

        self.to(device)
        self.mainLogger.info(f"Total number of particles: {particlesDataSet.n_partics}")
        self.mainLogger.info(
            f"Total number of poses to explore: {len(pose_to_particleIdx.keys())}/{self.so3_discretizer.grid_size}"
        )
        idxBatcher = IdxsBatcher(
            self.so3_discretizer,
            pose_to_particleIdx,
            self.n_cpus,
            sorting_method="none",
            verbose=self.verbose,
        )

        total_num_orientation_batches = n_items // batch_size + bool(n_items % batch_size)
        self.mainLogger.info(f"Total number of batches: {total_num_orientation_batches}")

        minimal_projs = self._compute_projections(torch.randn(1, 3))[0]
        renumbered_to_ori_poseIdxs = idxBatcher.renumbered_to_ori_poseIdxs

        class TensorCacheProjs(LookAheadTensorCache):
            def compute_idxs(other, idxs):
                degs = self.so3_discretizer.idx_to_eulerDegs(
                    renumbered_to_ori_poseIdxs[idxs]
                )
                return self._compute_projections(degs)[0]

        projs_cache = TensorCacheProjs(
            cache_size=batch_size,
            max_index=idxBatcher.maxindex,
            tensor_shape=minimal_projs.shape[1:],
            dtype=minimal_projs.dtype,
            data_device=device,
        )
        self.mainLogger.info("Projections cache initialized. Starting!!")

        statsRecorder = OnlineStatsRecorder(particlesDataSet.n_partics, dtype=torch.float64)
        delta_so3_size = self._get_so3_delta(device).shape[-1]

        for (flatten_poses_idxs, flatten_img_idxs) in tqdm(
            idxBatcher.yield_batchIdxs(batch_size),
            total=total_num_orientation_batches,
            desc="Aligning orientations",
            disable=not self.verbose,
        ):
            parts, ctfs = data_cache[flatten_img_idxs]
            projs = projs_cache[flatten_poses_idxs]

            unique_flatten_img_idxs, local_img_idxs = flatten_img_idxs.unique(
                sorted=False, return_inverse=True
            )
            local_img_idxs = local_img_idxs.to(device, non_blocking=True)

            if self.correct_ctf:
                projs = self._apply_ctfF(projs, ctfs)

            perImgCorr, pixelShiftsXY = self.correlateF(parts, projs)

            statsRecorder.update(perImgCorr.cpu(), flatten_img_idxs)

            (
                results_corr_matrix,
                results_info_matrix,
                softmax_denominator,
            ) = self.update_results_status(
                perImgCorr,
                pixelShiftsXY,
                results_corr_matrix,
                results_info_matrix,
                softmax_denominator,
                unique_flatten_img_idxs,
                local_img_idxs,
                renumbered_to_ori_poseIdxs,
                flatten_poses_idxs,
            )

        results_corr_matrix, idxs = results_corr_matrix.sort(dim=1)
        rows = torch.arange(results_info_matrix.size(0)).unsqueeze(1).expand(
            -1, results_info_matrix.size(1)
        )
        results_info_matrix = results_info_matrix[rows, idxs]

        predEulerDegs = self.so3_discretizer.idx_to_eulerDegs(
            results_info_matrix[..., 0].view(-1)
        ).view(*results_info_matrix.shape[:2], 3)

        predShiftsAngs = -(results_info_matrix[..., 1:].float() - half_particle_size) * pixel_size

        if delta_so3_size > 3:
            _mean, _std = statsRecorder.get_mean(), statsRecorder.get_standard_deviation()
            prob_x_y = torch.distributions.normal.Normal(
                _mean.unsqueeze(-1), _std.unsqueeze(-1)
            ).cdf(results_corr_matrix)
        elif delta_so3_size == 1:
            prob_x_y = torch.ones_like(results_corr_matrix)
        else:
            raise NotImplementedError("I don't know how to compute p(x|y) with very little examples")

        n_topK = predEulerDegs.shape[1]
        finalParticlesStar = None
        for complement_topK in range(n_topK):
            topK = n_topK - 1 - complement_topK
            particlesStar = particlesDataSet.get_particles_starstack(drop_rlnImageId=True)

            parts_range = range(results_info_matrix.shape[0])

            if predEulerDegs.shape[1] > 1:
                colname2change = {"copyNumber": topK}
                particlesStar.updateMd(None, None, colname2change)

            confidence = particlesDataSet.dataDict["confidences"].sum(-1)
            particlesStar.updateMd(idxs=parts_range, colname2change={"previousConfidenceScore": confidence.numpy()})

            confidence *= prob_x_y[:, topK]
            particlesStar.setPose(
                parts_range,
                eulerDegs=predEulerDegs[:, topK, ...].numpy(),
                shiftsAngst=predShiftsAngs[:, topK, ...].numpy(),
                confidence=confidence.numpy(),
            )

            particlesStar.updateMd(
                idxs=parts_range, colname2change={"bruteForceScore": results_corr_matrix[:, topK].numpy()}
            )
            particlesStar.updateMd(
                idxs=parts_range, colname2change={"bruteForceScoreNormalized": prob_x_y[:, topK].numpy()}
            )
            if complement_topK > 0:
                finalParticlesStar += particlesStar
            else:
                finalParticlesStar = particlesStar

        if starFnameOut is not None:
            finalParticlesStar.save(starFname=starFnameOut)
            self.mainLogger.info(f"particles were saved at {starFnameOut}")

        if REPORT_ALIGNMENT_DISPLACEMENT:
            ori_eulers = particlesDataSet.dataDict.get("eulerDegs")[
                range(predEulerDegs.shape[0]), ...
            ]
            ori_shifts = particlesDataSet.dataDict.get("shiftsAngs")[
                range(predEulerDegs.shape[0]), ...
            ]
            best_value = float("inf")
            for i in range(ori_eulers.shape[1]):
                for j in range(predEulerDegs.shape[1]):
                    degs_angular_displacement = euler_degs_diff(
                        predEulerDegs[:, j, :], ori_eulers[:, i, :]
                    )
                    angst_translation_displacement = shifst_angs_diff(
                        predShiftsAngs[:, j, :], ori_shifts[:, i, :]
                    )
                    degs_mean = degs_angular_displacement.mean()
                    if best_value > degs_mean:
                        best_value = degs_mean
                        degs_mean, degs_std = (
                            degs_angular_displacement.mean(),
                            degs_angular_displacement.std(),
                        )
                        degs_median = np.median(degs_angular_displacement)
                        degs_iqr = np.quantile(degs_angular_displacement, 0.75) - np.quantile(
                            degs_angular_displacement, 0.25
                        )

            agst_mean, agst_std = angst_translation_displacement.mean(), angst_translation_displacement.std()
            agst_median = np.median(angst_translation_displacement)
            agst_iqr = (np.quantile(angst_translation_displacement, 0.75) -
                        np.quantile(angst_translation_displacement, 0.25))
            self.mainLogger.info(
                "Alignment displacements mean (std)\tmedian (iqr)\n"
                f"Angle: {degs_mean:2.3f} +/- {degs_std:2.3f}\t{degs_median:2.3f} +/- {degs_iqr:2.3f}\n"
                f"Shift: {agst_mean:2.3f} +/- {agst_std:2.3f}\t{agst_median:2.3f} +/- {agst_iqr:2.3f} "
            )

        del particlesDataSet, projs_cache, data_cache, pose_to_particleIdx
        gc.collect()
        torch.cuda.empty_cache()
        return finalParticlesStar

    def update_results_status(self, perImgCorr, pixelShiftsXY, results_corr_matrix, results_info_matrix,
                              softmax_denominator,
                              unique_flatten_img_idxs, local_img_idxs, renumbered_to_ori_poseIdxs, flatten_poses_idxs):
        #TODO: This is not super efficient, but it is not the bottleneck

        _denominator = scatter_sum(perImgCorr.double().exp(), local_img_idxs)
        softmax_denominator[unique_flatten_img_idxs] += _denominator.cpu()  # This goes to +inf. We need normalized values.

        prevCorr = results_corr_matrix[unique_flatten_img_idxs]
        prevInfo = results_info_matrix[unique_flatten_img_idxs, ...]
        for k in range(self.keep_top_k_values):
            bestLocalCorrs, bestLocalImgIdxs = scatter_max(perImgCorr, local_img_idxs)
            bestLocalpixelShiftsXY = pixelShiftsXY[bestLocalImgIdxs]
            bestLocalCorrs = bestLocalCorrs.cpu()
            bestLocalImgIdxs = bestLocalImgIdxs.cpu()
            bestLocalpixelShiftsXY = bestLocalpixelShiftsXY.cpu()

            bestLocalPoseIdx = renumbered_to_ori_poseIdxs[
                flatten_poses_idxs[bestLocalImgIdxs]]

            corrDiff = (bestLocalCorrs.unsqueeze(-1) - prevCorr)
            replacebleMask = corrDiff > 0
            corrDiffMask = torch.count_nonzero(replacebleMask,
                                               -1).bool()  # Why bool? Beacause we will be indexing the dim=0 in prevCorr, with shape[1], and we only want to update the results if there is any positive case
            replaceIdxs = (corrDiff + (~replacebleMask) * -1e7).argmax(-1)  # 1e7 is ~ inf, since we cannot use inf, that generates nan
            replaceIdxs = replaceIdxs[corrDiffMask]  # This is to rule out rows that do not need to be updated
            prevCorr[corrDiffMask, replaceIdxs] = bestLocalCorrs[corrDiffMask]

            newInfo = torch.cat([bestLocalPoseIdx.unsqueeze(-1), bestLocalpixelShiftsXY], -1)
            prevInfo[corrDiffMask, replaceIdxs, :] = newInfo[corrDiffMask, :]
            perImgCorr[bestLocalImgIdxs] *= 0

        results_corr_matrix[unique_flatten_img_idxs] = prevCorr
        results_info_matrix[unique_flatten_img_idxs, ...] = prevInfo
        return results_corr_matrix, results_info_matrix, softmax_denominator

    def _extract_ccor_max_unconstrained(self, corrs):
        pixelShiftsXY = torch.empty(corrs.shape[0], 2, device=corrs.device, dtype=torch.int64)
        maxCorrsJ, maxIndxJ = corrs.max(-1)
        perImgCorr, maxIndxI = maxCorrsJ.max(-1)
        pixelShiftsXY[:, 1] = maxIndxI
        pixelShiftsXY[:, 0] = torch.gather(maxIndxJ, 1, maxIndxI.unsqueeze(1)).squeeze(1)
        return perImgCorr, pixelShiftsXY

    @classmethod
    @lru_cache(1)
    def _get_begin_end_from_max_shift(cls, image_shape, max_shift_fraction):
        h, w = image_shape
        one_minux_max_shift = 1 - max_shift_fraction
        delta_h = math.ceil((h * one_minux_max_shift) / 2)
        h0 = delta_h
        h1 = h - delta_h

        delta_w = math.ceil((w * one_minux_max_shift) / 2)
        w0 = delta_w
        w1 = w - delta_w

        return h0, h1, w0, w1

    def _extract_ccor_max(self, corrs, max_shift_fraction):
        pixelShiftsXY = torch.empty(corrs.shape[0], 2, device=corrs.device, dtype=torch.int64)
        h0, h1, w0, w1 = self._get_begin_end_from_max_shift(corrs.shape[-2:], max_shift_fraction)
        corrs = corrs[..., h0:h1, w0:w1]
        maxCorrsJ, maxIndxJ = corrs.max(-1)
        perImgCorr, maxIndxI = maxCorrsJ.max(-1)
        pixelShiftsXY[:, 1] = h0 + maxIndxI
        pixelShiftsXY[:, 0] = w0 + torch.gather(maxIndxJ, 1, maxIndxI.unsqueeze(1)).squeeze(1)
        return perImgCorr, pixelShiftsXY


def align_star(
    reference_vol: str,
    star_fname: str,
    out_fname: str,
    particles_dir: Optional[str],
    particle_radius_angs: Optional[float] = None,
    grid_distance_degs: float = 8.0,
    grid_step_degs: float = 2.0,
    return_top_k_poses: int = 1,
    padding_factor: float = 0.0,
    filter_resolution_angst: Optional[float] = None,
    n_cpus_per_job: int = 1,
    batch_size: int = 1024,
    cache_factor: float = 5.0,
    use_cuda: bool = True,
    verbose: bool = True,
    torch_matmul_precision: Literal["highest", "high", "medium"] = "high",
    fft_in_cuda: bool = True,
    gpu_id: Optional[int] = None,
    n_first_particles: Optional[int] = None,
    correct_ctf: bool = True,
):
    """

    :param reference_vol:
    :param star_fname:
    :param out_fname:
    :param particles_dir:
    :param particle_radius_angs:
    :param grid_distance_degs:
    :param grid_step_degs:
    :param return_top_k_poses:
    :param padding_factor:
    :param filter_resolution_angst:
    :param n_cpus_per_job:
    :param batch_size:
    :param cache_factor:
    :param use_cuda:
    :param verbose:
    :param torch_matmul_precision:
    :param fft_in_cuda:
    :param gpu_id:
    :param n_first_particles:
    :param correct_ctf:
    :return:
    """


    import torch.multiprocessing as mp

    # Torch setup
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    torch.set_float32_matmul_precision(torch_matmul_precision)
    _n_cpus_per_job = 1 if n_cpus_per_job == 0 else n_cpus_per_job
    torch.set_num_interop_threads(_n_cpus_per_job)
    torch.set_num_threads(_n_cpus_per_job)

    # Paths
    reference_vol = os.path.expanduser(reference_vol)
    star_fname = os.path.expanduser(star_fname)
    out_fname = os.path.expanduser(out_fname)
    data_rootdir = os.path.expanduser(particles_dir) if particles_dir else None

    with tempfile.TemporaryDirectory() as tmpdir:
        # Optional limit subset
        ori_star_fname = None
        if n_first_particles is not None:
            star_data = starfile.read(star_fname)
            particles_df = star_data["particles"]
            optics_df = star_data["optics"]
            particles_df = particles_df[:n_first_particles]
            star_in_limited = os.path.join(tmpdir, f"input_particles_{os.path.basename(star_fname)}")
            starfile.write({"optics": optics_df, "particles": particles_df}, star_in_limited)
            star_fname = star_in_limited


        # Optional low-pass filter of the reference volume
        if filter_resolution_angst is not None:
            new_reference_vol = os.path.join(tmpdir, f"input_vol_{os.path.basename(reference_vol)}")
            low_pass_filter_fname(reference_vol, resolution=filter_resolution_angst, out_fname=new_reference_vol)
            reference_vol = new_reference_vol

        # Build aligner and run
        aligner = Aligner(
            reference_vol=reference_vol,
            grid_distance_degs=grid_distance_degs,
            grid_step_degs=grid_step_degs,
            keep_top_k_values=return_top_k_poses,
            padding_factor=padding_factor,
            max_resolution_A=filter_resolution_angst,
            n_cpus=_n_cpus_per_job,
            verbose=verbose,
            correct_ctf=correct_ctf,
        )

        device = "cuda" if use_cuda else "cpu"
        if gpu_id is not None and use_cuda:
            device = f"cuda:{gpu_id}"

        aligner.align_star(
            star_fname,
            out_fname,
            data_rootdir=data_rootdir,
            batch_size=batch_size,
            particle_radius_angs=particle_radius_angs,
            particles_gpu_cache_size=int(cache_factor * batch_size),
            device=device,
            fft_in_device=fft_in_cuda,
        )


# CLI entry
if __name__ == "__main__":
    import sys, shlex
    print(' '.join(shlex.quote(arg) for arg in sys.argv[1:]))
    from argParseFromDoc import parse_function_and_call
    parse_function_and_call(align_star)
