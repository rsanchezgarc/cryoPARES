import gc
import os
import tempfile
import math
from functools import cached_property, lru_cache
from typing import Tuple, Iterable, Optional, Literal

import numpy as np
import starfile
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader, default_collate
from cryoPARES.constants import (RELION_EULER_CONVENTION, BATCH_POSE_NAME, RELION_PRED_POSE_CONFIDENCE_NAME,
                                 BATCH_ORI_IMAGE_NAME, BATCH_ORI_CTF_NAME, RELION_ANGLES_NAMES, RELION_SHIFTS_NAMES,
                                 BATCH_IDS_NAME, RELION_IMAGE_FNAME)
from torch import nn
from cryoPARES.configs.mainConfig import main_config
from cryoPARES.geometry.convert_angles import euler_angles_to_matrix, matrix_to_euler_angles
from cryoPARES.geometry.grids import so3_near_identity_grid_cartesianprod
from cryoPARES.geometry.metrics_angles import rotation_error_with_sym
from cryoPARES.utils.reconstructionUtils import get_vol
from .loggers import getWorkerLogger
from .myProgressBar import myTqdm as tqdm
from joblib.externals.loky.backend import get_context

from .dataUtils.filterToResolution import low_pass_filter_fname

from .dataUtils.dataTypes import IMAGEFNAME_MRCIMAGE, FNAME
from starstack import ParticlesStarSet
from .metrics import euler_degs_diff, shifst_angs_diff

# Fourier-side ops and dataset
from .fourierOperations import (
    correlate_dft_2d,
    extract_central_slices_rfft,
    compute_dft,
)
from .preprocessParticles import ParticlesFourierDataset, _compute_one_batch_fft, _getMask

REPORT_ALIGNMENT_DISPLACEMENT = True
def get_rotmat(degAngles, convention:str=RELION_EULER_CONVENTION, device="cpu"):
    return euler_angles_to_matrix(torch.deg2rad(degAngles), convention=convention).to(device)

class ProjectionMatcher(nn.Module):
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
        keep_top_k_values: int = 1,
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
        if not hasattr(self, "_so3_delta") or self._so3_delta.device != device:
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

            n_angles = math.ceil(self.grid_distance_degs * 2 / self.grid_step_degs)
            n_angles = n_angles if n_angles % 2 == 1 else n_angles + 1  # We always want an odd number
            self._so3_delta = so3_near_identity_grid_cartesianprod(self.grid_distance_degs, n_angles,
                                                             transposed=False, degrees=True,
                                                             remove_duplicates=False).to(device)
            #TODO: Remove duplicates=True makes the whole things broken. Probably due to euler angles singularities
        return self._so3_delta

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
        assert len(self.vol_shape) == 3
        assert len(set(self.vol_shape)) == 1, "Only cubic volumes are allowed"
        assert self.image_shape[-2] % 2 == 0, "Only even boxsizes are allowed"
        self.half_particle_size = 0.5 * self.image_shape[-2]

        # Prepare mask
        radius_px = self.image_shape[-2] // 2
        self.register_buffer("rmask",_getMask(radius_px, self.image_shape, device="cpu"))

    def projectF(self, rotMats: torch.Tensor) -> torch.Tensor:
        return extract_central_slices_rfft(
            self.reference_vol,
            image_shape=self.vol_shape,
            rotation_matrices=rotMats,
            rotation_matrix_zyx=False,
        )


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
        corrs = correlate_dft_2d(
            parts, projs, max_freq_pixels=self.max_resoluton_freq_pixels
        )
        b, options, l0, l1 = corrs.shape
        maxcorr, maxcorrIdxs = self._extract_ccor_max(corrs.reshape(-1, *corrs.shape[-2:]),
                                      max_shift_fraction=self.max_shift_fraction)
        del corrs
        return maxcorr.reshape(b, options), maxcorrIdxs.reshape(b, options, 2)

    # ----------------------- Shared helpers originally in base -----------------------

    def _compute_projections(self, so3_degs_grid: torch.Tensor):
        so3_degs_grid = so3_degs_grid.to(self.reference_vol.device)
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
        self.particlesDataset = particlesDataset

        from cryoPARES.datamanager.datamanager import DataManager
        dm = DataManager(particles, #TODO: this does not apply circular mask to the particle
                     symmetry="C1",
                     particles_dir=data_rootdir,
                     halfset=None,
                     batch_size=batch_size,
                     save_train_val_partition_dir=None,
                     is_global_zero=True,
                     num_augmented_copies_per_batch=1,
                     num_data_workers = self.n_cpus,
                     return_ori_imagen = True,
                     subset_idxs=None
                     )

        ds = dm.create_dataset(None)
        self.ds = ds


    def _fourier_forward(self, fparts, ctfs, eulerDegs):
        expanded_eulerDegs = self._get_so3_delta(eulerDegs.device).unsqueeze(0) + eulerDegs.unsqueeze(1)
        bsize = expanded_eulerDegs.size(0)
        expanded_eulerDegs = expanded_eulerDegs.reshape(bsize, -1, 3)  # unrolling all the input angles into one
        # TODO: Remove duplicate angles if n_input_angles > 1
        projs = self._compute_projections(expanded_eulerDegs.reshape(-1, 3))[0]

        if self.correct_ctf:
            projs = self._apply_ctfF(projs.reshape(bsize, -1, *projs.shape[-2:]), ctfs.unsqueeze(1))

        perImgCorr, pixelShiftsXY = self.correlateF(fparts.unsqueeze(1), projs)

        maxCorrs, maxCorrsIdxs = perImgCorr.topk(self.keep_top_k_values, sorted=True, largest=True, dim=-1)

        batch_idxs_range = torch.arange(pixelShiftsXY.size(0)).unsqueeze(1)
        pixelShiftsXY = pixelShiftsXY[batch_idxs_range, maxCorrsIdxs]
        predEulerDegs = expanded_eulerDegs[batch_idxs_range, maxCorrsIdxs]
        mean_corr = perImgCorr.mean(-1, keepdims=True)
        std_corr = perImgCorr.std(-1, keepdims=True)
        comparedWeight = torch.distributions.Normal(mean_corr, std_corr + 1e-6).cdf(maxCorrs)  # 1-P(I_i > All_images)
        return maxCorrs, predEulerDegs, pixelShiftsXY, comparedWeight

    def forward(self, imgs, ctfs, rotmats):

        imgs = imgs * self.rmask
        fparts = _compute_one_batch_fft(imgs)
        eulerDegs = torch.rad2deg(matrix_to_euler_angles(rotmats, RELION_EULER_CONVENTION))
        maxCorrs, predEulerDegs, pixelShiftsXY, comparedWeight = self._fourier_forward(fparts, ctfs, eulerDegs)
        predShiftsAngsXY = -(pixelShiftsXY.float() - self.half_particle_size) * self.vol_voxel_size
        predRotMats = euler_angles_to_matrix(torch.deg2rad(predEulerDegs), RELION_EULER_CONVENTION)
        return maxCorrs, predRotMats, predShiftsAngsXY, comparedWeight

    @torch.inference_mode()
    def align_star(
        self,
        particles: FNAME | ParticlesStarSet,
        starFnameOut: FNAME,
        data_rootdir: str | None = None,
        particle_radius_angs=None,
        batch_size=256,
        device="cuda",
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

        self.preprocess_particles(
            particles,
            data_rootdir,
            particle_radius_angs,
            batch_size,
            device if fft_in_device else "cpu",
        )

        particlesDataSet = self.ds #self.particlesDataset
        try:
            pixel_size = particlesDataSet.sampling_rate
        except AttributeError:
            pixel_size = particlesDataSet.datasets[0].original_sampling_rate()
        half_particle_size = self.image_shape[-1] / 2
        assert half_particle_size == self.half_particle_size
        n_particles = len(particlesDataSet)

        results_corr_matrix = torch.zeros(n_particles, self.keep_top_k_values)
        results_shifts_matrix = torch.zeros(n_particles, self.keep_top_k_values, 2, dtype=torch.int64)
        results_eulerDegs_matrix = torch.zeros(n_particles, self.keep_top_k_values, 3)
        stats_corr_matrix = torch.zeros(n_particles, self.keep_top_k_values)
        idds_list = [None] * n_particles

        particlesDataSet.device = "cpu"

        self.to(device)
        self.mainLogger.info(f"Total number of particles: {n_particles}")

        try:
            particlesStar = particlesDataSet.get_particles_starstack(drop_rlnImageId=True)
            confidence = particlesDataSet.dataDict["confidences"].sum(-1)
        except AttributeError:
            particlesStar = particlesDataSet.datasets[0].particles.copy()
            try:
                confidence = torch.tensor(particlesStar.particles_md.loc[:,RELION_PRED_POSE_CONFIDENCE_NAME].values,
                                          dtype=torch.float32)
            except KeyError:
                confidence = torch.ones(len(particlesStar.particles_md))
        if "rlnImageId" in particlesStar.particles_md.columns:
            particlesStar.particles_md.drop("rlnImageId", axis=1)

        dl = DataLoader(
                        #particlesDataSet, 
                        self.ds, 
                        batch_size=batch_size,
                        num_workers=self.n_cpus, shuffle=False, pin_memory=True,
                        multiprocessing_context='fork') #get_context('loky')
        non_blocking = False
        _partIdx = 0
        for batch in tqdm(dl, desc="Aligning particles", disable=not self.verbose):

            n_items = len(batch[BATCH_ORI_IMAGE_NAME])
            partIdx = torch.arange(_partIdx, _partIdx + n_items); _partIdx += n_items
            idds = batch[BATCH_IDS_NAME]
            rotmats = batch[BATCH_POSE_NAME][0].to(device, non_blocking=non_blocking)
            parts = batch[BATCH_ORI_IMAGE_NAME].to(device, non_blocking=non_blocking)
            ctfs = batch[BATCH_ORI_CTF_NAME].to(device, non_blocking=non_blocking)
            fparts = _compute_one_batch_fft(parts * self.rmask)
            eulerDegs = torch.rad2deg(matrix_to_euler_angles(rotmats, RELION_EULER_CONVENTION)).unsqueeze(1)


            maxCorrs, predEulerDegs, pixelShiftsXY, comparedWeight = self._fourier_forward(fparts, ctfs, eulerDegs)

            results_corr_matrix[partIdx, :] = maxCorrs.detach().cpu()
            results_shifts_matrix[partIdx, :] = pixelShiftsXY.detach().cpu()
            results_eulerDegs_matrix[partIdx, :] = predEulerDegs.detach().cpu()
            stats_corr_matrix[partIdx, :] = comparedWeight.detach().cpu()
            for _i, i_partIdx in enumerate(partIdx): idds_list[i_partIdx] = idds[_i]

        predEulerDegs = results_eulerDegs_matrix
        predShiftsAngs = -(results_shifts_matrix.float() - half_particle_size) * pixel_size

        prob_x_y = stats_corr_matrix
        n_topK = predEulerDegs.shape[1]
        # finalParticlesStar = None
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
                ######## Debug code
                r1 = torch.FloatTensor(Rotation.from_euler(RELION_EULER_CONVENTION,
                                                           eulerdegs,
                                                           degrees=True).as_matrix())
                r2 = torch.FloatTensor(Rotation.from_euler(RELION_EULER_CONVENTION,
                                                           particles_md.loc[idds_list, angles_names],
                                                           degrees=True).as_matrix())
                ang_err = torch.rad2deg(rotation_error_with_sym(r1, r2, symmetry="C1"))# C1 since we do not use symemtry in local refinement. Ideally we would like to use the proper symmetry for eval purposes

                s2 = particles_md.loc[idds_list, shiftsXYangs_names].values
                shift_error = np.sqrt(((predShiftsAngs - s2) ** 2).sum(-1))
                print(f"Median Ang   Error degs (top-{k + 1}):", np.median(ang_err))
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
        aligner = ProjectionMatcher(
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
            device=device,
            fft_in_device=fft_in_cuda,
        )


# CLI entry
if __name__ == "__main__":
    import sys, shlex
    print(' '.join(shlex.quote(arg) for arg in sys.argv[1:]))
    from argParseFromDoc import parse_function_and_call
    parse_function_and_call(align_star)

"""

--reference_vol /home/sanchezg/cryo/data/preAlignedParticles/EMPIAR-10166/data/allparticles_reconstruct.mrc 
--star_fname ~/cryo/data/preAlignedParticles/EMPIAR-10166/data/allparticles.star 
--particles_dir ~/cryo/data/preAlignedParticles/EMPIAR-10166/data/
--out_fname /tmp/pruebaCryoparesProjMatch.star
--n_first_particles 100
--grid_distance_degs 15
--grid_step_degs 5
--filter_resolution_angst 6
--batch_size 2

# --grid_distance_degs 15 --grid_step_degs 5 ==> Grid goes from -15 to + 15
"""
