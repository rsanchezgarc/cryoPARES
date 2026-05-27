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
from cryoPARES.utils.torch_grid_utils_compat import circle

from autoCLI_config import inject_defaults_from_config, CONFIG_PARAM
from cryoPARES.constants import (RELION_EULER_CONVENTION, BATCH_POSE_NAME, RELION_PRED_POSE_CONFIDENCE_NAME,
                                 BATCH_ORI_IMAGE_NAME, BATCH_ORI_CTF_NAME, RELION_ANGLES_NAMES, RELION_SHIFTS_NAMES,
                                 BATCH_IDS_NAME)
from cryoPARES.utils.torch_fourier_slice_compat import extract_central_slices_rfft_3d
from cryoPARES.utils.paths import MAP_AS_ARRAY_OR_FNAME_TYPE, FNAME_TYPE

from cryoPARES.configs.mainConfig import main_config
from cryoPARES.geometry.convert_angles import euler_angles_to_matrix, matrix_to_euler_angles
from cryoPARES.geometry.grids import so3_near_identity_grid_cartesianprod
from cryoPARES.geometry.metrics_angles import rotation_error_with_sym
from cryoPARES.utils.reconstructionUtils import get_vol
from cryoPARES.projmatching.projmatchingUtils.loggers import getWorkerLogger
from cryoPARES.projmatching.projmatchingUtils.myProgressBar import myTqdm as tqdm

from cryoPARES.projmatching.projmatchingUtils.filterToResolution import low_pass_filter_fname
from cryoPARES.projmatching.projmatchingUtils.fourierOperations import (
    correlate_dft_2d, build_ccorr_sign_grid, compute_dft_3d, _real_to_fourier_2d,
    _mask_for_dft_2d,
)

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

        self.use_subpixel_shifts = main_config.projmatching.use_subpixel_shifts
        self.zero_dc = main_config.projmatching.zero_dc

        # Noise-PSD matched filter (#8a) — warm-up state reset in align_star().
        self.noise_psd_whitening = main_config.projmatching.noise_psd_whitening
        self.noise_psd_warmup_batches = main_config.projmatching.noise_psd_warmup_batches
        self._noise_psd_amp_sum: torch.Tensor | None = None
        self._noise_psd_batches_seen: int = 0

        # SO(3) sub-step interpolation (#7): precomputed neighbor tables for fast O(1) lookup.
        self.use_so3_interpolation = main_config.projmatching.use_so3_interpolation
        if self.use_so3_interpolation:
            # Coarse stage (used in single-stage and two-stage coarse)
            nb_idx, nb_valid = _precompute_so3_interp_neighbors(
                self.grid_distance_degs, self.grid_step_degs
            )
            self._so3_interp_nb_idx = nb_idx      # (nCoarse, 6) long, CPU
            self._so3_interp_nb_valid = nb_valid  # (nCoarse, 6) bool, CPU

        # Two-stage coarse-to-fine search
        self.use_two_stage_search = main_config.projmatching.use_two_stage_search
        self.fine_grid_distance_degs = main_config.projmatching.fine_grid_distance_degs
        self.fine_grid_step_degs = main_config.projmatching.fine_grid_step_degs
        self.fine_top_k = main_config.projmatching.fine_top_k
        if self.use_two_stage_search:
            assert self.fine_top_k >= self.top_k_poses_localref, (
                f"fine_top_k ({self.fine_top_k}) must be >= top_k_poses_localref "
                f"({self.top_k_poses_localref})"
            )
            # Fine-stage SO(3) interpolation neighbor tables
            if self.use_so3_interpolation:
                fine_nb_idx, fine_nb_valid = _precompute_so3_interp_neighbors(
                    self.fine_grid_distance_degs, self.fine_grid_step_degs
                )
                self._fine_so3_interp_nb_idx = fine_nb_idx      # (nFine, 6) long, CPU
                self._fine_so3_interp_nb_valid = fine_nb_valid  # (nFine, 6) bool, CPU

        self._store_reference_vol(reference_vol, pixel_size)

        # Noise PSD buffer (image_shape known after _store_reference_vol).
        if self.noise_psd_whitening:
            H, W_half = self.image_shape[-2], self.image_shape[-2] // 2 + 1
            self.register_buffer("noise_psd_map", torch.ones(1, H, W_half))

        self.verbose = verbose
        self.correct_ctf = correct_ctf
        self.mainLogger = getWorkerLogger(self.verbose)

        if main_config.projmatching.disable_inductor_shape_padding:
            try:
                import torch._inductor.config as _inductor_cfg
                _inductor_cfg.shape_padding = False
            except Exception:
                pass

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

    def _get_so3_delta_rotmats(self, device: torch.device) -> torch.Tensor:
        """Return the coarse Cartesian delta grid as rotation matrices (nDelta, 3, 3). Cached."""
        if (not hasattr(self, "_so3_delta_rotmats_cache") or
                self._so3_delta_rotmats_cache.device != device):
            euler_grid = self._get_so3_delta(device)  # (nDelta, 3) in degrees
            self._so3_delta_rotmats_cache = get_rotmat(euler_grid, device=device)
        return self._so3_delta_rotmats_cache

    def _get_fine_delta_rotmats(self, device: torch.device) -> torch.Tensor:
        """Return fine-pass delta grid as rotation matrices (nFine, 3, 3).

        Uses Cartesian Euler grid (same structure as coarse grid) to enable fast SO(3)
        interpolation via precomputed neighbor tables. Benchmarks show Cartesian consistently
        outperforms Fibonacci on all real targets. Cached on self.
        """
        if (not hasattr(self, "_fine_delta_rotmats_cache") or
                self._fine_delta_rotmats_cache.device != device):
            # Use Cartesian grid for fine stage (enables neighbor table lookup)
            euler_grid = self._get_so3_delta_fine(device)  # (nFine, 3) in degrees
            self._fine_delta_rotmats_cache = get_rotmat(euler_grid, device=device)
        return self._fine_delta_rotmats_cache

    def _get_so3_delta_fine(self, device: torch.device) -> torch.Tensor:
        """Return fine-stage Cartesian Euler delta grid (nFine, 3) in degrees. Cached."""
        if (not hasattr(self, "_so3_delta_fine_cache") or
                self._so3_delta_fine_cache.device != device):
            n_angles = math.ceil(self.fine_grid_distance_degs * 2 / self.fine_grid_step_degs)
            if n_angles % 2 == 0:
                n_angles += 1
            euler_grid = so3_near_identity_grid_cartesianprod(
                self.fine_grid_distance_degs, n_angles,
                transposed=False, degrees=True, remove_duplicates=False
            )
            self._so3_delta_fine_cache = euler_grid.to(device, dtype=torch.float32)
        return self._so3_delta_fine_cache

    # ── Coarse-stage composition functions (set dynamically in __init__) ────────

    def _compose_coarse_euler_add(self, rotmats: torch.Tensor) -> torch.Tensor:
        """Expand input poses over the Cartesian delta grid via Euler angle addition.

        Args:
            rotmats: (B, topK, 3, 3) — input rotation matrices
        Returns:
            (B, topK*nCoarse, 3, 3) — all candidate rotation matrices
        """
        bsize = rotmats.size(0)
        eulerDegs = get_eulers(rotmats)                    # (B, topK, 3)
        euler_delta = self._get_so3_delta(rotmats.device)  # (nCoarse, 3)
        expanded = (euler_delta.unsqueeze(0).unsqueeze(0) +
                    eulerDegs.unsqueeze(2))                 # (B, topK, nCoarse, 3)
        return get_rotmat(
            expanded.reshape(-1, 3), device=rotmats.device
        ).reshape(bsize, -1, 3, 3)

    # ── Delta-grid helpers ───────────────────────────────────────────────────

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

    # ----------------------- Batch-size helpers -----------------------

    @staticmethod
    def _count_rotations(distance_deg: float, step_deg: float, fibo: bool = False) -> int:
        """Return the number of SO(3) grid points without allocating GPU tensors."""
        if fibo:
            from cryoPARES.geometry.grids import so3_grid_near_identity_fibo
            rotmats, _ = so3_grid_near_identity_fibo(
                distance_deg=distance_deg,
                spacing_deg=step_deg,
                use_small_aprox=True,
                output="matrix",
            )
            return len(rotmats) + 1  # +1 for prepended identity
        else:
            n_angles = math.ceil(distance_deg * 2 / step_deg)
            n_angles = n_angles if n_angles % 2 == 1 else n_angles + 1
            grid = so3_near_identity_grid_cartesianprod(
                distance_deg, n_angles, transposed=False, degrees=True, remove_duplicates=False
            )
            return len(grid)

    def _check_batch_size(self, batch_size: int, device: str) -> None:
        """Warn if batch_size is too large (OOM risk) or too small (throughput penalty).

        Memory model: the compiled projection kernel materialises a
            (batch_size × n_rotations × n_valid_coords × 9) float32 buffer
        where n_valid_coords ≈ π/4 × H × (H//2+1) is the number of rfft grid points
        within the Nyquist sphere. This dominates peak GPU memory.

        For a GPU with V GB of VRAM:
            safe_bs = floor(0.85 × V / (n_rots × n_valid × 9 × 4 bytes))

        Memory scales with BOTH batch_size and n_rotations.
        """
        if not torch.cuda.is_available():
            return

        try:
            dev = torch.device(device)
            if dev.type != "cuda":
                return
            gpu_idx = dev.index if dev.index is not None else torch.cuda.current_device()
            total_vram_gb = torch.cuda.get_device_properties(gpu_idx).total_memory / 1e9
        except Exception:
            return  # can't query GPU properties; skip

        import math
        H = self.image_shape[-2]
        n_valid = int(math.pi / 4 * H * (H // 2 + 1))   # rfft pts within Nyquist sphere

        if self.use_two_stage_search:
            n_fine = self._count_rotations(self.fine_grid_distance_degs,
                                           self.fine_grid_step_degs, fibo=False)
            n_rots = self.fine_top_k * n_fine      # fine pass materialises the largest buffer
        else:
            n_rots = self._count_rotations(self.grid_distance_degs, self.grid_step_degs)

        gb_per_batch = n_rots * n_valid * 9 * 4 / 1e9     # GB per batch element

        safe_bs    = max(1, int(total_vram_gb * 0.85 / gb_per_batch))
        underuse_bs = max(1, safe_bs // 8)  # below 12.5 % of safe capacity is wasteful

        est_peak_gb = batch_size * gb_per_batch
        if batch_size > safe_bs:
            print(
                f"[projmatching] WARNING: batch_size={batch_size} may exhaust GPU memory "
                f"({total_vram_gb:.0f} GB, estimated peak {est_peak_gb:.1f} GB). "
                f"Recommended: --batch_size {safe_bs}."
            )
        elif batch_size < underuse_bs:
            safe_peak_gb = safe_bs * gb_per_batch
            print(
                f"[projmatching] NOTE: batch_size={batch_size} uses only "
                f"{est_peak_gb:.1f}/{total_vram_gb:.0f} GB GPU memory. "
                f"Consider --batch_size {safe_bs} ({safe_peak_gb:.1f} GB) "
                f"for better GPU throughput."
            )

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

        # Band mask (low-pass at fftfreq_max): zeroes out-of-band positions in projs which may
        # contain uninitialised values from torch.empty in the projection kernel.
        img_size = vol_shape[-1]
        max_freq_px = int(round(self.fftfreq_max * img_size))
        band_mask = _mask_for_dft_2d(
            img_shape=tuple(self.image_shape),
            max_freq_pixels=max_freq_px,
            min_freq_pixels=None,
            rfft=True,
            fftshifted=True,
            device=torch.device("cpu"),
        )
        self.register_buffer("band_mask", band_mask)  # (1, H, W//2+1)

        # Phase-ramp sign buffer: folds the post-irfftn ifftshift_2d in correlate_dft_2d
        # into a frequency-domain pre-multiply, saving one O(B·nCand·H·W) pass.
        ccorr_sign_grid = build_ccorr_sign_grid(img_size, img_size, device=torch.device("cpu"))
        self.register_buffer("ccorr_sign_grid", ccorr_sign_grid)  # (1, 1, H, W//2+1)

    def _projectF(self, rotMats: torch.Tensor) -> torch.Tensor:

        return extract_central_slices_rfft_3d(
            self.reference_vol,
            rotation_matrices=rotMats,
            image_shape=self.vol_shape,
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

    def correlateF(self, parts: torch.Tensor, projs: torch.Tensor,
                   ctf: torch.Tensor | None = None):
        # Build projection-side whitening filter.
        # noise_psd_whitening: particles pre-whitened in _preprocess_particles_to_F;
        # noise_psd_map applied here gives the full 1/σ²_noise matched-filter weight.
        # band_mask is always included: zeroes out-of-band positions in projs (which may
        # contain uninitialised values from torch.empty in the projection kernel).
        if self.noise_psd_whitening:
            whitening_filter = self.noise_psd_map * self.band_mask  # bake band_mask in — no extra HBM pass
        else:
            whitening_filter = self.band_mask

        if ctf is not None:
            # Fold CTF into the projection-side whitening filter to eliminate the separate
            # O(B·nCand·H·W//2+1) CTF-apply pass (~3.9 ms/batch at B=32, nCand=343).
            # ctf may have a trailing size-1 dim (USE_TWO_FLOAT32_FOR_COMPLEX padding); squeeze it.
            ctf_real = ctf.squeeze(-1) if ctf.shape[-1] == 1 else ctf
            whitening_filter = ctf_real if whitening_filter is None else whitening_filter * ctf_real

        return self._correlateCrossCorrelation(parts, projs,
                                               whitening_filter=whitening_filter,
                                               ccorr_sign_grid=self.ccorr_sign_grid)

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

    def _correlateCrossCorrelation(self, parts: torch.Tensor, projs: torch.Tensor,
                                    whitening_filter: torch.Tensor | None = None,
                                    ccorr_sign_grid: torch.Tensor | None = None):
        corrs = self.correlate_dft_2d(parts, projs,
                                       whitening_filter=whitening_filter,
                                       ccorr_sign_grid=ccorr_sign_grid)
        b, options, l0, l1 = corrs.shape
        h0, h1, w0, w1 = _get_begin_end_from_max_shift(corrs.shape[-2:], self.max_shift_fraction)
        maxcorr, maxcorrIdxs = self._extract_ccor_max(corrs.reshape(-1, *corrs.shape[-2:]),
                                                      h0, h1, w0, w1,
                                                      use_subpixel=self.use_subpixel_shifts)

        del corrs
        return maxcorr.reshape(b, options), maxcorrIdxs.reshape(b, options, 2)

    def _compute_projections_from_rotmats(self, rotMats: torch.Tensor):
        return self.projectF(rotMats)

    def preprocess_particles(
            self,
            particles: FNAME_TYPE,
            data_rootdir,
            batch_size,
            n_cpus,
            halfset=None,
            subset_idxs=None
    ):
        # Auto-configure dataset pixel size and box size to match the reference volume.
        ref_voxel_size = self.vol_voxel_size
        ref_box_size = int(self.ori_vol_shape[0])
        ds_cfg = main_config.datamanager.particlesdataset
        if not np.isclose(ds_cfg.sampling_rate_angs_for_nnet, ref_voxel_size, atol=1e-2):
            print(f"[projmatching] Auto-setting sampling_rate_angs_for_nnet="
                  f"{ref_voxel_size:.4f} from reference volume (was {ds_cfg.sampling_rate_angs_for_nnet})")
            ds_cfg.sampling_rate_angs_for_nnet = ref_voxel_size
        if ds_cfg.image_size_px_for_nnet is None:
            print(f"[projmatching] Auto-setting image_size_px_for_nnet={ref_box_size} from reference volume")
            ds_cfg.image_size_px_for_nnet = ref_box_size

        # Auto-configure dataset pixel size and box size to match the reference volume.
        # Projmatching uses the original full-size images (BATCH_ORI_IMAGE_NAME), so
        # sampling_rate_angs_for_nnet must equal the reference voxel size and
        # image_size_px_for_nnet just needs to be a valid non-None value (unused by projmatching).
        # TODO: Long-term, implement a simpler ParticlesDataset variant for standalone projmatching
        #  that does not require NN-specific parameters (image_size_px_for_nnet,
        #  sampling_rate_angs_for_nnet) and reads pixels at their native size/rate directly.
        ref_voxel_size = self.vol_voxel_size
        ref_box_size = int(self.ori_vol_shape[0])
        ds_cfg = main_config.datamanager.particlesdataset
        if not np.isclose(ds_cfg.sampling_rate_angs_for_nnet, ref_voxel_size, atol=1e-2):
            print(f"[projmatching] Auto-setting sampling_rate_angs_for_nnet="
                  f"{ref_voxel_size:.4f} from reference volume (was {ds_cfg.sampling_rate_angs_for_nnet})")
            ds_cfg.sampling_rate_angs_for_nnet = ref_voxel_size
        if ds_cfg.image_size_px_for_nnet is None:
            print(f"[projmatching] Auto-setting image_size_px_for_nnet={ref_box_size} from reference volume")
            ds_cfg.image_size_px_for_nnet = ref_box_size

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

    # ----------------------- Two-stage search helpers -----------------------

    def _preprocess_particles_to_F(self, imgs: torch.Tensor) -> torch.Tensor:
        """Convert real-space particle images to Fourier domain.

        Applies circular mask, converts to rfft, then noise-PSD whitening warm-up.
        Returns fparts: shape (B, H, W//2+1[, 2]).
        """
        fparts = _real_to_fourier_2d(imgs * self.rmask,
                                     view_complex_as_two_float32=USE_TWO_FLOAT32_FOR_COMPLEX)

        # Noise-PSD matched filter: estimate σ_noise(freq) from background ring, whiten
        # particles here by 1/σ_noise and projections in correlateF → 1/σ²_noise weight.
        if self.noise_psd_whitening:
            with torch.no_grad():
                if self._noise_psd_batches_seen < self.noise_psd_warmup_batches:
                    # FFT of background (outside mask) — always compute in complex (not two-float32)
                    background = imgs * (1.0 - self.rmask)  # (B, H, W) — outside-mask region
                    fbackground = _real_to_fourier_2d(background, view_complex_as_two_float32=False)  # (B, H, W//2+1) complex
                    bg_amp = fbackground.abs().mean(dim=0)  # (H, W//2+1) batch-mean amplitude

                    if self._noise_psd_amp_sum is None:
                        self._noise_psd_amp_sum = bg_amp
                    else:
                        self._noise_psd_amp_sum = self._noise_psd_amp_sum + bg_amp
                    self._noise_psd_batches_seen += 1

                    avg_amp = self._noise_psd_amp_sum / self._noise_psd_batches_seen
                    in_band = self.band_mask[0] > 0.1
                    amp_in_band = avg_amp * in_band.float()
                    nonzero = amp_in_band[in_band]
                    eps = nonzero.mean() * 1e-3 + 1e-8 if nonzero.numel() > 0 else 1e-8
                    nmap = torch.where(in_band, 1.0 / (amp_in_band + eps), torch.zeros_like(avg_amp))
                    self.noise_psd_map = nmap.unsqueeze(0)  # (1, H, W//2+1)

            # Apply noise-PSD whitening to particles (particle side of the symmetric matched filter).
            if USE_TWO_FLOAT32_FOR_COMPLEX:
                fparts = fparts * self.noise_psd_map.unsqueeze(-1)
            else:
                fparts = fparts * self.noise_psd_map

        # Zero DC bin in-place (no clone needed — fparts is a freshly allocated tensor).
        # Callers rely on DC being zeroed here so correlate_dft_2d can skip the clone.
        if self.zero_dc:
            dc_row = fparts.shape[-3] // 2  # H//2 in (B, H, W//2+1[, 2])
            if USE_TWO_FLOAT32_FOR_COMPLEX:
                fparts[..., dc_row, 0, :] = 0
            else:
                fparts[..., dc_row, 0] = 0

        return fparts

    def _extract_cc_peaks(
            self,
            fparts: torch.Tensor,
            ctfs: torch.Tensor,
            expanded_rotmats: torch.Tensor,
            bsize: int,
            topk: int,
            apply_ctf: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project all candidates, correlate with particles, return top-k. No SO3 refinement.

        Args:
            fparts:           (B, H, W//2+1[, 2]) — Fourier-domain particle images
            ctfs:             (B, H, W//2+1[, 2]) — CTF envelopes
            expanded_rotmats: (B, nCand, 3, 3)    — all candidate rotation matrices
            bsize:            batch size B
            topk:             number of winners to return

        Returns:
            corrs_full:    (B, nCand) — full correlation array (for confidence scoring)
            maxCorrs:      (B, topk)
            predRotMats:   (B, topk, 3, 3)
            pixelShiftsXY: (B, topk, 2)
        """
        ncand = expanded_rotmats.shape[1]
        _shapeIdx = -3 if USE_TWO_FLOAT32_FOR_COMPLEX else -2

        ctfs_view = None
        if self.correct_ctf and apply_ctf:
            ctfs_view = ctfs.unsqueeze(1)
            if USE_TWO_FLOAT32_FOR_COMPLEX:
                ctfs_view = ctfs_view.unsqueeze(-1)

        projs = self._compute_projections_from_rotmats(expanded_rotmats.reshape(-1, 3, 3))
        projs = projs.reshape(bsize, ncand, *projs.shape[_shapeIdx:])

        # Zero DC bin in-place (projs is a freshly owned tensor from the projection kernel).
        # CTF is folded into correlateF's whitening filter — no separate CTF-apply pass.
        if self.zero_dc:
            dc_row = projs.shape[-3] // 2
            if USE_TWO_FLOAT32_FOR_COMPLEX:
                projs[..., dc_row, 0, :] = 0
            else:
                projs[..., dc_row, 0] = 0

        perImgCorr, pixelShiftsXY = self.correlateF(fparts.unsqueeze(1), projs, ctf=ctfs_view)

        maxCorrs, maxCorrsIdxs = perImgCorr.topk(topk, sorted=True, largest=True, dim=-1)

        batch_range = torch.arange(bsize, device=fparts.device).unsqueeze(1)
        pixelShiftsXY_topk = pixelShiftsXY[batch_range, maxCorrsIdxs]   # (B, topk, 2)
        predRotMats = expanded_rotmats[batch_range, maxCorrsIdxs]         # (B, topk, 3, 3)

        return perImgCorr, maxCorrs, predRotMats, pixelShiftsXY_topk


    def _forward_two_stage(
            self,
            fparts: torch.Tensor,
            ctfs: torch.Tensor,
            rotmats: torch.Tensor,
            bsize: int,
            apply_ctf: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Two-stage coarse-to-fine search.

        Stage 1 (coarse): Cartesian 6°/2° grid → top fine_top_k winners.
        Stage 2 (fine):   Cartesian fine grid around each coarse winner → best pose,
                          with O(1) table-based SO(3) parabolic sub-step refinement.
        Confidence is computed against the coarse correlation distribution.
        """
        coarse_expanded = self._compose_coarse_euler_add(rotmats)      # (B, nCoarse, 3, 3)
        coarse_corrs_full, _, coarse_rotmats, _ = self._extract_cc_peaks(
            fparts, ctfs, coarse_expanded, bsize, topk=self.fine_top_k, apply_ctf=apply_ctf
        )
        del coarse_expanded

        fine_delta = self._get_fine_delta_rotmats(rotmats.device)      # (nFine, 3, 3)
        fine_expanded = (fine_delta[None, None] @
                         coarse_rotmats.unsqueeze(2)).reshape(bsize, -1, 3, 3)

        # Extract top-k winners from fine stage (or top-1 if using SO3 interpolation)
        topk = 1 if self.use_so3_interpolation else self.top_k_poses_localref
        fine_corrs_grid, fine_corrs, fine_rotmats, fine_shifts = self._extract_cc_peaks(
            fparts, ctfs, fine_expanded, bsize, topk=topk, apply_ctf=apply_ctf
        )
        del fine_expanded, ctfs

        # Apply O(1) table-based SO3 parabolic sub-step refinement to fine-stage winner
        if self.use_so3_interpolation:
            # fine_corrs_grid: (B, nCoarse_topk * nFine)
            # Reshape to (B, nCoarse_topk, nFine) for per-coarse-winner processing
            nFine = fine_delta.size(0)
            fine_corrs_reshaped = fine_corrs_grid.reshape(bsize, self.fine_top_k, nFine)  # (B, K, nFine)

            # Find best coarse winner and its local fine winner
            coarse_winner_idx = fine_corrs_reshaped.max(dim=-1)[0].argmax(dim=-1)  # (B,) which coarse
            batch_range = torch.arange(bsize, device=fparts.device)
            fine_corrs_local = fine_corrs_reshaped[batch_range, coarse_winner_idx]  # (B, nFine)
            fine_winner_idx = fine_corrs_local.argmax(dim=-1)  # (B,) which fine

            # Get winner Euler angles and apply parabolic refinement
            winner_euler = get_eulers(fine_rotmats[:, 0])  # (B, 3) degrees

            # Parabolic refinement via precomputed neighbor table (ZERO extra projections)
            nb_idx = self._fine_so3_interp_nb_idx.to(fparts.device)
            nb_valid = self._fine_so3_interp_nb_valid.to(fparts.device)
            delta_euler = _so3_interpolate_euler_winner(
                fine_corrs_local, fine_winner_idx, nb_idx, nb_valid, self.fine_grid_step_degs
            )  # (B, 3) degrees - sub-step correction

            # Add sub-step correction to winner Euler angles and convert back to rotmat
            refined_euler = winner_euler + delta_euler  # (B, 3) degrees
            fine_rotmats = get_rotmat(refined_euler, device=fparts.device).unsqueeze(1)  # (B, 1, 3, 3)

            # Keep winner's correlation and shift
            fine_corrs = fine_corrs[:, :1]  # (B, 1)
            fine_shifts = fine_shifts[:, :1]  # (B, 1, 2)

        mean_corr = coarse_corrs_full.mean(-1, keepdims=True)
        std_corr  = coarse_corrs_full.std(-1, keepdims=True)
        comparedWeight = torch.distributions.Normal(mean_corr, std_corr + 1e-6).cdf(fine_corrs)
        predShiftsAngsXY = -(fine_shifts.float() - self.half_particle_size) * self.vol_voxel_size
        return fine_corrs, fine_rotmats, predShiftsAngsXY, comparedWeight

    # ----------------------- Main forward -----------------------

    def forward(self, imgs, ctfs, rotmats):
        assert imgs.shape[1:] == self.rmask.shape, (
            f"Error, particle images and reference maps must have same shape "
            f"({imgs.shape[1:], self.rmask.shape}). Make sure that you are using the same "
            f"box size and sampling rate for the particles and the reference volume."
        )
        fparts = self._preprocess_particles_to_F(imgs)
        bsize = rotmats.size(0)

        if self.use_two_stage_search:
            return self._forward_two_stage(fparts, ctfs, rotmats, bsize)

        # Single-stage: Cartesian euler_add grid
        expanded_rotmats = self._compose_coarse_euler_add(rotmats)

        # Extract top-k winners (or top-1 if using SO3 interpolation)
        topk = 1 if self.use_so3_interpolation else self.top_k_poses_localref
        perImgCorr, maxCorrs, predRotMats, pixelShiftsXY = self._extract_cc_peaks(
            fparts, ctfs, expanded_rotmats, bsize, topk=topk
        )
        del ctfs

        # Apply O(1) table-based SO3 parabolic sub-step refinement to the winner
        if self.use_so3_interpolation:
            # Get winner index and Euler angles
            winner_idx = perImgCorr.argmax(dim=-1)  # (B,)
            winner_euler = get_eulers(predRotMats[:, 0])  # (B, 3) degrees

            # Parabolic refinement via precomputed neighbor table (ZERO extra projections)
            nb_idx = self._so3_interp_nb_idx.to(fparts.device)
            nb_valid = self._so3_interp_nb_valid.to(fparts.device)
            delta_euler = _so3_interpolate_euler_winner(
                perImgCorr, winner_idx, nb_idx, nb_valid, self.grid_step_degs
            )  # (B, 3) degrees - sub-step correction

            # Add sub-step correction to winner Euler angles and convert back to rotmat
            refined_euler = winner_euler + delta_euler  # (B, 3) degrees
            predRotMats = get_rotmat(refined_euler, device=fparts.device).unsqueeze(1)  # (B, 1, 3, 3)

            # Keep winner's correlation and shift
            maxCorrs = maxCorrs[:, :1]  # (B, 1)
            pixelShiftsXY = pixelShiftsXY[:, :1]  # (B, 1, 2)

        mean_corr = perImgCorr.mean(-1, keepdims=True)
        std_corr  = perImgCorr.std(-1, keepdims=True)
        comparedWeight = torch.distributions.Normal(mean_corr, std_corr + 1e-6).cdf(maxCorrs)
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

        # Reset noise-PSD warm-up state so each call re-estimates from its own particles
        self._noise_psd_amp_sum = None
        self._noise_psd_batches_seen = 0

        n_cpus = max(0, n_cpus)  # 0 = inline loading in main process (no subprocess spawn)
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

        # TODO: pixel_size here is sampling_rate_angs_for_nnet (the NN target rate), not the
        #  original particle pixel size — see TODO in particlesDataset.py::sampling_rate property.
        #  For standalone projmatching, the user must set sampling_rate_angs_for_nnet to match
        #  the reference voxel size, even though projmatching uses the original full-size images.
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

        self._check_batch_size(batch_size, device)

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

                print(f"Median Ang   Displacement degs (top-{k + 1}):", np.median(ang_err.numpy()))
                print(f"Median Shift Displacement Angs (top-{k + 1}):", np.median(shift_error))
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


# ══════════════════════════════════════════════════════════════════════════════
# SO(3) sub-step interpolation helpers (Change #7)
# ══════════════════════════════════════════════════════════════════════════════

def _precompute_so3_interp_neighbors(
        grid_distance_degs: float,
        grid_step_degs: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build axis-aligned neighbor index tables for Cartesian SO(3) parabolic interpolation.

    For the Cartesian Euler grid (n_angles³ points), each grid point has up to 6 axis-aligned
    neighbors (rot±, tilt±, psi±). This function precomputes their flat indices and a validity
    mask once at init, so the hot interpolation path only does cheap index lookups.

    Args:
        grid_distance_degs: half-width of the search ball (degrees).
        grid_step_degs:     angular step between grid points (degrees).

    Returns:
        neighbor_idx:   (nDelta, 6) long — flat indices of the 6 neighbors in order
                        (rot+, rot-, tilt+, tilt-, psi+, psi-). Boundary entries are
                        clamped to the point itself; check neighbor_valid before using.
        neighbor_valid: (nDelta, 6) bool — True where the neighbor lies inside the grid.
    """
    n_angles = math.ceil(grid_distance_degs * 2 / grid_step_degs)
    if n_angles % 2 == 0:
        n_angles += 1
    nDelta = n_angles ** 3

    flat_idx = torch.arange(nDelta, dtype=torch.long)
    rot_i  = flat_idx // (n_angles * n_angles)
    tilt_j = (flat_idx // n_angles) % n_angles
    psi_l  = flat_idx % n_angles

    strides     = [n_angles * n_angles, n_angles, 1]
    axis_coords = [rot_i, tilt_j, psi_l]

    neighbor_idx   = torch.zeros(nDelta, 6, dtype=torch.long)
    neighbor_valid = torch.zeros(nDelta, 6, dtype=torch.bool)

    for ax, (stride, coord) in enumerate(zip(strides, axis_coords)):
        col_p = 2 * ax      # positive-step neighbor column
        col_m = 2 * ax + 1  # negative-step neighbor column

        valid_p = coord < n_angles - 1
        neighbor_valid[:, col_p] = valid_p
        neighbor_idx[:, col_p] = torch.where(valid_p, flat_idx + stride, flat_idx)

        valid_m = coord > 0
        neighbor_valid[:, col_m] = valid_m
        neighbor_idx[:, col_m] = torch.where(valid_m, flat_idx - stride, flat_idx)

    return neighbor_idx, neighbor_valid


def _so3_interpolate_euler_winner(
        perImgCorr: torch.Tensor,       # (B, nDelta)
        winner_idx: torch.Tensor,        # (B,) long
        neighbor_idx: torch.Tensor,      # (nDelta, 6) long
        neighbor_valid: torch.Tensor,    # (nDelta, 6) bool
        grid_step_degs: float,
) -> torch.Tensor:                       # (B, 3) delta Euler angles in degrees
    """Parabolic sub-step refinement of the winning Cartesian Euler grid point.

    For each of the 3 Euler axes independently, fits a 1D parabola through the CC values at
    the winner and its two axis-aligned neighbors and returns the sub-step offset to the
    parabola peak. Identical in spirit to the sub-pixel shift refinement in _extract_ccor_max.

    Corrections are clamped to ±step/2 and zeroed wherever a neighbor is missing (boundary)
    or the CC surface is not concave along that axis (no well-defined local maximum).

    Only meaningful for the Cartesian euler_add path (structured axis-aligned grid).
    """
    B = winner_idx.shape[0]
    batch_range = torch.arange(B, device=perImgCorr.device)

    cc_0 = perImgCorr[batch_range, winner_idx]  # (B,)

    nb_idx   = neighbor_idx[winner_idx]    # (B, 6)
    nb_valid = neighbor_valid[winner_idx]  # (B, 6)

    delta_euler = torch.zeros(B, 3, device=perImgCorr.device, dtype=perImgCorr.dtype)

    for ax in range(3):
        col_p, col_m = 2 * ax, 2 * ax + 1

        cc_p = perImgCorr[batch_range, nb_idx[:, col_p]]  # (B,)
        cc_m = perImgCorr[batch_range, nb_idx[:, col_m]]  # (B,)

        # Parabolic interpolation: same formula as _extract_ccor_max
        # delta = (cc_m - cc_p) / (2*cc_m - 4*cc_0 + 2*cc_p)  in grid-step units
        denom = 2.0 * cc_m - 4.0 * cc_0 + 2.0 * cc_p          # (B,)
        safe_denom = torch.where(denom < -1e-8, denom, torch.full_like(denom, -1e-8))
        delta_ax = (cc_m - cc_p) / safe_denom * grid_step_degs  # (B,) in degrees

        delta_ax = delta_ax.clamp(-grid_step_degs * 0.5, grid_step_degs * 0.5)

        # Only apply where both neighbors exist AND the surface is concave (denom < 0)
        apply = nb_valid[:, col_p] & nb_valid[:, col_m] & (denom < -1e-8)
        delta_euler[:, ax] = torch.where(apply, delta_ax, torch.zeros_like(delta_ax))

    return delta_euler


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

def _extract_ccor_max(corrs, h0, h1, w0, w1, use_subpixel: bool = False):
    # corrs is always (N, H, W) — reshaped by _correlateCrossCorrelation before this call
    if h0 > 0:
        corrs_crop = corrs[..., h0:h1, w0:w1]
    else:
        corrs_crop = corrs
    maxCorrsJ, maxIndxJ = corrs_crop.max(-1)
    perImgCorr, maxIndxI = maxCorrsJ.max(-1)
    int_i = maxIndxI  # integer peak row in cropped array, shape (N,)
    int_j = torch.gather(maxIndxJ, -1, maxIndxI.unsqueeze(-1)).squeeze(-1)  # shape (N,)

    if use_subpixel:
        H = corrs_crop.shape[-2]
        W = corrs_crop.shape[-1]
        N = corrs_crop.shape[0]
        n_idx = torch.arange(N, device=corrs.device)  # (N,)

        # Clamp to interior so ±1 neighbours always exist
        safe_i = int_i.clamp(1, H - 2)
        safe_j = int_j.clamp(1, W - 2)

        # --- sub-pixel refinement along row axis (at col = int_j) ---
        fi_m = corrs_crop[n_idx, safe_i - 1, int_j]  # (N,)
        fi_0 = corrs_crop[n_idx, safe_i,     int_j]
        fi_p = corrs_crop[n_idx, safe_i + 1, int_j]
        denom_i = 2 * fi_m - 4 * fi_0 + 2 * fi_p
        delta_i = ((fi_m - fi_p) / denom_i.clamp(max=-1e-8)) * (int_i == safe_i).float()
        sub_i = safe_i.float() + delta_i

        # --- sub-pixel refinement along col axis (at row = int_i) ---
        fj_m = corrs_crop[n_idx, int_i, safe_j - 1]  # (N,)
        fj_0 = corrs_crop[n_idx, int_i, safe_j    ]
        fj_p = corrs_crop[n_idx, int_i, safe_j + 1]
        denom_j = 2 * fj_m - 4 * fj_0 + 2 * fj_p
        delta_j = ((fj_m - fj_p) / denom_j.clamp(max=-1e-8)) * (int_j == safe_j).float()
        sub_j = safe_j.float() + delta_j
    else:
        sub_i = int_i.float()
        sub_j = int_j.float()

    pixelShiftsXY = torch.empty(corrs.shape[:-2] + (2,), device=corrs.device, dtype=torch.float32)
    pixelShiftsXY[..., 1] = h0 + sub_i
    pixelShiftsXY[..., 0] = w0 + sub_j
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
        batch_size: int = 32,
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
