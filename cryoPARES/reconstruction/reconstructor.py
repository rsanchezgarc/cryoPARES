import os
import sys
from functools import lru_cache
from typing import List, Optional, Tuple, Union, Literal

import numpy as np
import torch
import tqdm
from starstack import ParticlesStarSet
from cryoPARES.constants import (
    RELION_ANGLES_NAMES,
    RELION_SHIFTS_NAMES,
    RELION_EULER_CONVENTION,
    RELION_IMAGE_FNAME,
    RELION_PRED_POSE_CONFIDENCE_NAME,
    float32_matmul_precision
)
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch_fourier_shift import fourier_shift_dft_2d

from cryoPARES.configManager.inject_defaults import inject_defaults_from_config, CONFIG_PARAM
from cryoPARES.configs.mainConfig import main_config
from cryoPARES.datamanager.ctf.rfft_ctf import compute_ctf_rfft
from cryoPARES.geometry.convert_angles import euler_angles_to_matrix
from cryoPARES.geometry.symmetry import getSymmetryGroup

from cryoPARES.utils.paths import FNAME_TYPE
from cryoPARES.utils.reconstructionUtils import write_vol


class Reconstructor(nn.Module):
    @inject_defaults_from_config(main_config.reconstruct, update_config_with_args=True)
    def __init__(
        self,
        symmetry: str,
        correct_ctf: bool = CONFIG_PARAM(),
        eps: float = CONFIG_PARAM(),
        min_denominator_value: Optional[float] = None,
        weight_with_confidence: bool = CONFIG_PARAM(),
        *args,
        **kwargs,
    ):
        numerator = kwargs.pop("numerator", None)
        weights = kwargs.pop("weights", None)
        ctfsq = kwargs.pop("ctfsq", None)

        super().__init__(*args, **kwargs)
        self.symmetry = symmetry.upper()
        self.has_symmetry = self.symmetry != "C1"
        self.correct_ctf = correct_ctf
        self.eps = eps  # The Tikhonov constant. Should be ~1/SNR; could be estimated per frequency.
        if min_denominator_value is None:
            min_denominator_value = eps * 0.1
        self.min_denominator_value = min_denominator_value
        self.weight_with_confidence = weight_with_confidence

        self.register_buffer("dummy_buffer", torch.ones(1))
        if self.has_symmetry:
            self.register_buffer(
                "sym_matrices", getSymmetryGroup(self.symmetry, as_matrix=True)
            )
        else:
            self.sym_matrices = None

        self.box_size: Optional[int] = None
        self.sampling_rate: Optional[float] = None

        self.register_buffer("numerator", numerator)
        self.register_buffer("weights", weights)
        self.register_buffer("ctfsq", ctfsq)

        self._initialized = False

        from cryoPARES.reconstruction.insert_central_slices_rfft_3d import \
            insert_central_slices_rfft_3d_multichannel, compiled_insert_central_slices_rfft_3d_multichannel

        if not main_config.reconstruct.disable_compile_insert_central_slices_rfft_3d_multichannel:
            print("Compiling insert_central_slices_rfft_3d_multichannel")
            self.insert_central_slices_rfft_3d_multichannel = compiled_insert_central_slices_rfft_3d_multichannel
        else:
            self.insert_central_slices_rfft_3d_multichannel = insert_central_slices_rfft_3d_multichannel

        # TODO(RING-CALIBRATION):
        # Add ring-wise EMA scaling to stabilize on-the-fly recon by keeping the
        # running mean of (alpha * CTF^2) per frequency ring near ~1.0.
    def get_device(self):
        return self.dummy_buffer.device

    def move_buffers_to_share_mem(self):
        if self.numerator is not None:
            self.numerator.share_memory_()
        if self.weights is not None:
            self.weights.share_memory_()
        if self.ctfsq is not None:
            self.ctfsq.share_memory_()

    def get_buffers(self):
        return dict(numerator=self.numerator, weights=self.weights, ctfsq=self.ctfsq)

    def zero_buffers(self):
        if self.numerator is not None: self.numerator.zero_()
        if self.weights is not None: self.weights.zero_()
        if self.ctfsq is not None: self.ctfsq.zero_()

    def set_metadata_from_particles(self, particlesDataset: "ReconstructionParticlesDataset"):
        box_size: int = particlesDataset.particle_shape[-1]
        sampling_rate = particlesDataset.sampling_rate

        if self.sampling_rate is not None:
            assert (
                sampling_rate == self.sampling_rate
            ), "Error, mismatch between the previous and current sampling_rate"
        else:
            self.sampling_rate = sampling_rate

        if self.box_size is not None:
            assert (
                box_size == self.box_size
            ), "Error, mismatch between the previous and current box_size"
        else:
            self.box_size = box_size
            self.particle_shape = (box_size, box_size)
            nky, nkx = self.box_size, self.box_size // 2 + 1

            self.numerator = torch.zeros(
                (2, self.box_size, self.box_size, nkx),
                dtype=torch.float32,
                device=self.get_device(),
            )
            self.weights = torch.zeros(
                (self.box_size, self.box_size, nkx),
                dtype=torch.float32,
                device=self.get_device(),
            )
            self.ctfsq = torch.zeros_like(self.weights, dtype=torch.float32, device=self.get_device())

        self._initialized = True

    def _expand_with_symmetry(
        self,
        imgs: torch.Tensor,
        ctf: Union[torch.Tensor, int],
        rotMats: torch.Tensor,
        hwShiftAngs: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, int], torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Expand images and metadata with symmetry operations following RELION.

        Args:
            imgs: [B, H, W] tensor (spatial-domain before rFFT) or [B, Hy, Hx_rfft] after rFFT as used here
            ctf:  [B, H, W] tensor (or 0 if no CTF correction)
            rotMats: [B, 3, 3] active rotation matrices
            hwShiftAngs: [B, 2] pixel shifts (kept UNCHANGED for symmetry expansion)
            confidence: [B] per-particle scalar (optional)

        Returns:
            Expanded tensors; confidence is replicated if provided (else None).
        """
        if not self.has_symmetry:
            return imgs, ctf, rotMats, hwShiftAngs, confidence

        B = rotMats.shape[0]
        K = self.sym_matrices.shape[0]  # number of symmetry operations

        # 1) Expand images & CTF by tiling
        expanded_imgs = imgs.unsqueeze(1).expand(-1, K, *([*imgs.shape[1:]])).reshape(-1, *imgs.shape[1:])
        if isinstance(ctf, torch.Tensor):
            expanded_ctf = ctf.unsqueeze(1).expand(-1, K, *([*ctf.shape[1:]])).reshape(-1, *ctf.shape[1:])
        else:
            expanded_ctf = ctf  # sentinel 0

        # 2) Compose orientations (RELION uses NewE = OldE * R for point groups; active back via transpose)
        device = rotMats.device
        E = rotMats.transpose(-1, -2)  # [B, 3, 3] (RELION/XMIPP Euler matrix)
        R = self.sym_matrices.to(device=device)  # [K, 3, 3]
        E = E.unsqueeze(1).expand(-1, K, -1, -1).reshape(-1, 3, 3)  # [B*K, 3, 3]
        Rexp = R.unsqueeze(0).expand(B, -1, -1, -1).reshape(-1, 3, 3)  # [B*K, 3, 3]
        E_new = E @ Rexp
        rotMats_new = E_new.transpose(-1, -2).contiguous()

        # 3) Shifts are NOT modified (non-helical)
        expanded_hwShiftAngs = hwShiftAngs.unsqueeze(1).expand(-1, K, -1).reshape(-1, 2)

        # 4) Confidence replicated per symmetry mate (if provided)
        if confidence is not None:
            expanded_conf = confidence.unsqueeze(1).expand(-1, K).reshape(-1)
        else:
            expanded_conf = None

        return expanded_imgs, expanded_ctf, rotMats_new, expanded_hwShiftAngs, expanded_conf

    def _get_reconstructionParticlesDataset(
        self,
        particles_star_fname,
        particles_dir,
        subset_idxs=None,
        halfmap_subset=None
    ):
        particlesDataset = ReconstructionParticlesDataset(
            particles_star_fname,
            particles_dir,
            correct_ctf=self.correct_ctf,
            subset_idxs=subset_idxs,
            halfmap_subset=halfmap_subset,
            return_confidence=self.weight_with_confidence,
        )
        self.set_metadata_from_particles(particlesDataset)
        return particlesDataset

    def backproject_particles(
        self,
        particles_star_fname: FNAME_TYPE,
        particles_dir: Optional[FNAME_TYPE] = None,
        batch_size=1,
        num_dataworkers=0,
        use_only_n_first_batches=None,
        subset_idxs=None,
        halfmap_subset=None
    ):
        for _ in self._backproject_particles(
            particles_star_fname,
            particles_dir,
            batch_size,
            num_dataworkers,
            use_only_n_first_batches,
            subset_idxs,
            halfmap_subset
        ):
            pass

    def _backproject_particles(
        self,
        particles_star_fname: FNAME_TYPE,
        particles_dir: Optional[FNAME_TYPE] = None,
        batch_size=1,
        num_dataworkers=0,
        use_only_n_first_batches=None,
        subset_idxs=None,
        halfmap_subset=None,
        verbose=True,
    ):
        particlesDataset = self._get_reconstructionParticlesDataset(particles_star_fname, particles_dir,
                                                                    subset_idxs=subset_idxs,
                                                                    halfmap_subset=halfmap_subset)
        dl = DataLoader(
            particlesDataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_dataworkers,
            pin_memory=num_dataworkers > 0,
            multiprocessing_context="fork" if num_dataworkers > 0 else None,
        )

        zyx_matrices = False
        for bidx, batch in enumerate(
            tqdm.tqdm(dl, desc=f"backprojecting PID({os.getpid()})", disable=not verbose)
        ):
            if self.weight_with_confidence:
                ids, imgs, ctf, rotMats, hwShiftAngs, confidence = batch
                self._backproject_batch(
                    imgs, ctf, rotMats, hwShiftAngs, confidence=confidence, zyx_matrices=zyx_matrices
                )
            else:
                ids, imgs, ctf, rotMats, hwShiftAngs = batch
                self._backproject_batch(imgs, ctf, rotMats, hwShiftAngs, confidence=None, zyx_matrices=zyx_matrices)

            if use_only_n_first_batches and bidx >= use_only_n_first_batches:
                break
            yield imgs.shape[0]

    def _backproject_batch(
        self,
        imgs: torch.Tensor,
        ctf: torch.Tensor,
        rotMats: torch.Tensor,
        hwShiftAngs: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
        zyx_matrices: bool = False,
    ):
        """
        Backprojects a single batch of particles into the volume.

        Args:
            imgs: [B, H, W] real-valued images (spatial).
            ctf: [B, H, W] real CTF images (if correct_ctf) else 0 sentinel.
            rotMats: [B,(M,) 3, 3] active rotation matrices (M optional).
            hwShiftAngs: [B,(M,) 2] half-way pixel shifts (x,y) in pixels.
            confidence: Optional [B] scalar weights in [0,1].
            zyx_matrices: if True, interpret rotation matrices as ZYX; else standard.
        """
        assert self._initialized, "Error, Reconstructor was not initialized"
        rotMats_shape = rotMats.shape
        assert (
            rotMats_shape[:-2] == hwShiftAngs.shape[:-1] and rotMats.shape[0] == imgs.shape[0]
        ), f"{rotMats_shape}, {hwShiftAngs.shape}, {imgs.shape}"

        device = self.get_device()
        imgs = imgs.to(device, non_blocking=True)
        rotMats = rotMats.to(device, non_blocking=True)
        hwShiftAngs = hwShiftAngs.to(device, non_blocking=True)

        # Pre-processing to rFFT (fftshift -> rfftn -> fftshift on y)
        imgs = torch.fft.fftshift(imgs, dim=(-2, -1))
        imgs = torch.fft.rfftn(imgs, dim=(-2, -1))
        imgs = torch.fft.fftshift(imgs, dim=(-2,))

        if len(rotMats_shape) == 4:
            M = rotMats_shape[1]
            # Expand to [B*M, ...]
            imgs = imgs.unsqueeze(1).expand(-1, M, -1, -1).reshape(-1, *imgs.shape[-2:])
            if isinstance(ctf, torch.Tensor):
                ctf = ctf.unsqueeze(1).expand(-1, M, -1, -1).reshape(-1, *ctf.shape[-2:])
            rotMats = rotMats.reshape(-1, 3, 3)
            hwShiftAngs = hwShiftAngs.reshape(-1, 2)
            if confidence is not None:
                confidence = confidence.unsqueeze(1).expand(-1, M).reshape(-1)

        # Symmetry expansion (replicates confidence if present)
        if self.has_symmetry:
            imgs, ctf, rotMats, hwShiftAngs, confidence = self._expand_with_symmetry(
                imgs, ctf, rotMats, hwShiftAngs, confidence
            )
            imgs = imgs.contiguous()
            if isinstance(ctf, torch.Tensor):
                ctf = ctf.contiguous()
            rotMats = rotMats.contiguous()
            hwShiftAngs = hwShiftAngs.contiguous()
            # confidence may be None; if not, keep contiguous for view()
            if confidence is not None:
                confidence = confidence.contiguous()

        # Apply translational shifts in Fourier domain
        imgs = fourier_shift_dft_2d(
            dft=imgs,
            image_shape=self.particle_shape,
            shifts=hwShiftAngs / self.sampling_rate,
            rfft=True,
            fftshifted=True,
        )

        if self.correct_ctf:
            ctf = ctf.to(imgs.device)
            imgs = imgs * ctf

            # Prepare channels: [real, imag, ctf^2]
            # we only build alpha if confidence is provided (to avoid overhead).
            if confidence is not None:
                confidence = torch.nan_to_num(confidence, nan=0.0,
                                              posinf=1.0, neginf=0.0).clamp(0.0, 1.5)
                alpha = confidence.view(-1, 1, 1, 1).to(ctf.device)  # broadcast over channels and pixels
                stacked = torch.stack([imgs.real, imgs.imag, ctf**2], dim=1) * alpha
            else:
                stacked = torch.stack([imgs.real, imgs.imag, ctf**2], dim=1)

            dft_ctf, weights = self.insert_central_slices_rfft_3d_multichannel(
                image_rfft=stacked,
                volume_shape=(self.box_size,) * 3,
                rotation_matrices=rotMats,
                fftfreq_max=None,
                zyx_matrices=zyx_matrices,
            )

            self.numerator += dft_ctf[:2, ...]
            self.ctfsq += dft_ctf[-1, ...]
            # Geometric sampling weights remain unscaled by confidence (standard practice)
            self.weights += weights
        else:
            raise NotImplementedError("Backprojection without CTF correction is not implemented.")

    @staticmethod
    @lru_cache(maxsize=1)
    def get_sincsq(shape, device, eps=1e-3):
        """
        Separable de-apodization for trilinear gridding:
        PSF(z,y,x) = sinc^2(z) * sinc^2(y) * sinc^2(x), evaluated on the voxel grid.

        torch.sinc(x) = sin(pi*x)/(pi*x).
        We use centered voxel coordinates normalized by the length in each axis so the
        arguments live roughly in [-0.5, 0.5]; add a floor to avoid division blow-ups.
        """
        D, H, W = shape
        z = torch.linspace(-(D - 1) / 2, (D - 1) / 2, D, device=device) / D  # (D,)
        y = torch.linspace(-(H - 1) / 2, (H - 1) / 2, H, device=device) / H  # (H,)
        x = torch.linspace(-(W - 1) / 2, (W - 1) / 2, W, device=device) / W  # (W,)

        sz = torch.sinc(z).pow(2)  # (D,)
        sy = torch.sinc(y).pow(2)  # (H,)
        sx = torch.sinc(x).pow(2)  # (W,)

        S = sz[:, None, None] * sy[None, :, None] * sx[None, None, :]  # (D,H,W)
        return S.clamp_min(eps)

    def generate_volume(
        self,
        fname: Optional[FNAME_TYPE] = None,
        overwrite_fname: bool = True,
        device: Optional[str] = "cpu",
    ):
        dft = torch.zeros_like(self.numerator)

        mask = self.weights > self.min_denominator_value
        if self.correct_ctf:
            denominator = self.ctfsq[mask] + self.eps * self.weights[mask]
            denominator[denominator.abs() < self.min_denominator_value] = self.min_denominator_value
            dft[:, mask] = self.numerator[:, mask] / denominator[None, ...]
        else:
            dft[:, mask] = self.numerator[:, mask] / self.weights[mask][None, ...]
        dft = torch.complex(real=dft[0, ...], imag=dft[1, ...])

        dft = torch.fft.ifftshift(dft, dim=(-3, -2))
        dft = torch.fft.irfftn(dft, dim=(-3, -2, -1))
        dft = torch.fft.ifftshift(dft, dim=(-3, -2, -1))
        sincsq = self.get_sincsq(dft.shape, device, self.eps)
        vol = dft.to(device) / sincsq

        if fname is not None:
            write_vol(vol.detach().cpu(), fname, self.sampling_rate, overwrite=overwrite_fname)
        return vol


class ReconstructionParticlesDataset(Dataset):
    def __init__(
        self,
        particles_star_fname: FNAME_TYPE,
        particles_dir: Optional[FNAME_TYPE] = None,
        correct_ctf: bool = True,
        subset_idxs=None,
        halfmap_subset=None,
        return_confidence: bool = False,
    ):

        self.particles_star_fname = particles_star_fname
        self.particles_dir = particles_dir
        self.subset_idxs = subset_idxs
        self.halfmap_subset = halfmap_subset


        self.correct_ctf = correct_ctf
        self.return_confidence = return_confidence

        self._particles = None
        self._sampling_rate = None
        self._particle_shape = None

    @property
    def sampling_rate(self):
        if self._sampling_rate is None:
            self._sampling_rate = self.particles.sampling_rate
        return self._sampling_rate

    @property
    def particle_shape(self):
        if self._particle_shape is None:
            self._particle_shape = self.particles.particle_shape
        return self._particle_shape

    @property
    def particles(self):
        if self._particles is None:
            self._particles = ParticlesStarSet(starFname=self.particles_star_fname,
                                              particlesDir=self.particles_dir)
            if self.subset_idxs is not None:
                self._particles = self.particles.createSubset(idxs=self.subset_idxs)
            if self.halfmap_subset:
                halfsets = self.particles.particles_md["rlnRandomSubset"]
                assert halfsets is not None, "Error, no halfset found"
                idxs = np.where(halfsets == self.halfmap_subset)[0].tolist()
                self._particles = self.particles.createSubset(idxs=idxs)
        return self._particles

    def __len__(self):
        return len(self.particles)

    def __getitem__(self, item):
        try:
            img, md_row = self.particles[item]
        except ValueError:
            print(f"Error retrieving item {item}")
            raise

        degEuler = torch.FloatTensor([md_row[name] for name in RELION_ANGLES_NAMES])
        rotMat = euler_angles_to_matrix(torch.deg2rad(degEuler), convention=RELION_EULER_CONVENTION)
        hwShiftAngs = torch.FloatTensor([md_row[name] for name in RELION_SHIFTS_NAMES[::-1]])

        img = torch.FloatTensor(img)

        if self.correct_ctf:
            dfu = md_row["rlnDefocusU"]
            dfv = md_row["rlnDefocusV"]
            dfang = md_row["rlnDefocusAngle"]
            volt = float(self.particles.optics_md["rlnVoltage"][0])
            cs = float(self.particles.optics_md["rlnSphericalAberration"][0])
            w = float(self.particles.optics_md["rlnAmplitudeContrast"][0])
            iid = md_row[RELION_IMAGE_FNAME]
            ctf = compute_ctf_rfft(
                img.shape[-2],
                self.sampling_rate,
                dfu,
                dfv,
                dfang,
                volt,
                cs,
                w,
                phase_shift=0,
                bfactor=None,
                fftshift=True,
                device="cpu",
            )
        else:
            iid = md_row[RELION_IMAGE_FNAME]
            ctf = 0  # sentinel when not applying CTF correction

        if self.return_confidence:
            # Use 1.0 default if key missing
            conf = float(md_row.get(RELION_PRED_POSE_CONFIDENCE_NAME, 1.0))
            confidence = torch.tensor(conf, dtype=torch.float32)
            return iid, img, ctf, rotMat, hwShiftAngs, confidence
        else:
            return iid, img, ctf, rotMat, hwShiftAngs

@inject_defaults_from_config(main_config.reconstruct, update_config_with_args=True)
def reconstruct_starfile(
    particles_star_fname: str,
    symmetry: str,
    output_fname: str,
    particles_dir: Optional[str] = None,
    num_dataworkers: int = 1,
    batch_size: int = 128,
    use_cuda: bool = True,
    correct_ctf: bool = CONFIG_PARAM(),
    eps: float = CONFIG_PARAM(),
    min_denominator_value: Optional[float] = None,
    use_only_n_first_batches: Optional[int] = None,
    float32_matmul_precision: Optional[str] = float32_matmul_precision,
    weight_with_confidence: bool = CONFIG_PARAM(),
    halfmap_subset: Optional[Literal["1", "2"]] = None
):
    """
    :param particles_star_fname: The particles to reconstruct
    :param symmetry: The symmetry of the volume (e.g. C1, D2, ...)
    :param output_fname: The name of the output filename (star)
    :param particles_dir: The particles directory (root of the starfile fnames)
    :param num_dataworkers: Num workers for data loading
    :param batch_size: The number of particles to be simultaneously backprojected
    :param use_cuda: if NOT, it will not use cuda devices
    :param correct_ctf: if NOT, it will not correct CTF
    :param eps: The regularization constant (ideally, this is 1/SNR)
    :param min_denominator_value: Used to prevent division by 0. By default is 0.1*eps
    :param use_only_n_first_batches: Use only the n first batches to reconstruct
    :param float32_matmul_precision: Set it to high or medium for speed up at a precision cost
    :param weight_with_confidence: If True, read and apply per-particle confidence. If False (default),
                           do NOT fetch/pass confidence (zero overhead).
    :param halfmap_subset: The random subset of particles to use
    """
    if float32_matmul_precision is not None:
        torch.set_float32_matmul_precision(float32_matmul_precision)
    if halfmap_subset is not None:
        halfmap_subset = int(halfmap_subset)
    device = "cpu" if not use_cuda else "cuda"
    reconstructor = Reconstructor(
        symmetry=symmetry, correct_ctf=correct_ctf, eps=eps, min_denominator_value=min_denominator_value,
        weight_with_confidence=weight_with_confidence
    )
    reconstructor.to(device=device)
    reconstructor.backproject_particles(
        particles_star_fname,
        particles_dir,
        batch_size,
        num_dataworkers,
        use_only_n_first_batches=use_only_n_first_batches,
        halfmap_subset=halfmap_subset
    )
    print(f"Saving map at {output_fname}")
    reconstructor.generate_volume(output_fname)


if __name__ == "__main__":
    from argParseFromDoc import parse_function_and_call
    parse_function_and_call(reconstruct_starfile)
    """
    
python -m cryoPARES.reconstruction.reconstructor --symmetry C1  --particles_star_fname ~/cryo/data/preAlignedParticles/EMPIAR-10166/data/donwsampled/down1000particles.star  --particles_dir  ~/cryo/data/preAlignedParticles/EMPIAR-10166/data/donwsampled/ --output_fname /tmp/reconstruction.mrc  --weight_with_confidence    

    """