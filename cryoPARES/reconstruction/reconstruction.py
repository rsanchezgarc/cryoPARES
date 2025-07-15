"""Cryo‑EM 3‑D reconstruction –
====================================================
A pipeline that reconstructs a Coulomb potential
volume from a RELION‑style particle STAR file using the open‑source
"""
import os
import sys
from functools import lru_cache
from typing import List, Optional

import mrcfile
import torch
import tqdm
from starstack import ParticlesStarSet
from starstack.constants import RELION_ANGLES_NAMES, RELION_SHIFTS_NAMES, RELION_EULER_CONVENTION
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch_fourier_shift import fourier_shift_dft_2d
from torch_grid_utils import fftfreq_grid

from cryoPARES.datamanager.ctf.rfft_ctf import compute_ctf_rfft
from cryoPARES.geometry.convert_angles import euler_angles_to_matrix
from cryoPARES.geometry.symmetry import getSymmetryGroup
from cryoPARES.reconstruction.insert_central_slices_rfft_3d import insert_central_slices_rfft_3d_multichannel
# from torch_fourier_slice import insert_central_slices_rfft_3d_multichannel
from cryoPARES.utils.paths import FNAME_TYPE


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # I noticed that torch.com tries to compile in the GPU as well even if all the tensors are in the CPU
compiled_insert_central_slices_rfft_3d_multichannel = torch.compile(insert_central_slices_rfft_3d_multichannel, mode=None)  # mode="max-autotune")
# compiled_insert_central_slices_rfft_3d_multichannel = insert_central_slices_rfft_3d_multichannel

class Reconstructor(nn.Module):
    def __init__(self, symmetry: str, correct_ctf: bool = True, eps=1e-3,
                 min_denominator_value=1e-4, *args, **kwargs):

        numerator = kwargs.pop("numerator", None)
        weights = kwargs.pop("weights", None)
        ctfsq = kwargs.pop("ctfsq", None)

        super().__init__(*args, **kwargs)
        self.symmetry = symmetry.upper()
        self.has_symmetry = self.symmetry != "C1"
        self.correct_ctf = correct_ctf
        self.eps = eps # The Tikhonov constant. Should be 1/SNR, we might want to estimate it per frequency
        self.min_denominator_value = min_denominator_value

        self.register_buffer("dummy_buffer", torch.ones(1))
        if self.has_symmetry:
            self.register_buffer('sym_matrices', getSymmetryGroup(self.symmetry, as_matrix=True))
        else:
            self.sym_matrices = None

        self.box_size = None
        self.sampling_rate = None

        self.register_buffer('numerator', numerator)
        self.register_buffer('weights', weights)
        self.register_buffer('ctfsq', ctfsq)

        self._initialized = False

    def get_device(self):
        return self.dummy_buffer.device

    def move_buffers_to_share_mem(self):
        if self.numerator:
            self.numerator.share_memory_()
        if self.weights:
            self.weights.share_memory_()
        if self.ctfsq:
            self.ctfsq.share_memory_()

    def get_buffers(self):
        return dict(numerator=self.numerator, weights=self.weights, ctfsq=self.ctfsq)

    def set_metadata_from_particles(self, particlesDataset):

        box_size:int = particlesDataset.particle_shape[-1]
        sampling_rate = particlesDataset.sampling_rate

        if self.sampling_rate is not None:
            assert sampling_rate == self.sampling_rate, "Error, mismatch between the previous and current sampling_rate"
        else:
            self.sampling_rate = sampling_rate

        if self.box_size is not None:
            assert box_size == self.box_size, "Error, mismatch between the previous and current box_size"
        else:
            self.box_size = box_size
            self.particle_shape = (box_size, box_size)
            nky, nkx = self.box_size, self.box_size // 2 + 1

            # self.numerator = torch.zeros((self.box_size, self.box_size, nkx, 2), dtype=torch.float32, device=self.get_device())
            self.numerator = torch.zeros((2, self.box_size, self.box_size, nkx), dtype=torch.float32, device=self.get_device())

            self.weights = torch.zeros((self.box_size, self.box_size, nkx), dtype=torch.float32, device=self.get_device())
            self.ctfsq = torch.zeros_like(self.weights, dtype=torch.float32, device=self.get_device())

        self._initialized = True

    def _expand_with_symmetry(self, imgs, ctf, rotMats, hwShiftAngs):
        """
        Expand images and metadata with symmetry operations.

        Args:
            imgs: [B, H, W] tensor of images in Fourier space
            ctf: [B, H, W] tensor of CTF values (or 0 if no CTF correction)
            rotMats: [B, 3, 3] tensor of rotation matrices
            hwShiftAngs: [B, 2] tensor of shifts

        Returns:
            Expanded tensors with symmetry operations applied
        """
        if not self.has_symmetry:
            return imgs, ctf, rotMats, hwShiftAngs

        batch_size = imgs.shape[0]
        num_sym_ops = self.sym_matrices.shape[0]

        # Expand images - no change needed as Fourier transforms are rotationally invariant
        # for the image data itself
        expanded_imgs = imgs.unsqueeze(1).expand(-1, num_sym_ops, -1, -1).reshape(-1, *imgs.shape[1:])

        # Expand CTF - same as images since CTF is applied in image space
        if isinstance(ctf, torch.Tensor):
            expanded_ctf = ctf.unsqueeze(1).expand(-1, num_sym_ops, -1, -1).reshape(-1, *ctf.shape[1:])
        else:
            # Handle the case where ctf is 0 (no CTF correction)
            expanded_ctf = ctf

        # Expand shifts - symmetry operations don't change shifts in image space
        expanded_hwShiftAngs = hwShiftAngs.unsqueeze(1).expand(-1, num_sym_ops, -1).reshape(-1, hwShiftAngs.shape[-1])

        # Apply symmetry operations to rotation matrices
        # sym_matrices: [num_sym_ops, 3, 3]
        # rotMats: [batch_size, 3, 3]
        # Result: [batch_size * num_sym_ops, 3, 3]
        expanded_rotMats = torch.matmul(
            self.sym_matrices.unsqueeze(1),  # [num_sym_ops, 1, 3, 3]
            rotMats.unsqueeze(0)  # [1, batch_size, 3, 3]
        ).reshape(-1, 3, 3)  # [batch_size * num_sym_ops, 3, 3]

        return expanded_imgs, expanded_ctf, expanded_rotMats, expanded_hwShiftAngs

    def _get_reconstructionParticlesDataset(self, particles_star_fname, particles_dir, subset_idxs=None):
        particlesDataset = ReconstructionParticlesDataset(particles_star_fname, particles_dir,
                                                          correct_ctf=self.correct_ctf, subset_idxs=subset_idxs)

        self.set_metadata_from_particles(particlesDataset)
        return particlesDataset

    def backproject_particles(self, particles_star_fname: FNAME_TYPE,
                              particles_dir: Optional[FNAME_TYPE] = None,
                              batch_size=1, num_dataworkers=0, use_only_n_first_batches=None, subset_idxs=None):

        for _ in self._backproject_particles(particles_star_fname, particles_dir, batch_size,
                                                   num_dataworkers, use_only_n_first_batches, subset_idxs):
            pass

    def _backproject_particles(self, particles_star_fname: FNAME_TYPE,
                               particles_dir: Optional[FNAME_TYPE] = None,
                               batch_size=1, num_dataworkers=0, use_only_n_first_batches=None, subset_idxs=None,
                               verbose=True):


        particlesDataset = self._get_reconstructionParticlesDataset(particles_star_fname,
                                                               particles_dir, subset_idxs=subset_idxs)
        dl = DataLoader(particlesDataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_dataworkers,
                        pin_memory=num_dataworkers > 0,
                        multiprocessing_context="fork" if num_dataworkers > 0 else None)

        zyx_matrices = False
        for bidx, (imgs, ctf, rotMats, hwShiftAngs) in enumerate(tqdm.tqdm(dl, desc=f"backprojecting PID({os.getpid()})",
                                                                           disable=not verbose)):
            self._backproject_batch(imgs, ctf, rotMats, hwShiftAngs, zyx_matrices)
            if use_only_n_first_batches and bidx > use_only_n_first_batches:
                break
            yield imgs.shape[0]

    def _backproject_batch(self, imgs: torch.Tensor, ctf: torch.Tensor,
                           rotMats: torch.Tensor, hwShiftAngs: torch.Tensor,
                           zyx_matrices: bool):
        """
        Backprojects a single batch of particles into the volume.

        Args:
            imgs (torch.Tensor): The particle images.
            ctf (torch.Tensor): The CTF information for each particle.
            rotMats (torch.Tensor): The rotation matrices for each particle.
            hwShiftAngs (torch.Tensor): The translational shifts for each particle.
            zyx_matrices (bool): Flag indicating the rotation matrix format.
        """
        #TODO: Implement confidence weighting
        assert self._initialized, "Error, Reconstructor was not initialized"
        rotMats_shape = rotMats.shape
        assert rotMats_shape[:2] == hwShiftAngs.shape[:2] and rotMats.shape[0] == imgs.shape[0], \
                        f"{rotMats_shape}, {hwShiftAngs.shape}, {imgs.shape}"

        device = self.get_device()
        imgs = imgs.to(device, non_blocking=True)
        rotMats = rotMats.to(device, non_blocking=True)
        hwShiftAngs = hwShiftAngs.to(device, non_blocking=True)

        # Pre-processing
        imgs = torch.fft.fftshift(imgs, dim=(-2, -1))  # Volume center to array origin
        imgs = torch.fft.rfftn(imgs, dim=(-2, -1))
        imgs = torch.fft.fftshift(imgs, dim=(-2,))  # Actual fftshift

        if len(rotMats_shape) == 4:
            imgs = imgs.unsqueeze(1).expand(-1, rotMats_shape[1], -1, -1).view(-1, *imgs.shape[-2:])
            ctf = ctf.unsqueeze(1).expand(-1, rotMats_shape[1], -1, -1).view(-1, *ctf.shape[-2:])
            rotMats = rotMats.view(-1, 3, 3)
            hwShiftAngs = hwShiftAngs.view(-1, 2)
        # Apply translational shifts
        imgs = fourier_shift_dft_2d(
            dft=imgs,
            image_shape=self.particle_shape,
            shifts=hwShiftAngs / self.sampling_rate,  # Shifts need to be in pixels
            rfft=True,
            fftshifted=True,
        )

        if self.has_symmetry:
            imgs, ctf, rotMats, hwShiftAngs = self._expand_with_symmetry(imgs, ctf, rotMats, hwShiftAngs)

        if self.correct_ctf:
            ctf = ctf.to(imgs.device)
            imgs *= ctf

            # Stack for multichannel insertion
            imgs = torch.stack([imgs.real, imgs.imag, ctf**2], dim=1)

            dft_ctf, weights = compiled_insert_central_slices_rfft_3d_multichannel(
                image_rfft=imgs,
                volume_shape=(self.box_size,) * 3,
                rotation_matrices=rotMats,
                fftfreq_max=None,
                zyx_matrices=zyx_matrices,
            )

            # Update the volume Fourier space components
            # self.numerator += torch.view_as_complex(dft_ctf[:2, ...].permute(1, 2, 3, 0).contiguous())
            # self.numerator[...,0] += dft_ctf[0,...]
            # self.numerator[...,1] += dft_ctf[1,...]
            # self.numerator += dft_ctf[:2, ...].permute(1, 2, 3, 0)
            self.numerator += dft_ctf[:2, ...]
            self.ctfsq += dft_ctf[-1, ...]
            self.weights += weights
        else:
            raise NotImplementedError("Backprojection without CTF correction is not implemented.")

    @staticmethod
    @lru_cache(maxsize=1)
    def get_sincsq(shape, device):
        grid = fftfreq_grid(image_shape=shape, rfft=False, fftshift=True, norm=True, device=device)
        return torch.sinc(grid) ** 2

    def generate_volume(self, fname: Optional[FNAME_TYPE] = None, overwrite_fname: bool = True,
                        device: Optional[str] = "cpu"):

        dft = torch.zeros_like(self.numerator)

        # Only divide where we have sufficient weights
        mask = self.weights > self.min_denominator_value
        if self.correct_ctf:
            denominator = self.ctfsq[mask] + self.eps * self.weights[mask]
            # Handle potential near-zero denominators
            denominator[denominator.abs() < self.min_denominator_value] = self.min_denominator_value
            dft[:, mask] = self.numerator[:, mask] / denominator[None, ...]

        else:
            dft[:, mask] = self.numerator[:, mask] / self.weights[mask][None, ...]
        dft = torch.complex(real=dft[0,...], imag=dft[1,...])
        # back to real space
        dft = torch.fft.ifftshift(dft, dim=(-3, -2))  # actual ifftshift
        dft = torch.fft.irfftn(dft, dim=(-3, -2, -1))
        dft = torch.fft.ifftshift(dft, dim=(-3, -2, -1))  # center in real space

        # correct for convolution with linear interpolation kernel
        vol = dft.to(device) / self.get_sincsq(dft.shape, device)

        if fname is not None:
            mrcfile.write(fname, data=vol.detach().cpu().numpy(), overwrite=overwrite_fname,
                          voxel_size=self.sampling_rate)
        return vol

class ReconstructionParticlesDataset(Dataset):
    def __init__(self, particles_star_fname: FNAME_TYPE,
                 particles_dir: Optional[FNAME_TYPE] = None,
                 correct_ctf=True, subset_idxs=None):
        self.particles = ParticlesStarSet(starFname=particles_star_fname, particlesDir=particles_dir)
        if subset_idxs is not None:
            self.particles = self.particles.createSubset(idxs=subset_idxs)
        self.sampling_rate = self.particles.sampling_rate
        self.particle_shape = self.particles.particle_shape
        self.correct_ctf = correct_ctf

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

            ctf = compute_ctf_rfft(img.shape[-2], self.sampling_rate, dfu, dfv, dfang, volt, cs, w,
                                   phase_shift=0, bfactor=None, fftshift=True,
                                   device="cpu")
        else:
            ctf = 0 #This is just an indicator that no ctf is going to be used. None cannot be used due to torch DataLoader

        return img, ctf, rotMat, hwShiftAngs

def reconstruct_starfile(particles_star_fname: str, symmetry: str, output_fname: str, particles_dir:Optional[str]=None,
                         num_dataworkers: int = 1, batch_size: int = 64, use_cuda: bool = True,
                         correct_ctf: bool = True, eps: float = 1e-3, min_denominator_value: float = 1e-4,
                         use_only_n_first_batches: Optional[int] = None,
                         float32_matmul_precision: Optional[str] = None):
    """

    :param particles_star_fname: The particles to reconstruct
    :param symmetry: The symmetry of the volume (e.g. C1, D2, ...)
    :param output_fname: The name of the output filename
    :param particles_dir: The particles directory (root of the starfile fnames)
    :param num_dataworkers: Num workers for data loading
    :param batch_size: The number of particles to be simultaneusly backprojected
    :param use_cuda:
    :param correct_ctf:
    :param eps: The regularization constant (ideally, this is 1/SNR)
    :param min_denominator_value: Used to prevent division by 0
    :param use_only_n_first_batches: Use only the n first batches to reconstruct
    :param float32_matmul_precision: Set it to high or medium for speed up at a precision cost
    :return:
    """
    if float32_matmul_precision is not None:
        torch.set_float32_matmul_precision(float32_matmul_precision)  # 'high'

    device = "cpu" if not use_cuda else "cuda"
    reconstructor = Reconstructor(symmetry=symmetry, correct_ctf=correct_ctf, eps=eps,
                                  min_denominator_value=min_denominator_value)
    reconstructor.to(device=device)
    reconstructor.backproject_particles(particles_star_fname, particles_dir, batch_size, num_dataworkers,
                                        use_only_n_first_batches=use_only_n_first_batches)
    print("")
    reconstructor.generate_volume(output_fname)


def __backproject_test():
    reconstructor = Reconstructor(symmetry="C1", correct_ctf=True, eps=1e-3, min_denominator_value=1e-4, )
    reconstructor.to(device="cpu")
    reconstructor.backproject_particles(
        particles_star_fname="/home/sanchezg/cryo/data/preAlignedParticles/EMPIAR-10166/data/projections/1000proj_with_ctf.star",
        particles_dir="/home/sanchezg/cryo/data/preAlignedParticles/EMPIAR-10166/data/", batch_size=96,
        num_dataworkers=0)
    return reconstructor

def _test():
    reconstructor = __backproject_test()
    vol = reconstructor.generate_volume(fname="/tmp/reconstructed_volume.mrc", device="cpu")

    relion_vol = torch.as_tensor(
        mrcfile.read("/home/sanchezg/cryo/myProjects/cryoPARES/cryoPARES/reconstruction/relion_reconstruct.mrc"),
        dtype=torch.float32)
    from torch_fourier_shell_correlation import fsc
    fsc_result = fsc(vol, relion_vol)
    print(fsc_result)

    from matplotlib import pyplot as plt
    f, axes = plt.subplots(2,3, squeeze=False)
    axes[0, 0].imshow(vol[..., vol.shape[-1]//2])
    axes[0, 1].imshow(vol[..., vol.shape[-2]//2, :])
    axes[0, 2].imshow(vol[vol.shape[0]//2, ...])

    axes[1, 0].imshow(relion_vol[..., relion_vol.shape[-1]//2])
    axes[1, 1].imshow(relion_vol[..., relion_vol.shape[-2]//2, :])
    axes[1, 2].imshow(relion_vol[relion_vol.shape[0]//2, ...])

    plt.show()
    print()

def _test_real_insertion():
    from scipy.spatial.transform import Rotation
    from lightning import seed_everything
    seed_everything(32)

    b, h, w = 32, 8, 5
    img_rfft = torch.rand(b, h, w, dtype=torch.complex64)
    ctf = torch.rand(b, h, w, dtype=torch.float32)
    rotmats = torch.FloatTensor(Rotation.random(b).as_matrix())
    img_as_real = torch.stack([img_rfft.real, img_rfft.imag, ctf], dim=1)
    dft_ctf0, weights = insert_central_slices_rfft_3d_multichannel(
        image_rfft=img_as_real,
        volume_shape=(h,) * 3,
        rotation_matrices=rotmats,
        fftfreq_max=None,
        zyx_matrices=False,
    )

    img_ctf = torch.stack([img_rfft, ctf], dim=1)
    dft_ctf1, weights = insert_central_slices_rfft_3d_multichannel(
        image_rfft=img_ctf,
        volume_shape=(h,) * 3,
        rotation_matrices=rotmats,
        fftfreq_max=None,
        zyx_matrices=False,
    )
    print(dft_ctf0[0].allclose(dft_ctf1[0].real, atol=1e-4))
    print(dft_ctf0[1].allclose(dft_ctf1[0].imag, atol=1e-4))
    print("DONE")

def _profile___backproject_test():
    import cProfile
    import pstats

    __backproject_test() #Warmup
    # Save profile data to a file
    cProfile.run('__backproject_test()', '/tmp/profile_output.prof')

    # Load and analyze the saved profile
    stats = pstats.Stats('/tmp/profile_output.prof')
    stats.sort_stats('cumulative')  # Sort by cumulative time
    stats.print_stats()

if __name__ == "__main__":
    # _profile___backproject_test(); sys.exit()
    # _test(); sys.exit()
    # _test_real_insertion()
    from argParseFromDoc import parse_function_and_call
    parse_function_and_call(reconstruct_starfile)

    """
--symmetry C1 --particles_star_fname /home/sanchezg/cryo/data/preAlignedParticles/EMPIAR-10166/data/allparticles.star --output_fname /tmp/reconstruction.mrc --use_only_n_first_batches 100    
    """