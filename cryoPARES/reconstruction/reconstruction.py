"""Cryo‑EM 3‑D reconstruction –
====================================================
A pipeline that reconstructs a Coulomb potential
volume from a RELION‑style particle STAR file using the open‑source
**TeamTomo** ecosystem:

* [`starstack`](https://github.com/rsanchezgarc/starstack) – zero‑copy image
  streaming + optics‑table handling.
* [`torch‑fourier‑slice`](https://github.com/teamtomo/torch-fourier-slice) –
  Fourier‑slice back‑projection on GPU.
* [`torch‑fourier‑shift`](https://github.com/teamtomo/torch-fourier-shift) –
  sub‑pixel recentering in Fourier space.

"""
from typing import List, Optional

import mrcfile
import torch
import tqdm
from starstack import ParticlesStarSet
from starstack.constants import RELION_ANGLES_NAMES, RELION_SHIFTS_NAMES, RELION_EULER_CONVENTION
from torch.utils.data import Dataset, DataLoader
from torch_fourier_shift import fourier_shift_dft_2d
from torch_fourier_slice import insert_central_slices_rfft_3d, insert_central_slices_rfft_3d_multichannel
from torch_grid_utils import fftfreq_grid

from cryoPARES.datamanager.ctf.rfft_ctf import compute_ctf_rfft
from cryoPARES.geometry.convert_angles import euler_angles_to_matrix
from cryoPARES.geometry.symmetry import getSymmetryGroup
from cryoPARES.utils.paths import FnameType

compiled_insert_central_slices_rfft_3d_multichannel = torch.compile(insert_central_slices_rfft_3d_multichannel)
# compiled_insert_central_slices_rfft_3d_multichannel = insert_central_slices_rfft_3d_multichannel

class Reconstructor():
    def __init__(self, symmetry: str, device:str,
                 correct_ctf: bool = True, eps=1e-3, min_denominator_value=1e-4):

        self.symmetry = symmetry.upper()
        self.has_symmetry = self.symmetry != "C1"
        self.device = device

        self.correct_ctf = correct_ctf
        self.eps = eps # The Tikhonov constant. Should be 1/SNR, we might want to estimate it per frequency
        self.min_denominator_value = min_denominator_value

        self.box_size = None
        self.sampling_rate = None
        self.f_num = None
        self.f_den = None


    def set_metadata_from_particles(self, particlesDataset):

        box_size = particlesDataset.particle_shape[-1]
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

            self.f_num = torch.zeros((self.box_size, self.box_size, nkx), dtype=torch.complex64, device=self.device)
            self.weights = torch.zeros_like(self.f_num, dtype=torch.float32) #TODO: Separate weights from ctf**2
            self.ctfs = torch.zeros_like(self.f_num, dtype=torch.float32)


    def backproject_particles(self, particles_star_fname: FnameType,
                              particles_dir: Optional[FnameType] = None,
                              batch_size=1, num_workers=0, use_only_n_first_batches=None):
        particlesDataset = ReconstructionParticlesDataset(particles_star_fname, particles_dir,
                                                          correct_ctf=self.correct_ctf)

        self.set_metadata_from_particles(particlesDataset)

        dl = DataLoader(particlesDataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers,
                        pin_memory=num_workers > 0 and str(self.device).startswith("cuda"))


        symMat = getSymmetryGroup(self.symmetry, as_matrix=True, device=self.device)
        zyx_matrices = False
        for bidx, (imgs, ctf, rotMats, hwShiftAngs) in enumerate(tqdm.tqdm(dl, desc="backprojecting", disable=False)):

            imgs = imgs.to(self.device, non_blocking=True)
            rotMats = rotMats.to(self.device, non_blocking=True)
            hwShiftAngs = hwShiftAngs.to(self.device, non_blocking=True)

            imgs = torch.fft.fftshift(imgs, dim=(-2, -1))  # volume center to array origin
            imgs = torch.fft.rfftn(imgs, dim=(-2, -1))
            imgs = torch.fft.fftshift(imgs, dim=(-2,))  # actual fftshift

            # from matplotlib import pyplot as plt
            # f, axes = plt.subplots(1, 2)
            # axes[0].imshow(imgs[0].abs().log())
            # axes[1].imshow(ctf[0])
            # plt.show()

            if self.has_symmetry:
                extended_rotMats = symMat @ rotMats
                raise NotImplementedError()

            imgs = fourier_shift_dft_2d(dft=imgs,
                                       image_shape=self.particle_shape,
                                       shifts=hwShiftAngs / self.sampling_rate,  # Shifts need to be passed in pixels
                                       rfft=True,
                                       fftshifted=True,
                                       )

            if self.correct_ctf:
                ctf = ctf.to(imgs.device)
                imgs *= ctf
                # #The following is for phase-flip only
                # imgs *= ctf.sign()
                # ctf = torch.ones_like(ctf)

                imgs = torch.stack([imgs.real, imgs.imag, ctf**2], dim=1)
                # imgs = torch.stack([imgs, ctf**2], dim=1)

                dft_ctf, weights = compiled_insert_central_slices_rfft_3d_multichannel(
                    image_rfft=imgs,
                    volume_shape=(self.box_size,) * 3,
                    rotation_matrices=rotMats,
                    fftfreq_max=None,
                    zyx_matrices=zyx_matrices,
                )

                # self.f_num += dft_ctf[0, ...]
                # self.ctfs += dft_ctf[1, ...].real

                self.f_num += torch.view_as_complex(dft_ctf[:2, ...].permute(1,2,3,0).contiguous())
                self.ctfs += dft_ctf[-1, ...]

                self.weights += weights
            else:
                dft_3d, weights = insert_central_slices_rfft_3d(
                    image_rfft=imgs,
                    volume_shape=(self.box_size,) * 3,
                    rotation_matrices=rotMats,
                    fftfreq_max=None,
                    zyx_matrices=zyx_matrices,
                )
                self.f_num += dft_3d
                self.weights += weights

            # if self.correct_ctf:
            #     pass
            #     ctf_sq_3d, _ = insert_central_slices_rfft_3d(
            #         image_rfft=ctf ** 2,
            #         volume_shape=(self.box_size,) * 3,
            #         rotation_matrices=rotMats,
            #         fftfreq_max=None,
            #         zyx_matrices=zyx_matrices,
            #     )
            #     self.ctfs += ctf_sq_3d

            if use_only_n_first_batches and bidx > use_only_n_first_batches: #TODO: Remove this debug code
                break

    def generate_volume(self, fname: Optional[FnameType] = None, overwrite_fname: bool = True,
                        device: Optional[str] = "cpu"):

        dft = torch.zeros_like(self.f_num)

        # Only divide where we have sufficient weights
        mask = self.weights > self.min_denominator_value
        if self.correct_ctf:
            denominator = self.ctfs[mask] + self.eps * self.weights[mask]
            # Handle potential near-zero denominators
            denominator[denominator.abs() < self.min_denominator_value] = self.min_denominator_value
            dft[mask] = self.f_num[mask] / denominator

        else:
            dft[mask] = self.f_num[mask] / self.weights[mask]

        # back to real space
        dft = torch.fft.ifftshift(dft, dim=(-3, -2))  # actual ifftshift
        dft = torch.fft.irfftn(dft, dim=(-3, -2, -1))
        dft = torch.fft.ifftshift(dft, dim=(-3, -2, -1))  # center in real space

        # correct for convolution with linear interpolation kernel
        grid = fftfreq_grid( image_shape=dft.shape, rfft=False, fftshift=True, norm=True, device=dft.device)
        vol = dft / torch.sinc(grid) ** 2
        vol = vol.to(device)

        if fname is not None:
            mrcfile.write(fname, data=vol.detach().cpu().numpy(), overwrite=overwrite_fname,
                          voxel_size=self.sampling_rate)
        return vol

class ReconstructionParticlesDataset(Dataset):
    def __init__(self, particles_star_fname: FnameType,
                 particles_dir: Optional[FnameType] = None,
                 correct_ctf=True):
        self.particles = ParticlesStarSet(starFname=particles_star_fname, particlesDir=particles_dir)
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
                         num_workers: int = 1, batch_size: int = 64, use_cuda: bool = False,
                         correct_ctf: bool = True, eps: float = 1e-3, min_denominator_value: float = 1e-4,
                         use_only_n_first_batches: Optional[int] = None):
    """

    :param particles_star_fname: The particles to reconstruct
    :param symmetry: The symmetry of the volume (e.g. C1, D2, ...)
    :param output_fname: The name of the output filename
    :param particles_dir: The particles directory (root of the starfile fnames)
    :param num_workers:
    :param batch_size: The number of particles to be simultaneusly backprojected
    :param use_cuda:
    :param correct_ctf:
    :param eps: The regularization constant (ideally, this is 1/SNR)
    :param min_denominator_value: Used to prevent division by 0
    :param use_only_n_first_batches: Use only the n first batches to reconstruct
    :return:
    """

    device = "cpu" if not use_cuda else "cuda"
    reconstructor = Reconstructor(symmetry=symmetry, device=device, correct_ctf=correct_ctf, eps=eps,
                                  min_denominator_value=min_denominator_value)
    reconstructor.backproject_particles(particles_star_fname, particles_dir, batch_size, num_workers,
                                  use_only_n_first_batches=use_only_n_first_batches)
    reconstructor.generate_volume(output_fname)


def _test():
    reconstructor = Reconstructor(symmetry="C1", device="cpu", correct_ctf=True, eps=1e-3, min_denominator_value=1e-4)

    reconstructor.backproject_particles(
        particles_star_fname="/home/sanchezg/cryo/data/preAlignedParticles/EMPIAR-10166/data/projections/1000proj_with_ctf.star",
        particles_dir="/home/sanchezg/cryo/data/preAlignedParticles/EMPIAR-10166/data/",
        batch_size=96, num_workers=0, )

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

if __name__ == "__main__":
    # _test()
    # _test_real_insertion()
    from argParseFromDoc import parse_function_and_call
    parse_function_and_call(reconstruct_starfile)