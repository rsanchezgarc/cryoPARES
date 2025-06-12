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

```text
V̂(k) = Σ_i   CTF_i(k) · Î_i(k)
        ─────────────────────────
        Σ_i   CTF_i(k)^2 + ε
```
"""
from typing import List, Optional

import torch
import torch.fft as _fft
from starstack import ParticlesStarSet
from starstack.constants import RELION_ANGLES_NAMES, RELION_SHIFTS_NAMES, RELION_IMAGE_FNAME, RELION_EULER_CONVENTION
from torch.utils.data import Dataset, DataLoader
from torch_fourier_shift import fourier_shift_dft_2d
from torch_fourier_slice import backproject_2d_to_3d, insert_central_slices_rfft_3d
from torch_grid_utils import fftfreq_grid

from cryoPARES.datamanager.relionStarDataset import ParticlesRelionStarDataset
from cryoPARES.geometry.convert_angles import euler_angles_to_matrix


class Reconstructor():
    def __init__(self, symmetry: str, image_size_px: int,
                 correct_ctf: bool = True, eps=1e-3):

        self.symmetry = symmetry
        self.box_size = image_size_px
        nky, nkx = self.box_size, self.box_size // 2 + 1
        self.f_num = torch.zeros((self.box_size, self.box_size, nkx), dtype=torch.complex64, device=self.device)
        self.f_den = torch.zeros_like(self.f_num, dtype=torch.float32)
        self.correct_ctf = correct_ctf
        self.eps = eps # The Tikhonov constant

    def _load_particles(self, particles_star_fname: List[str],
                 particles_dir: Optional[List[str]] = None):
        particles = ParticlesStarSet(starFname=particles_star_fname, particlesDir=particles_dir)
        particlesDataset = ReconstructionParticlesDataset(particles)



    def backproject_particles(self, particles_star_fname: List[str],
                              particles_dir: Optional[List[str]] = None,
                              batch_size=1, num_workers=0,):
        particlesDataset = ReconstructionParticlesDataset(particles_star_fname, particles_dir,
                                                          correct_ctf=self.correct_ctf)
        dl = DataLoader(particlesDataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        for imgs, rotMats in dl:
            dft_3d, weights = insert_central_slices_rfft_3d(
                image_rfft=imgs,
                volume_shape=(imgs.shape[-1], )*3,
                rotation_matrices=rotMats,
                fftfreq_max=None,
                zyx_matrices=True,
            )
            self.f_num += dft_3d

            if self.correct_ctf:
                ctf_sq_3d, _ = insert_central_slices_rfft_3d(
                    image_rfft=imgs,
                    volume_shape=(imgs.shape[-1],) * 3,
                    rotation_matrices=rotMats,
                    fftfreq_max=None,
                    zyx_matrices=True,
                )
                self.f_den += (ctf_sq_3d * weights)
            else:
                self.f_den += weights

    def generate_volume(self, device="cpu"):
        dft = self.f_num / (self.f_den + self.eps)
        # back to real space
        dft = torch.fft.ifftshift(dft, dim=(-3, -2))  # actual ifftshift
        dft = torch.fft.irfftn(dft, dim=(-3, -2, -1))
        dft = torch.fft.ifftshift(dft, dim=(-3, -2, -1))  # center in real space

        # correct for convolution with linear interpolation kernel
        grid = fftfreq_grid(
            image_shape=dft.shape, rfft=False, fftshift=True, norm=True, device=dft.device
        )
        vol = dft / torch.sinc(grid) ** 2
        return vol

class ReconstructionParticlesDataset(Dataset):
    def __init__(self, particles_star_fname: List[str],
                 particles_dir: Optional[List[str]] = None,
                 correct_ctf=True):
        self.particles = ParticlesStarSet(starFname=particles_star_fname, particlesDir=particles_dir)
        self.sampling_rate = self.particles.sampling_rate
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
        rotMat = euler_angles_to_matrix(degEuler, convention=RELION_EULER_CONVENTION)
        xyShiftAngs = torch.FloatTensor([md_row[name] for name in RELION_SHIFTS_NAMES])

        iid = md_row[RELION_IMAGE_FNAME]
        img = torch.FloatTensor(img)
        img_shape = img.shape
        img = torch.fft.fftshift(img, dim=(-2, -1))  # volume center to array origin
        img = torch.fft.rfftn(img, dim=(-2, -1))
        img = torch.fft.fftshift(img, dim=(-2,))  # actual fftshift

        img = fourier_shift_dft_2d(dft = img,
            image_shape=img_shape,
            shifts = xyShiftAngs/self.sampling_rate, #Shifts are in pixels
            rfft=True,
            fftshifted=True,
        )

        if self.correct_ctf:
            raise NotImplementedError()
        else:
            ctf = None

        return img, ctf, rotMat


if __name__ == "__main__":
    raise NotImplementedError()