import copy
import functools
import os
import os.path as osp
import shutil
from functools import lru_cache

import numpy as np
import torch
from joblib.externals.loky.backend import get_context
from tensordict import TensorDict
from torch.utils.data import Dataset

from libtilt.ctf.ctf_2d import calculate_ctf_cached
from .dataUtils.starUtils import _get_repeated_image_idxs
from .loggers import getWorkerLogger
from .myProgressBar import myTqdm as tqdm

from libtilt.shapes import circle
from starstack import ParticlesStarSet
from .dataUtils.dataTypes import DEFAULT_DTYPE
from .dataUtils.compute_ctf import compute_ctf


class ParticlesDatasetBase(Dataset):
    IMAGES_AS_FOURIER = NotImplemented
    def __init__(self, data: ParticlesStarSet, mmap_dirname: str | None, particle_radius_angs: float | None = None,
                 pad_length: int | None = None, device: torch.device = "cpu", batch_size: int = 1024, n_jobs = 4,
                 verbose: bool = False):

        self.part_data = data
        self.pad_length = pad_length
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.mainLogger = getWorkerLogger(verbose)

        self._device = device

        # Prepare masks
        if particle_radius_angs is None:
            radius_px = self.part_data.particle_shape[-1] // 2
        else:
            radius_px = particle_radius_angs / self.sampling_rate

        self.radius_px = radius_px

        self.mmap_dirname = mmap_dirname
        if mmap_dirname is not None:
            if os.path.exists(mmap_dirname):
                shutil.rmtree(mmap_dirname)
            os.mkdir(mmap_dirname)

        self.dataDict, self.n_partics, duplicated_indices = self._precompute_data()
        self.selected_idxs = [idxs[0] for idxs in duplicated_indices]

        self.mainLogger.info("Particles were pre-processed!\n")


    def clone(self, copy_dataDict=True):

        # Create a shallow copy of the current instance
        new_instance = copy.copy(self)

        # If required, deepcopy the dataDict to ensure a completely separate copy
        if copy_dataDict:
            new_instance.dataDict = copy.deepcopy(self.dataDict)

        # Ensure the device setting is correctly applied to the new instance
        new_instance._device = self._device
        new_instance.part_data = copy.deepcopy(self.part_data)
        return new_instance

    def get_particles_starstack(self, drop_rlnImageId=True):
        particlesStar = self.part_data.createSubset(idxs=self.selected_idxs)
        if drop_rlnImageId and "rlnImageId" in particlesStar.particles_md.columns:
            particlesStar.particles_md.drop("rlnImageId", axis=1)
        return particlesStar

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        self._device = device
        if self.mmap_dirname is not None:
            self.dataDict.to(device)

    @property
    def sampling_rate(self):
        sampling_rate = self.part_data.sampling_rate
        return sampling_rate

    def preprocess_particles(self, starFname, root_dir, pad_length, radius_px, compute_device,
                            batch_size, num_workers, verbose):
        raise NotImplementedError()

    @torch.inference_mode()
    def _precompute_data(self):

        self.mainLogger.info("Precomputing  fft, ctf and moving them to RAM...")

        # starFname = self.part_data.starFname
        root_dir = self.part_data.particlesDir
        pad_length = self.pad_length
        radius_px = self.radius_px
        n_jobs = self.n_jobs if self.n_jobs > 0 else 1
        batch_size = self.batch_size
        if str(self.device).startswith("cuda"):
            batch_size = batch_size * 2
            batch_size = (batch_size // n_jobs + int(bool(batch_size % n_jobs)))
        else:
            batch_size = batch_size

        imgs, ctfs, (eulerDegs,
                     xyShiftAngs, confidences), duplicated_indices = self.preprocess_particles(self.part_data, root_dir,
                                                                            pad_length, radius_px,
                                                                            compute_device=self.device,
                                                                            batch_size=batch_size,
                                                                            num_workers=self.n_jobs,
                                                                            verbose=self.verbose)

        dataDict = TensorDict(
            dict(
                imgs_reprep=imgs,
                ctfs=ctfs,
                eulerDegs=eulerDegs,
                shiftsAngs=xyShiftAngs,
                confidences = confidences,
            ), batch_size=[imgs.shape[0]], device="cpu",
        )

        if self.mmap_dirname is not None:
            dataDict = dataDict.to(self.device)
        return dataDict, imgs.shape[0], duplicated_indices

    def __getitem__(self, item):
        dataDict = self.dataDict[item]
        return item, dataDict.get("imgs_reprep").to(self.device), \
            dataDict.get("ctfs").to(self.device), \
            dataDict.get("eulerDegs").to(self.device)

    def getMd(self, item):
        raise NotImplementedError("Error, if there is duplicated particles, we need to remap them")
        # return self.part_data[item]

    def __len__(self):
        return self.n_partics

class ParticlesFourierDataset(ParticlesDatasetBase):
    IMAGES_AS_FOURIER = True
    def _fourier_proj_to_real(self, projections):
        return _fourier_proj_to_real(projections, pad_length=self.pad_length)

    def preprocess_particles(self, particles, root_dir, pad_length, radius_px, compute_device,
                            batch_size, num_workers, verbose):
        return compute_inputs_fourier(particles, root_dir, pad_length, radius_px, compute_device, batch_size,
                                      num_workers, verbose=verbose)


def _fourier_proj_to_real(projections, pad_length=None):
    projections = torch.fft.ifftshift(projections, dim=(-2,))  # ifftshift of rfft
    projections = torch.fft.irfftn(projections, dim=(-2, -1))
    projections = torch.fft.ifftshift(projections, dim=(-2, -1))  # recenter real space

    if pad_length is not None:
        projections = projections[..., pad_length: -pad_length, pad_length: -pad_length]
    return projections

class ParticlesRealSpaceDataset(ParticlesDatasetBase):
    IMAGES_AS_FOURIER = False


@lru_cache(1)
def _getMask(radius_px, particle_shape, device):
    return circle(radius_px, image_shape=particle_shape, smoothing_radius=radius_px * .05).to(device)


@torch.inference_mode()
def compute_inputs_fourier(particles, root_dir, pad_length: int, radius_px, compute_device, batch_size, num_workers, verbose=True):


    fftdataset = TorchParticlesForFftDataset(particles, root_dir, radius_px, pad_length, device="cpu")

    one_img_example = _compute_one_batch_fft(fftdataset[0][1]) #TODO: make this agnostic to Fourier/real space

    fft_img_shape = one_img_example.shape
    full_fft_img_shape = tuple([fft_img_shape[-2]]*2)

    fftdataset.reset_dataset()  #Reseting the dataset to avoid multiprocessing pickle issue
    dl = torch.utils.data.DataLoader(fftdataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                     persistent_workers=True if num_workers > 0 else False,
                                     pin_memory=str(compute_device).startswith("cuda"),
                                     multiprocessing_context=get_context('loky') #TODO: make this more robust in case we don't use loky
                                     )
    #TODO: How to deal with multiple copies of the same particle. At the moment we just recompute them, wasting compute
    n_particles = len(fftdataset)

    images = torch.zeros(n_particles, *one_img_example.shape, dtype=one_img_example.dtype, device="cpu")
    ctfs = torch.zeros(n_particles, *fft_img_shape, dtype=one_img_example.dtype, device="cpu")
    anglesDegs = torch.zeros(n_particles, fftdataset.n_copies, 3, dtype=DEFAULT_DTYPE, device="cpu")
    xyShiftAngs = torch.zeros(n_particles, fftdataset.n_copies, 2, dtype=DEFAULT_DTYPE, device="cpu")
    confidence = torch.zeros(n_particles, fftdataset.n_copies, dtype=DEFAULT_DTYPE, device="cpu")

    ctf_constant_params = fftdataset.getCtfConstantParameters()

    if compute_device != "cpu":
        for k in ctf_constant_params:
            if isinstance(ctf_constant_params[k], torch.Tensor):
                ctf_constant_params[k] = ctf_constant_params[k].to(compute_device)

    _compute_ctf = functools.partial(calculate_ctf_cached,
                                     **ctf_constant_params, b_factor=0., phase_shift=0.,
                                     image_shape=full_fft_img_shape,
                                     rfft=False, fftshift=True, device=compute_device)

    for idxs, _imgs, (_anglesDegs, _xyShiftAngs, _conf), ctf_params in tqdm(dl, desc="Computing fft", disable=not verbose):
        imgs = _imgs.to(compute_device, non_blocking=True)
        images[idxs] = _compute_one_batch_fft(imgs).cpu()
        anglesDegs[idxs] = _anglesDegs
        xyShiftAngs[idxs] = _xyShiftAngs
        confidence[idxs] = _conf

        defocus_u = torch.as_tensor(ctf_params["defocus_u"] * 1e-4, dtype=DEFAULT_DTYPE, device=compute_device)  # micrometers
        defocus_v = torch.as_tensor(ctf_params["defocus_v"] * 1e-4, dtype=DEFAULT_DTYPE, device=compute_device)  # micrometers
        astigmatism_angle = torch.as_tensor(ctf_params["defocus_angle"], dtype=DEFAULT_DTYPE, device=compute_device)
        defocus = 0.5 * (defocus_u + defocus_v)
        astigmatism = 0.5 * (defocus_u - defocus_v)


        _ctfs = _compute_ctf(defocus, astigmatism, astigmatism_angle).cpu()

        # _ctfs = compute_ctf(image_size=full_fft_img_shape[-2],
        #                     sampling_rate= ctf_constant_params["pixel_size"],
        #                     dfu= torch.as_tensor(ctf_params["defocus_u"], dtype=DEFAULT_DTYPE, device=compute_device),
        #                     dfv= torch.as_tensor(ctf_params["defocus_v"], dtype=DEFAULT_DTYPE, device=compute_device),
        #                     dfang=torch.as_tensor(ctf_params["defocus_angle"], dtype=DEFAULT_DTYPE, device=compute_device),
        #                     volt=ctf_constant_params["voltage"],
        #                     cs=ctf_constant_params['spherical_aberration'], w=ctf_constant_params['amplitude_contrast'],
        #                     phase_shift=0, bfactor=None, device=compute_device).unsqueeze(1).cpu()

        centre = _ctfs.shape[-2] // 2
        _ctfs = _ctfs.to(ctfs.dtype)
        #Next two lines are to convert a fft into a rfft. I should be able to generate ctfs in rrft using rfftfreq instead
        ctfs[idxs, ..., :-1] = _ctfs[:,0,..., centre:]
        ctfs[idxs, ..., -1] = _ctfs[:,0,..., 0]


    return images, ctfs, (anglesDegs, xyShiftAngs, confidence), fftdataset.repeated_indices

def _compute_one_batch_fft(imgs): #TODO: Is it worth it to compile it?
    imgs = torch.fft.fftshift(imgs, dim=(-2, -1))
    imgs = torch.fft.rfftn(imgs, dim=(-2, -1))
    imgs = torch.fft.fftshift(imgs, dim=(-2,))
    # imgs = imgs / imgs.abs().sum() #Normalization
    return imgs

class TorchParticlesForFftDataset(torch.utils.data.Dataset):
    def __init__(self, particles, root_dir, radius_px, pad_length, device="cpu"):

        self.root_dir = root_dir
        self.device = device
        if isinstance(particles, ParticlesStarSet):
            dataset = particles
            self.clean_dataset = dataset.copy()
        else:
            self.starFname = particles #TODO: Check if this is a legacy code
            dataset = ParticlesStarSet(particles, root_dir)
            self.clean_dataset = dataset.copy()


        repeated_indices, n_copies = _get_repeated_image_idxs(dataset.particles_md)
        self.repeated_indices = repeated_indices
        self.n_copies = n_copies
        self.n_unique_indices = len(self.repeated_indices)

        particle_shape = dataset.particle_shape

        self.particle_shape = particle_shape

        if pad_length is not None:
            self.pad_list = [pad_length] * 4
            self.pad_fun = self._pad_fun
        else:
            self.pad_fun = self._identity

        self.rmask = _getMask(radius_px, particle_shape, self.device)
        self._dataset = None
        # self._set_dataset(dataset)
        # self.clean_dataset = None #reset_dataset sets the clean_dataset property
        # self.reset_dataset(dataset.copy())


    def _identity(self, imgs):
        return imgs

    def _pad_fun(self, imgs):
        return torch.nn.functional.pad(imgs, pad=self.pad_list, mode='constant', value=0)

    @property
    def dataset(self):  # We want a different particlesStarSet in each subprocess

        if self._dataset is None:
            # self._set_dataset(ParticlesStarSet(self.starFname, self.root_dir))
            self.reset_dataset()
        return self._dataset

    # def _set_dataset(self, particlesStarSet):
    #     self._dataset = particlesStarSet
    #     self._poses = [torch.as_tensor(x, dtype=DEFAULT_DTYPE)
    #                    for x in self.dataset.getPoseFromMd(self.dataset.particles_md)]
    #     if "rlnParticleFigureOfMerit" in self.dataset.particles_md.columns:
    #         self._confidences = torch.as_tensor(self.dataset.particles_md["rlnParticleFigureOfMerit"].values,
    #                                             dtype=DEFAULT_DTYPE)
    #     else:
    #         self._confidences = torch.ones(len(self.dataset), dtype=DEFAULT_DTYPE) / self.n_copies

    def reset_dataset(self, dataset=None):
        if dataset is None:
            dataset = self.clean_dataset.copy()
        else:
            dataset = dataset.copy()
            self.clean_dataset = dataset
        self._dataset = dataset
        self._poses = [torch.as_tensor(x, dtype=DEFAULT_DTYPE)
                       for x in self.dataset.getPoseFromMd(self.dataset.particles_md)]
        if "rlnParticleFigureOfMerit" in self.dataset.particles_md.columns:
            self._confidences = torch.as_tensor(self.dataset.particles_md["rlnParticleFigureOfMerit"].values,
                                                dtype=DEFAULT_DTYPE)
        else:
            self._confidences = torch.ones(len(self.dataset), dtype=DEFAULT_DTYPE) / self.n_copies

    def getCtfConstantParameters(self):
        voltage = set(self.dataset.optics_md.rlnVoltage.values)
        assert len(voltage) == 1, "Error, only particles with the same voltage can be processed"
        voltage = voltage.pop()

        amplitude_contrast = set(self.dataset.optics_md.rlnAmplitudeContrast.values)
        assert len(amplitude_contrast) == 1, "Error, only particles with the same amplitude_contrast can be processed"
        amplitude_contrast = amplitude_contrast.pop()

        spherical_aberration = set(self.dataset.optics_md.rlnSphericalAberration.values)
        assert len(spherical_aberration) == 1, "Error, only particles with the same spherical_aberration can be processed"
        spherical_aberration = spherical_aberration.pop()

        pixel_size = set(self.dataset.optics_md.rlnImagePixelSize.values)
        assert len(pixel_size) == 1, "Error, only particles with the same pixel_size can be processed"
        pixel_size = pixel_size.pop()

        return dict(
            voltage=voltage,
            amplitude_contrast=amplitude_contrast,
            spherical_aberration=spherical_aberration,
            pixel_size=pixel_size,
        )

    def __len__(self):
        return self.n_unique_indices

    @property
    def pixel_size(self):
        return self.dataset.sampling_rate

    def __getitem__(self, item):
        idxs = self.repeated_indices[item]

        img, md = self.dataset[idxs[0]]
        img = torch.as_tensor(img, dtype=DEFAULT_DTYPE, device=self.device)
        img *= self.rmask
        img = self.pad_fun(img) #TODO: Check how padding affect CTF.

        anglesDegs = self._poses[0][idxs]
        xyShiftAngs = self._poses[1][idxs]
        ctf_params = self._dataset.getCtfParamsFromMd(md)
        confs = self._confidences[idxs]
        return torch.as_tensor(item), img, (anglesDegs, xyShiftAngs, confs), ctf_params


COMPILE = False #TODO: move this to config
if COMPILE:
    _compute_one_batch_fft = torch.compile(_compute_one_batch_fft)
    # _calculate_ctf = torch.compile(_calculate_ctf)
    # process_particle_idxs = torch.compile(process_particle_idxs)

if __name__ == "__main__":
    root_dir = osp.expanduser("~/ScipionUserData/projects/simulated_bgal/")
    dirname = osp.join(root_dir, "Runs/000422_ProtRelionRefine3D/extra/")
    starFname = osp.join(dirname, "relion_data_with_dupl.star") #"relion_data.star")
    pad_length = 128
    n_partics = 2000
    device = "cpu"
    mmap_dirname = None #"/tmp/mmamp_trial"
    data = ParticlesStarSet(starFname, root_dir)
    dataset = ParticlesFourierDataset(data, mmap_dirname=mmap_dirname, particle_radius_angs=80, pad_length=pad_length,
                                      batch_size=512)
    print(dataset.dataDict["eulerDegs"].numpy())
    c_dataset = dataset.clone()
    print()