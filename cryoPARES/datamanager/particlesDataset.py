import functools
import warnings
from abc import ABC, abstractmethod
from functools import cached_property

import torch
import numpy as np

from scipy.spatial.transform import Rotation as R
from sklearn.model_selection import train_test_split
from starstack.constants import RELION_IMAGE_FNAME
from starstack.particlesStar import ParticlesStarSet
from torch.utils.data import Dataset
from typing import Union, Literal, Optional, List, Tuple, Any, Dict

from cryoPARES.cacheManager import get_cache
from autoCLI_config import inject_defaults_from_config, CONFIG_PARAM
from cryoPARES.configs.datamanager_config.particlesDataset_config import CtfCorrectionType, ImgNormalizationType
from cryoPARES.configs.mainConfig import main_config
from cryoPARES.constants import RELION_ANGLES_NAMES, RELION_SHIFTS_NAMES, \
    RELION_PRED_POSE_CONFIDENCE_NAME, RELION_EULER_CONVENTION, RELION_ORI_POSE_CONFIDENCE_NAME, BATCH_PARTICLES_NAME, \
    BATCH_IDS_NAME, BATCH_POSE_NAME, BATCH_MD_NAME, BATCH_ORI_IMAGE_NAME, BATCH_ORI_CTF_NAME

warnings.filterwarnings("ignore", "Gimbal lock detected. Setting third angle to zero since it "
                                  "is not possible to uniquely determine all angles.")
warnings.filterwarnings("ignore", message="The torchvision.datapoints and torchvision.transforms.v2 namespaces")


from cryoPARES.datamanager.augmentations import AugmenterBase
from cryoPARES.datamanager.ctf.rfft_ctf import correct_ctf
from cryoPARES.utils.torchUtils import data_to_numpy

class ParticlesDataset(Dataset, ABC):
    #TODO: This class still has several Relion-specific features
    @inject_defaults_from_config(main_config.datamanager.particlesdataset)
    def __init__(self,
                 symmetry: str,
                 halfset: Optional[int],
                 sampling_rate_angs_for_nnet: float = CONFIG_PARAM(),
                 image_size_px_for_nnet: int = CONFIG_PARAM(),
                 store_data_in_memory: bool = CONFIG_PARAM(),
                 mask_radius_angs: Optional[float] = CONFIG_PARAM(),
                 apply_mask_to_img: bool = CONFIG_PARAM(),
                 min_maxProb: Optional[float] = CONFIG_PARAM(),
                 perImg_normalization: Literal["none", "noiseStats", "subtractMean"] = CONFIG_PARAM(),
                 ctf_correction: Literal["none", "phase_flip", "ctf_multiply",
                                         "concat_phase_flip", "concat_ctf_multiply"] = CONFIG_PARAM(),
                 reduce_symmetry_in_label:bool = CONFIG_PARAM(),
                 return_ori_imagen: bool = False,
                 subset_idxs: Optional[List[int]] = None
                 ):

        super().__init__()

        self.sampling_rate_angs_for_nnet = sampling_rate_angs_for_nnet
        self.image_size_px_for_nnet = image_size_px_for_nnet
        self.store_data_in_memory = store_data_in_memory
        self.mask_radius_angs = mask_radius_angs
        self.apply_mask_to_img = apply_mask_to_img
        self.min_maxProb = min_maxProb
        self.reduce_symmetry_in_label = reduce_symmetry_in_label
        self.return_ori_imagen = return_ori_imagen
        
        self.symmetry = symmetry.upper()
        self.halfset = halfset
        self.subset_idxs = subset_idxs

        assert perImg_normalization in (item.value for item in ImgNormalizationType)
        if perImg_normalization == "none":
            self._normalize = self._normalizeNone
        elif perImg_normalization == "noiseStats":
            self._normalize = self._normalizeNoiseStats
        elif perImg_normalization == "subtractMean":
            self._normalize = self.confidences_normalizeSubtractMean
        else:
            ValueError(f"Error, perImg_normalization {perImg_normalization} wrong option")

        assert ctf_correction in (item.value for item in CtfCorrectionType)

        if ctf_correction == "none":
            self._correctCtf = self._correctCtfNone
        elif ctf_correction.endswith("phase_flip"):
            self._correctCtf = self._correctCtfPhase
        elif ctf_correction.endswith("ctf_multiply"):
            raise NotImplementedError("Error, ctf_multiply was not implemented")
        else:
            ValueError(f"Error, perImg_normalization {ctf_correction} wrong option")

        self.ctf_correction_do_concat = ctf_correction.startswith("concat")
        self.ctf_correction = ctf_correction.removeprefix("concat_")

        if self.store_data_in_memory:
            self.memory = get_cache(cache_name=None, verbose=0)
            self._getIdx = self.memory.cache(self._getIdx, ignore=["self"], verbose=0)

        self._particles = None
        self._augmenter = None
        self._image_size = None

    @property
    def nnet_image_size_px(self) -> int:
        """The image size in pixels"""
        if self.image_size_px_for_nnet is None:
            return self.particles.particle_shape[-1]
        else:
            return self.image_size_px_for_nnet

    @abstractmethod
    def load_ParticlesStarSet(self):
        raise NotImplementedError()

    def _load_ParticlesStarSet(self):
        part_set = self.load_ParticlesStarSet()
        self._particles = part_set
        assert len(part_set) > 0, "Error, no particles were found in the star file"

        if self.subset_idxs is not None:
            self._particles = self._particles.createSubset(idxs=self.subset_idxs)

        if self.halfset is not None:
            if "rlnRandomSubset" not in self._particles.particles_md:
                half1, half2 = train_test_split(self._particles.particles_md.index, test_size=0.5,
                                              random_state=11, #Using the same seed to ensure that we always split the same way
                                              shuffle=True)
                self._particles.particles_md.loc[:, "rlnRandomSubset"] = 1
                self._particles.particles_md.loc[half2, "rlnRandomSubset"] = 2

            subsetNums = self._particles.particles_md["rlnRandomSubset"].values
            _subsetNums = set(subsetNums)
            assert min(_subsetNums) >= 1 and max(_subsetNums) <= 2
            idxs = np.where(subsetNums == self.halfset)[0]
            self._particles = self._particles.createSubset(idxs=idxs)



        if self.min_maxProb is not None:
            maxprob = self._particles.particles_md[RELION_ORI_POSE_CONFIDENCE_NAME]
            idxs = np.where(maxprob >= self.min_maxProb)[0]
            self._particles = self.particles.createSubset(idxs=idxs)

        return self._particles

    @property
    def particles(self) -> ParticlesStarSet:
        """
        a starstack.particlesStar.ParticlesStarSet representing the loaded particles
        """
        if self._particles is None:
            self._particles = self._load_ParticlesStarSet()
        return self._particles

    @property
    def sampling_rate(self) -> float:
        """The particle image sampling rate in A/pixels"""
        if self.image_size_px_for_nnet is None:
            return self.particles.sampling_rate
        else:
            return self.sampling_rate_angs_for_nnet

    def original_sampling_rate(self) -> float:
        return self.particles.sampling_rate

    def original_image_size(self) -> int:
        images_sizes = set([int(x) for x in self.particles.optics_md["rlnImageSize"].values])
        assert len(images_sizes) == 1, "Error, several rlnImageSize contained in the starfile. Only one rlnImageSize per starfile is supported"
        return images_sizes.pop()

    @property
    def augmenter(self) -> AugmenterBase:
        """The data augmentator object to be applied"""
        return self._augmenter

    @augmenter.setter
    def augmenter(self, augmenterObj: AugmenterBase):
        """

        Args:
            augmenter: he data augmentator object to be applied

        """
        self._augmenter = augmenterObj

    @staticmethod
    @functools.lru_cache(2)
    def _getParticleMask(image_size_px, sampling_rate, mask_radius_angs,
                                      device: Optional[Union[torch.device, str]] = None) -> Tuple[torch.Tensor,
                                                                                                  torch.Tensor]:

        radius = image_size_px / 2
        if mask_radius_angs is None:
            normalizationRadiusPixels = image_size_px / 2
        else:
            normalizationRadiusPixels = mask_radius_angs / sampling_rate

        ies, jes = torch.meshgrid(
            torch.linspace(-1 * radius, 1 * radius, image_size_px, dtype=torch.float32),
            torch.linspace(-1 * radius, 1 * radius, image_size_px, dtype=torch.float32),
            indexing="ij"
        )
        r = (ies ** 2 + jes ** 2) ** 0.5
        normalizationMask = (r > normalizationRadiusPixels)
        normalizationMask = normalizationMask.to(device)
        particleMask = ~ normalizationMask
        return normalizationMask, particleMask

    def _normalizeNoiseStats(self, img):
        """

        Args:
            img: 1XSxS tensor

        Returns:

        """
        backgroundMask = self._getParticleMask(self.nnet_image_size_px, sampling_rate=self.sampling_rate, mask_radius_angs=self.mask_radius_angs)[0]
        noiseRegion = img[:, backgroundMask]
        meanImg = noiseRegion.mean()
        stdImg = noiseRegion.std()
        return (img - meanImg) / stdImg

    def _normalizeSubtractMean(self, img):
        return (img - img.mean())

    def _normalizeNone(self, img):
        return img

    def _correctCtfPhase(self, img, md_row, optics_data):

        ctf, wimg = correct_ctf(img, float(optics_data["rlnImagePixelSize"].item()),
                                dfu=md_row["rlnDefocusU"], dfv=md_row["rlnDefocusV"],
                                dfang=md_row["rlnDefocusAngle"],
                                volt=float(optics_data["rlnVoltage"].item()),
                                cs=float(optics_data["rlnSphericalAberration"].item()),
                                w=float(optics_data["rlnAmplitudeContrast"].item()),
                                mode=self.ctf_correction, fftshift=True)
        wimg = torch.clamp(wimg, img.min(), img.max())
        wimg = torch.nan_to_num(wimg, nan=img.mean())
        if self.ctf_correction_do_concat:
            img = torch.concat([img, wimg], dim=0)
        else:
            img = wimg
        ctf = ctf.real
        return img, ctf

    def _correctCtfNone(self, img, md_row, optics_group_num):
        return img, None

    def _getIdx(self, item: int) -> Tuple[str, torch.Tensor, Tuple[torch.Tensor,torch.Tensor,torch.Tensor],
                                         Dict[str, Any], Tuple[torch.Tensor, Optional[torch.Tensor]]]:

        try:
            img_ori, md_row = self.particles[item]
        except ValueError:
            print(f"Error retrieving item {item}")
            raise

        optics_group_num = int(md_row['rlnOpticsGroup'])
        optics_data = self.particles.optics_md.query(f'rlnOpticsGroup == {optics_group_num}')

        iid = md_row[RELION_IMAGE_FNAME]
        img_ori = torch.FloatTensor(img_ori)
        img, ctf_ori = self._correctCtf(img_ori.unsqueeze(0), md_row, optics_data)

        if img.isnan().any():
            raise RuntimeError(f"Error, img with idx {item} has NAN")

        img = self.resizeImage(img, optics_data)
        img = self._normalize(img) #I changed the order of the normalization call, in cesped it was before ctf correction

        degEuler = torch.FloatTensor([md_row.get(name, 0) for name in RELION_ANGLES_NAMES])
        xyShiftAngs = torch.FloatTensor([md_row.get(name, 0) for name in RELION_SHIFTS_NAMES])
        confidence = torch.FloatTensor([md_row.get(RELION_ORI_POSE_CONFIDENCE_NAME, 1)])

        return iid, img, (degEuler, xyShiftAngs, confidence), md_row.to_dict(), (img_ori, ctf_ori)

    @cached_property
    def symmetry_group(self):
        return R.create_group(self.symmetry.upper())

    def resizeImage(self, img, optics_data):

        ori_pixelSize = float(optics_data["rlnImagePixelSize"].item())
        img, pad_info, crop_info = resize_and_padCrop_tensorBatch(img.unsqueeze(0),
                                                                  ori_pixelSize,
                                                                  self.sampling_rate_angs_for_nnet,
                                                                  self.nnet_image_size_px,
                                                                  padding_mode="constant")
        img = img.squeeze(0)
        return img

    def __getitem(self, item):
        iid, prepro_img, (degEuler, xyShiftAngs, confidence), md_dict, (img_ori, ctf_ori)= self._getIdx(item)
        if self.augmenter is not None:
            prepro_img, degEuler, shift, _ = self.augmenter(prepro_img,  # 1xSxS image expected
                                                            degEuler,
                                                            shiftFraction=xyShiftAngs / (self.nnet_image_size_px * self.sampling_rate))
            xyShiftAngs = shift * (self.nnet_image_size_px * self.sampling_rate)

        r = R.from_euler(RELION_EULER_CONVENTION, degEuler, degrees=True)

        if self.symmetry != "C1" and self.reduce_symmetry_in_label:
            r = r.reduce(self.symmetry_group)
        rotMat = r.as_matrix()
        rotMat = torch.FloatTensor(rotMat)

        if self.apply_mask_to_img:
            mask = self._getParticleMask(self.nnet_image_size_px, sampling_rate=self.sampling_rate,
                                         mask_radius_angs=self.mask_radius_angs)[1]
            prepro_img *= mask

        batch = {BATCH_IDS_NAME: iid,
                 BATCH_PARTICLES_NAME: prepro_img,
                 BATCH_POSE_NAME: (rotMat, xyShiftAngs, confidence),
                 BATCH_MD_NAME: md_dict}

        if self.return_ori_imagen:
            batch[BATCH_ORI_IMAGE_NAME] = img_ori
            batch[BATCH_ORI_CTF_NAME] = ctf_ori

        return batch

    def __getitem__(self, item):
        return self.__getitem(item)

    def __len__(self):
        return len(self.particles)

    def updateMd(self, ids: List[str], angles: Optional[Union[torch.Tensor, np.ndarray]] = None,
                 shifts: Optional[Union[torch.Tensor, np.ndarray]] = None,
                 confidence: Optional[Union[torch.Tensor, np.ndarray]] = None,
                 angles_format: Literal["rotmat", "ZYZEulerDegs"] = "rotmat",
                 shifts_format: Literal["Angst"] = "Angst"):
        """
        Updates the metadata of the particles with selected ids

        Args:
            ids (List[str]): The ids of the entries to be updated e.g. ["1@particles_0.mrcs", "2@particles_0.mrcs]
            angles (Optional[Union[torch.Tensor, np.ndarray]]): The particle pose angles to update
            shifts (Optional[Union[torch.Tensor, np.ndarray]]): The particle shifts
            confidence (Optional[Union[torch.Tensor, np.ndarray]]): The prediction confidence
            angles_format (Literal[rotmat, zyzEulerDegs]): The format for the argument angles
            shifts_format (Literal[rotmat, zyzEulerDegs]): The format for the argument shifts

        """

        assert angles_format in ["rotmat", "ZYZEulerDegs"], \
            'Error, angle_format should be in ["rotmat", "ZYZEulerDegs"]'

        assert shifts_format in ["Angst"], \
            'Error, shifts_format should be in ["Angst"]'

        col2val = {}

        if angles is not None:
            angles = data_to_numpy(angles)
            if angles_format == "rotmat":
                r = R.from_matrix(angles)
                rots, tilts, psis = r.as_euler(RELION_EULER_CONVENTION, degrees=True).T
            else:
                rots, tilts, psis = [angles[:, i] for i in range(3)]

            col2val.update({  # RELION_ANGLES_NAMES
                RELION_ANGLES_NAMES[0]: rots,
                RELION_ANGLES_NAMES[1]: tilts,
                RELION_ANGLES_NAMES[2]: psis
            })

        if shifts is not None:
            shifts = data_to_numpy(shifts)
            col2val.update({
                RELION_SHIFTS_NAMES[0]: shifts[:, 0],
                RELION_SHIFTS_NAMES[1]: shifts[:, 1],
            })

        if confidence is not None:
            confidence = data_to_numpy(confidence)
            col2val.update({
                RELION_PRED_POSE_CONFIDENCE_NAME: confidence,
            })
        assert col2val, "Error, no editing values were provided"
        self.particles.updateMd(ids=ids, colname2change=col2val)



def resize_and_padCrop_tensorBatch(array, current_sampling_rate, new_sampling_rate, new_n_pixels=None,
                                   padding_mode='reflect'):
    ndims = array.ndim - 2
    if isinstance(array, np.ndarray):
        wasNumpy = True
        array = torch.from_numpy(array)

    else:
        wasNumpy = False

    if isinstance(current_sampling_rate, tuple):
        current_sampling_rate = torch.tensor(current_sampling_rate)
    if isinstance(new_sampling_rate, tuple):
        new_sampling_rate = torch.tensor(new_sampling_rate)

    scaleFactor = current_sampling_rate / new_sampling_rate
    if isinstance(scaleFactor, (int, float)):
        scaleFactor = (scaleFactor,) * ndims
    else:
        scaleFactor = tuple(scaleFactor)
    # Resize the array
    if ndims == 2:
        mode = 'bilinear'
    elif ndims == 3:
        mode = 'trilinear'
    else:
        raise ValueError(f"Option not valid. ndims={ndims}")
    resampled_array = torch.nn.functional.interpolate(array, scale_factor=scaleFactor, mode=mode, antialias=False)
    pad_width = []
    crop_positions = []
    if new_n_pixels is not None:
        if isinstance(new_n_pixels, int):
            new_n_pixels = [new_n_pixels] * ndims
        for i in range(ndims):
            new_n_pix = new_n_pixels[i]
            old_n_pix = resampled_array.shape[i + 2]
            if new_n_pix < old_n_pix:
                # Crop the tensor
                crop_start = (old_n_pix - new_n_pix) // 2
                resampled_array = resampled_array.narrow(i + 2, crop_start, new_n_pix)
                crop_positions.append((crop_start, crop_start + new_n_pix))
            elif new_n_pix > old_n_pix:
                # Pad the tensor
                pad_before = (new_n_pix - old_n_pix) // 2
                pad_after = new_n_pix - old_n_pix - pad_before
                pad_width.extend((pad_before, pad_after))

        if len(pad_width) > 0:
            resampled_array = torch.nn.functional.pad(resampled_array, pad_width, mode=padding_mode)

    if wasNumpy:
        resampled_array = resampled_array.numpy()
    return resampled_array, pad_width, crop_positions
