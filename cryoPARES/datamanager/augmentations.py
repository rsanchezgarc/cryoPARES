import abc
import os.path as osp
import functools
import random
from dataclasses import asdict
from typing import Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from kornia.augmentation import RandomElasticTransform
from scipy.spatial.transform import Rotation
from torchvision.transforms.v2 import RandomErasing
import torchvision.transforms.functional as transformF

from cryoPARES.configManager.config_searcher import inject_config
from cryoPARES.configs.mainConfig import main_config
from cryoPARES.constants import BATCH_PARTICLES_NAME


# TODO: Implement augmentations in a better way, defining custom torchvision operations so that they can be used in batch mode seamingly.

class AugmenterBase(abc.ABC):
    @abc.abstractmethod
    def applyAugmentation(self, imgs, degEulerList, shiftFractionList):
        raise NotImplementedError()


@inject_config()
class Augmenter(AugmenterBase):
    def __init__(self,
                 min_n_augm_per_img: Optional[int],
                 max_n_augm_per_img: Optional[int]):
        augmentConfig = main_config.datamanager.augmenter
        self.imageSize = main_config.datamanager.particlesdataset.desired_image_size_px
        self._augmentationTypes = asdict(augmentConfig.operations)

        self.augmentationTypes = self._augmentationTypes.copy()  # We have self._augmentationTypes in case we want to reset probs

        self.min_n_augm_per_img = augmentConfig.min_n_augm_per_img if (
                    min_n_augm_per_img is None) else min_n_augm_per_img
        self.max_n_augm_per_img = augmentConfig.max_n_augm_per_img if (
                    max_n_augm_per_img is None) else max_n_augm_per_img

        self.probSchedulers = {name: Scheduler(vals.get("probScheduler")).generate() for name, vals in
                               self.augmentationTypes.items()}

        self.augmentation_count = 0

    @staticmethod
    @functools.lru_cache(1)
    def _getRandomEraser(**kwargs):
        return RandomErasing(p=1., **kwargs)

    @staticmethod
    @functools.lru_cache(1)
    def _getElasticTransformer(imsSize, **kwargs):
        kernel_size_fraction = kwargs["kernel_size_fraction"]
        sigma_fraction = kwargs["sigma_fraction"]
        kernel_size = 1 + 2 * int(imsSize * kernel_size_fraction // 2)
        sigma = imsSize * sigma_fraction
        return RandomElasticTransform(p=1., kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma),
                                      alpha=(kwargs["alpha"], kwargs["alpha"]), padding_mode='zeros')


    def _randomErase(self, img, **kwargs):
        eraser = self._getRandomEraser(**kwargs)
        return eraser(img)

    def _randomElastic(self, img, **kwargs):
        eraser = self._getElasticTransformer(self.imageSize, **kwargs)
        return eraser(img).squeeze(0)


    def _get_nrounds(self):
        return random.randint(self.min_n_augm_per_img, self.max_n_augm_per_img)

    def _get_rand(self):
        return random.random()

    def applyAugmentation(self, imgs, degEulerList, shiftFractionList):
        if len(imgs.shape) > 3: #TODO: Better batch mode
            transformed_batch = []
            degEulerList_ = []
            shiftFractionList_ = []
            applied_transforms_ = []
            for img, euler, shift in zip(imgs, degEulerList, shiftFractionList):
                (transformed_img, euler, shift,
                 applied_transforms) = self._applyAugmentation(img, euler, shift)
                transformed_batch.append(transformed_img)
                degEulerList_.append(euler)
                shiftFractionList_.append(shift)
                applied_transforms_ += [applied_transforms]
            return torch.stack(transformed_batch, dim=0), torch.stack(degEulerList_, dim=0), \
                torch.stack(shiftFractionList_, dim=0), applied_transforms_
        else:
            return self._applyAugmentation(imgs, degEulerList, shiftFractionList)

    def _applyAugmentation(self, img, degEuler, shiftFraction):
        """

        Args:
            img: A tensor of shape 1XLxL
            degEuler:
            shiftFraction:

        Returns:

        """
        img = img.clone()
        applied_transforms = []
        n_rounds = self._get_nrounds()
        for round in range(n_rounds):
            for aug, aug_kwargs in self.augmentationTypes.items():
                p = self.probSchedulers[aug](aug_kwargs["p"], self.augmentation_count)
                aug_kwargs= aug_kwargs.copy()
                del aug_kwargs["p"]
                if self._get_rand() < p:
                    if aug == "randomGaussNoise":
                        scale = random.random() * aug_kwargs["scale"]
                        applied_transforms.append((aug, dict(scale=scale)))
                        img += torch.randn_like(img) * scale
                    elif aug == "randomUnifNoise":
                        scale = random.random() * aug_kwargs["scale"]
                        img += (torch.rand_like(img) - 0.5) * scale
                        applied_transforms.append((aug, dict(scale=scale)))

                    elif aug == "inPlaneRotations90": #rot, tilt, psi
                        rotOrder = random.randint(0, 3)
                        img = torch.rot90(img, rotOrder, [-2, -1])
                        degEuler[-1] = (degEuler[-1] + 90. * rotOrder) % 360
                        applied_transforms.append((aug, dict(rotOrder=rotOrder)))

                    elif aug == "inPlaneRotations":
                        randDeg = (torch.rand(1) - 0.5) * aug_kwargs["maxDegrees"]
                        img, theta = rotTransImage(img.unsqueeze(0), randDeg, translationFract=torch.zeros(1),
                                                   scaling=1)
                        img = img.squeeze(0)
                        degEuler[-1] = (degEuler[-1] + randDeg.item()) % 360
                        applied_transforms.append((aug, dict(randDeg=randDeg)))

                    elif aug == "inPlaneShifts":  # It is important to do rotations before shifts

                        randFractionShifts = (torch.rand(2) - 0.5) * aug_kwargs["maxShiftFraction"]
                        img = rotTransImage(img.unsqueeze(0), torch.zeros(1), translationFract=randFractionShifts,
                                            scaling=1)[0].squeeze(0)
                        shiftFraction += randFractionShifts
                        applied_transforms.append((aug, dict(randFractionShifts=randFractionShifts)))

                    elif aug == "sizePerturbation":
                        scale = 1 + (random.random() - 0.5) * aug_kwargs["maxSizeFraction"]
                        img = rotTransImage(img.unsqueeze(0), torch.zeros(1), translationFract=torch.zeros(2),
                                            scaling=torch.FloatTensor([scale]))[0].squeeze(0)
                        applied_transforms.append((aug, dict(scale=scale)))

                    elif aug == "gaussianBlur":
                        scale = 1e-3 + (1 + (random.random() - 0.5) * aug_kwargs["scale"])
                        img = transformF.gaussian_blur(img, kernel_size=3 + 2 * int(scale), sigma=scale)
                        applied_transforms.append((aug, dict(scale=scale)))

                    elif aug == "erasing":
                        kwargs = {k: tuple(v) for k, v in aug_kwargs.items()}
                        img = self._randomErase(img, **kwargs)
                        applied_transforms.append((aug, dict(kwargs=kwargs)))
                    elif aug == "randomElastic":
                        img = self._randomElastic(img, **aug_kwargs)
                    else:
                        raise ValueError(f"Error, unknown augmentation {aug}")
        self.augmentation_count += 1
        return img, degEuler, shiftFraction, applied_transforms

    def __call__(self, img, eulersDeg, shiftFraction):
        return self.applyAugmentation(img, eulersDeg, shiftFraction)



def rotTransImage(image, degrees, translationFract, scaling=1., padding_mode='reflection',
                  interpolation_mode="bilinear", rotation_first=True) -> Tuple[torch.Tensor, torch.Tensor]: #TODO: Move to
    """

    :param image: BxCxNxN
    :param degrees:
    :param translationFract: The translation to be applied as a fraction of the total size in pixels
    :param scaling:
    :param padding_mode:
    :param interpolation_mode:
    :param rotation_first: if using to compute Relion alignment parameters, set it to True
    :return:
    """
    align_corners = True  # If set to True, the extrema (-1 and 1) are considered as referring to the center points of the input’s corner pixels. If set to False, they are instead considered as referring to the corner points of the input’s corner pixels, making the sampling more resolution agnostic.
    assert ((-1 < translationFract) & (translationFract < 1)).all(), \
        (f"Error, translation should be provided as a fraction of the image."
         f" {translationFract.min()} {translationFract.max()} ")
    radians = torch.deg2rad(degrees)
    cosAngle = torch.cos(radians)
    sinAngle = torch.sin(radians)

    # theta = torch.stack([cosAngle, -sinAngle, translation[..., 0:1], sinAngle, cosAngle, translation[..., 1:2]], -1).view(-1, 2, 3)

    noTransformation = torch.eye(3).unsqueeze(0).repeat(sinAngle.shape[0], 1, 1).to(sinAngle.device)
    rotMat = noTransformation.clone()
    rotMat[:, :2, :2] = torch.stack([cosAngle, -sinAngle, sinAngle, cosAngle], -1).view(-1, 2, 2)

    transMat = noTransformation.clone()
    transMat[:, :2, -1] = translationFract

    if rotation_first:
        theta = torch.bmm(rotMat, transMat)[:, :2, :]
    else:
        theta = torch.bmm(transMat, rotMat)[:, :2, :]

    # raise NotImplementedError("TODO: check if this is how to do it, rotTrans rather than transRot")
    if scaling != 1:
        theta[:, 0, 0] *= scaling
        theta[:, 1, 1] *= scaling

    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    # Generate the grid for the transformation
    grid = F.affine_grid(
        theta,
        size=image.shape,
        align_corners=align_corners,
    )

    # Perform the affine transformation with automatic padding
    image = F.grid_sample(
        image,
        grid,
        padding_mode=padding_mode,
        align_corners=align_corners,
        mode=interpolation_mode

    )
    return image, theta

def _generate_scheduler(schedulerInfo):
    if schedulerInfo is None:
        def _identity(x, current_step):
            return x
        return _identity
    else:
        schedulerName = schedulerInfo["type"]
        schedulerKwargs = schedulerInfo["kwargs"]
        if schedulerName == "linear_up":
            maxProb = schedulerKwargs.get("max_prob")
            scheduler_steps = schedulerKwargs.get("scheduler_steps")

            def linear_up(p, current_step):
                # Linearly increase from 0 to p over scheduler_steps
                increment = (maxProb - p) / scheduler_steps
                new_p = min(p + increment * current_step, maxProb)
                return new_p

            return linear_up
        elif schedulerName == "linear_down":
            scheduler_steps = schedulerKwargs.get("scheduler_steps")
            minProb = schedulerKwargs.get("min_prob")

            def linear_down(p, current_step):
                # Linearly decrease from p to min_prob over scheduler_steps
                decrement = (p - minProb) / scheduler_steps
                new_p = max(p - decrement * current_step, minProb)
                return new_p

            return linear_down
        else:
            raise NotImplementedError(f"False {schedulerName} is not valid")

class Scheduler:
    def __init__(self, schedulerInfo):
        self.schedulerInfo = schedulerInfo

    def identity(self, x, current_step):
        return x

    def linear_up(self, p, current_step):
        maxProb = self.schedulerInfo["kwargs"].get("max_prob")
        scheduler_steps = self.schedulerInfo["kwargs"].get("scheduler_steps")
        increment = (maxProb - p) / scheduler_steps
        return min(p + increment * current_step, maxProb)

    def linear_down(self, p, current_step):
        scheduler_steps = self.schedulerInfo["kwargs"].get("scheduler_steps")
        minProb = self.schedulerInfo["kwargs"].get("min_prob")
        decrement = (p - minProb) / scheduler_steps
        return max(p - decrement * current_step, minProb)

    def generate(self):
        if self.schedulerInfo is None:
            return self.identity
        else:
            schedulerName = self.schedulerInfo["type"]
            if schedulerName == "linear_up":
                return self.linear_up
            elif schedulerName == "linear_down":
                return self.linear_down
            else:
                raise NotImplementedError(f"{schedulerName} is not valid")

if __name__ == "__main__":


    from torchvision.datasets import CIFAR100
    from cryoPARES.datamanager.relionStarDataset import ParticlesRelionStarDataset

    # dataset = CIFAR100(root="/tmp/cifcar", transform=ToTensor(), download=True)
    dataset = ParticlesRelionStarDataset(star_fname=osp.expanduser("~/cryo/data/preAlignedParticles/EMPIAR-10166/data/1000particles.star"), particles_dir=None, symmetry="c1")

    augmenter = Augmenter()
    # augmenter._get_rand = lambda: 0
    from torch.utils.data import DataLoader

    dl = DataLoader(dataset, batch_size=4, num_workers=0, shuffle=False)
    for batch in dl:
        if isinstance(batch, Dict):
            img = batch[BATCH_PARTICLES_NAME][:, 0:1, ...] # We will show only the first channel
            cmap = "gray"
        else:
            img = batch[0]
            cmap = None

        eulers = torch.from_numpy(Rotation.random(img.shape[0]).as_matrix().astype(np.float32))
        shiftFrac = torch.zeros(img.shape[0], 2)
        print(img.shape)
        img_, eulers_, shiftFrac_, applied_transforms = augmenter.applyAugmentation(img, eulers, shiftFrac)
        from matplotlib import pyplot as plt

        f, axes = plt.subplots(1, 2)
        for i in range(img.shape[0]):
            print(applied_transforms[i], f"#applied transforms= {len(applied_transforms[i])}")
            axes.flat[0].imshow(img[i, ...].permute(1, 2, 0), cmap=cmap)
            axes.flat[1].imshow(img_[i, ...].permute(1, 2, 0), cmap=cmap)
            plt.show()
            print()
            print()
        # break
