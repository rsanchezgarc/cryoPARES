import torch
import os
from os import PathLike

import torch
from torch.utils.data import DataLoader, BatchSampler, Sampler, RandomSampler
from typing import Union, Literal, Optional, Tuple, Iterable

from lightning import pytorch as pl

from cryoPARES.configs.mainConfig import main_config
from cryoPARES.constants import BATCH_PARTICLES_NAME


def get_number_image_channels():
     if main_config.datamanager.ctf_correction_mode.startswith("concat"):
         return 2
     else:
         return 1

def get_example_random_batch(batch_size=1):
    #TODO: batch size should be in the config
    imgsize = main_config.datamanager.desired_image_size_pixels
    batch = {BATCH_PARTICLES_NAME:torch.randn(batch_size, get_number_image_channels(), imgsize, imgsize)}
    return batch




class ParticlesDataModule(pl.LightningDataModule):
    """
    ParticlesDataModule: A LightningDataModule that wraps a ParticlesDataset
    """

    def __init__(self, targetName: Union[PathLike, str], halfset: Optional[Literal[0, 1]], image_size: int,
                 apply_perImg_normalization: bool = True,
                 ctf_correction: Literal["none", "phase_flip"] = "phase_flip", image_size_factor_for_crop: float = 0.25,
                 num_augmented_copies_per_batch: int = 2,
                 augmenter: Optional[Augmenter] = None, train_validaton_split_seed: int = 113,
                 train_validation_split: Tuple[float, float] = (0.7, 0.3), batch_size: int = 8,
                 num_data_workers: int = 0):

        super().__init__()
        # self.save_hyperparameters()  #Not needed since we are using CLI
        self.targetName = targetName
        self.halfset = halfset
        self.benchmarkDir = os.path.expanduser(benchmarkDir)
        self.image_size = image_size
        self.apply_perImg_normalization = apply_perImg_normalization
        self.ctf_correction = ctf_correction
        self.image_size_factor_for_crop = image_size_factor_for_crop
        self.num_augmented_copies_per_batch = num_augmented_copies_per_batch
        self.augmenter = augmenter
        self.train_validaton_split_seed = train_validaton_split_seed
        self.train_validation_split = train_validation_split
        self.batch_size = batch_size
        self.num_data_workers = num_data_workers

        self._symmetry = None

    @property
    def symmetry(self):
        """The point symmetry of the dataset"""
        if self._symmetry is None:
            dataset = self.createDataset()
            self._symmetry = dataset.symmetry
        return self._symmetry

    def createDataset(self):
        return ParticlesDataset(self.targetName, halfset=self.halfset, benchmarkDir=self.benchmarkDir,
                                image_size=self.image_size, apply_perImg_normalization=self.apply_perImg_normalization)

    def _create_dataloader(self, partitionName: Optional[str]):

        dataset = self.createDataset()
        if partitionName in ["train", "val"]:
            assert self.train_validation_split is not None, "Error, self.train_validation_split required"
            dataset.augmenter = self.augmenter if partitionName == "train" else None
            generator = torch.Generator().manual_seed(self.train_validaton_split_seed)
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, self.train_validation_split,
                                                                       generator=generator)

            if partitionName == "train":
                dataset = train_dataset
                print(f"Train dataset {len(train_dataset)}")

                batch_sampler = MultiInstanceSampler(sampler=RandomSampler(dataset), batch_size=self.batch_size,
                                                     drop_last=True,
                                                     num_copies_to_sample=self.num_augmented_copies_per_batch)
                return DataLoader(
                                dataset,
                                batch_sampler=batch_sampler,
                                num_workers=self.num_data_workers,
                                persistent_workers=True if self.num_data_workers > 0 else False)
            else:
                dataset = val_dataset
                print(f"Validation dataset {len(val_dataset)}")

        return DataLoader(
            dataset, batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_data_workers,
            persistent_workers=True if self.num_data_workers > 0 else False)

    def train_dataloader(self):
        return self._create_dataloader(partitionName="train")

    def val_dataloader(self):
        return self._create_dataloader(partitionName="val")

    def test_dataloader(self):
        return self._create_dataloader(partitionName="test")

    def predict_dataloader(self):
        return self._create_dataloader(partitionName=None)


class MultiInstanceSampler(BatchSampler):
    def __init__(self, sampler: Union[Sampler[int], Iterable[int]], batch_size: int, drop_last: bool,
                 num_copies_to_sample:int=1):
        assert  batch_size % num_copies_to_sample == 0, "Error, batch_size % num_copies_to_sample == 0 required"
        super().__init__(sampler, batch_size//num_copies_to_sample, drop_last)
        self.num_copies_to_sample = num_copies_to_sample

    def __iter__(self):
        for idx in super(MultiInstanceSampler, self).__iter__():
            yield idx * self.num_copies_to_sample
