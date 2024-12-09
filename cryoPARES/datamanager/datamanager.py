import collections
from dataclasses import asdict

import torch
import os
import os.path as osp
from os import PathLike

import torch
from torch.utils.data import DataLoader, BatchSampler, Sampler, RandomSampler, ConcatDataset
from typing import Union, Literal, Optional, Tuple, Iterable, List

import pytorch_lightning as pl

from cryoPARES.configManager.config_searcher import inject_config
from cryoPARES.configs.mainConfig import main_config
from cryoPARES.constants import BATCH_PARTICLES_NAME
from cryoPARES.datamanager.augmentations import Augmenter
from cryoPARES.datamanager.relionStarDataset import ParticlesRelionStarDataset

FnameType = Union[PathLike, str]


def get_number_image_channels():

    if main_config.datamanager.particlesdataset.ctf_correction.startswith("concat"):
        return 2
    else:
        return 1

def get_example_random_batch(batch_size):
    imgsize = main_config.datamanager.particlesdataset.desired_image_size_px
    batch = {BATCH_PARTICLES_NAME:torch.randn(batch_size, get_number_image_channels(), imgsize, imgsize)}
    return batch

@inject_config()
class DataManager(pl.LightningDataModule):
    """
    DataManager: A LightningDataModule that wraps a ParticlesDataset
    """

    def __init__(self, star_fnames: List[FnameType],
                 symmetry:str,
                 particles_dir: Optional[List[FnameType]],
                 halfset: Optional[Literal[1, 2]], #TODO: particlesDataset does not handle halfsets
                 batch_size: int,
                 save_train_val_partition_dir: Optional[FnameType],
                 is_global_zero: bool,
                 # The following arguments have a default config value
                 num_augmented_copies_per_batch: int,
                 train_validaton_split_seed: int,
                 train_validation_split: Tuple[float, float],
                 num_data_workers: int,
                 augment_train: bool,
                 onlfy_first_dataset_for_validation: bool,
                 ):

        super().__init__()

        self.star_fnames = self._expand_fname(star_fnames)
        self.symmetry = symmetry.upper()
        self.particles_dir = self._expand_fname(particles_dir)
        if self.particles_dir is None:
            self.particles_dir = [None] * len(self.star_fnames)
        self.halfset = halfset #TODO: halfset is not used
        if self.halfset:
            raise NotImplementedError()
        self.num_augmented_copies_per_batch = num_augmented_copies_per_batch
        self.train_validaton_split_seed = train_validaton_split_seed
        self.train_validation_split = train_validation_split
        self.batch_size = batch_size
        self.num_data_workers = num_data_workers
        self.is_global_zero = is_global_zero
        self.save_train_val_partition_dir = save_train_val_partition_dir
        self.augment_train = augment_train
        self.onlfy_first_dataset_for_validation = onlfy_first_dataset_for_validation

        if self.augment_train:
            self.augmenter = Augmenter()
        else:
            self.augmenter = None

    @staticmethod
    def _expand_fname(fnameOrList):
            if fnameOrList is None:
                return None
            elif isinstance(fnameOrList, str):
                return [osp.expanduser(fnameOrList)]
            elif isinstance(fnameOrList, collections.abc.Iterable):
                return [osp.expanduser(fname) if fname else None for fname in fnameOrList]
            else:
                raise ValueError(f"Not valid fname {fnameOrList}")

    def prepare_data(self) -> None:
        return

    def create_dataset(self, partitionName):
        datasets = []
        for i, (partFname, partDir) in enumerate(zip(self.star_fnames, self.particles_dir)):
            mrcsDataset = ParticlesRelionStarDataset(star_fname=partFname,  particles_dir=partDir,
                                                     symmetry=self.symmetry)

            if self.is_global_zero and self.save_train_val_partition_dir is not None:
                dirname = osp.join(self.save_train_val_partition_dir, partitionName if partitionName is not None else "full")
                os.makedirs(dirname, exist_ok=True)
                fname = osp.join(dirname, f"{i}-particles.star")
                if not osp.isfile(fname):
                    mrcsDataset.saveMd(fname, overwrite=False)
            datasets.append(mrcsDataset)
            if self.onlfy_first_dataset_for_validation and partitionName != "train":
                break
        dataset = ConcatDataset(datasets)
        return dataset



    def _create_dataloader(self, partitionName: Optional[str]):

        dataset = self.create_dataset(partitionName)
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

def _test():

    main_config.datamanager.num_data_workers = 0
    dm = DataManager(star_fnames=["~/cryo/data/preAlignedParticles/EMPIAR-10166/data/1000particles.star"],
                     symmetry="c1",
                     augment_train=False,
                     particles_dir=None,
                     halfset=None,
                     batch_size=2,
                     save_train_val_partition_dir=None
                     )
    dl = dm.train_dataloader()
    # dl = dm.val_dataloader()
    for batch in dm.val_dataloader():
        print(batch.keys())

if __name__ == "__main__":
    _test()