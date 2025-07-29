import collections
import os
import os.path as osp

import torch
from torch.utils.data import DataLoader, BatchSampler, Sampler, RandomSampler, ConcatDataset, DistributedSampler, \
    SequentialSampler
from typing import Union, Literal, Optional, Tuple, Iterable, List

import pytorch_lightning as pl
from cryoPARES.configManager.inject_defaults import inject_defaults_from_config, CONFIG_PARAM
from cryoPARES.configs.mainConfig import main_config
from cryoPARES.constants import BATCH_PARTICLES_NAME, BATCH_IDS_NAME, BATCH_MD_NAME, BATCH_POSE_NAME
from cryoPARES.utils.paths import FNAME_TYPE


def get_number_image_channels():

    if main_config.datamanager.particlesdataset.ctf_correction.startswith("concat"):
        return 2
    else:
        return 1

def get_example_random_batch(batch_size, n_channels=None, seed=None):
    imgsize = main_config.datamanager.particlesdataset.desired_image_size_px
    n_channels = n_channels if n_channels is not None else get_number_image_channels()
    seed = torch.Generator().manual_seed(seed) if seed is not None else None
    batch = {
        BATCH_IDS_NAME: [str(i) for i in range(batch_size)],
        BATCH_PARTICLES_NAME: torch.randn(batch_size, n_channels, imgsize, imgsize, generator=seed),
        BATCH_POSE_NAME: [torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1),
                          torch.randn(batch_size, 2, generator=seed),
                          torch.rand(batch_size, generator=seed)],
        BATCH_MD_NAME: {"mdField1": torch.rand(batch_size, generator=seed), "mdField2": ["a"*batch_size]}
    }
    return batch

class DataManager(pl.LightningDataModule):
    """
    DataManager: A LightningDataModule that wraps a ParticlesDataset
    """
    @inject_defaults_from_config(main_config.datamanager, update_config_with_args=False)
    def __init__(self, star_fnames: List[FNAME_TYPE] | FNAME_TYPE,
                 symmetry:str,
                 particles_dir: Optional[List[FNAME_TYPE]] | FNAME_TYPE,
                 halfset: Optional[Literal[1, 2]],
                 batch_size: int,
                 save_train_val_partition_dir: Optional[FNAME_TYPE],
                 is_global_zero: bool,
                 # The following arguments have a default config value
                 num_augmented_copies_per_batch: int = CONFIG_PARAM(),
                 train_validaton_split_seed: int = CONFIG_PARAM(),
                 train_validation_split: Tuple[float, float] = CONFIG_PARAM(),
                 num_data_workers: int = CONFIG_PARAM(),
                 augment_train: bool = CONFIG_PARAM(),
                 only_first_dataset_for_validation: bool = CONFIG_PARAM(),
                 return_ori_imagen: bool = False,
                 subset_idxs: Optional[List[int]] = None
                 ):

        super().__init__()

        self.star_fnames = self._expand_fname(star_fnames)
        self.symmetry = symmetry.upper()
        self.particles_dir = self._expand_fname(particles_dir)
        if self.particles_dir is None:
            self.particles_dir = [None] * len(self.star_fnames)
        self.halfset = halfset
        self.num_augmented_copies_per_batch = num_augmented_copies_per_batch
        self.train_validaton_split_seed = train_validaton_split_seed
        self.train_validation_split = train_validation_split
        self.batch_size = batch_size
        self.num_data_workers = num_data_workers
        self.is_global_zero = is_global_zero
        self.save_train_val_partition_dir = save_train_val_partition_dir
        self.augment_train = augment_train
        self.only_first_dataset_for_validation = only_first_dataset_for_validation
        self.return_ori_imagen = return_ori_imagen
        if self.augment_train:
            from cryoPARES.datamanager.augmentations import Augmenter
            self.augmenter = Augmenter()
        else:
            self.augmenter = None
        self._subset_idxs = subset_idxs

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

        from cryoPARES.datamanager.relionStarDataset import ParticlesRelionStarDataset
        if partitionName in ["train", "val"]:
            store_data_in_memory = main_config.datamanager.particlesdataset.store_data_in_memory
        else:
            store_data_in_memory = None
        datasets = []
        for i, (partFname, partDir) in enumerate(zip(self.star_fnames, self.particles_dir)):
            mrcsDataset = ParticlesRelionStarDataset(particles_star_fname=partFname, particles_dir=partDir,
                                                     symmetry=self.symmetry, halfset=self.halfset,
                                                     store_data_in_memory=store_data_in_memory,
                                                     return_ori_imagen=self.return_ori_imagen,
                                                     subset_idxs=self._subset_idxs)

            if self.is_global_zero and self.save_train_val_partition_dir is not None:
                dirname = osp.join(self.save_train_val_partition_dir, partitionName
                                                        if partitionName is not None else "full")
                os.makedirs(dirname, exist_ok=True)
                fname = osp.join(dirname, f"{i}-particles.star")
                if not osp.isfile(fname):
                    mrcsDataset.saveMd(fname, overwrite=False)
            mrcsDataset.augmenter = self.augmenter if partitionName == "train" else None
            datasets.append(mrcsDataset)
            if self.only_first_dataset_for_validation and partitionName == "val":
                break
        dataset = ConcatDataset(datasets)
        if partitionName in ["train", "val"]:
            assert self.train_validation_split is not None
            generator = torch.Generator().manual_seed(self.train_validaton_split_seed)
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset,
                self.train_validation_split,
                generator=generator
            )
            if partitionName == "train":
                dataset = train_dataset
            else:
                dataset = val_dataset
        return dataset


    def _create_dataloader(self, partitionName: Optional[str]=None):

        dataset = self.create_dataset(partitionName)

        if self.trainer is not None:
            distributed_world_size = getattr(self.trainer, 'world_size', 1)
            rank = getattr(self.trainer, 'local_rank', 0)
        else:
            distributed_world_size = 1
            rank = 0
        assert  (rank == 0) == self.is_global_zero
        if partitionName not in ["train", "val"]:
            # Logic for test/predict
            sampler = DistributedSampler(dataset, num_replicas=distributed_world_size, rank=rank) \
                                        if distributed_world_size > 1 else None
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=False,
                              num_workers=self.num_data_workers, sampler=sampler, pin_memory=True)

        if partitionName == "train":
            print(f"Train dataset {len(dataset)}")

            if distributed_world_size > 1:
                sampler = DistributedSampler(dataset, num_replicas=distributed_world_size, rank=rank,
                                             shuffle=not self.trainer.overfit_batches > 0)
            else:  # Not distributed
                if self.trainer is not None and self.trainer.overfit_batches > 0:
                    sampler = SequentialSampler(dataset)
                else:
                    sampler = RandomSampler(dataset)

            batch_sampler = MultiInstanceSampler(
                sampler=sampler,
                batch_size=self.batch_size,
                drop_last=True,
                num_copies_to_sample=self.num_augmented_copies_per_batch
            )
            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=self.num_data_workers,
                persistent_workers=True if self.num_data_workers > 0 else False,
                pin_memory=True if self.num_data_workers > 0 else False,
            )
        elif partitionName == "val":
            print(f"Validation dataset {len(dataset)}")

            sampler = DistributedSampler(dataset, num_replicas=distributed_world_size, rank=rank, shuffle=False)\
                                        if distributed_world_size > 1 else None

            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_data_workers,
                persistent_workers=True if self.num_data_workers > 0 else False,
                sampler=sampler,
                pin_memory=True if self.num_data_workers > 0 else False,
            )
        else:
            raise ValueError("Error, wrong partition")

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
        assert batch_size % num_copies_to_sample == 0, "Error, batch_size % num_copies_to_sample == 0 required"
        super().__init__(sampler, batch_size//num_copies_to_sample, drop_last)
        self.num_copies_to_sample = num_copies_to_sample

    def __iter__(self):
        for idx in super(MultiInstanceSampler, self).__iter__():
            yield idx * self.num_copies_to_sample

def _test():

    main_config.datamanager.num_data_workers = 0
    dm = DataManager(star_fnames=["~/cryo/data/preAlignedParticles/EMPIAR-10166/data/1000particles.star"],
                     symmetry="c1",
                     augment_train=True,
                     particles_dir=None,
                     is_global_zero=True,
                     halfset=1,
                     batch_size=8,
                     save_train_val_partition_dir=None
                     )
    dl = dm.train_dataloader()
    # dl = dm.val_dataloader()
    for batch in dl:
        # print(batch.keys())
        print(batch["idd"])
if __name__ == "__main__":
    _test()