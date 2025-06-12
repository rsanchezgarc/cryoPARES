from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple

from cryoPARES.configs.datamanager_config.augmentations_config import Augmenter_config
from cryoPARES.configs.datamanager_config.particlesDataset_config import ParticlesDataset_config


@dataclass
class DataManager_config:
    num_augmented_copies_per_batch: int = 4
    train_validaton_split_seed: int = 113
    train_validation_split: Tuple[float, float] = (0.7, 0.3)
    num_data_workers: int = 4
    augment_train: bool = True
    only_first_dataset_for_validation: bool = True

    augmenter: Augmenter_config = field(default_factory=Augmenter_config)
    particlesdataset: ParticlesDataset_config = field(default_factory=ParticlesDataset_config)
