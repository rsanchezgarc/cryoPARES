from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple

from cryoPARES.configs.datamanager_config.augmentations_config import Augmenter_config
from cryoPARES.configs.datamanager_config.particlesDataset_config import ParticlesDataset_config


@dataclass
class DataManager_config:
    """Data management configuration parameters."""

    # Centralized parameter documentation
    PARAM_DOCS = {
        'num_dataworkers': 'Number of parallel data loading workers per GPU. Each worker is a separate CPU process. Set to 0 to load data in the main thread (useful for debugging)',
        'num_augmented_copies_per_batch': 'Number of augmented copies per particle in each batch. Each copy undergoes different data augmentation. Batch size must be divisible by this value',
    }

    num_augmented_copies_per_batch: int = 4
    train_validaton_split_seed: int = 113
    train_validation_split: Tuple[float, float] = (0.7, 0.3)
    num_dataworkers: int = 8
    augment_train: bool = True
    only_first_dataset_for_validation: bool = True

    augmenter: Augmenter_config = field(default_factory=Augmenter_config)
    particlesdataset: ParticlesDataset_config = field(default_factory=ParticlesDataset_config)
