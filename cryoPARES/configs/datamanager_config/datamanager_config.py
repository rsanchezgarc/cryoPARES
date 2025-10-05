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
        'num_dataworkers': 'Number of parallel data loading workers per GPU. Each worker is a separate CPU process. Set to 0 to load data in the main thread (useful only for debugging). Try not to oversubscribe by asking more workers than CPUs',
        'num_augmented_copies_per_batch': 'Number of augmented copies per particle in each batch. Each copy undergoes different data augmentation. Batch size must be divisible by this value',
        'train_validaton_split_seed': 'Random seed for splitting data into training and validation sets',
        'train_validation_split': 'Fraction of data to use for training and validation (tuple of two floats that sum to 1.0)',
        'augment_train': 'Apply data augmentation to training set',
        'only_first_dataset_for_validation': 'If multiple datasets provided, use only the first one for validation (useful when combining datasets)',
    }

    num_augmented_copies_per_batch: int = 4
    train_validaton_split_seed: int = 113
    train_validation_split: Tuple[float, float] = (0.7, 0.3)
    num_dataworkers: int = 8
    augment_train: bool = True
    only_first_dataset_for_validation: bool = True

    augmenter: Augmenter_config = field(default_factory=Augmenter_config)
    particlesdataset: ParticlesDataset_config = field(default_factory=ParticlesDataset_config)
