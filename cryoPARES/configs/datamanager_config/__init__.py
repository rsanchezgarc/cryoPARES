from dataclasses import dataclass
from enum import Enum
from typing import Tuple


@dataclass
class Datamanager_fields:
    num_augmented_copies_per_batch: int = 2
    train_validaton_split_seed: int = 113
    train_validation_split: Tuple[float, float] = (0.7, 0.3)
    num_data_workers: int = 0
    augment_train: bool = True