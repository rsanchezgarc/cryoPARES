from dataclasses import dataclass
from typing import Optional


@dataclass
class Train_config:
    n_epochs: int = 2
    learning_rate: float = 1e-3
    batch_size: int = 2
    accumulate_grad_batches: int = 1
    weight_decay: float = 1e-5

    use_cuda: bool = True
    n_cpus_if_no_cuda: int = 2
    random_seed: int = 12313
    monitor_metric : str = "val_geo_degs" #"val_loss"
    patient_reduce_lr_plateau_n_epochs: int = 5

    gradient_clip_value: Optional[float] = 5.


    swalr_begin_epoch: float = 0.8
    swalr_annelaing_n_epochs: int = 15
    min_learning_rate_factor: float = 1e-3 #The smallest learning rate we want is 1000 smaller than the initial learning rate


    train_precision: str = "32"  # "32" "16" "bf16"
    sync_batchnorm: bool = True  # Consider setting it to False if the batch size is large to speed it up

    default_optimizer: str = "RAdam"
    warmup_n_epochs: Optional[int] = None