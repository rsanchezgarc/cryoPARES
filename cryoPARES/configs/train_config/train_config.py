from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class Train_config:
    """Training configuration parameters."""

    # Centralized parameter documentation (both config and CLI-exposed params)
    PARAM_DOCS = {
        # Config parameters
        'n_epochs': 'Number of training epochs. More epochs allow better convergence, although it does not help beyond a certain point',
        'learning_rate': 'Initial learning rate for optimizer. Tune this for optimal convergence (typical range: 1e-4 to 1e-2)',
        'batch_size': 'Number of particles per batch. Try to make it as large as possible before running out of GPU memory. We advice using batch sizes of at least 32 images',
        'accumulate_grad_batches': 'Number of batches to accumulate gradients over. Effective batch size = batch_size × accumulate_grad_batches. We advice to use effective batch sizes of  effective batch sizes of 512 to 2048 images',
        'weight_decay': 'L2 regularization weight decay for optimizer. Increase if overfitting occurs',

        # CLI-exposed parameters (not in config, but used in train.py)
        'symmetry': 'Point group symmetry of the molecule (e.g., C1, D7, I, O, T)',
        'particles_star_fname': 'Path(s) to RELION 3.1+ format .star file(s) containing pre-aligned particles. Can accept multiple files',
        'train_save_dir': 'Output directory where model checkpoints, logs, and training artifacts will be saved',
        'particles_dir': 'Root directory for particle image paths. If paths in .star file are relative, this directory is prepended (similar to RELION project directory concept)',
        'split_halfs': 'If True (default), trains two separate models on data half-sets for cross-validation. Use --NOT_split_halfs to train single model on all data',
        'continue_checkpoint_dir': 'Path to checkpoint directory to resume training from a previous run',
        'finetune_checkpoint_dir': 'Path to checkpoint directory to fine-tune a pre-trained model on new dataset',
        'compile_model': 'Enable torch.compile for faster training (experimental)',
        'val_check_interval': 'Fraction of epoch between validation checks. You generally don\'t want to touch it, but you can set it to smaller values (0.1-0.5) for large datasets to get quicker feedback',
        'overfit_batches': 'Number of batches to use for overfitting test (debugging feature to verify model can memorize small dataset)',
        'map_fname_for_simulated_pretraining': 'Path(s) to reference map(s) for simulated projection warmup before training on real data. The number of maps must match number of particle star files',
        'junk_particles_star_fname': 'Optional star file(s) with junk-only particles for estimating confidence z-score thresholds',
        'junk_particles_dir': 'Root directory for junk particle image paths (analogous to particles_dir)',

        # TrainerPartition-specific parameters
        'partition': 'Data partition to train on: "half1", "half2", or "allParticles". Used for half-set cross-validation',
        'continue_checkpoint_fname': 'Path to specific checkpoint file to resume training from previous run',
        'finetune_checkpoint_fname': 'Path to specific checkpoint file to fine-tune pre-trained model on new data',
        'find_lr': 'Enable automatic learning rate finder to suggest optimal learning rate (single GPU only). Not recommended',
    }

    n_epochs: int = 100
    learning_rate: float = 1e-3
    batch_size: int = 64
    accumulate_grad_batches: int = 16
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

    cuda_for_reconstruct: bool = True
    batch_size_for_reconstruct: int = 64
    float32_matmul_precision_for_reconstruct: Optional[str] = "high"

    float32_matmul_precision: str = "high"

    pl_plugin: Literal["LightningEnvironment", "none"] = "LightningEnvironment"  #Try none to submit to slurm
    num_computer_nodes: int = 1 #It has not been tried with values different from 1

    expandable_segments_GPU_mem: bool = False #Set it to true if there are fragmentation problems

    snr_for_simulation: float = 0.05
    n_epochs_simulation: int = 10