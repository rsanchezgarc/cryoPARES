from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class Inference_config:
    """Inference configuration parameters."""

    # Centralized parameter documentation
    PARAM_DOCS = {
        # Config parameters
        'batch_size': 'Number of particles per batch for inference',
        'use_cuda': 'Enable GPU acceleration for inference. If False, runs on CPU only',
        'n_cpus_if_no_cuda': 'Maximum CPU threads per worker when CUDA is disabled',
        'random_seed': 'Random seed for reproducibility during inference',
        'top_k_poses_nnet': 'Number of top pose predictions to retrieve from neural network before local refinement',
        'pl_plugin': 'PyTorch Lightning plugin: "LightningEnvironment" or "none" (try "none" for Slurm submission)',
        'num_computer_nodes': 'Number of compute nodes for distributed inference (only tested with 1)',
        'skip_localrefinement': 'Skip local pose refinement step and use only neural network predictions',
        'skip_reconstruction': 'Skip 3D reconstruction step and output only predicted poses',
        'update_progressbar_n_batches': 'Update progress bar every N batches',
        'directional_zscore_thr': 'Confidence z-score threshold for filtering particles. Particles with scores below this are discarded as low-confidence',
        'before_refiner_buffer_size': 'TODO: Document this parameter',
        'float32_matmul_precision': 'PyTorch float32 matrix multiplication precision mode ("highest", "high", or "medium")',

        # CLI-exposed parameters (not in config, but used in distributed_inference)
        'particles_star_fname': 'Path to input RELION particles .star file',
        'checkpoint_dir': 'Path to training directory (or .zip file) containing half-set models with checkpoints and hyperparameters. By default they are called version_0, version_1, etc.',
        'results_dir': 'Output directory for inference results including predicted poses and optional reconstructions',
        'data_halfset': 'Which particle half-set(s) to process: "half1", "half2", or "allParticles"',
        'model_halfset': 'Model half-set selection policy: "half1", "half2", "allCombinations", or "matchingHalf" (uses matching data/model pairs)',
        'particles_dir': 'Root directory for particle image paths. If provided, overrides paths in the .star file',
        'n_jobs': 'Number of worker processes. Defaults to number of GPUs if CUDA enabled, otherwise 1',
        'compile_model': 'Compile model with torch.compile for faster inference (experimental, requires PyTorch 2.0+)',
        'reference_map': 'Path to reference map (.mrc) for FSC computation during validation',
        'reference_mask': 'Path to reference mask (.mrc) for masked FSC calculation',
        'subset_idxs': 'List of particle indices to process (for debugging or partial processing)',
        'n_first_particles': 'Process only the first N particles from dataset (debug feature)',
        'check_interval_secs': 'Polling interval in seconds for parent loop in distributed processing',
    }

    batch_size: int = 64
    use_cuda: bool = True
    n_cpus_if_no_cuda: int = 4
    random_seed: int = 12313

    top_k_poses_nnet: int = 1

    pl_plugin: Literal["LightningEnvironment", "none"] = "LightningEnvironment"  #Try none to submit to slurm
    num_computer_nodes: int = 1 #It has not been tried with values different from 1

    skip_localrefinement: bool = False
    skip_reconstruction: bool = False
    update_progressbar_n_batches: int = 20
    directional_zscore_thr: Optional[float] = None

    before_refiner_buffer_size: int = 16

    float32_matmul_precision: str = "high"
