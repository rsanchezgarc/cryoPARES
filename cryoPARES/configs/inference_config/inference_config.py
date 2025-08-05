from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class Inference_config:

    batch_size: int = 64
    use_cuda: bool = True
    n_cpus_if_no_cuda: int = 4
    random_seed: int = 12313

    top_k: int = 1

    pl_plugin: Literal["LightningEnvironment", "none"] = "LightningEnvironment"  #Try none to submit to slurm
    num_computer_nodes: int = 1 #It has not been tried with values different from 1

    perform_localrefinement: bool = True
    perform_reconstruction: bool = True
    update_progessbar_n_batches: int = 20
    directional_zscore_thr: Optional[float] = None

    before_refiner_buffer_size: int = 16
