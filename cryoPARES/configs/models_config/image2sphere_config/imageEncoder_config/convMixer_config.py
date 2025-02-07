from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class ConvMixer_config:
    hidden_dim: int = 512
    n_blocks: int = 12
    kernel_size: int= 9
    patch_size: int= 7
    add_stem : bool = False
    out_channels: int = 512
    dropout_rate: float = 0.
    normalization: Literal["Batch"] = "Batch"