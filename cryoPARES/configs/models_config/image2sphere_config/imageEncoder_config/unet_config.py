from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class Unet_config:
    n_blocks: int = 5
    out_channels: Optional[int] = None
    out_channels_first: int= 16
    n_decoder_blocks_removed: Optional[int] = 1
    kernel_size: int = 5
    pooling: Literal["max", "avg"] = 'max'
    padding: str = "same"
    activation: str = "LeakyReLU"
    normalization: str = "Batch"
    upsampling_type: str = 'bilinear'  # 'conv'
    dropout: float = 0.