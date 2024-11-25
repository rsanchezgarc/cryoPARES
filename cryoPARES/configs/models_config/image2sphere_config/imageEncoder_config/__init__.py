from dataclasses import dataclass
from enum import Enum
from typing import Optional, Literal


class EncoderArchitecture(str, Enum):
    ResNet = "resNet"
    ConvMixer = "convMixer"
    Unet = "unet"


@dataclass
class ImageEncoder_fields:
    encoderArtchitecture: EncoderArchitecture = EncoderArchitecture.ResNet
    out_channels: Optional[int] = 512


