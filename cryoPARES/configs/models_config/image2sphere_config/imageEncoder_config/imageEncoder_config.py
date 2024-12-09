from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Literal

from cryoPARES.configs.models_config.image2sphere_config.imageEncoder_config.convMixer_config import ConvMixer_config
from cryoPARES.configs.models_config.image2sphere_config.imageEncoder_config.resNet_config import ResNet_config
from cryoPARES.configs.models_config.image2sphere_config.imageEncoder_config.unet_config import Unet_config


class EncoderArchitecture(str, Enum):
    ResNet = "resNet"
    ConvMixer = "convMixer"
    Unet = "unet"


@dataclass
class ImageEncoder_config:
    encoderArtchitecture: EncoderArchitecture = EncoderArchitecture.ResNet #EncoderArchitecture.Unet
    out_channels: Optional[int] = 512

    convmixer: ConvMixer_config = field(default_factory=ConvMixer_config)
    resnet: ResNet_config = field(default_factory=ResNet_config)
    unet: Unet_config = field(default_factory=Unet_config)


