from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class ResNet_config:
    resnetName: str = "resnet50"
    load_imagenetweights: bool = True