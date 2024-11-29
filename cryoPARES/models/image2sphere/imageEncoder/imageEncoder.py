import re
from math import ceil
from typing import Optional, Tuple, List

import hydra.utils
import torch
import torchvision

from torch import nn

from cryoPARES.configs.mainConfig import pyObjectFromStr, main_config
from cryoPARES.configs.models_config.image2sphere_config.imageEncoder_config.imageEncoder_config import \
    EncoderArchitecture
from cryoPARES.datamanager.datamanager import get_number_image_channels, get_example_random_batch
from cryoPARES.models.image2sphere.gaussianFilters import GaussianFilterBank

from cryoPARES.configManager.config_searcher import inject_config
from cryoPARES.constants import BATCH_PARTICLES_NAME

def instatiateEncoder(encoderArtchitecture): #instatiateEncoder cannot be included in the __init__ because of the @inject_config
    artchName = encoderArtchitecture.value
    return pyObjectFromStr(f".{artchName}.{artchName[0].upper() + artchName[1:]}")

@inject_config()
class ImageEncoder(nn.Module):
    def __init__(self, encoderArtchitecture:Optional[EncoderArchitecture], out_channels: Optional[int],
                 **kwargs):
        super().__init__()
        self.in_channels = get_number_image_channels()
        images = get_example_random_batch(batch_size=1)[BATCH_PARTICLES_NAME]
        self.filterBank = GaussianFilterBank(in_channels=self.in_channels)
        out = self.filterBank(images)
        if encoderArtchitecture is None:
            encoderArtchitecture = main_config.models.image2sphere.imageencoder.encoderArtchitecture
        encoderClass = instatiateEncoder(encoderArtchitecture)
        self.imageEncoder = encoderClass(out.shape[1], image_size=images.shape[-2], out_channels=out_channels)

    def forward(self, x):
        x = self.filterBank(x)
        return self.imageEncoder(x)

if __name__ == "__main__":
    from cryoPARES.datamanager.datamanager import get_example_random_batch
    from cryoPARES.constants import BATCH_PARTICLES_NAME
    batch = get_example_random_batch(1)
    x = batch[BATCH_PARTICLES_NAME]
    model = ImageEncoder()
    out = model(x)
    print(out.shape)
