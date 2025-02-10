import re
from math import ceil
from typing import Optional, Tuple, List

import hydra.utils
import torch
import torchvision

from torch import nn

from cryoPARES.constants import BATCH_PARTICLES_NAME
from cryoPARES.configs.mainConfig import pyObjectFromStr, main_config
from cryoPARES.configs.models_config.image2sphere_config.imageEncoder_config.imageEncoder_config import \
    EncoderArchitecture
from cryoPARES.datamanager.datamanager import get_number_image_channels, get_example_random_batch
from cryoPARES.models.image2sphere.gaussianFilters import GaussianFilterBank

from cryoPARES.configManager.inject_defaults import inject_defaults_from_config, CONFIG_PARAM

class ImageEncoder(nn.Module):
    @inject_defaults_from_config(main_config.models.image2sphere.imageencoder)
    def __init__(self, encoderArtchitecture:EncoderArchitecture = CONFIG_PARAM(),
                 out_channels: Optional[int]=CONFIG_PARAM(),
                 **kwargs):
        super().__init__()
        self.in_channels = get_number_image_channels()

        images = get_example_random_batch(batch_size=1)[BATCH_PARTICLES_NAME]
        self.filterBank = GaussianFilterBank(in_channels=self.in_channels)
        out = self.filterBank(images)
        if encoderArtchitecture is None:
            encoderArtchitecture = main_config.models.image2sphere.imageencoder.encoderArtchitecture
        encoderClass = self.instantiateEncoder(encoderArtchitecture)
        self.imageEncoder = encoderClass(out.shape[1], image_size=images.shape[-2], out_channels=out_channels)

    def instantiateEncoder(self, encoderArtchitecture):
        artchName = encoderArtchitecture.value
        return pyObjectFromStr(f".{artchName}.{artchName[0].upper() + artchName[1:]}")

    def forward(self, x):
        x = self.filterBank(x)
        return self.imageEncoder(x)

if __name__ == "__main__":
    from cryoPARES.datamanager.datamanager import get_example_random_batch
    batch = get_example_random_batch(1)
    x = batch[BATCH_PARTICLES_NAME]
    model = ImageEncoder()
    out = model(x)
    print(out.shape)
