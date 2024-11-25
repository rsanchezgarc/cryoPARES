import re
from math import ceil
from typing import Optional, Tuple, List

import hydra.utils
import torch
import torchvision

from torch import nn

from cryoPARES.configs.mainConfig import pyObjectFromStr, main_config
from cryoPARES.datamanager.datamanager import get_number_image_channels, get_example_random_batch
from cryoPARES.models.image2sphere.gaussianFilters import GaussianFilterBank


class ImageEncoder(nn.Module):
    def __init__(self, out_channels: Optional[int] = None, **kwargs):
        super().__init__()
        self.in_channels = get_number_image_channels()
        images = get_example_random_batch(batch_size=1)["particle"] #TODO: use something like constants.PARTICLE_NAME

        self.filterBank = GaussianFilterBank(in_channels=self.in_channels)
        out = self.filterBank(images)
        imgencoder_config = main_config.models.image2sphere.imageencoder
        artchName = imgencoder_config.encoderArtchitecture
        encoderClass = pyObjectFromStr(f".{artchName}.{artchName[0].upper()+artchName[1:]}")
        self.imageEncoder = encoderClass(out.shape[1], image_size=images.shape[-2], out_channels=out_channels)

    def forward(self, x):
        x = self.filterBank(x)
        return self.imageEncoder(x)

if __name__ == "__main__":
    from cryoPARES.datamanager.datamanager import get_example_random_batch
    batch = get_example_random_batch()
    x = batch["particle"]
    model = ImageEncoder()
    out = model(x)
    print(out.shape)
