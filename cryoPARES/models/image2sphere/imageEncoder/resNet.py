import re
import torchvision
from torch import nn
from typing import Optional

from autoCLI_config import inject_defaults_from_config, CONFIG_PARAM
from cryoPARES.configs.mainConfig import main_config


class ResNet(nn.Module):
    @inject_defaults_from_config(main_config.models.image2sphere.imageencoder.resnet, update_config_with_args=False)
    def __init__(self, in_channels:int, resnetName: str= CONFIG_PARAM(), load_imagenetweights: bool=CONFIG_PARAM(),
                 out_channels: int =CONFIG_PARAM(), **kwargs):

        super().__init__()
        cls = getattr(torchvision.models, resnetName)
        num = re.findall(r"Resnet(\d+)", resnetName, re.IGNORECASE)[0]

        base_Resnet = cls(weights=None if not load_imagenetweights \
            else getattr(torchvision.models, f"ResNet{num}_Weights").IMAGENET1K_V1)

        layers = [
            nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=1),
            *list(base_Resnet.children())[:-2]
        ]
        feature_channels = 2048 if int(num) >= 50 else 512  # Resnet50+ uses 2048, earlier versions use 512
        if out_channels is not None:
            layers.append(nn.Conv2d(feature_channels, out_channels, kernel_size=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    from cryoPARES.datamanager.datamanager import get_example_random_batch
    batch = get_example_random_batch(1)
    x = batch["particle"]
    model = ResNet(in_channels=x.shape[1], out_channels=8)
    out = model(x)
    print(out.shape)
