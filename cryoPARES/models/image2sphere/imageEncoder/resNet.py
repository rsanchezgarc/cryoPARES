import re
import torchvision
from torch import nn
from typing import Optional

from cryoPARES.configManager.config_searcher import inject_config


@inject_config()
class ResNet(nn.Module):
    def __init__(self, in_channels:int, resnetName: str, load_imagenetweights: bool,
                 out_channels: Optional[int] = None, **kwargs):


        super().__init__()
        cls = getattr(torchvision.models, resnetName)
        num = re.findall(r"Resnet(\d+)", resnetName, re.IGNORECASE)[0]

        base_Resnet = cls(weights=None if not load_imagenetweights \
            else getattr(torchvision.models, f"ResNet{num}_Weights").IMAGENET1K_V2)

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
    batch = get_example_random_batch()
    x = batch["particle"]
    model = ResNet(in_channels=x.shape[1], out_channels=8)
    out = model(x)
    print(out.shape)
