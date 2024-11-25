import re
from math import ceil
from typing import Optional, Tuple, List

import torch
import torchvision

from torch import nn

from cryoPARES.models.image2sphere.gaussianFilters import GaussianFilterBank


class BaseEncoder(nn.Module):
    def __init__(self, model:nn.Module, in_channels, guassian_sigmas: List[float], output_dim_filters:int):
        """

        Args:
            model (str): The model to use as encoder
            sigma0 (float): The standard deviation of the 1st filter to expand the number of channels from 1 to 3
            sigma1 (float): The standard deviation of the 2nd filter to expand the number of channels from 1 to 3
        """
        super().__init__()
        self.in_channels = in_channels
        self.imageEncoder = model
        self.filterBank = GaussianFilterBank(in_channels, sigma_values=guassian_sigmas, out_dim=output_dim_filters)

    def forward(self, x):
        x = self.filterBank(x)
        return self.imageEncoder(x)

class ResNetImageEncoder(BaseEncoder):
    def __init__(self, resnetName: str = "resnet50", load_imagenetweights: bool = True, sigma0: float = 1.,
                 sigma1: float = 2., in_channels:int=1): #TODO: It should read in_channels automatically
        """

        Args:
            resnetName (str): The name of the torchvision.models resnet to be used
            sigma0 (float): The standard deviation of the 1st filter to expand the number of channels from 1 to 3
            sigma1 (float): The standard deviation of the 2nd filter to expand the number of channels from 1 to 3
        """
        self.in_channels = in_channels
        cls = getattr(torchvision.models, resnetName)
        num = re.findall(r"resnet(\d+)", resnetName, re.IGNORECASE)[0]
        resnet = nn.Sequential(*list(cls(weights=None if not load_imagenetweights \
                                else getattr(torchvision.models, f"ResNet{num}_Weights").IMAGENET1K_V2
                                         ).children())[:-2])
        super().__init__(model=resnet, sigma0 = sigma0, sigma1 = sigma1, in_channels= in_channels)


from unet.unet import Encoder, Decoder, EncodingBlock
from unet.conv import ConvolutionalBlock

class _Unet(nn.Module):
    def __init__(self, in_channels, n_blocks, out_channels, out_channels_first=64, n_decoder_blocks_removed=1):

        super().__init__()
        assert n_blocks > n_decoder_blocks_removed + 1

        pooling = 'max'  # 'avg'
        padding = "same"
        activation = "LeakyReLU"
        normalization = "Batch"
        upsampling_type =  'bilinear' #'conv'
        self.encoder = Encoder(
            in_channels,
            out_channels_first = out_channels_first,
            dimensions = 2,
            pooling_type= pooling,
            num_encoding_blocks=n_blocks,
            normalization=normalization,
            preactivation=False,
            residual=True,
            padding=padding,
            padding_mode="zeros",
            activation=activation,
            initial_dilation=None,
            dropout=0,
        )

        self.bottom_block = EncodingBlock(
            self.encoder.out_channels,
            out_channels_first = 2 * self.encoder.out_channels,
            dimensions=2,
            normalization=normalization,
            pooling_type=None,
            preactivation=False,
            residual=True,
            padding=padding,
            padding_mode="zeros",
            activation=activation,
            dilation=self.encoder.dilation,
        )

        self.decoder = Decoder(
            out_channels_first * 2**(n_blocks-1),
            dimensions=2,
            upsampling_type=upsampling_type,
            num_decoding_blocks = (n_blocks-n_decoder_blocks_removed),
            normalization=normalization,
            preactivation=False,
            residual=False,
            padding=padding,
            padding_mode="zeros",
            activation=activation,
            initial_dilation=None,
            dropout=0,
        )

        if out_channels is not None:
            self.last_layer = ConvolutionalBlock(
                2, out_channels_first * (2**(n_decoder_blocks_removed)), out_channels,
                kernel_size=1, activation=None,
            )
        else:
            self.last_layer = nn.Identity()

    def forward(self, x):
        skip_connections, encoding = self.encoder(x)
        encoding = self.bottom_block(encoding)
        x = self.decoder(skip_connections, encoding)
        return self.last_layer(x)


class UnetLikeImageEncoder(BaseEncoder):
    def __init__(self, n_blocks: int = 5, sigma0: float = 1.,
                 sigma1: float = 2., in_channels:int=1, out_channels:Optional[int]=None, out_channels_first:int=64,
                 n_decoder_blocks_removed:int=1):
        """

        Args:
            resnetName (str): The name of the torchvision.models resnet to be used
            sigma0 (float): The standard deviation of the 1st filter to expand the number of channels from 1 to 3
            sigma1 (float): The standard deviation of the 2nd filter to expand the number of channels from 1 to 3
        """

        model = _Unet(in_channels*3, n_blocks, out_channels=out_channels, out_channels_first=out_channels_first,
                      n_decoder_blocks_removed=n_decoder_blocks_removed)
        super().__init__(model= model, sigma0 = sigma0, sigma1 = sigma1, in_channels= in_channels)

class ResidualForConvMixer(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class ConvMixer(nn.Module):

    def __init__(self, in_dim, hidden_dim, depth, kernel_size=9, patch_size=7,
                 global_pooling=True, flatten_if_no_global_pooling=True, flatten_start_dim=1,
                 num_classes=1000, add_stem=False, normalization=None, **kwargs):
        super().__init__()
        if normalization is None:
            normalization = lambda *args: nn.BatchNorm2d(*args, momentum=0.01) ##nn.InstanceNorm2d #lambda *args: nn.BatchNorm2d(*args, momentum=0.01)
            #normalization = lambda c,h,w, *args: nn.LayerNorm([c, h, w])

        if add_stem:
            steam = nn.Sequential(
                nn.Conv2d(in_dim, hidden_dim, kernel_size=kernel_size, stride=2, padding=kernel_size//2),
                nn.GELU(),
                normalization(hidden_dim))
            in_dim = hidden_dim
        else:
            steam = nn.Identity()

        self.cnn = nn.Sequential(
            steam,
            nn.Conv2d(in_dim, hidden_dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            normalization(hidden_dim),
            *[nn.Sequential(
                    ResidualForConvMixer(nn.Sequential(
                        nn.Conv2d(hidden_dim, hidden_dim, kernel_size, groups=hidden_dim, padding="same"),
                        nn.GELU(),
                        normalization(hidden_dim)
                    )),
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                    nn.GELU(),
                    normalization(hidden_dim),
            ) for _ in range(depth)],
        )
        self.global_pooling = global_pooling
        if self.global_pooling:
            self.trailModel = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            trailModel = nn.Conv2d(hidden_dim, num_classes, kernel_size=1, padding="same")
            layers = []
            if flatten_if_no_global_pooling:
                layers += [trailModel, nn.Flatten(start_dim=flatten_start_dim)]
                # if flatten_start_dim > 1:
                #     layers += [SwapLastDims()]
                self.trailModel = nn.Sequential(*layers)
            else:
                self.trailModel = trailModel

    def forward(self, x):
        feats = self.cnn(x)
        out = self.trailModel(feats)
        return out

class ConvMixerImageEncoder(BaseEncoder):
    def __init__(self, n_blocks: int = 12, sigma0: float = 1.,
                 sigma1: float = 2., in_channels:int=1, out_channels:Optional[int]=None, hidden_dim:int=512,
                 kernel_size=9, patch_size=7):
        """

        Args:
            resnetName (str): The name of the torchvision.models resnet to be used
            sigma0 (float): The standard deviation of the 1st filter to expand the number of channels from 1 to 3
            sigma1 (float): The standard deviation of the 2nd filter to expand the number of channels from 1 to 3
        """

        # model = UNet2D(in_channels*3, out_classes=out_channels, padding="same")


        model = ConvMixer(in_channels*3, hidden_dim, depth=n_blocks, kernel_size=kernel_size, patch_size=patch_size,
                          add_stem=False, global_pooling=False, flatten_if_no_global_pooling=False,
                          num_classes=out_channels if out_channels else hidden_dim)
        super().__init__(model=model, sigma0=sigma0, sigma1=sigma1, in_channels=in_channels)

