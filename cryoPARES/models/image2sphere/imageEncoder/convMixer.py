from torch import nn

from cryoPARES.configManager.config_searcher import inject_config


class ResidualForConvMixer(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

@inject_config()
class ConvMixer(nn.Module):

    def __init__(self, in_channels, hidden_dim, n_blocks, kernel_size, patch_size,
                 out_channels, add_stem, dropout_rate, normalization="Batch",
                 global_pooling=False, flatten_if_no_global_pooling=False, flatten_start_dim=1, **kwargs):
        super().__init__()
        if normalization == "Batch":
            normalization = lambda *args: nn.BatchNorm2d(*args, momentum=0.01)
        else:
            raise NotImplementedError()

        if add_stem:
            steam = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=kernel_size, stride=2, padding=kernel_size//2),
                nn.GELU(),
                normalization(hidden_dim))
            in_channels = hidden_dim
        else:
            steam = nn.Identity()

        self.cnn = nn.Sequential(
            steam,
            nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            normalization(hidden_dim),
            *[nn.Sequential(
                    ResidualForConvMixer(nn.Sequential(
                        nn.Conv2d(hidden_dim, hidden_dim, kernel_size, groups=hidden_dim, padding="same"),
                        nn.GELU(),
                        nn.Dropout2d(dropout_rate),  # Dropout after token mixing
                        normalization(hidden_dim)
                    )),
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                    nn.GELU(),
                    nn.Dropout2d(dropout_rate),  # Dropout after channel mixing
                    normalization(hidden_dim),
            ) for _ in range(n_blocks)],
        )
        self.global_pooling = global_pooling
        if self.global_pooling:
            self.trailModel = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(hidden_dim, out_channels)
            )
        else:
            trailModel = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, padding="same")
            layers = []
            if flatten_if_no_global_pooling:
                layers += [trailModel, nn.Flatten(start_dim=flatten_start_dim)]
                self.trailModel = nn.Sequential(*layers)
            else:
                self.trailModel = trailModel

    def forward(self, x):
        feats = self.cnn(x)
        out = self.trailModel(feats)
        return out

if __name__ == "__main__":
    from cryoPARES.datamanager.datamanager import get_example_random_batch
    batch = get_example_random_batch(1)
    x = batch["particle"]
    model = ConvMixer(x.shape[1])
    out = model(x)
    print(out.shape)