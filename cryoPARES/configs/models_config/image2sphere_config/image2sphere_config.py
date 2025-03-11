from dataclasses import dataclass, field

from cryoPARES.configs.models_config.image2sphere_config.gaussianFilters_config import GaussianFilters_config
from cryoPARES.configs.models_config.image2sphere_config.imageEncoder_config.imageEncoder_config import \
    ImageEncoder_config
from cryoPARES.configs.models_config.image2sphere_config.so3Components_config import So3Components_config


@dataclass
class Image2Sphere_config:
    lmax: int = 12
    label_smoothing: float = 0.05 #TODO: Move to loss config
    enforce_symmetry: bool = True
    use_simCLR: bool = False
    imageencoder: ImageEncoder_config = field(default_factory=ImageEncoder_config)
    so3components: So3Components_config = field(default_factory=So3Components_config)
    gaussianfilters: GaussianFilters_config = field(default_factory=GaussianFilters_config)
