from dataclasses import dataclass, field

from cryoPARES.configs.models_config.directionalnormalizer_config.directionalnormalizer_config import \
    Directionalnormalizer_config
from cryoPARES.configs.models_config.image2sphere_config.image2sphere_config import Image2Sphere_config


@dataclass
class Models_config:
    image2sphere: Image2Sphere_config = field(default_factory=Image2Sphere_config)
    directionalNormalizer: Directionalnormalizer_config = field(default_factory=Directionalnormalizer_config)