from dataclasses import dataclass, field
from typing import Optional

from cryoPARES.configs.models_config.image2sphere_config.image2sphere_config import Image2Sphere_config


@dataclass
class Projmatching_config:
    compile_projectVol: bool = True #TODO: I believe that this config is not going to work in a dynamic manner, as it is used with a inject_config
    compile_projectVol_mode: Optional[str] = "max-autotune"