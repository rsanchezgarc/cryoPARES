from dataclasses import dataclass, field

from cryoPARES.configs.models_config.image2sphere_config.gaussianFilters_config import GaussianFilters_config
from cryoPARES.configs.models_config.image2sphere_config.imageEncoder_config.imageEncoder_config import \
    ImageEncoder_config
from cryoPARES.configs.models_config.image2sphere_config.so3Components_config import So3Components_config


@dataclass
class Image2Sphere_config:
    """Image to SO(3) sphere configuration parameters."""

    # Centralized parameter documentation
    PARAM_DOCS = {
        'lmax': 'Maximum spherical harmonic degree for SO(3) representation. Higher values allow finer angular resolution but increase memory and computation',
        'label_smoothing': 'Label smoothing factor for loss function to prevent overconfidence (TODO: Move this to a future loss config)',
        'enforce_symmetry': 'Apply point group symmetry constraints during pose prediction',
        'use_simCLR': 'Enable SimCLR contrastive learning loss for improved representation learning',
        'simCLR_temperature': 'Temperature parameter for SimCLR contrastive loss (controls distribution sharpness)',
        'simCLR_loss_weight': 'Weight for SimCLR contrastive loss relative to main pose prediction loss',
        'average_neigs_for_pred': 'Average predictions from neighboring orientations for smoother results (slightly slower but more accurate)',
        'n_neigs_to_compute': 'Number of neighboring orientations to use for averaging when average_neigs_for_pred is True',
    }

    lmax: int = 12
    label_smoothing: float = 0.05 #TODO: Move this to a future loss config
    enforce_symmetry: bool = True
    use_simCLR: bool = False
    simCLR_temperature: float = 0.5  # Temperature parameter for contrastive loss
    simCLR_loss_weight: float = 0.1  # Weight for contrastive loss relative to main loss
    average_neigs_for_pred: bool = False #Sligthly slower and slightly more accurate
    n_neigs_to_compute: int = 10 #Number of neigs used for averaging

    imageencoder: ImageEncoder_config = field(default_factory=ImageEncoder_config)
    so3components: So3Components_config = field(default_factory=So3Components_config)
    gaussianfilters: GaussianFilters_config = field(default_factory=GaussianFilters_config)
