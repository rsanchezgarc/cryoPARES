from dataclasses import dataclass, field



@dataclass
class Directionalnormalizer_config:
    """Directional normalizer configuration for confidence scoring."""

    # Centralized parameter documentation
    PARAM_DOCS = {
        'hp_order': 'HEALPix order for spherical grid resolution used in directional normalization. Higher values give finer resolution',
    }

    hp_order: int = 2