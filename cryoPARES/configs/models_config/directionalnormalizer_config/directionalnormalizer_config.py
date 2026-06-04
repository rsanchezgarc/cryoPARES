from dataclasses import dataclass, field



@dataclass
class Directionalnormalizer_config:
    """Directional normalizer configuration for confidence scoring."""

    # Centralized parameter documentation
    PARAM_DOCS = {
        'hp_order': 'HEALPix order for spherical grid resolution used in directional normalization. Higher values give finer resolution',
        'good_particles_percentile': 'When ground truth is unavailable, only particles scoring above this percentile are used to estimate normalization statistics',
        'min_particles_per_cone': 'Minimum number of particles required per orientation cone for reliable statistics. Cones with fewer particles fall back to global statistics',
    }

    hp_order: int = 2
    good_particles_percentile: float = 95.0
    min_particles_per_cone: int = 10