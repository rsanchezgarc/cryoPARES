from dataclasses import dataclass, asdict, field, MISSING
from typing import Optional

import numpy as np


@dataclass
class I2SProjector_config:
    """Image to sphere projector configuration."""

    # Centralized parameter documentation
    PARAM_DOCS = {
        'sphere_fdim': 'Feature dimension for spherical representation',
        'hp_order': 'HEALPix order for spherical grid resolution',
        'coverage': 'Fraction of sphere to cover during projection',
        'sigma': 'Gaussian kernel width for projection smoothing',
        'max_beta': 'Maximum beta angle (in radians) for projection coverage',
        'taper_beta': 'Beta angle (in radians) at which to begin tapering projection weights',
        'rand_fraction_points_to_project': 'Fraction of points to randomly sample for projection (reduces computation)',
    }

    sphere_fdim: int = 512
    hp_order: int = 3
    coverage: float = 0.9
    sigma: float = 0.2
    max_beta: float = float(np.radians(90))
    taper_beta: float = float(np.radians(75))
    rand_fraction_points_to_project: Optional[float] = 0.5

@dataclass
class S2Conv_config:
    """S2 (sphere) convolution configuration."""

    # Centralized parameter documentation
    PARAM_DOCS = {
        'f_out': 'Number of output features from S2 convolution',
        'hp_order': 'HEALPix order for S2 convolution grid resolution',
    }

    f_out: int = 64
    hp_order: int = 4

@dataclass
class SO3Activation_config:
    """SO(3) activation function configuration."""

    # Centralized parameter documentation
    PARAM_DOCS = {
        'so3_act_resolution': 'Resolution for SO(3) activation grid sampling',
    }

    so3_act_resolution: int = 10

@dataclass
class SO3Conv_config:
    """SO(3) convolution configuration."""

    # Centralized parameter documentation
    PARAM_DOCS = {
        'f_out': 'Number of output features from SO(3) convolution',
        'max_rads': 'Maximum radius in radians for SO(3) convolution kernel',
        'n_angles': 'Number of angular samples for SO(3) convolution',
    }

    f_out = 1
    max_rads: float = np.pi/12
    n_angles: int = 8

@dataclass
class SO3OuptutGrid_config:
    """SO(3) output grid configuration."""

    # Centralized parameter documentation
    PARAM_DOCS = {
        'hp_order': 'HEALPix order for SO(3) output grid resolution',
    }

    hp_order: int = 4


@dataclass
class So3Components_config:
    """SO(3) neural network components configuration."""

    i2sprojector: I2SProjector_config = field(default_factory=I2SProjector_config)
    s2conv: S2Conv_config = field(default_factory=S2Conv_config)
    so3activation: SO3Activation_config = field(default_factory=SO3Activation_config)
    so3conv: SO3Conv_config = field(default_factory=SO3Conv_config)
    so3outputgrid: SO3OuptutGrid_config = field(default_factory=SO3OuptutGrid_config)