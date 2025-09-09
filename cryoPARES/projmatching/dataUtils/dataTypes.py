import os

import numpy as np
import torch

from starstack import ParticlesStarSet
from .dataClasses import ImageTensor

FNAME = os.PathLike | str
IMAGEFNAME_MRCIMAGE = FNAME | ImageTensor | torch.Tensor | np.ndarray
DEFAULT_DTYPE = torch.float32
PARTICLES_SET_OR_STARFNAME = FNAME | ParticlesStarSet
