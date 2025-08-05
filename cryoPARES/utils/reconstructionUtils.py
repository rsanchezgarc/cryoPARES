import shutil
from typing import Tuple

import mrcfile
import numpy as np
import torch

from cryoPARES.constants import DEFAULT_DTYPE_PROJMATCHING, RELION_EULER_CONVENTION
from cryoPARES.geometry.convert_angles import euler_angles_to_matrix
from cryoPARES.utils.paths import MAP_AS_ARRAY_OR_FNAME_TYPE, FNAME_TYPE


def get_vol(vol: MAP_AS_ARRAY_OR_FNAME_TYPE, pixel_size: float | None,
            device: torch.device|str="cpu") -> Tuple[torch.Tensor, float]:
    if isinstance(vol, FNAME_TYPE):
        with mrcfile.open(vol, permissive=True) as f:
            data = torch.from_numpy(f.data.copy())
            _pixel_size = float(f.voxel_size.x)
            if pixel_size:
                assert np.isclose(pixel_size, _pixel_size)
            else:
                pixel_size = _pixel_size
    else:
        data = vol
    data = data.to(DEFAULT_DTYPE_PROJMATCHING).to(device)
    return data, pixel_size

def write_vol(vol:torch.tensor, fname, pixel_size, overwrite=True):
    mrcfile.write(fname, vol.numpy().astype(np.float32), overwrite=overwrite, voxel_size=pixel_size)


def get_rotmat(degAngles, convention:str=RELION_EULER_CONVENTION, device="cpu"):
    return euler_angles_to_matrix(torch.deg2rad(degAngles), convention=convention).to(device)
