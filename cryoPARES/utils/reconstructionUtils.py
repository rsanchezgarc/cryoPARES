import shutil
from typing import Tuple

import mrcfile
import numpy as np
import torch

from cryoPARES.constants import DEFAULT_DTYPE_PROJMATCHING, RELION_EULER_CONVENTION
from cryoPARES.geometry.convert_angles import euler_angles_to_matrix
from cryoPARES.utils.paths import MAP_AS_ARRAY_OR_FNAME_TYPE, FNAME_TYPE

def get_mrc_metadata(mrc_fname: str) -> Tuple[Tuple[int, int, int], float]:
    """
    Read MRC file metadata without loading the full volume data.

    :param mrc_fname: Path to the MRC file
    :return: Tuple of (shape, sampling_rate) where shape is (nx, ny, nz) and sampling_rate is in Angstroms/pixel
    :raises FileNotFoundError: If the file does not exist
    :raises ValueError: If the file is not a valid MRC file
    """
    with mrcfile.open(mrc_fname, permissive=True) as f:
        shape = (f.header.nx.item(), f.header.ny.item(), f.header.nz.item())
        sampling_rate = float(f.voxel_size.x)
    return shape, sampling_rate

#TODO: This code is used in other places, so move to a common place
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
