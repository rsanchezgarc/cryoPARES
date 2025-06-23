import torch

from cryoPARES.cacheManager import get_cache
from scipy.spatial.transform import Rotation as R

cache = get_cache(cache_name=None)
@cache.cache
def getSymmetryGroup(symmetry, as_matrix=False, device:str="cpu"):

    group = R.create_group(symmetry.upper())
    if as_matrix:
        group = torch.stack([torch.FloatTensor(x)
                     for x in getSymmetryGroup(symmetry).as_matrix()])
        group = group.to(device)
    return group