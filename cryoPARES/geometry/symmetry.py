from cryoPARES.cacheManager import get_cache
from scipy.spatial.transform import Rotation as R

cache = get_cache(cache_name=None)
@cache.cache
def getSymmetryGroup(symmetry):
    # from cryoUtils.geometry.symmetry import getSymmetryMatrices
    # group = getSymmetryMatrices(self.symmetrize_contraction)
    # group = torch.FloatTensor(group)
    group = R.create_group(symmetry.upper())
    return group