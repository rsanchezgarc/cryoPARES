import numpy as np
import torch
import healpy as hp
from typing import Literal, Optional

from cryoPARES.cacheManager import get_cache
from cryoPARES.constants import RELION_EULER_CONVENTION
from cryoPARES.geometry.convert_angles import matrix_to_euler_angles
from cryoPARES.geometry.metrics_angles import rotation_magnitude

cache = get_cache(cache_name=__name__)

cache.cache()
def s2_healpix_grid(hp_order, max_beta):
    """Returns healpix grid up to a max_beta

    Alternative implementation:
            from escnn.group import so3_group
            so2grid = torch.stack([torch.as_tensor(x.to("ZYZ"), dtype=torch.float32).rad2deg() for x in so3_group(maximum_frequency=l_max).sphere_grid(N_side=2**2, type="healpix")])

    """
    n_side = 2**hp_order
    m = hp.query_disc(nside=n_side, vec=(0,0,1), radius=max_beta)
    beta, alpha = hp.pix2ang(n_side, m)
    alpha = torch.from_numpy(alpha)
    beta = torch.from_numpy(beta)
    return torch.stack((alpha, beta)).float()

cache.cache()
def so3_healpix_grid_equiangular(hp_order: int = 3):
    """Returns healpix grid over so3 of equally spaced rotations

    alpha: 0-2pi around Y
    beta: 0-pi around X
    gamma: 0-2pi around Y
    hp_order | num_points | bin width (deg)  | N inplane
    ----------------------------------------
         0    |         72 |    60           |
         1    |        576 |    30
         2    |       4608 |    15           | 24
         3    |      36864 |    7.5          | 48
         4    |     294912 |    3.75         | 96
         5    |    2359296 |    1.875

    :return: tensor of shape (3, npix)
    """
    n_side = 2 ** hp_order
    npix = hp.nside2npix(n_side)
    beta, alpha = hp.pix2ang(n_side, torch.arange(npix))
    alpha = alpha.float()
    beta = beta.float()
    psi = torch.linspace(0, 2 * np.pi, 6 * n_side + 1)[:-1] # order3:48 order4:96

    alpha_beta = torch.stack([alpha, beta])
    n_cones = alpha_beta.shape[1]
    n_psis = psi.shape[0]

    # Reshape and expand gamma to shape (2, n_cones, M)
    alpha_beta_expanded = alpha_beta.unsqueeze(2).expand(2, n_cones, n_psis)

    # Reshape and expand gamma to shape (1, n_cones, M)
    gamma_expanded = psi.unsqueeze(0).unsqueeze(1).expand(1, n_cones, n_psis)

    # Concatenate along the first dimension and then reshape
    zyz_grid = torch.cat((alpha_beta_expanded, gamma_expanded), dim=0)

    zyz_grid = zyz_grid.reshape(3, n_cones * n_psis)
    result = torch.as_tensor(zyz_grid, dtype=torch.float)

    # #If we were doing things with the angles we would NEED TO GO FROM ZYZ TO YXY to keep it consistent with e3nn, but
    # #since we are not doing it, it is not necessary
    # from scipy.spatial.transform import Rotation as R
    # r = R.from_euler("ZYZ", zyz_grid.T, degrees=False)
    # result = torch.FloatTensor(r.as_euler("YXY", degrees=False)).T.contiguous() #TODO. Is his needed after all?

    return result, n_cones

# try:
#     from escnn.group import so3_group
#     def so3_healpix_grid_escnn(hp_order: int, method:Literal["thomson", "hopf", "thomson_cube", "fibonacci"] = "hopf",
#                                representation:Optional[Literal["MAT", "Q", "ZYZ"]]="ZYZ"):
#         n_cones = hp.order2npix(hp_order) * 6
#         gridB = so3_group(maximum_frequency=1).grid(N=n_cones, type=method)
#         if representation is not None:
#             gridB = [x.to(representation) for x in gridB]
#             gridB = torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in gridB]).T.contiguous()
#         return gridB, n_cones
#
#     def so3_near_identity_grid_escnn(max_rads,
#                                      hp_order):  # TODO. It is hard to set parameters that lead to good results
#         fullSo3 = so3_healpix_grid_escnn(hp_order, representation="MAT")[0]
#         magnitudes = rotation_magnitude(fullSo3)
#         angles = matrix_to_euler_angles(fullSo3[magnitudes < max_rads], convention=RELION_EULER_CONVENTION)
#         assert angles.shape[0] > 0
#         return angles
#
# except (ImportError, AttributeError):
#     pass

so3_healpix_grid = so3_healpix_grid_equiangular #so3_healpix_grid_escnn

cache.cache()
def so3_near_identity_grid_cartesianprod(max_rads, n_angles): #TODO: It is probably better to use something like healpy rather than a cartesian product.
    """Spatial grid over SO3 used to parametrize localized filter

    :return: a local grid of SO(3) points
           size of the kernel = n_alpha**3
    """

    angles_range = torch.linspace(-max_rads, max_rads, n_angles)
    grid = torch.cartesian_prod(angles_range, angles_range, angles_range)
    return grid.T.contiguous()


def so3_near_identity_grid_ori(max_alpha=np.pi / 12, max_beta=np.pi / 12, max_gamma=np.pi / 12,
                           n_alpha=8, n_beta=3, n_gamma=8):  # New version
    """Spatial grid over SO3 used to parametrize localized filter

    :return: a local grid of SO(3) points
           size of the kernel = n_alpha * n_beta * n_gamma
    """

    alpha = torch.linspace(-max_alpha, max_alpha, n_alpha)
    beta = torch.linspace(-max_beta, max_beta, n_beta)
    gamma = torch.linspace(-max_gamma, max_gamma, n_gamma)
    grid = torch.cartesian_prod(alpha, beta, gamma)
    return grid.T

