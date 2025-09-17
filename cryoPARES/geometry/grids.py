import numpy as np
import torch
import healpy as hp
from typing import Literal, Optional

from scipy.spatial.transform import Rotation as R

from cryoPARES.cacheManager import get_cache
from cryoPARES.constants import RELION_EULER_CONVENTION
from cryoPARES.geometry.convert_angles import matrix_to_euler_angles
from cryoPARES.geometry.metrics_angles import rotation_magnitude

cache = get_cache(cache_name=__name__)


def hp_order_to_degs(hp_order):
    return hp.nside2resol(hp.order2nside(hp_order), arcmin=True) / 60

def pick_hp_order(grid_resolution_degs):
    for i in range(14):  # We generally want i to be in the 4 to 6 range.
        curr_degs = hp_order_to_degs(i)
        if curr_degs <= grid_resolution_degs:
            return i
    raise RuntimeError(f"Error, discretization required for {grid_resolution_degs} is beyond precision limit")

@cache.cache()
def s2_healpix_grid(hp_order, max_beta):
    """Returns healpix grid up to a max_beta
    :param max_beta: Radians
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

@cache.cache()
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
    # result = torch.FloatTensor(r.as_euler("YXY", degrees=False)).T.contiguous()
    #IMPORTANT!!! At the moment, directionalNormalizer assumes ZYZ, DO NOT CHANGE IT

    return result, n_cones


so3_healpix_grid = so3_healpix_grid_equiangular

@cache.cache()
def so3_near_identity_grid_cartesianprod(max_angle, n_angles,
                                         transposed=True, degrees=False,
                                         remove_duplicates=True): #TODO: It is  better to use something like healpy rather than a cartesian product.
    """Spatial grid over SO3 used to parametrize localized filter

    :return: a local grid of SO(3) points
    """

    angles_range = torch.linspace(-max_angle, max_angle, n_angles)
    grid = torch.cartesian_prod(angles_range, angles_range, angles_range)
    if remove_duplicates:
        grid = _filter_duplicate_angles_in_grid(grid, degrees=degrees) #TODO: This seems working poorly
    if transposed:
        grid = grid.T.contiguous()
    return grid

def _filter_duplicate_angles_in_grid(grid, degrees:bool, atol=None):
    """

    :param grid: A tensor of shape Bx3, representing euler angles
    :param degrees:
    :return:
    """
    if atol is None:
        if degrees:
            atol = 0.1
        else:
            atol = np.deg2rad(0.1)
    rots = R.from_euler(RELION_EULER_CONVENTION, grid, degrees=degrees)
    duplicates = set()
    for i in range(grid.shape[0]):

        relative_duplicate_indices = np.where(rots[i].approx_equal(
            rots[range(i + 1, grid.shape[0])],
            atol=atol, degrees=degrees))[0]
        # Convert relative indices to original absolute indices
        absolute_duplicate_indices = relative_duplicate_indices + (i + 1)
        duplicates.update(absolute_duplicate_indices.tolist())

    idxs = [i for i in range(len(rots)) if i not in duplicates]
    grid = grid[idxs]
    return grid

from cryoPARES.geometry.utilsGrid import so3_grid_near_identity_fibo as _so3_grid_near_identity_fibo

@cache.cache()
def so3_grid_near_identity_fibo(
                    distance_deg: float,
                    spacing_deg: float,
                    use_small_aprox: bool = False,
                    device=None,
                    dtype=torch.float32,
                    return_weights: bool = True,
                    output: str = "matrix"):
    return _so3_grid_near_identity_fibo(
                distance_deg, spacing_deg, use_small_aprox, device,
                dtype, return_weights, output)

if __name__ == "__main__":
    # out = so3_near_identity_grid_cartesianprod(15, 3, transposed=False, degrees=True)
    out = so3_near_identity_grid_cartesianprod(np.pi/12, 8, transposed=False, degrees=False)
    print(out)