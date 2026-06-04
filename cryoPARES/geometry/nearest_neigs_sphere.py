import os
import warnings
import numba
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation as R
from cryoPARES.constants import RELION_EULER_CONVENTION


@numba.njit(parallel=False)
def quaternion_distance(q1, q2):
    """
    Compute the distance between two rotations represented by unit quaternions.

    Parameters:
    - q1: A numpy array of shape (4,) representing the first quaternion.
    - q2: A numpy array of shape (4,) representing the second quaternion.

    Returns:
    - d(q1,q2)=1−⟨q1,q2⟩**2 which is equal to (1−cosθ)/2 and serves as a distance
    """
    dist = 1 - np.dot(q1, q2) ** 2

    return dist

def compute_nearest_neighbours(eulerDegs, k, cache_dir, n_jobs):
    eulerDegs = torch.transpose(eulerDegs, 0,1)
    fname = os.path.join(cache_dir, f"knn_{eulerDegs.shape[0]}.pt")
    if os.path.exists(fname):
        return torch.load(fname, weights_only=True)

    print(f"Computing nearest neighbours. For n > 294912, it will take a long time (Current n={eulerDegs.shape[0]})")

    rs = R.from_euler(RELION_EULER_CONVENTION, eulerDegs, degrees=True)
    X = rs.as_quat()
    X = X / np.linalg.norm(X, axis=-1, keepdims=True)

    neigh = NearestNeighbors(n_neighbors=k, metric=quaternion_distance, algorithm="ball_tree", n_jobs=n_jobs)
    neigh.fit(X)
    dists, nearest_neigbours = neigh.kneighbors(X)
    dists = torch.as_tensor(dists, dtype=torch.float32)
    nearest_neigbours = torch.as_tensor(nearest_neigbours, dtype=torch.int64)
    out = dict(dists=dists, nearest_neigbours=nearest_neigbours)

    try:
        torch.save(out, fname)
    except (FileNotFoundError, IOError, PermissionError) as e:
        print(e)
        warnings.warn(f"The CACHE_DIR {cache_dir} is not available ({e}), skipping cache. "
                      f"Some weights will be recomputed in each execution, wasting compute")

    print(f"Computing nearest neighbours... Done")
    return out



