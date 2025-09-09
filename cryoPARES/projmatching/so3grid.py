import os
import timeit
from collections import deque
from typing import List

import joblib
import numpy as np
import torch
import healpy as hp
from scipy.spatial.transform import Rotation as R

from cryoPARES.configs.mainConfig import main_config
from cryoPARES.constants import RELION_EULER_CONVENTION
from .loggers import getWorkerLogger
from .numbaUtils import quaternion_distance


class SO3_discretizer():
    def __init__(self, hp_order, verbose=False):

        self.hp_order = hp_order
        self.nside = hp.order2nside(hp_order)
        self.degs_step = hp.nside2resol(self.nside, arcmin=True) / 60

        self.num_inplane = int(360. / self.degs_step)
        self.num_hp = hp.nside2npix(self.nside)
        self.mainLogger = getWorkerLogger(verbose)

    @property
    def grid_size(self):
        return self.num_hp * self.num_inplane

    @staticmethod
    def hp_order_to_degs(hp_order):
        return hp.nside2resol(hp.order2nside(hp_order), arcmin=True) / 60

    @staticmethod
    def pick_hp_order(grid_resolution_degs):
        for i in range(14): #We generally want i to be in the 4 to 6 range.
            curr_degs = SO3_discretizer.hp_order_to_degs(i)
            if curr_degs <= grid_resolution_degs:
                return i
        raise RuntimeError(f"Error, discretization required for {grid_resolution_degs} is beyond precision limit")

    def _move_tilt_to_0_pi(self, eulerDegs):
        # angles is expected to be an Nx3 tensor where each row is (a, b, c)

        # Convert angles from degrees to radians for computation

        # Extract a, b, c components
        a, b, c = eulerDegs[..., 0], eulerDegs[..., 1], eulerDegs[..., 2]

        # Adjust b to be within 0 to 180 degrees
        mask_b_neg = b < 0
        mask_b_gt_pi = b > 180

        b[mask_b_neg] = -b[mask_b_neg]
        a[mask_b_neg] += 180
        c[mask_b_neg] += 180

        b[mask_b_gt_pi] = 360 - b[mask_b_gt_pi]
        a[mask_b_gt_pi] += 180
        c[mask_b_gt_pi] += 180

        # Normalize a and c to be within the range [0, 2*pi)
        a = a % 360
        c = c % 360

        return a, b, c

    def eulerDegs_to_idx(self, eulerDegs):
        """

        :param eulerDegs: First two angles are the S2 angles and last one is the in-plane angle (e.g., ZYZ as used in cryo-EM)
        :return:
        """
        if isinstance(eulerDegs, torch.Tensor):
            bn = torch
            _hp = hp
            to_long = lambda x: x.long()
        else:
            bn = np
            _hp = hp
            to_long = lambda x: x.astype(np.int64)
            # raise NotImplementedError()

        phi_deg, theta_deg, inplanes = self._move_tilt_to_0_pi(eulerDegs)


        # from scipy.spatial.transform import Rotation as R
        # assert np.isclose(
        #                      R.from_euler("ZYZ", torch.stack([phi_deg, theta_deg, inplanes], -1).reshape(-1, 3),
        #                                   degrees=True).as_matrix(),
        #                      R.from_euler("ZYZ", eulerDegs.reshape(-1, 3), degrees=True).as_matrix(), atol=1e-3).all()


        # Convert adjusted degrees to radians
        theta_rad = bn.deg2rad(theta_deg)
        phi_rad = bn.deg2rad(phi_deg)

        # Get the HEALPix index
        hp_idx = _hp.ang2pix(self.nside, theta_rad, phi_rad)
        inplane_idx = to_long((inplanes / self.degs_step) % self.num_inplane)

        so3_idxs = self._compose_idx(inplane_idx, hp_idx)
        # bad_idxs = torch.where(so3_idxs > self.grid_size)
        assert (so3_idxs < self.grid_size).all()

        return so3_idxs

    def _compose_idx(self, inplane_idx, hp_idx):

        return inplane_idx * self.num_hp + hp_idx

    def _decompose_idx(self, idx):
        inplane_idx = (idx // self.num_hp)
        hp_idx = (idx % self.num_hp)
        if isinstance(idx, torch.Tensor):
            inplane_idx = inplane_idx.long()
            hp_idx = hp_idx.long()
        else:
            inplane_idx = inplane_idx.astype(np.int64)
            hp_idx = hp_idx.astype(np.int64)
        return inplane_idx, hp_idx

    def idx_to_eulerDegs(self, idx):
        if isinstance(idx, torch.Tensor):
            backend = torch
        else:
            backend = np
        inplane_idx, hp_idx = self._decompose_idx(idx)

        eulerDegs = backend.rad2deg(backend.stack(hp.pix2ang(self.nside, hp_idx),-1))
        inplane_degs = (inplane_idx * self.degs_step) % 360
        eulerDegs = backend.stack([eulerDegs[...,1], eulerDegs[...,0], inplane_degs], -1)

        eulerDegs = (eulerDegs + 180) % 360 - 180
        if isinstance(idx, torch.Tensor):
            eulerDegs = eulerDegs.to(torch.float32)
        else:
            eulerDegs = eulerDegs.astype(np.float32)
        return eulerDegs

    def _get_neig_idx(self, idx):
        was_int = False
        if isinstance(idx, int):
            idx = np.array([idx], dtype=np.int64)
            was_int = True
        if len(idx.shape) > 1:
            raise NotImplementedError("Only implemented for len(idx.shape) > 1")
        inplane_idx, hp_idx = self._decompose_idx(idx)
        hp_neigs = np.zeros((idx.shape[0], 9), dtype=np.int64)
        hp_neigs[:, 1:] = hp.get_all_neighbours(self.nside, hp_idx).T
        hp_neigs[:, 0] = idx

        inplane_neigs = np.stack([inplane_idx, (inplane_idx-1)%self.num_inplane, (inplane_idx+1)%self.num_inplane], -1)
        cartesian_prod = _cartesian_product_rowwise(inplane_neigs, hp_neigs)
        goodNeigsMask = (cartesian_prod >= 0).any(-1)
        neigIdxs = self._compose_idx(cartesian_prod[..., 0], cartesian_prod[..., 1])
        neigsList = [row[goodNeigsMask[i]][1:].tolist() for i, row in enumerate(neigIdxs)]
        if was_int:
            neigsList = neigsList[0]
        return neigsList

    def _datadir(self):
        return  os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    def compute_nearest_neighbours(self, k=10, n_jobs=1):

        fname = os.path.join(self._datadir(), f"knn_{self.hp_order}.joblib")
        if os.path.exists(fname):
            return joblib.load(fname)

        self.mainLogger(f"Computing nearest neighbors. For hp_order > 4, it will take a long time (hp_order={self.hp_order})")

        from sklearn.neighbors import NearestNeighbors
        ids = torch.arange(self.grid_size)
        eulerDegs = self.idx_to_eulerDegs(ids)
        rs = R.from_euler(RELION_EULER_CONVENTION, eulerDegs, degrees=True)
        X = rs.as_quat()
        X = X / np.linalg.norm(X, axis=-1, keepdims=True)

        neigh = NearestNeighbors(n_neighbors=k, metric=quaternion_distance, algorithm="ball_tree", n_jobs=n_jobs)
        neigh.fit(X)
        dists, nearest_neigbours = neigh.kneighbors(X)
        out = dict(dists=dists, nearest_neigbours=nearest_neigbours, hp_order=self.hp_order)
        joblib.dump(out, fname)
        return out

    def traverse_so3_sphere(self, orientation_idxs=None, n_jobs=1):

        fname = os.path.join(self._datadir(), f"so3Traversal_{self.hp_order}.joblib")
        if orientation_idxs is None:
            if os.path.isfile(fname):
                order = joblib.load(fname)
                return iter(order)
            orientation_idxs = np.arange(self.grid_size)
            visted_ordered = []
        else:
            visted_ordered = None
        out = self.compute_nearest_neighbours(n_jobs=n_jobs)
        dists, neighbours = out["dists"], out["nearest_neigbours"]
        toCancelMask = ~ np.isin(neighbours, orientation_idxs)
        dists[toCancelMask] = np.inf
        neighbours[toCancelMask] = -1

        visited = set()  # Track visited nodes
        queue = deque()
        queue.appendleft(orientation_idxs[0])
        non_visited = set(orientation_idxs)
        while queue:  # While queue is not empty
            node = queue.pop()
            # print(node)
            if node not in visited:
                visited.add(node)
                non_visited.remove(node)
                yield node
                if visted_ordered is not None:
                    visted_ordered.append(node)
                # Create a list of (distance, child_idx) tuples for unvisited children
                _neigs = [neighbours[node, i] for i in range(1, neighbours.shape[1])] #The first item is the nodeID
                dists_childIdx = [(dists[node, i], n_) for i, n_ in enumerate(_neigs) if
                                  n_ != -1 and not n_ in visited and n_ in non_visited]
                dists_childIdx = [(d,n) for d,n in dists_childIdx if not np.isinf(d)]
                if dists_childIdx:
                    # Sort the list by distance to prioritize closer nodes
                    dists_childIdx.sort(key=lambda x: x[0])
                    for dist, child_idx in dists_childIdx:
                        queue.appendleft(child_idx)
                elif non_visited: #We need to go for the next connected component
                    one_example = non_visited.pop()
                    queue.appendleft(one_example)
                    non_visited.add(one_example)

        if visted_ordered is not None and not os.path.isfile(fname):
            joblib.dump(np.array(visted_ordered), fname)
        # print("Finished!")

def _cartesian_product_rowwise(A, B):

    # Reshape A and B for broadcasting

    A_reshaped = A[:, :, np.newaxis]
    B_reshaped = B[:, np.newaxis, :]

    # Use broadcasting to create the Cartesian product

    A_broadcasted = np.broadcast_to(A_reshaped, (A.shape[0], A.shape[1], B.shape[1]))
    B_broadcasted = np.broadcast_to(B_reshaped, (B.shape[0], A.shape[1], B.shape[1]))

    # Stack along a new axis to create pairs
    result = np.stack((A_broadcasted, B_broadcasted), axis=-1)
    result = result.reshape(result.shape[0], -1, 2)
    return result


def _spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return np.stack([x, y, z], -1)

def _test0():

    so3 = SO3_discretizer(hp_order=16)
    n = 10
    # eulerDegs = torch.rand(n, 3) * 360
    eulerDegs = (torch.rand(n, 3)-0.5) * 360 * 4

    # eulerDegs = torch.FloatTensor(([[ 16.0341, 123.0934, 177.9880],
    #                                 [128.4075,  98.1952, 124.2592],
    #                                 [155.2062,  17.2833,  98.2104],
    #                                 [ 49.5994,   3.3769,  86.3761],
    #                                 [151.0677,  49.6523, 106.8809]]))
    # eulerDegs = torch.FloatTensor(([[ 355.0, 355.0, 355.0],
    #                                 [   0.0, 355.0, 355.0],
    #                                 [95.0, 95.0, 355.0],
    #                                 [0.0, 95.0, 355.0],
    #                                 [ 275.0, 275.0, 275.0],
    #                                 [ 0.0, 275.0, 275.0]]))

    # eulerDegs = torch.FloatTensor([[125.0000, -7.0759, -152.5000], [125.0000, -7.0759, -150.5000]])
    print("eulerDegs\n", np.round(eulerDegs, decimals=3))
    idx = so3.eulerDegs_to_idx(eulerDegs)
    recoveredDegs = so3.idx_to_eulerDegs(idx)
    print("recoveredDegs\n", np.round(recoveredDegs, decimals=3))
    ori_points = np.round(_spherical_to_cartesian(1, torch.deg2rad(eulerDegs[...,1]), torch.deg2rad(eulerDegs[...,0])), decimals=3)
    recov_points = np.round(_spherical_to_cartesian(1, torch.deg2rad(recoveredDegs[...,1]), torch.deg2rad(recoveredDegs[...,0])), decimals=3)

    from scipy.spatial.transform import Rotation as R
    recoverdDegsNorm = R.from_euler("ZYZ", recoveredDegs, degrees=True).as_euler("ZYZ", degrees=True)
    oriDegsNorm = R.from_euler("ZYZ", eulerDegs, degrees=True).as_euler("ZYZ", degrees=True)
    print("oriDegsNorm\n", oriDegsNorm)
    print(recoverdDegsNorm)
    assert np.isclose(ori_points[:,:2], recov_points[:,:2], atol=1e-2).all()
    assert np.isclose(oriDegsNorm, recoverdDegsNorm, atol=1e-2).all()


def _test1():
    so3 = SO3_discretizer(hp_order=4)
    initPoints = np.array([0, 999, 1032, 1221])
    neigsList = so3._get_neig_idx(initPoints)
    for i, init in enumerate(initPoints):
        print(init, "\n", neigsList[i])
        print(so3.idx_to_eulerDegs(init), "\n", so3.idx_to_eulerDegs(np.array(neigsList[i])))
        print("----------------")

if not main_config.projmatching.disable_compile_projectVol:
    #I cannot see a difference between mode='reduce-overhead' and mode='max-autotune', and they are quite close to not compiling
    SO3_discretizer.eulerDegs_to_idx = torch.compile(SO3_discretizer.eulerDegs_to_idx, fullgraph=False, dynamic=True, mode=config.COMPILE_MODE)
    SO3_discretizer.idx_to_eulerDegs = torch.compile(SO3_discretizer.idx_to_eulerDegs, fullgraph=False, dynamic=True, mode=config.COMPILE_MODE)
    # so3_d = SO3_discretizer(5)
    # out_eulerDegs_to_idx = torch._dynamo.explain(so3_d.eulerDegs_to_idx)(so3_d, torch.tensor([[ 0., 0., 0.]]))
    # out_idx_to_eulerDegs = torch._dynamo.explain(so3_d.idx_to_eulerDegs)(so3_d, torch.tensor([[1, 1, 1]]))
    # print(out_eulerDegs_to_idx)
    # print()

def _test2():
    eulersDeg = torch.tensor([[ 15.0884, 1.6385, -59.6726]])
    so3_discretizer = SO3_discretizer(hp_order=5)
    idxs = so3_discretizer.eulerDegs_to_idx(eulersDeg)
    eulersDeg_disc = so3_discretizer.idx_to_eulerDegs(idxs)
    from torchCryoAlign.metrics import euler_degs_diff
    print(eulersDeg)
    print(eulersDeg_disc)
    print(euler_degs_diff(eulersDeg, eulersDeg_disc))
    print()

def _test3(n=10000, m=100):
    for i in range(m):
        eulersDeg = torch.rand(n, 3)
        so3_discretizer = SO3_discretizer(hp_order=5)
        idxs = so3_discretizer.eulerDegs_to_idx(eulersDeg)
        eulersDeg_disc = so3_discretizer.idx_to_eulerDegs(idxs)
        # from torchCryoAlign.metrics import _euler_degs_diff
        # print(_euler_degs_diff(eulersDeg, eulersDeg_disc))
        return idxs

if __name__ == "__main__":
    # _test0()
    # _test1()
    # _test2()

    _test3(n=1)
    print("n=1 finished")
    _test3()
    print("n=1000 finished")
    out = _test3()
    print("n=1000 2 finished")
    out = timeit.timeit(_test3, number=1000, )
    print(out)