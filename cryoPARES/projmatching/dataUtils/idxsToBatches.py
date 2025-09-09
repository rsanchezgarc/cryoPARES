import time
from typing import Literal

import numpy
import numpy as np
import torch
from more_itertools import chunked

from .starUtils import traverse_node_subset
from ..loggers import getWorkerLogger
from ..numbaUtils import typed_dict_read_keys, typed_dict_flatten_keys_vals_asarray


class IdxsBatcher():
    def __init__(self, so3_discretizer, pose_to_particleIdx, n_cpus,
                 sorting_method: Literal["traversal", "idx_sorting", "none"]="traversal", verbose=False):

        self.verbose = verbose
        n_max_poses = so3_discretizer.grid_size
        keys = typed_dict_read_keys(pose_to_particleIdx)
        self.pose_to_particleIdx = pose_to_particleIdx
        self.logger = getWorkerLogger(verbose)

        #valid_poseIdxs_renumbered
        to = time.time()
        self.logger.info("Sorting nodes...")
        if sorting_method == "traversal":
            sorted_keys = np.fromiter(so3_discretizer.traverse_so3_sphere(keys, n_jobs=n_cpus), dtype=np.int64) #unique keys, in the order that should be traversed
        elif sorting_method == "idx_sorting":
            sorted_keys = np.sort(keys)
        elif sorting_method == "traversal_precomputed":
            full_nodes_traversal_order = np.fromiter(so3_discretizer.traverse_so3_sphere(n_jobs=n_cpus), dtype=np.int64)
            sorted_keys = traverse_node_subset(full_nodes_traversal_order, keys)
        else:
            sorted_keys = keys

        t1 = time.time()
        self.logger.info(f"Sorting nodes took {round(t1-to, 2)} s")

        old2newMap = numpy.ones(n_max_poses, dtype=np.int64) + n_max_poses  #We add n_max_poses to cause index error if we are using unexpected indices
        old2newMap[sorted_keys] = np.arange(sorted_keys.shape[0]) #  m_i is the renumbered id for old i-th position
        keys, vals = typed_dict_flatten_keys_vals_asarray(pose_to_particleIdx)
        valid_poseIdxs_renumbered = old2newMap[keys]

        self.renumbered_to_ori_poseIdxs = torch.as_tensor(sorted_keys) #This is used to map back self.valid_poseIdxs_renumbered to the original poses

        sorted_idxs = np.argsort(valid_poseIdxs_renumbered)
        self.valid_poseIdxs_renumbered = torch.as_tensor(valid_poseIdxs_renumbered[sorted_idxs])
        self.associated_particlIdxs = torch.as_tensor(vals[sorted_idxs])

        # for i in range(sorted_keys.shape[0]):
        #     print(i)
        #     assert  np.unique(self.valid_poseIdxs_renumbered[np.where(keys == sorted_keys[i])]) == i
        # print()

    def yield_batchIdxs(self, batch_size):

        for batchIdxs in chunked(range(self.valid_poseIdxs_renumbered.shape[0]), n=batch_size):
            flatten_poses_idxs = self.valid_poseIdxs_renumbered[batchIdxs]
            flatten_img_idxs = self.associated_particlIdxs[batchIdxs]
            yield flatten_poses_idxs, flatten_img_idxs

    @property
    def maxindex(self):
        return self.valid_poseIdxs_renumbered[-1]