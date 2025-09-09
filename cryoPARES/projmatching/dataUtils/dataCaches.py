
from ..tensorCache.tensorListLRUCache import TensorListLRUCache


def create_particles_cache(particlesDataSet, l1_cache_size, l2_cache_size, l1_device):

    onePart, oneCtf = particlesDataSet[0][1:3]
    assert  str(particlesDataSet.device) == "cpu", "Error, the particlesDataSet needs to be in the CPU device"
    class TensorListCachePartsCtfL2(TensorListLRUCache): #This is the CPU cache
            def compute_idxs(self, idxs):
                # print(f"New particles {len(idxs)}")
                parts, ctfs = particlesDataSet[idxs][1:3]
                return [parts.contiguous(), ctfs.contiguous()]


    data_cache_l2 = TensorListCachePartsCtfL2(cache_size=l2_cache_size, max_index=particlesDataSet.n_partics,
                                         tensor_shapes=[onePart.shape, oneCtf.shape],
                                         dtypes=[onePart.dtype, oneCtf.dtype],
                                         data_device="cpu")

    class TensorListCachePartsCtfL1(TensorListLRUCache): #This is the GPU cache
            def compute_idxs(self, idxs):
                return [x.to(self.data_device, non_blocking=True) for x in data_cache_l2[idxs]]

    data_cache_l1 = TensorListCachePartsCtfL1(cache_size=l1_cache_size, max_index=particlesDataSet.n_partics,
                                         tensor_shapes=[onePart.shape, oneCtf.shape],
                                         dtypes=[onePart.dtype, oneCtf.dtype],
                                         data_device=l1_device)

    return data_cache_l1


# from torch.utils.data import Dataset
# import torch
#
# class PoseIdxParticleDataset(Dataset): #TODO: Remove this
#     def __init__(self, particlesDataSet, valid_poseIdxs_renumbered, associated_particlIdxs, device="cpu"):
#         self.valid_poseIdxs_renumbered = valid_poseIdxs_renumbered
#         self.associated_particlIdxs = associated_particlIdxs
#         self.particles = particlesDataSet.dataDict["imgs_reprep"]
#         self.ctfs = particlesDataSet.dataDict["ctfs"]
#         self.device = torch.device(device)
#
#     def __len__(self):
#         return len(self.valid_poseIdxs_renumbered)
#
#     def __getitem__(self, index):
#
#         pose_idx = self.valid_poseIdxs_renumbered[index]
#         particle_idx = self.associated_particlIdxs[index]
#
#         particle = self.particles[particle_idx].to(self.device, non_blocking=True)
#         ctf = self.ctfs[particle_idx].to(self.device, non_blocking=True)
#
#         return pose_idx, particle_idx, (particle, ctf)

