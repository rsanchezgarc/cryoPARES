from typing import Union, Any, Dict, Tuple, List

import torch
from torch import nn, ScriptModule

from cryoPARES.configs.mainConfig import main_config
from cryoPARES.projmatching.projMatching import ProjectionMatcher
from tensorDataBuffer import StreamingBuffer
from cryoPARES.models.model import RotationPredictionMixin
from cryoPARES.constants import (BATCH_IDS_NAME, BATCH_PARTICLES_NAME, BATCH_ORI_IMAGE_NAME,
                                 BATCH_ORI_CTF_NAME, BATCH_MD_NAME)

class InferenceModel(RotationPredictionMixin, nn.Module):
    def __init__(self,
                 so3model: Union[nn.Module, ScriptModule],
                 scoreNormalizer: Union[nn.Module, ScriptModule],
                 normalizedScore_thr: float,
                 localRefiner: ProjectionMatcher,
                 buffer_size: int = 4, #TODO: Move this to CONFIG
                 return_top_k: int = 1
                 ):
        super().__init__()
        self.__init_mixin__()
        self.so3model = so3model
        self.symmetry = self.so3model.symmetry
        self.scoreNormalizer = scoreNormalizer
        self.normalizedScore_thr = normalizedScore_thr
        self.localRefiner = localRefiner
        self.buffer_size = buffer_size
        self.return_top_k = return_top_k

        self.buffer = StreamingBuffer(
            buffer_size=buffer_size,
            processing_fn=self._run_stage2,
        )

    def forward(self, ids, imgs, fullSizeImg, fullSizeCtfs, top_k):
        if self.normalizedScore_thr:
            _top_k = 10 if top_k < 10 else top_k
        else:
            _top_k = top_k
        _, _, _, pred_rotmats, maxprobs = self.so3model(imgs, _top_k)
        #TODO: Check if we are injecting the top10 scores
        norm_nn_score = self.scoreNormalizer(pred_rotmats[:, 0, ...], maxprobs[:,:10].sum(1))
        pred_rotmats = pred_rotmats[:,:top_k]
        maxprobs = maxprobs[:,:top_k]
        if self.normalizedScore_thr is not None:
            passing_mask = (norm_nn_score > self.normalizedScore_thr).squeeze()
            if not passing_mask.any():
                return None
            valid_indices = torch.where(passing_mask)[0].cpu().tolist()
            #fimg, ctf, rotmats
            batch_to_add = {
                'ids': [ids[i] for i in valid_indices],
                'imgs': fullSizeImg[passing_mask],  # Use the device-specific tensor
                'ctfs': fullSizeCtfs[passing_mask],
                'rotmats': pred_rotmats[passing_mask],
                'maxprobs': maxprobs[passing_mask],
                'norm_nn_score': norm_nn_score[passing_mask],
            }
            if self.localRefiner is not None:
                out = self.buffer.add_batch(batch_to_add)
                return out
            else:
                return (batch_to_add['ids'], batch_to_add['rotmats'], None, batch_to_add['maxprobs'],
                        batch_to_add['norm_nn_score'])
        else:
            return ids, pred_rotmats, None, maxprobs, norm_nn_score

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        ids = batch[BATCH_IDS_NAME]
        imgs = batch[BATCH_PARTICLES_NAME]
        fullSizeImg = batch[BATCH_ORI_IMAGE_NAME]
        fullSizeCtfs = batch[BATCH_ORI_CTF_NAME]
        # metadata = batch[BATCH_MD_NAME]
        results = self.forward(ids, imgs, fullSizeImg, fullSizeCtfs, top_k=self.return_top_k)
        # return ids, pred_rotmats, pred_shifts, maxprobs, norm_nn_score, metadata
        return results


    def _run_stage2(self, tensors, md):
        #tensors.keys() -> ids, imgs, ctfs, rotmats, maxprobs, norm_nn_score
        (maxCorrs, predRotMats, predShiftsAngs,
         comparedWeight) = self.localRefiner.forward(tensors['imgs'], tensors['ctfs'], tensors['rotmats'])
        score = tensors['maxprobs'] * comparedWeight
        return md['ids'], predRotMats, predShiftsAngs, score, tensors['norm_nn_score']

    def flush(self):
        return self.buffer.flush()

def _update_config_for_test():
    main_config.models.image2sphere.lmax = 6
    main_config.models.image2sphere.so3components.i2sprojector.sphere_fdim = 512
    main_config.models.image2sphere.so3components.i2sprojector.rand_fraction_points_to_project = 1.
    main_config.models.image2sphere.so3components.i2sprojector.hp_order = 2
    main_config.models.image2sphere.so3components.s2conv.hp_order = 2
    main_config.models.image2sphere.so3components.s2conv.f_out = 16
    main_config.models.image2sphere.so3components.so3outputgrid.hp_order = 3

    main_config.datamanager.particlesdataset.desired_image_size_px = 336
    main_config.models.image2sphere.so3components.i2sprojector.rand_fraction_points_to_project = 1
    main_config.models.image2sphere.label_smoothing = 0.1

def _test():
    _update_config_for_test()
    from cryoPARES.models.directionalNormalizer.directionalNormalizer import DirectionalPercentileNormalizer
    from cryoPARES.models.image2sphere.image2sphere import Image2Sphere
    from cryoPARES.projmatching.projMatching import ProjectionMatcher
    from cryoPARES.datamanager.datamanager import get_example_random_batch
    from cryoPARES.constants import BATCH_PARTICLES_NAME, BATCH_IDS_NAME
    from cryoPARES.datamanager.ctf.rfft_ctf import compute_ctf_rfft

    from scipy.spatial.transform import Rotation

    b = 4
    example_batch = get_example_random_batch(b, seed=42)
    imgs = example_batch[BATCH_PARTICLES_NAME]
    ids = example_batch[BATCH_IDS_NAME]

    ctfs = compute_ctf_rfft(imgs.shape[-2], sampling_rate=1.27, dfu=5000, dfv=5000, dfang=0,
                             volt=300, cs=2.7, w=0.1, phase_shift=0, bfactor=None, fftshift=True,
                             device="cpu").unsqueeze(0).expand(b,-1,-1)

    symmetry = "C1"
    top_k = 1

    so3Model = Image2Sphere(symmetry=symmetry, num_augmented_copies_per_batch=1)
    percentilemodel = DirectionalPercentileNormalizer(symmetry=symmetry)
    percentilemodel.fit(torch.FloatTensor(Rotation.random(5000).as_matrix()), torch.rand(5000))
    normalizedScore_thr = -50

    localRefiner = ProjectionMatcher(
        reference_vol="/home/sanchezg/cryo/data/preAlignedParticles/EMPIAR-10166/data/allparticles_reconstruct.mrc",
        grid_distance_degs=10, grid_step_degs=5)
    model = InferenceModel(so3Model, percentilemodel, normalizedScore_thr, localRefiner, buffer_size=1)
    out = model.forward(ids, imgs, imgs[:, 0, ...], ctfs, top_k)
    print(out)
    print("First was out")
    out = model.flush()
    print(out)
    print("Done")



if __name__ == "__main__":
    _test()