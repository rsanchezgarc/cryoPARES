from typing import Union, Any, Dict, Tuple, List

import torch
from torch import nn, ScriptModule

from cryoPARES.configManager.inject_defaults import inject_defaults_from_config, CONFIG_PARAM
from cryoPARES.configs.mainConfig import main_config
from cryoPARES.projmatching.projMatching import ProjectionMatcher
from cryoPARES.reconstruction.reconstruction import Reconstructor
from cryoPARES.inference.nnetWorkers.tensorDataBuffer import StreamingBuffer
from cryoPARES.models.model import RotationPredictionMixin
from cryoPARES.constants import (BATCH_IDS_NAME, BATCH_PARTICLES_NAME, BATCH_ORI_IMAGE_NAME,
                                 BATCH_ORI_CTF_NAME, BATCH_MD_NAME)

class InferenceModel(RotationPredictionMixin, nn.Module):
    @inject_defaults_from_config(main_config.inference, update_config_with_args=False)
    def __init__(self,
                 so3model: Union[nn.Module, ScriptModule],
                 scoreNormalizer: Union[nn.Module, ScriptModule, None],
                 normalizedScore_thr: float | None,
                 localRefiner: ProjectionMatcher | None,
                 reconstructor: Reconstructor | None = None,
                 before_refiner_buffer_size: int = CONFIG_PARAM(),
                 return_top_k: int = 1
                 ):
        super().__init__()
        self.__init_mixin__()
        self.so3model = so3model
        self.symmetry = self.so3model.symmetry
        self.scoreNormalizer = scoreNormalizer
        self.normalizedScore_thr = normalizedScore_thr
        self.localRefiner = localRefiner
        self.reconstructor = reconstructor
        self.buffer_size = before_refiner_buffer_size
        self.return_top_k = return_top_k

        if self.normalizedScore_thr is not None:
            self.buffer = StreamingBuffer(
                buffer_size=before_refiner_buffer_size,
                processing_fn=self._run_stage2,
            )
        else:
            self.buffer = None

    def _firstforward(self, imgs, top_k):
        _top_k = 10 if top_k < 10 else top_k
        _, _, _, pred_rotmats, maxprobs = self.so3model(imgs, _top_k)
        if self.scoreNormalizer:
            norm_nn_score = self.scoreNormalizer(pred_rotmats[:, 0, ...], maxprobs[:,:10].sum(1))
        else:
            norm_nn_score = torch.nan * torch.ones_like(maxprobs[:,0])
        pred_rotmats = pred_rotmats[:,:top_k]
        maxprobs = maxprobs[:,:top_k]
        return pred_rotmats, maxprobs, norm_nn_score

    def forward(self, ids, imgs, fullSizeImg, fullSizeCtfs, top_k):

        pred_rotmats, maxprobs, norm_nn_score = self._firstforward(imgs, top_k)
        if self.normalizedScore_thr is not None:
            passing_mask = (norm_nn_score > self.normalizedScore_thr).squeeze()
            if not passing_mask.any():
                return None
            valid_indices = torch.where(passing_mask)[0].to("cpu", non_blocking=False)
            #fimg, ctf, rotmats
            batch_to_add = {
                'imgs': fullSizeImg[passing_mask],
                'ctfs': fullSizeCtfs[passing_mask],
                'rotmats': pred_rotmats[passing_mask],
                'maxprobs': maxprobs[passing_mask],
                'norm_nn_score': norm_nn_score[passing_mask],
                'ids': [ids[i] for i in valid_indices.tolist()],
            }
        else:
            batch_to_add = {
                'ids': ids,
                'imgs': fullSizeImg,
                'ctfs': fullSizeCtfs,
                'rotmats': pred_rotmats,
                'maxprobs': maxprobs,
                'norm_nn_score': norm_nn_score,
            }
        if self.localRefiner is not None:
            if self.buffer is not None:
                out = self.buffer.add_batch(batch_to_add)
            else:
                out = self._run_stage2(**batch_to_add)
        else:
            out = (batch_to_add['ids'], batch_to_add['rotmats'], None, batch_to_add['maxprobs'],
                    batch_to_add['norm_nn_score'])
            if self.reconstructor is not None:
                raise NotImplementedError("Error, at the moment, local refinement is needed before"
                                          "reconstruction")
        return out

    def forward_without_buffer(self, ids, imgs, fullSizeImg, fullSizeCtfs, top_k):
        """
        A more efficient forward pass that bypasses the streaming buffer.
        This is useful when no z-score filtering is applied and we want to process
        batches directly without buffering.
        """
        pred_rotmats, maxprobs, norm_nn_score = self._firstforward(imgs, top_k)
        if self.localRefiner is not None:
            kwargs = {
                'imgs': fullSizeImg,
                'ctfs': fullSizeCtfs,
                'rotmats': pred_rotmats,
                'maxprobs': maxprobs,
                'norm_nn_score': norm_nn_score,
                'ids': ids}
            out = self._run_stage2(**kwargs)
        else:
            out = (ids, pred_rotmats, None, maxprobs, norm_nn_score)
            if self.reconstructor is not None:
                raise NotImplementedError("Error, at the moment, local refinement is needed before"
                                          "reconstruction")
        return out

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        ids = batch[BATCH_IDS_NAME]
        imgs = batch[BATCH_PARTICLES_NAME]
        fullSizeImg = batch[BATCH_ORI_IMAGE_NAME]
        fullSizeCtfs = batch[BATCH_ORI_CTF_NAME]
        # metadata = batch[BATCH_MD_NAME]
        if self.normalizedScore_thr is not None:
            results = self.forward(ids, imgs, fullSizeImg, fullSizeCtfs, top_k=self.return_top_k)
        else:

            results = self.forward_without_buffer(ids, imgs, fullSizeImg, fullSizeCtfs, top_k=self.return_top_k)
        ## return ids, pred_rotmats, pred_shifts, maxprobs, norm_nn_score
        # from scipy.spatial.transform import Rotation
        # from cryoPARES.constants import  BATCH_MD_NAME, RELION_IMAGE_FNAME,RELION_ANGLES_NAMES
        # import pandas as pd
        # print(pd.DataFrame(batch[BATCH_MD_NAME])[[RELION_IMAGE_FNAME] + RELION_ANGLES_NAMES])
        # print(Rotation.from_matrix(results[1][:,0,...].detach().cpu()).as_euler("ZYZ", degrees=True).round(0))
        # breakpoint()
        return results

    def _run_stage2(self, **kwargs): #TODO: This could be speeded-up if localRefiner and reconstructor
                                        #TODO: are fed with fourier transformed images
        #tensors.keys() -> imgs, ctfs, rotmats, maxprobs, norm_nn_score
        (maxCorrs, predRotMats, predShiftsAngsXY,
         comparedWeight) = self.localRefiner.forward(kwargs['imgs'], kwargs['ctfs'], kwargs['rotmats'])
        score = torch.where(torch.isnan(comparedWeight), kwargs['maxprobs']*0.5, kwargs['maxprobs'] * comparedWeight)

        if self.reconstructor is not None:
            self.reconstructor._backproject_batch(kwargs['imgs'], kwargs['ctfs'],
                           rotMats=predRotMats, hwShiftAngs=predShiftsAngsXY.flip(-1),
                                                  confidence=None, zyx_matrices=False)

        return kwargs['ids'], predRotMats, predShiftsAngsXY, score, kwargs['norm_nn_score']

    def flush(self):
        if self.buffer:
            results = self.buffer.flush()
            return results
        else:
            return None

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
    normalizedScore_thr = None #-50

    localRefiner = ProjectionMatcher(
        reference_vol="/home/sanchezg/cryo/data/preAlignedParticles/EMPIAR-10166/data/allparticles_reconstruct.mrc",
        grid_distance_degs=10, grid_step_degs=5)
    model = InferenceModel(so3Model, percentilemodel, normalizedScore_thr, localRefiner, before_refiner_buffer_size=1)
    out = model.forward(ids, imgs, imgs[:, 0, ...], ctfs, top_k)
    print(out)
    print("First was out")
    out = model.flush()
    print(out)
    print("Done")



if __name__ == "__main__":
    _test()