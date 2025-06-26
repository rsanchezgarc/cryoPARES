from typing import Union, Any, Dict, Tuple, List

import torch
from torch import nn, ScriptModule

from cryoPARES.geometry.metrics_angles import rotation_error_rads
from cryoPARES.models.model import RotationPredictionMixin


class InferenceModel(RotationPredictionMixin, nn.Module):
    def __init__(self,
                 so3model: Union[nn.Module, ScriptModule],
                 scoreNormalizer: Union[nn.Module, ScriptModule],
                 top_k: int,
                 ):
        super().__init__()
        self.__init_mixin__()
        self.so3model = so3model
        self.symmetry = self.so3model.symmetry
        self.scoreNormalizer = scoreNormalizer
        self.top_k = top_k

    def forward(self, imgs, top_k):
        wD, rotMat_logits, pred_rotmat_id, pred_rotmats, maxprobs = self.so3model(imgs, top_k)
        #TODO: Check if we are injecting the top10 scores
        norm_score = self.scoreNormalizer(pred_rotmats, maxprobs)

        return wD, rotMat_logits, pred_rotmat_id, pred_rotmats, maxprobs, norm_score


    def predict_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0):
        idd, imgs, poses, metadata = self.resolve_batch(batch)
        wD, rotMat_logits, pred_rotmat_id, pred_rotmats, maxprobs, norm_score = self(imgs, self.top_k)

        pred_shifts = pred_rotmats.new_full((pred_rotmats.shape[0], self.top_k, 2), torch.nan) #TODO: Predict this as well
        shifts_probs = maxprobs * torch.nan

        if poses is not None:
            (rotMats, xyShiftAngs, conf) = poses
            errors = rotation_error_rads(rotMats, pred_rotmats[:,0,...])
            errors = torch.rad2deg(errors)
            errors = errors.detach().cpu()
        else:
            errors = None
        return idd, (pred_rotmats, maxprobs, norm_score), (pred_shifts, shifts_probs), errors, metadata