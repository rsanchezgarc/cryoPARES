from typing import Union

import torch
import lightning.pytorch as pl
from fontTools.misc.cython import returns

from cryoPARES.models.image2sphere.image2sphere import Image2Sphere


class PlModel(pl.LightningModule):
    def __init__(self, symmetry: str, num_augmented_copies_per_batch:int):

        super().__init__()

        self.symmetry = symmetry
        self.num_augmented_copies_per_batch = num_augmented_copies_per_batch


        self.model = self.build_components(symmetry, num_augmented_copies_per_batch)

    @staticmethod
    def build_components(symmetry, num_augmented_copies_per_batch):
        return Image2Sphere(
                         symmetry=symmetry,
                         # num_augmented_copies_per_batch=num_augmented_copies_per_batch
        )
    def _step(self, batch, batch_idx):

        idd, imgs, (rotMats, shifts, conf), metadata = self.resolve_batch(batch)
        loss, error_rads, pred_rotmats, maxprob, probs = self.model.forward_and_loss(imgs, rotMats, conf)
        return loss, error_rads, pred_rotmats, maxprob, probs

    def training_step(self, batch, batch_idx):
        loss, ang_error, pred_rotmats, maxprob, probs = self._step(batch, batch_idx)
        loss = loss.mean()
        self.log("loss", loss, prog_bar=True, batch_size=pred_rotmats.shape[0])
        self.log("error_degs", torch.rad2deg(ang_error.mean()), prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=pred_rotmats.shape[0], sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, ang_error, pred_rotmats, maxprob, probs = self._step(batch, batch_idx)
        loss = loss.mean()
        self.log("val_loss", loss, prog_bar=True, on_epoch=True,
                 batch_size=pred_rotmats.shape[0], sync_dist=True)
        self.log("val_error_degs", torch.rad2deg(ang_error.mean()), prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=pred_rotmats.shape[0], sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):

        idd, imgs, (rotMats, shifts, conf), metadata = self.resolve_batch(batch)
        grid_signal, pred_rotmats, maxprob, probs = self.model(imgs)
        if rotMats is not None:
            errors = self.model.rotation_error_rads(rotMats, pred_rotmats)
            errors = torch.rad2deg(errors)
            metadata["pred_degs_error"] = errors.detach().cpu().numpy()
        return idd, (pred_rotmats, maxprob), metadata


    def resolve_batch(self, batch):

        idd, imgs, (rotMats, shifts, conf), metadata = batch

        # # #Modified for the old datamanager
        # imgs = batch.particle
        # idd = str(batch.itemIdx)
        # rotMats = batch.poseRepresentation
        # shifts = batch.shiftsFraction * 1.5 #1.5 is the sampling rate, only for debug here #TODO: Remove this
        # conf = batch.poseProbability

        return idd, imgs, (rotMats, shifts, conf), metadata