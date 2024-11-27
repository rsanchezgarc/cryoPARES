from typing import Union, Any

import torch
import lightning.pytorch as pl

from cryoPARES.datamanager.datamanager import get_example_random_batch
from cryoPARES.geometry.metrics_angles import rotation_error_rads
from cryoPARES.models.image2sphere.image2sphere import Image2Sphere


class PlModel(pl.LightningModule):
    def __init__(self, symmetry: str, num_augmented_copies_per_batch:int): #TODO: num_augmented_copies_per_batch should be in a config

        super().__init__()

        self.symmetry = symmetry
        self.num_augmented_copies_per_batch = num_augmented_copies_per_batch

        i2s = self.build_components(symmetry, num_augmented_copies_per_batch)
        self.model = i2s

    @staticmethod
    def build_components(symmetry, num_augmented_copies_per_batch): #TODO: num_augmented_copies_per_batch should be in a config
        return Image2Sphere(
                         symmetry=symmetry,
                         # num_augmented_copies_per_batch=num_augmented_copies_per_batch
        )
    def _step(self, batch, batch_idx):

        idd, imgs, (rotMats, shifts, conf), metadata = self.resolve_batch(batch)
        (wD, rotMat_logits, pred_rotmat_ids,
         pred_rotmats, maxprobs), loss, error_rads = self.model.forward_and_loss(imgs, rotMats, conf)
        return loss, error_rads, pred_rotmats, maxprobs

    def training_step(self, batch, batch_idx):
        loss, ang_error, pred_rotmats, maxprobs = self._step(batch, batch_idx)
        loss = loss.mean()
        self.log("loss", loss, prog_bar=True, batch_size=pred_rotmats.shape[0])
        self.log("error_degs", torch.rad2deg(ang_error.mean()), prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=pred_rotmats.shape[0], sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, ang_error, pred_rotmats, maxprobs = self._step(batch, batch_idx)
        loss = loss.mean()
        self.log("val_loss", loss, prog_bar=True, on_epoch=True,
                 batch_size=pred_rotmats.shape[0], sync_dist=True)
        self.log("val_error_degs", torch.rad2deg(ang_error.mean()), prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=pred_rotmats.shape[0], sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):

        idd, imgs, (rotMats, shifts, conf), metadata = self.resolve_batch(batch)
        wD, rotMat_logits, pred_rotmat_id, pred_rotmats, maxprobs = self.model(imgs)
        if rotMats is not None: #TODO: we need to refactor this to remove the if
            errors = rotation_error_rads(rotMats, pred_rotmats)
            errors = torch.rad2deg(errors)
            metadata["pred_degs_error"] = errors.detach().cpu().numpy()
        return idd, (pred_rotmats, maxprobs), metadata


    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.predict_step(*args, **kwargs)

    def resolve_batch(self, batch):

        idd, imgs, (rotMats, shifts, conf), metadata = batch

        # # #Modified for the old datamanager
        # imgs = batch.particle
        # idd = str(batch.itemIdx)
        # rotMats = batch.poseRepresentation
        # shifts = batch.shiftsFraction * 1.5 #1.5 is the sampling rate, only for debug here #TODO: Remove this
        # conf = batch.poseProbability

        return idd, imgs, (rotMats, shifts, conf), metadata


def _update_config_for_test():
    from cryoPARES.configs.mainConfig import main_config
    main_config.models.image2sphere.lmax = 6
    main_config.models.image2sphere.so3components.i2sprojector.sphere_fdim = 128
    main_config.models.image2sphere.so3components.i2sprojector.rand_fraction_points_to_project = 1.
    main_config.models.image2sphere.so3components.i2sprojector.hp_order = 2
    main_config.models.image2sphere.so3components.s2conv.hp_order = 2
    main_config.models.image2sphere.so3components.so3ouptutgrid.hp_order = 3

if __name__ == "__main__":
    b = 1
    batch = get_example_random_batch(b)
    plmodel = PlModel(symmetry="c1", num_augmented_copies_per_batch=1)
    out = plmodel(batch, batch_idx=0)
    print()