import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch.optim.lr_scheduler')
import torch
import pytorch_lightning as pl
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch.optim import Optimizer
from typing import Union, Any, Callable, Optional

from cryoPARES.configs.mainConfig import main_config
from cryoPARES.constants import BATCH_IDS_NAME, BATCH_PARTICLES_NAME, BATCH_POSE_NAME, BATCH_MD_NAME
from cryoPARES.datamanager.datamanager import get_example_random_batch
from cryoPARES.geometry.metrics_angles import rotation_error_rads
from cryoPARES.models.image2sphere.image2sphere import Image2Sphere
from cryoPARES.utils.plUtils import is_pylig2


#TODO: inject config for the optimizer?
class PlModel(pl.LightningModule):
    def __init__(self, lr: float, symmetry: str, num_augmented_copies_per_batch:int):

        super().__init__()

        self.lr = lr
        self.symmetry = symmetry
        self.num_augmented_copies_per_batch = num_augmented_copies_per_batch

        i2s = self.build_components(symmetry, num_augmented_copies_per_batch)
        self.model = i2s

        self.warmup_epochs = main_config.train.warmup_n_epochs
        if is_pylig2:
            self.optimizer_step = self.optimizer_step_v2
        else:
            self.optimizer_step = self.optimizer_step_v1

    @staticmethod
    def build_components(symmetry, num_augmented_copies_per_batch):
        return Image2Sphere(symmetry=symmetry, num_augmented_copies_per_batch=num_augmented_copies_per_batch)
    def _step(self, batch, batch_idx):

        idd, imgs, (rotMats, shifts, conf), metadata = self.resolve_batch(batch)
        (wD, rotMat_logits, pred_rotmat_ids,
         pred_rotmats, maxprobs), loss, error_rads = self.model.forward_and_loss(imgs, rotMats, conf)
        return loss, error_rads, pred_rotmats, maxprobs

    def training_step(self, batch, batch_idx):
        loss, ang_error, pred_rotmats, maxprobs = self._step(batch, batch_idx)
        loss = loss.mean()
        self.log("loss", loss, prog_bar=True, batch_size=pred_rotmats.shape[0])
        self.log("geo_degs", torch.rad2deg(ang_error.mean()), prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=pred_rotmats.shape[0], sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, ang_error, pred_rotmats, maxprobs = self._step(batch, batch_idx)
        loss = loss.mean()
        self.log("val_loss", loss, prog_bar=True, on_epoch=True,
                 batch_size=pred_rotmats.shape[0], sync_dist=True)
        self.log("val_geo_degs", torch.rad2deg(ang_error.mean()), prog_bar=True, on_step=False, on_epoch=True,
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

        idd = batch[BATCH_IDS_NAME]
        imgs = batch[BATCH_PARTICLES_NAME]
        (rotMats, xyShiftAngs, conf) = batch[BATCH_POSE_NAME]
        metadata = batch[BATCH_MD_NAME]
        return idd, imgs, (rotMats, xyShiftAngs, conf), metadata


    def optimizer_step_v1(self, epoch: int, batch_idx: int,
                            optimizer: Union[Optimizer, LightningOptimizer],
                            optimizer_idx: int = 0,
                            optimizer_closure: Optional[Callable[[], Any]] = None,
                            on_tpu: bool = False,
                            using_lbfgs: bool = False):

        optimizer.step(closure=optimizer_closure)
        if self.warmup_epochs:
            n_steps_for_warmup = self.fraction_for_warmup * self.trainer.estimated_stepping_batches
            if self.trainer.global_step < n_steps_for_warmup:
                lr_scale = min(1., float(self.trainer.global_step + 1) / n_steps_for_warmup)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr_scale * self.learning_rate
        optimizer.step(closure=optimizer_closure)

    #THIS IS FOR pytorch_lightning v2.
    def optimizer_step_v2(self, epoch: int, batch_idx: int,
                            optimizer: Union[Optimizer, LightningOptimizer],
                            optimizer_closure: Optional[Callable[[], Any]] = None):

        optimizer.step(closure=optimizer_closure)
        # print(f"\nepoch {epoch}, batch_idx {batch_idx} \n")
        if self.warmup_epochs:
            n_steps_for_warmup = self.fraction_for_warmup * self.trainer.estimated_stepping_batches
            if self.trainer.global_step < n_steps_for_warmup:
                lr_scale = min(1., float(self.trainer.global_step + 1) / n_steps_for_warmup)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr_scale * self.learning_rate

    def configure_optimizers(self):
        lr = self.lr
        configTrain = main_config.train
        optClass = getattr(torch.optim, configTrain.default_optimizer)
        if self.warmup_epochs is None:
            opt = optClass(self.parameters(), lr=lr, weight_decay=configTrain.weight_decay)

        else:
            opt = optClass(self.parameters(), lr=self.lr, weight_decay=configTrain.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, verbose=True,
                                                      factor=0.5,
                                                      threshold=5e-4,
                                                      min_lr=lr * configTrain.min_learning_rate_factor,
                                                      cooldown=1,
                                                      patience=configTrain.patient_reduce_lr_plateau_n_epochs)
        conf = {
            'optimizer': opt,
            'lr_scheduler': lr_scheduler,
            'monitor': configTrain.monitor_metric
        }
        return conf


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
    plmodel = PlModel(lr=1-5, symmetry="c1", num_augmented_copies_per_batch=1)
    out = plmodel(batch, batch_idx=0)
    print()