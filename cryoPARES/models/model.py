import os.path
import warnings

from tqdm import tqdm

from cryoPARES import constants
from torch import Tensor, nn, ScriptModule
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore', category=UserWarning, module='torch.optim.lr_scheduler')
import torch
import pytorch_lightning as pl
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch.optim import Optimizer
from typing import Union, Any, Callable, Optional, Dict, Tuple, List
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image

from cryoPARES.configs.mainConfig import main_config
from cryoPARES.datamanager.datamanager import get_example_random_batch
from cryoPARES.geometry.metrics_angles import rotation_error_rads
from cryoPARES.models.image2sphere.image2sphere import Image2Sphere
from cryoPARES.utils.plUtils import is_pylig2


class RotationPredictionMixin:
    def __init_mixin__(self):
        """Initialize mixin-specific attributes. Call this in the main class __init__."""
        from cryoPARES.constants import BATCH_IDS_NAME, BATCH_PARTICLES_NAME, BATCH_POSE_NAME, BATCH_MD_NAME
        self.BATCH_IDS_NAME = BATCH_IDS_NAME
        self.BATCH_PARTICLES_NAME = BATCH_PARTICLES_NAME
        self.BATCH_POSE_NAME = BATCH_POSE_NAME
        self.BATCH_MD_NAME = BATCH_MD_NAME

    def _visualize_rotmats(self, pred_rotmats, gt_rotmats, error_degs, max_samples=8, partition="val"):
        """
        Visualize predicted and ground truth rotation matrices.
        Uses Agg backend temporarily to avoid X server requirements.
        """
        # Save the current backend
        import matplotlib
        current_backend = matplotlib.get_backend()

        try:
            # Temporarily switch to Agg backend
            matplotlib.use('Agg', force=True)

            # Detach and move to CPU for visualization
            pred_rotmats_cpu = pred_rotmats[:, 0].detach().cpu()  # Only use top-1 prediction
            gt_rotmats_cpu = gt_rotmats.detach().cpu()

            # Calculate angular errors
            errors_deg = error_degs.detach().cpu().numpy()
            # Convert to numpy for visualization
            pred_rotmats_cpu = pred_rotmats_cpu.numpy()
            gt_rotmats_cpu = gt_rotmats_cpu.numpy()

            # Limit the number of samples to visualize
            n_samples = min(max_samples, pred_rotmats_cpu.shape[0])

            fig, axes = plt.subplots(n_samples, 2, figsize=(12, 2 * n_samples))

            # In case of single sample, ensure axes is 2D
            if n_samples == 1:
                axes = np.array([axes])

            for i in range(n_samples):
                # Visualize predicted rotation matrix
                self._plot_rotmat(axes[i, 0], pred_rotmats_cpu[i], title=f"Pred {i + 1}")

                # Visualize ground truth rotation matrix
                self._plot_rotmat(axes[i, 1], gt_rotmats_cpu[i], title=f"GT {i + 1}, Error: {errors_deg[i]:.2f}°")

            plt.tight_layout()

            # Convert plot to tensor for logging
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)

            # Convert PIL Image to tensor
            img = Image.open(buf)
            img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1)  # Convert to CxHxW format

            # Log the visualization tensor
            if self.logger is not None:
                if hasattr(self.logger, 'experiment') and hasattr(self.logger.experiment, 'add_image'):
                    self.logger.experiment.add_image(
                        f'rotmat_{partition}',
                        img_tensor,
                        self.global_step
                    )
                elif hasattr(self.logger, 'log_image'):
                    self.logger.log_image(
                        key=f'rotmat_{partition}',
                        images=[img_tensor]
                    )

            return img_tensor

        finally:
            # Switch back to the original backend
            matplotlib.use(current_backend, force=False)

    def _plot_rotmat(self, ax, rotmat, title):
        """Helper function to visualize a rotation matrix"""

        current_backend = matplotlib.get_backend()
        try:
            # Switch to non-interactive backend if needed
            if current_backend.lower() == 'tkagg' or 'headless' in str(current_backend).lower():
                matplotlib.use('Agg')
            im = ax.imshow(rotmat, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_title(title)
            ax.set_xticks([0, 1, 2])
            ax.set_yticks([0, 1, 2])
            ax.set_xticklabels(['x', 'y', 'z'])
            ax.set_yticklabels(['x', 'y', 'z'])
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f"{rotmat[i, j]:.2f}", ha="center", va="center", color="black")
            return im
        finally:
            matplotlib.use(current_backend)


class PlModel(RotationPredictionMixin, pl.LightningModule):
    def __init__(self, lr: float, symmetry: str, num_augmented_copies_per_batch: int,
                 top_k: int,
                 so3model: Optional[Union[nn.Module, ScriptModule]] = None,
                ):

        #TODO: Forward with neigs needs to be exposed and selectable via config
        super().__init__()
        self.__init_mixin__()
        self.lr = lr #The original LR
        self.symmetry = symmetry
        self.num_augmented_copies_per_batch = num_augmented_copies_per_batch
        self.top_k = top_k

        if so3model is None:
            so3model = self.build_components(symmetry, num_augmented_copies_per_batch)
        self.so3model = so3model

        self.warmup_epochs = main_config.train.warmup_n_epochs
        if is_pylig2:
            self.optimizer_step = self.optimizer_step_v2
        else:
            self.optimizer_step = self.optimizer_step_v1
        self.save_hyperparameters(ignore=['so3model'])


    @staticmethod
    def build_components(symmetry, num_augmented_copies_per_batch):
        return Image2Sphere(symmetry=symmetry, num_augmented_copies_per_batch=num_augmented_copies_per_batch)

    def _step(self, batch, batch_idx):

        # idd = batch[self.BATCH_IDS_NAME]
        imgs = batch[self.BATCH_PARTICLES_NAME]
        (gt_rotMats, xyShiftAngs, confid) = batch[self.BATCH_POSE_NAME]
        # metadata = batch[self.BATCH_MD_NAME]
        del batch
        (_, _, _, pred_rotmats, maxprobs
         ), loss, error_rads = self.so3model.forward_and_loss(imgs, gt_rotMats, confid, top_k=self.top_k)
        error_degs = torch.rad2deg(error_rads)
        return loss, error_degs, pred_rotmats, maxprobs, gt_rotMats

    def training_step(self, batch, batch_idx):
        loss, error_degs, pred_rotmats, maxprobs, gt_rotmats = self._step(batch, batch_idx)
        loss = loss.mean()

        self.log("loss", loss, prog_bar=True, batch_size=pred_rotmats.shape[0], sync_dist=False)
        self.log("geo_degs",error_degs.mean(), prog_bar=True, on_step=True, on_epoch=True,
                 batch_size=pred_rotmats.shape[0], sync_dist=True)

        if batch_idx == 0:
            # Visualize the predicted rotmats and the ground truth rotmats with error
            self._visualize_rotmats(pred_rotmats, gt_rotmats, error_degs=error_degs, partition="train")

        return loss

    def validation_step(self, batch, batch_idx):
        loss, error_degs, pred_rotmats, maxprobs, gt_rotmats = self._step(batch, batch_idx)
        loss = loss.mean()

        # Calculate and log average angular error in degrees
        self.log("val_loss", loss, prog_bar=True, on_epoch=True,
                 batch_size=pred_rotmats.shape[0], sync_dist=True)
        self.log("val_geo_degs", error_degs.mean(), prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=pred_rotmats.shape[0], sync_dist=True)

        if batch_idx == 0:
            # Visualize the predicted rotmats and the ground truth rotmats with error
            self._visualize_rotmats(pred_rotmats, gt_rotmats, error_degs=error_degs, partition="val")

        return loss

    def forward(self, imgs: torch.Tensor, batch_idx: int, dataloader_idx: int = 0, top_k: Optional[int] = None) -> Any:
        if top_k is None:
            top_k = self.top_k
        return self.so3model(imgs, top_k=top_k) #wD, rotMat_logits, pred_rotmat_id, pred_rotmats, maxprobs


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

    # THIS IS FOR pytorch_lightning v2.
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
        opt = optClass(self.parameters(), lr=lr, weight_decay=configTrain.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                                  opt,
                                                  factor=0.5,
                                                  threshold=5e-4,
                                                  min_lr=lr * configTrain.min_learning_rate_factor,
                                                  cooldown=1,
                                                  patience=configTrain.patient_reduce_lr_plateau_n_epochs)
        confid = {
            'optimizer': opt,
            'lr_scheduler': lr_scheduler,
            'monitor': configTrain.monitor_metric
        }
        return confid

    def on_train_end(self) -> None:
        from cryoPARES.models.directionalNormalizer.directionalNormalizer import DirectionalPercentileNormalizer
        NORMALIZER_HP_ORDER = 2  # TODO: Move to config

        print("Preparing directional percentiles")
        device = self.trainer.strategy.root_device
        self.so3model.to(device)

        if self.trainer is None or self.trainer.val_dataloaders is None:
            print("No val_data dataloaders found. Skipping prediction and stats computation.")
            return
        val_dataloaders = self.trainer.val_dataloaders
        if isinstance(self.trainer.val_dataloaders, DataLoader):
            val_dataloaders = [self.trainer.val_dataloaders]

        predRotMats = []
        gtRotmats = []
        scores = []
        self.eval()
        with torch.inference_mode():
            for dataloader_idx, dataloader in enumerate(val_dataloaders):
                for batch_idx, batch in enumerate(tqdm(dataloader)):
                    batch = self.transfer_batch_to_device(batch, device, dataloader_idx)
                    imgs = batch[self.BATCH_PARTICLES_NAME]
                    _, _, _, pred_rotmats, maxprobs = self.forward(imgs, batch_idx, dataloader_idx, top_k=10)
                    predRotMats.append(pred_rotmats[:,0,...]) # I am only using topK=1 for the clusters in the normalizer

                    topKscore = maxprobs.sum(-1) #But we use top-10 for confidence
                    scores.append(topKscore)

                    gt_rotmats = batch[self.BATCH_POSE_NAME][0]
                    gtRotmats.append(gt_rotmats)

                    if self.trainer.overfit_batches is not None and batch_idx > self.trainer.overfit_batches:
                        break
            predRotMats = torch.concat(predRotMats)
            gtRotmats = torch.concat(gtRotmats)
            scores = torch.concat(scores)

            dirname = os.path.dirname(self.trainer.checkpoint_callback.best_model_path)
            precentile_model_savename = os.path.join(dirname, constants.BEST_DIRECTIONAL_NORMALIZER)

            if self.trainer.world_size > 1:
                predRotMats = self.all_gather(predRotMats).reshape(-1, *predRotMats.shape[1:])
                gtRotmats = self.all_gather(gtRotmats).reshape(-1, *gtRotmats.shape[1:])
                scores = self.all_gather(scores).reshape(-1, *scores.shape[1:])

            if self.trainer.is_global_zero:
                normalizer = DirectionalPercentileNormalizer(symmetry=self.symmetry, hp_order=NORMALIZER_HP_ORDER)
                normalizer.fit(predRotMats.cpu(), scores.cpu(), gtRotmats.cpu())
                torch.save(normalizer, precentile_model_savename)


def _update_config_for_test():
    from cryoPARES.configs.mainConfig import main_config
    main_config.models.image2sphere.lmax = 6
    main_config.models.image2sphere.so3components.i2sprojector.sphere_fdim = 128
    main_config.models.image2sphere.so3components.i2sprojector.rand_fraction_points_to_project = 1.
    main_config.models.image2sphere.so3components.i2sprojector.hp_order = 2
    main_config.models.image2sphere.so3components.s2conv.hp_order = 2
    main_config.models.image2sphere.so3components.so3ouptutgrid.hp_order = 3

def _test0():
    b = 3
    batch = get_example_random_batch(b)
    model_kwargs = dict(lr=1e-5, symmetry="c1", num_augmented_copies_per_batch=1, top_k=1)
    plmodel = PlModel(**model_kwargs)
    imgs = batch[plmodel.BATCH_PARTICLES_NAME]
    (rotMats, xyShiftAngs, confid) = batch[plmodel.BATCH_POSE_NAME]
    plmodel.training_step(batch, batch_idx=0)

    plmodel(imgs, batch_idx=0)
    plmodel.predict_step(batch, batch_idx=0)
    scripted_model = torch.jit.script(plmodel.so3model)
    plmodel = PlModel(so3model=scripted_model, **model_kwargs)
    out = plmodel(imgs, batch_idx=0)
    print([o.shape for o in out])
    out = plmodel.predict_step(batch, batch_idx=0)
    print([o.shape for o in out if isinstance(o, Tensor)])

if __name__ == "__main__":
    _test0()