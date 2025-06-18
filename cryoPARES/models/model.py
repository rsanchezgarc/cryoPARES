import warnings

from cryoPARES.constants import RELION_EULER_CONVENTION
from torch import Tensor, nn, ScriptModule
from torch.utils.data import DataLoader

from cryoPARES.geometry.convert_angles import matrix_to_euler_angles

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


class PlModel(pl.LightningModule):
    def __init__(self, lr: float, symmetry: str, num_augmented_copies_per_batch: int,
                 top_k: int, model: Optional[Union[nn.Module, ScriptModule]] = None):

        super().__init__()

        self.lr = lr
        self.symmetry = symmetry
        self.num_augmented_copies_per_batch = num_augmented_copies_per_batch
        self.top_k = top_k

        if model is None:
            model = self.build_components(symmetry, num_augmented_copies_per_batch)
        self.model = model

        self.warmup_epochs = main_config.train.warmup_n_epochs
        if is_pylig2:
            self.optimizer_step = self.optimizer_step_v2
        else:
            self.optimizer_step = self.optimizer_step_v1
        self.save_hyperparameters(ignore=['model'])

        from cryoPARES.constants import BATCH_IDS_NAME, BATCH_PARTICLES_NAME, BATCH_POSE_NAME, BATCH_MD_NAME
        self.BATCH_IDS_NAME = BATCH_IDS_NAME
        self.BATCH_PARTICLES_NAME = BATCH_PARTICLES_NAME
        self.BATCH_POSE_NAME = BATCH_POSE_NAME
        self.BATCH_MD_NAME = BATCH_MD_NAME

    @staticmethod
    def build_components(symmetry, num_augmented_copies_per_batch):
        return Image2Sphere(symmetry=symmetry, num_augmented_copies_per_batch=num_augmented_copies_per_batch)

    def _step(self, batch, batch_idx):

        idd, imgs, (gt_rotMats, shifts, conf), metadata = self.resolve_batch(batch)
        (wD, rotMat_logits, pred_rotmat_ids,
         pred_rotmats, maxprobs), loss, error_rads = self.model.forward_and_loss(imgs, gt_rotMats, conf, top_k=self.top_k)
        error_degs = torch.rad2deg(error_rads)
        return loss, error_degs, pred_rotmats, maxprobs, gt_rotMats

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

    def training_step(self, batch, batch_idx):
        loss, error_degs, pred_rotmats, maxprobs, gt_rotmats = self._step(batch, batch_idx)
        loss = loss.mean()

        self.log("loss", loss, prog_bar=True, batch_size=pred_rotmats.shape[0], sync_dist=False)
        self.log("geo_degs",error_degs.mean(), prog_bar=True, on_step=True, on_epoch=True,
                 batch_size=pred_rotmats.shape[0], sync_dist=False)

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

    def predict_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0):
        idd, imgs, poses, metadata = self.resolve_batch(batch)
        wD, rotMat_logits, pred_rotmat_id, pred_rotmats, maxprobs = self(imgs, batch_idx)
        all_angles_probs = rotMat_logits.softmax(-1)

        pred_shifts = pred_rotmats.new_full((pred_rotmats.shape[0], self.top_k, 2), torch.nan)
        shifts_probs = maxprobs * torch.nan

        if poses is not None:
            (rotMats, xyShiftAngs, conf) = poses
            errors = rotation_error_rads(rotMats, pred_rotmats[:,0,...])
            errors = torch.rad2deg(errors)
            errors = errors.detach().cpu()
        else:
            errors = None
        return idd, (pred_rotmats, maxprobs, all_angles_probs), (pred_shifts, shifts_probs), errors, metadata

    def forward(self, imgs: torch.Tensor, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return self.model(imgs, top_k=self.top_k)

    def resolve_batch(self, batch: Dict[str, Union[torch.Tensor, List[str],
                                                   Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                                                   Dict[str, Any]]]
                      ) -> Tuple[torch.Tensor, torch.Tensor,
                                 Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Dict[str, Any]]:
        idd = batch[self.BATCH_IDS_NAME]
        imgs = batch[self.BATCH_PARTICLES_NAME]
        (rotMats, xyShiftAngs, conf) = batch[self.BATCH_POSE_NAME]
        metadata = batch[self.BATCH_MD_NAME]
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

    def on_fit_end(self) -> None:
        #TODO: You need to get the predictions of all the testing set and store the
        # ground truth and predicted rotation matrices, together with the prediction confidence.
        #Then, you need to compute the stats according to the other code I am pasting. The stats try to estimate
        #Robust z-scores for each of the projection directions of the cryo-em particles.

        if self.trainer is None or self.trainer.val_dataloaders is None:
            print("No val_data dataloaders found. Skipping prediction and stats computation.")
            return
        val_dataloaders = self.trainer.val_dataloaders
        if isinstance(self.trainer.val_dataloaders, DataLoader):
            val_dataloaders = [self.trainer.val_dataloaders]

        eulerDegs = []
        self.eval()
        with (torch.inference_mode()):
            for dataloader_idx, dataloader in enumerate(val_dataloaders):
                for batch_idx, batch in enumerate(dataloader):
                    predictions = self.predict_step(batch, batch_idx, dataloader_idx)
                    idd, (pred_rotmats, maxprobs, all_angles_probs), \
                    (pred_shifts, shifts_probs), errors, metadata = predictions
                    eulerDegs.append(torch.rad2deg(matrix_to_euler_angles(pred_rotmats, convention=RELION_EULER_CONVENTION)))

        # Aggregate results from all predictions
        all_ids = []
        all_pred_eulers = []
        all_maxprobs = []
        all_gt_eulers = []
        all_errors_deg = []
        all_metadata = []

        for pred_batch in all_preds:
            all_ids.extend(pred_batch['idd'])
            all_pred_eulers.extend(pred_batch['pred_rotmats_euler'])
            all_maxprobs.extend(pred_batch['maxprobs'])
            if pred_batch['gt_rotmats_euler'] is not None:
                all_gt_eulers.extend(pred_batch['gt_rotmats_euler'])
            if pred_batch['errors_deg'] is not None:
                all_errors_deg.extend(pred_batch['errors_deg'].tolist()) # Convert tensor to list for extend
            all_metadata.extend(pred_batch['metadata'])


        # Create a DataFrame for STAR file
        pred_df = pd.DataFrame({
            '_rlnParticleName': [f'{_id:06}' for _id in all_ids], # Example particle name
            '_rlnAngleRot': [euler[0] for euler in all_pred_eulers],
            '_rlnAngleTilt': [euler[1] for euler in all_pred_eulers],
            '_rlnAnglePsi': [euler[2] for euler in all_pred_eulers],
            NNET_PRED_SCORE_NAME: all_maxprobs, # Use the defined score name
            '_rlnAngleRot_ori': [euler[0] for euler in all_gt_eulers] if all_gt_eulers else np.nan,
            '_rlnAngleTilt_ori': [euler[1] for euler in all_gt_eulers] if all_gt_eulers else np.nan,
            '_rlnAnglePsi_ori': [euler[2] for euler in all_gt_eulers] if all_gt_eulers else np.nan,
            'error_degs': all_errors_deg if all_errors_deg else np.nan
        })

        # Add other metadata fields if present and relevant, assuming they are dicts
        if all_metadata and isinstance(all_metadata[0], dict):
            # Flatten metadata dicts into columns
            for key in all_metadata[0].keys():
                # Check if the key exists in metadata and if it's a tensor, convert to numpy
                if isinstance(all_metadata[0][key], torch.Tensor):
                    pred_df[key] = [md[key].item() if isinstance(md[key], torch.Tensor) and md[key].numel() == 1 else md[key].numpy() for md in all_metadata]
                else:
                     pred_df[key] = [md[key] for md in all_metadata]

        # Define output STAR file path
        output_star_fname = os.path.join(self.trainer.log_dir, f"predictions_epoch_{self.trainer.current_epoch}.star")
        starfile.write({'particles': pred_df}, output_star_fname, overwrite=True)
        print(f"Predictions saved to {output_star_fname}")

        # Compute and apply robust z-scores
        print("Computing and applying robust z-scores...")
        try:
            # Need a reference STAR file for estimate_parameters.
            # For this example, we can use the generated prediction file itself as reference,
            # or you might have a dedicated validation set predictions file.
            # If using a separate validation set, ensure it's logged earlier.
            # For simplicity, using the generated prediction file as the reference.
            ref_star_fname = output_star_fname # Using the generated file as reference

            # Make sure this works with DDP by having the main process handle the file I/O
            if self.trainer.global_rank == 0:
                params = estimate_parameters(ref_star_fname, self.symmetry,
                                             pred_score_name=NNET_PRED_SCORE_NAME,
                                             show_plots=False) # Set show_plots to True for debugging if needed

                rescored_star_out_fname = os.path.join(self.trainer.log_dir, f"rescored_predictions_epoch_{self.trainer.current_epoch}.star")
                apply_parameters(output_star_fname, rescored_star_out_fname, params, self.symmetry,
                                 ori_score_name=NNET_PRED_SCORE_NAME,
                                 new_score_name=NEW_SCORE_NNET_NAME,
                                 show_plots=False, # Set show_plots to True for debugging if needed
                                 overwrite=True)
                print(f"Rescored predictions saved to {rescored_star_out_fname}")

        except Exception as e:
            print(f"Error during z-score computation: {e}")
            # If running with DDP, ensure this doesn't cause issues if only rank 0 tries to write files
            # You might want to handle this more robustly depending on your DDP setup.

        raise NotImplementedError()

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
    model_kwargs = dict(lr=1e-5, symmetry="c2", num_augmented_copies_per_batch=1, top_k=1)
    plmodel = PlModel(**model_kwargs)
    _, imgs, poses, _ = plmodel.resolve_batch(batch)
    plmodel.training_step(batch, batch_idx=0)

    plmodel(imgs, batch_idx=0)
    plmodel.predict_step(batch, batch_idx=0)
    scripted_model = torch.jit.script(plmodel.model)
    plmodel = PlModel(model=scripted_model, **model_kwargs)
    out = plmodel(imgs, batch_idx=0)
    print([o.shape for o in out])
    out = plmodel.predict_step(batch, batch_idx=0)
    print([o.shape for o in out if isinstance(o, Tensor)])

if __name__ == "__main__":
    _test0()