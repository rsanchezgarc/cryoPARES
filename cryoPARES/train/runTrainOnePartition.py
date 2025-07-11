import gc
import os
import subprocess
import sys
import warnings
import torch
import multiprocessing
import os.path as osp
from typing import Optional, List, Literal

import pytorch_lightning as pl
from argParseFromDoc import generate_command_for_argparseFromDoc
from pytorch_lightning.trainer.states import TrainerStatus
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import (
    TQDMProgressBar, EarlyStopping, ModelCheckpoint,
    LearningRateMonitor, StochasticWeightAveraging, LearningRateFinder
)
from lightning_fabric.utilities.seed import seed_everything

from cryoPARES import constants
from cryoPARES.configManager.inject_defaults import inject_defaults_from_config, CONFIG_PARAM
from cryoPARES.constants import DATA_SPLITS_BASENAME, TRAINING_DONE_TEMPLATE
from cryoPARES.configs.mainConfig import main_config
from cryoPARES.reconstruction.reconstruction import reconstruct_starfile


class TrainerPartition:
    @inject_defaults_from_config(main_config.train, update_config_with_args=True)
    def __init__(self, symmetry: str, particles_star_fname: List[str], train_save_dir: str,
                 particles_dir: Optional[List[str]] = None, n_epochs: int = CONFIG_PARAM(),
                 partition: Literal["allParticles", "half1", "half2"] = "allParticles",
                 continue_checkpoint_fname: Optional[str] = None, finetune_checkpoint_fname: Optional[str] = None,
                 find_lr: bool = False, compile_model: bool = False, val_check_interval: Optional[float] = None,
                 overfit_batches: Optional[int] = None):
        """Initialize trainer for a single partition.

        Args:
            symmetry: The point symmetry of the reconstruction
            particles_star_fname: The starfile containing the pre-aligned particles
            train_save_dir: Root directory for models and logs
            particles_dir: Directory containing particles (defaults to dirname of star file)
            n_epochs: Number of epochs to train
            partition: Partition to train (half1, half2, or allParticles)
            continue_checkpoint_fname: Path to continue training from checkpoint
            finetune_checkpoint_fname: Path to load weights but train new instance
            find_lr: Use automatic learning rate finder (GPU only)
            compile_model: Use torch 2.0 compilation
            val_check_interval: Fraction of epoch between validations
            overfit_batches: Number of batches to use if overfitting
        """
        self.symmetry = symmetry
        self.particles_star_fname = particles_star_fname
        self.train_save_dir = train_save_dir
        self.particles_dir = particles_dir
        self.partition = partition
        self.continue_checkpoint_fname = continue_checkpoint_fname
        self.finetune_checkpoint_fname = finetune_checkpoint_fname
        self.find_lr = find_lr
        self.compile_model = compile_model
        self.val_check_interval = val_check_interval
        self.n_epochs = n_epochs
        self.overfit_batches = overfit_batches

        self.train_config = main_config.train
        if self.n_epochs is not None:
            assert self.n_epochs >= 0
            self.train_config.n_epochs = self.n_epochs

        torch.set_float32_matmul_precision(constants.float32_matmul_precision)

    def _setup_accelerator(self):
        from cryoPARES.utils.torchUtils import accelerator_selector
        accel, dev_count = accelerator_selector(
            use_cuda=self.train_config.use_cuda,
            n_gpus_torch=None,
            n_cpus_torch=self.train_config.n_cpus_if_no_cuda
        )
        print(f'devices={dev_count} accelerator={accel}', flush=True)
        torch.set_num_threads(min(1, multiprocessing.cpu_count() // dev_count))
        return accel, dev_count

    def _setup_loggers(self):
        version_folder = self.partition
        logger1 = TensorBoardLogger(self.train_save_dir, name="", version=version_folder, sub_dir=None)
        _, version = osp.split(logger1.log_dir)
        logger2 = CSVLogger(self.train_save_dir, name="", version=version_folder)
        return logger1, logger2

    def _setup_callbacks(self, logger1):
        checkpointer = ModelCheckpoint(
            dirpath=osp.join(logger1.log_dir, 'checkpoints'),
            monitor=self.train_config.monitor_metric,
            filename=constants.PROJECT_NAME + "_model",
            save_last=True,
            save_top_k=2,
            verbose=True
        )

        callbacks = [
            TQDMProgressBar(refresh_rate=20),
            EarlyStopping(
                monitor=self.train_config.monitor_metric,
                patience=2 * self.train_config.patient_reduce_lr_plateau_n_epochs + 1,
                verbose=True
            ),
            checkpointer,
            LearningRateMonitor(logging_interval='epoch'),
            StochasticWeightAveraging(
                annealing_epochs=self.train_config.swalr_annelaing_n_epochs,
                swa_epoch_start=self.train_config.swalr_begin_epoch,
                swa_lrs=self.train_config.min_learning_rate_factor * 0.5 * self.train_config.learning_rate
            )
        ]

        if self.find_lr:
            callbacks.append(LearningRateFinder())

        return callbacks, checkpointer

    def _setup_trainer(self, accel, dev_count, loggers, callbacks):
        from cryoPARES.utils import plUtils
        return pl.Trainer(
            max_epochs=self.train_config.n_epochs,
            logger=loggers,
            devices=dev_count,
            accelerator=accel,
            num_nodes=plUtils.GET_PL_NUM_NODES(main_config.train.num_computer_nodes),
            strategy=plUtils.GET_PL_STRATEGY(dev_count, ),
            gradient_clip_val=self.train_config.gradient_clip_value,
            val_check_interval=self.val_check_interval,
            overfit_batches=self.overfit_batches if self.overfit_batches is not None else 0,
            accumulate_grad_batches=self.train_config.accumulate_grad_batches,
            precision=self.train_config.train_precision,
            sync_batchnorm=self.train_config.sync_batchnorm,
            plugins=plUtils.GET_PL_PLUGIN(main_config.train.pl_plugin),
            callbacks=callbacks,
            use_distributed_sampler=False #TODO: Check if this is required
        )

    def _kwargs_for_loading(self):
        return {
            "lr": self.train_config.learning_rate,
            "symmetry": self.symmetry,
            "num_augmented_copies_per_batch": main_config.datamanager.num_augmented_copies_per_batch,
            "top_k": main_config.inference.top_k
        }

    def _setup_model(self, trainer):
        from cryoPARES.models.model import PlModel
        kwargs = self._kwargs_for_loading()
        if self.continue_checkpoint_fname or self.finetune_checkpoint_fname:
            kwargs["map_location"] = torch.device("cpu")
            if self.continue_checkpoint_fname is None:
                load_fname = self.finetune_checkpoint_fname
                resume_from_checkpoint = None
            else:
                load_fname = self.continue_checkpoint_fname
                resume_from_checkpoint = self.continue_checkpoint_fname

            pl_model, _continue_checkpoint_fname = PlModel.load_from_checkpoint(load_fname, **kwargs)

            if (resume_from_checkpoint and
                    _continue_checkpoint_fname != self.continue_checkpoint_fname):
                raise RuntimeError(
                    "Checkpoint has been modified. Either load new weights and reset optimizer, "
                    "or use unmodified checkpoint."
                )
        else:
            pl_model = PlModel(**kwargs)
            resume_from_checkpoint = None

        if self.compile_model:
            print("Compiling model")
            pl_model.compile()

        return pl_model, resume_from_checkpoint

    def _setup_datamodule(self, trainer, logger1):
        from cryoPARES.datamanager.datamanager import DataManager

        halfset = {
            "half1": 1,
            "half2": 2,
            "allParticles": None
        }[self.partition]

        return DataManager(
            star_fnames=self.particles_star_fname,
            symmetry=self.symmetry,
            augment_train=main_config.datamanager.augment_train,
            particles_dir=self.particles_dir,
            halfset=halfset,
            batch_size=self.train_config.batch_size,
            save_train_val_partition_dir=osp.join(logger1.log_dir, DATA_SPLITS_BASENAME),
            is_global_zero=trainer.is_global_zero
        )

    def run(self):
        seed_everything(self.train_config.random_seed)

        accel, dev_count = self._setup_accelerator()
        logger1, logger2 = self._setup_loggers()
        callbacks, checkpointer = self._setup_callbacks(logger1)
        trainer = self._setup_trainer(accel, dev_count, [logger1, logger2], callbacks)
        pl_model, resume_from_checkpoint = self._setup_model(trainer)
        datamodule = self._setup_datamodule(trainer, logger1)

        if self.find_lr and isinstance(callbacks[-1], LearningRateFinder):
            suggested_lr = callbacks[-1].suggestion()
            print("lr_finder.results", suggested_lr)
            self.train_config.LEARNING_RATE = suggested_lr

        print("Training starts", flush=True)
        trainer.fit(pl_model, datamodule=datamodule, ckpt_path=resume_from_checkpoint)

        trainer.strategy.barrier() #TODO: We probably want to run reconstruct in the main loop
        if trainer.is_global_zero and trainer.state.status == TrainerStatus.FINISHED:
            print("Trained finished. Saving model ")
            self._save_training_completion(checkpointer)
            save_train_val_partition_dir = trainer.datamodule.save_train_val_partition_dir
            num_data_workers = datamodule.num_data_workers
            del trainer
            del pl_model, checkpointer
            del datamodule
            gc.collect()
            torch.cuda.empty_cache()

            torch.cuda.empty_cache()
            print("Trained finished. Lauching reconstruction")
            reconstructions_dir = get_reconstructions_dir(self.train_save_dir, self.partition)
            os.makedirs(reconstructions_dir, exist_ok=True)
            if os.path.isdir(save_train_val_partition_dir):
                saved_particles_dir = os.path.join(save_train_val_partition_dir, "train")
                for fnameIdx, fname in enumerate(os.listdir(saved_particles_dir)):
                    fname = os.path.join(saved_particles_dir, fname)
                    print(f"Reconstructing {fname}")
                    particles_dir = self.particles_dir[fnameIdx] if self.particles_dir is not None \
                                                                else os.path.dirname(self.particles_star_fname[fnameIdx])

                    output_fname = os.path.join(reconstructions_dir, "%d.mrc" % fnameIdx)
                    kwargs = dict(particles_star_fname=fname,
                                  symmetry=self.symmetry,
                                  output_fname=output_fname,
                                  particles_dir=particles_dir,
                                  num_workers=num_data_workers,
                                  batch_size=main_config.train.batch_size_for_reconstruct,
                                  use_cuda=main_config.train.cuda_for_reconstruct,
                                  correct_ctf=True, eps=1e-3, min_denominator_value=1e-4)
                    if self.overfit_batches is not None:
                        kwargs["use_only_n_first_batches"] = self.overfit_batches

                    cmd = generate_command_for_argparseFromDoc(
                        "cryoPARES.reconstruction.reconstruction",
                        fun=reconstruct_starfile,
                        use_module=True,
                        python_executable=sys.executable,
                        **kwargs
                    )
                    print(cmd)  # TODO: Use loggers
                    subprocess.run(
                        cmd.split(),
                        cwd=os.path.abspath(os.path.join(__file__, "..", "..", "..")), check=True
                    )
            else:
                warnings.warn("No validation particles found, directional precentiles were not computed")

    def _save_training_completion(self, checkpointer):
        dirname = osp.dirname(checkpointer.best_model_path)
        if dirname is None:
            return

        done_name = get_done_fname(self.train_save_dir, self.partition)
        best_model_basename = osp.basename(checkpointer.best_model_path)

        cwd = os.getcwd()
        os.chdir(dirname)
        if osp.isfile(constants.BEST_CHECKPOINT_BASENAME):
            os.unlink(constants.BEST_CHECKPOINT_BASENAME)
        os.symlink(best_model_basename, constants.BEST_CHECKPOINT_BASENAME)

        from cryoPARES.models.model import PlModel
        best_module = PlModel.load_from_checkpoint(constants.BEST_CHECKPOINT_BASENAME, map_location="cpu")
        best_model_script = torch.jit.script(best_module.so3model)
        torch.jit.save(best_model_script, constants.BEST_MODEL_SCRIPT_BASENAME)

        os.chdir(cwd)

        with open(done_name, "w") as f:
            f.write(
                f"Done (PID {os.getgid()}). Best score is {checkpointer.best_model_score}\n"
                f"Best checkpoint is: {best_model_basename}\n"
            )
        print("Training done!")


def get_done_fname(dirname: str, partition: str) -> str:
    """Get path to the DONE_TRAINING.txt file for a partition.

    Args:
        dirname: Root directory of the experiment
        partition: Partition name (half1, half2, or allParticles)
    """
    return osp.join(dirname, partition, "checkpoints", TRAINING_DONE_TEMPLATE)


def get_reconstructions_dir(dirname: str, partition: str):
    return osp.join(dirname, partition, "reconstructions")

def check_if_training_partion_done(dirname: str, partition: str):
    """Check if training for given partition is complete by looking for DONE_TRAINING.txt

    Args:
        dirname: Root directory of the experiment (generally xxxx_v1, xxxx_v2...)
        partition: Partition name (half1, half2, or allParticles)
    """
    return osp.isfile(get_done_fname(dirname, partition))


def execute_trainOnePartition(**kwargs):

    if check_if_training_partion_done(kwargs['train_save_dir'], kwargs['partition']):
        print(f"Training for partition {kwargs['partition']} already completed")
        return

    cmd = generate_command_for_argparseFromDoc(
        "cryoPARES.train.runTrainOnePartition",
        fun=TrainerPartition.__init__,
        use_module=True,
        python_executable=sys.executable,
        **kwargs
    )
    config_args = kwargs.get("config_args", None)
    if config_args is not None:
        cmd += " --config " + " ".join(config_args)
    print(cmd)  # TODO: Use loggers
    subprocess.run(
        cmd.split(),
        cwd=os.path.abspath(os.path.join(__file__, "..", "..", "..")), check=True
    )


if __name__ == "__main__":
    # from argParseFromDoc import AutoArgumentParser
    # parser = AutoArgumentParser(prog="train partition cryoPARES")

    from cryoPARES.configManager.configParser import ConfigArgumentParser, export_config_to_yaml

    parser = ConfigArgumentParser(prog="train_partition_cryoPARES", config_obj=main_config)

    parser.add_args_from_function(TrainerPartition.__init__)
    args, config_args = parser.parse_args()
    TrainerPartition(**vars(args)).run()

    """
python -m cryoPARES.train.runTrainOnePartition --symmetry C1 --particles_star_fname ~/cryo/data/preAlignedParticles/EMPIAR-10166/data/1000particles.star  --train_save_dir /tmp/CryoParesTrain
    
"""