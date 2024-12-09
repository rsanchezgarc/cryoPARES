import os

import torch
import multiprocessing
import os.path as osp

import pytorch_lightning as pl
from pytorch_lightning.trainer.states import TrainerStatus
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from typing import Optional, List
from pytorch_lightning.callbacks import TQDMProgressBar, EarlyStopping, ModelCheckpoint, LearningRateMonitor, \
    StochasticWeightAveraging
from lightning_fabric.utilities.seed import seed_everything

from cryoPARES.constants import DATA_SPLITS_BASENAME, TRAINING_DONE_TEMPLATE


def trainOnePartition(symmetry: str, particlesStarFname: List[str],
          trainSaveDir: str = None, particlesDir: Optional[List[str]] = None,
          partition: str = "allParticles",
          continueCheckpointFname: Optional[str] = None,
          finetuneCheckpointFname: Optional[str] = None,
          find_lr: bool = False, compileModel:bool=False,
          val_check_interval:Optional[float] = None, n_epochs: Optional[int]=None,
          overfit_batches: Optional[int] = None):
    """

    :param symmetry: The point symmetry of the reconstruction
    :param particlesStarFname: The starfile containing the pre-aligned particles
    :param trainSaveDir: The root directory where models and logs are saved.
    :param particlesDir: The directory where the particles of the particlesStarFname are located. If not, it is assumed os.dirname(particlesStarFname)
    :param partition: The partition that is going to be trainined (half1, half2, or all together)
    :param continueCheckpointFname: The path of a pre-trained model to continue training.
    :param finetuneCheckpointFname: The path of a pre-trained model to load the weights, but train a new instance
    :param find_lr: Use the automatic heuristic for learning rate. Works only for GPUS=1
    :param compileModel: If True, torch 2.0 will try to compile the model to make training faster.
    :param val_check_interval: The fraction of an epoch after which the validation set will be evaluated
    :param n_epochs: The number of epochs to run. If not provided, it will use trainConfig.N_EPOCHS
    :param overfit_batches: If provided, number of train and validation batches to use
    :return:
    """

    from cryoPARES import constants
    from cryoPARES.configs.mainConfig import main_config
    from cryoPARES.utils.torchUtils import accelerator_selector
    torch.set_float32_matmul_precision(constants.float32_matmul_precision)
    from cryoPARES.datamanager.datamanager import DataManager
    from cryoPARES.utils import plUtils
    from cryoPARES.models.model import PlModel

    trainConf = main_config.train
    if n_epochs is not None:
        assert n_epochs >= 0
        trainConf.n_epochs = n_epochs

    datamanagerConfig = main_config.datamanager
    accel, dev_count = accelerator_selector(use_cuda=trainConf.use_cuda, n_gpus_torch=None,
                                            n_cpus_torch=trainConf.n_cpus_if_no_cuda)
    print(f'devices={dev_count} accelerator={accel}', flush=True)

    torch.set_num_threads(min(1, multiprocessing.cpu_count() // dev_count))
    seed_everything(trainConf.random_seed)

    # log_dir, version_folder = osp.split(trainSaveDir)
    log_dir = trainSaveDir
    version_folder = partition
    logger1 = TensorBoardLogger(log_dir, name="", version=version_folder, sub_dir=None)

    _, version = osp.split(logger1.log_dir)
    logger2 = CSVLogger(log_dir, name="", version=version_folder)


    checkpointer = ModelCheckpoint(
        dirpath=osp.join(logger1.log_dir, 'checkpoints'),
        monitor=trainConf.monitor_metric, filename=constants.PROJECT_NAME+"_model",
        save_last=True, save_top_k=2, verbose=True)

    callbacks = [
        TQDMProgressBar(refresh_rate=20),
        EarlyStopping(monitor=trainConf.monitor_metric,
                      patience=2 * trainConf.patient_reduce_lr_plateau_n_epochs+1, verbose=True),
        checkpointer,
        LearningRateMonitor(logging_interval='epoch'),
    ]

    callbacks += [StochasticWeightAveraging(annealing_epochs=trainConf.swalr_annelaing_n_epochs,
                                            swa_epoch_start=trainConf.swalr_begin_epoch,
                                            swa_lrs=trainConf.min_learning_rate_factor * 0.5 * trainConf.learning_rate)]

    if find_lr:
        from pytorch_lightning.callbacks import LearningRateFinder
        lr_finder = LearningRateFinder()
        callbacks += [lr_finder]

    trainer = pl.Trainer(
        # default_root_dir=trainSaveDir,  #val_check_interval=100, #Remove val_check_interval=100 when in production
        max_epochs=trainConf.n_epochs,
        logger=[logger1, logger2],
        devices=dev_count,
        accelerator=accel,  #"auto",
        num_nodes=plUtils.GET_PL_NUM_NODES(),
        strategy=plUtils.GET_PL_STRATEGY(dev_count),
        gradient_clip_val=trainConf.gradient_clip_value,
        # detect_anomaly=True,
        # profiler="advanced", #"simple" #"advanced"
        val_check_interval = val_check_interval,
        overfit_batches=overfit_batches if overfit_batches is not None else 0,
        accumulate_grad_batches=trainConf.accumulate_grad_batches,
        precision=trainConf.train_precision,
        sync_batchnorm=trainConf.sync_batchnorm,  #Recommended == True when your batch size is small
        plugins=plUtils.GET_PL_PLUGIN(), callbacks=callbacks)

    if partition == "half1":
        halfset = 1
    elif partition == "half2":
        halfset = 2
    elif partition == "allParticles":
        halfset = None
    else:
        raise ValueError()
    datamodule = DataManager(star_fnames=particlesStarFname,
                             symmetry=symmetry,
                             augment_train=datamanagerConfig.augment_train,
                             particles_dir=particlesDir,
                             halfset=halfset,
                             batch_size=trainConf.batch_size,
                             save_train_val_partition_dir=osp.join(logger1.log_dir, DATA_SPLITS_BASENAME),
                             is_global_zero=trainer.is_global_zero,
                             )

    kwargs = dict(lr=trainConf.learning_rate,
                  symmetry=symmetry,
                  num_augmented_copies_per_batch=main_config.datamanager.num_augmented_copies_per_batch)


    if continueCheckpointFname or finetuneCheckpointFname:
        kwargs["map_location"] = torch.device("cpu")
        if continueCheckpointFname is None:
            _loadFname = finetuneCheckpointFname
            resume_from_checkpoint = None
        else:
            _loadFname = continueCheckpointFname
            resume_from_checkpoint = continueCheckpointFname
        pl_model, _continueCheckpointFname = PlModel.load_from_checkpoint(_loadFname, **kwargs)
        if resume_from_checkpoint and _continueCheckpointFname != continueCheckpointFname:
            raise RuntimeError("Error, you aim to resume_from_checkpoint, but the checkpoint has been modified, you can either"
                               " load the new weights and reset the optimizer, or use a non-modified checkpoint ")
    else:
        pl_model = PlModel(**kwargs)
        resume_from_checkpoint = None
    if compileModel:
        print("Compiling model")
        pl_model.compile()

    if find_lr:
        # Results can be found in
        suggested_lr = lr_finder.suggestion()
        print("lr_finder.results", suggested_lr)
        trainConf.LEARNING_RATE = suggested_lr


    print("Training starts", flush=True)
    trainer.fit(pl_model, datamodule=datamodule, ckpt_path=resume_from_checkpoint)
    # print(trainer.state.status)
    # print(trainer.should_stop)

    trainer.strategy.barrier()
    if trainer.is_global_zero and trainer.state.status == TrainerStatus.FINISHED:
        dirname = osp.dirname(checkpointer.best_model_path)
        if dirname is not None:
            doneName = osp.join(dirname, TRAINING_DONE_TEMPLATE)
            best_model_basename = os.path.basename(checkpointer.best_model_path)
            with open(doneName, "w") as f:
                f.write(f"Done (PID {os.getgid()}). Best score is {checkpointer.best_model_score}\n"
                        f"Best checkpoint is: {best_model_basename}\n")
            cwd = os.getcwd()
            os.chdir(dirname)
            if os.path.isfile(constants.BEST_CHECKPOINT_BASENAME):
                os.unlink(constants.BEST_CHECKPOINT_BASENAME)
            os.symlink(best_model_basename, constants.BEST_CHECKPOINT_BASENAME)
            os.chdir(cwd)

        print("Training done!")
        # if not checkpointer.best_model_path and continueCheckpointFname:
        #     checkpointer.best_model_path = continueCheckpointFname
        # print(f"Best model {checkpointer.best_model_path} ({checkpointer.best_model_score})")
        # _computePercentiles(checkpointer.best_model_path, datamodule.train_val_partition_dir, symmetry,
        #                     datamodule.get_particlesDir())


if __name__ == "__main__":
    from argParseFromDoc import parse_function_and_call
    parse_function_and_call(trainOnePartition)

    """
python -m cryoPARES.train.runTrainOnePartition --symmetry C1 --particlesStarFname ~/cryo/data/preAlignedParticles/EMPIAR-10166/data/1000particles.star    
"""