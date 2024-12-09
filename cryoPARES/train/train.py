import json
import os
import shutil
import warnings
import psutil
import torch
import subprocess
import sys
import os.path as osp
import tempfile
from typing import Optional, List

from cryoPARES import constants
from cryoPARES.utils.reproducibility import _copyCode



def train(symmetry: str, particlesStarFname: List[str],
          trainSaveDir: Optional[str] = None, particlesDir: Optional[List[str]] = None,
          split_halfs: bool = True,
          continueCheckpointDir: Optional[str] = None,
          finetuneCheckpointDir: Optional[str] = None,
          compileModel: bool = False,
          val_check_interval: Optional[float] = None,
          overfit_batches: Optional[int] = None,
          mapFname_for_simulated_pretraining: Optional[List[str]] = None):
    """

    :param symmetry: The point symmetry of the reconstruction
    :param particlesStarFname: The starfile containing the pre-aligned particles
    :param trainSaveDir: The root directory where models and logs are saved. If None, it is /tmp/PROJECT_NAME
    :param particlesDir: The directory where the particles of the particlesStarFname are located. If not, it is assumed os.dirname(particlesStarFname)
    :param split_halfs: If True, it trains a model for each half of the data
    :param continueCheckpointDir: The path of a pre-trained model to continue training.
    :param finetuneCheckpointDir: The path of a pre-trained model to do finetunning
    :param launch_tensorboard: Launch tensorboard as an independent process
    :param find_lr: Use the automatic heuristic for learning rate. Works only for GPUS=1
    :param compileModel: If True, torch 2.0 will try to compile the model to make training faster.
    :param val_check_interval: The fraction of an epoch after which the validation set will be evaluated
    :param overfit_batches: If provided, number of train and validation batches to use
    :param skip_training: If True, skip training and compute percentiles only
    :param mapFname_for_simulated_pretraining: If provided, it will run a warmup training on simulations using this maps. They need to match the particlesStarFname order
    :return:
    """

    assert 0 <= int(bool(continueCheckpointDir)) + int(bool(finetuneCheckpointDir)) <= 1, ("Error, "
                                                                                           "continueCheckpointDir and finetuneCheckpintDir are mutually exclusive")
    particlesStarFname = [osp.expanduser(fname) for fname in particlesStarFname]
    for fname in particlesStarFname:
        assert osp.isfile(fname), f"Error, fname {fname} not found"

    if trainSaveDir is not None:
        assert osp.isdir(trainSaveDir), f"Error, directory {trainSaveDir} not found"

    from cryoSolver.supervisedAngles.trainWorkers.checkpointUtils import get_version_to_use
    from cryoSolver.supervisedAngles.inferenceWorkers.runInference import splitHalfsFromName
    from cryoPARES.constants import TRAINING_DONE_TEMPLATE

    from cryoSolver.supervisedAngles.supervisedAnglesConfig import \
        config  # This needs to be imported within the function to be able to be updated with argument parser
    torch.set_float32_matmul_precision(constants.float32_matmul_precision)

    dataConfig = config.getDataConfig()
    print(config.all_parameters_dict)

    if continueCheckpointDir is not None:
        _trainSaveDir, versionDir = osp.split(continueCheckpointDir)
        if trainSaveDir is not None:
            assert _trainSaveDir == trainSaveDir, ("Error, when continuing a checkpoint, please, do not provide a "
                                                   "trainSaveDir")
        trainSaveDir = _trainSaveDir

    else:
        # In all other cases we will get a new directory
        trainSaveDir = _getTrainSaveDir(trainSaveDir)
        versionDir = get_version_to_use(trainSaveDir, "version_", dir_only=True)

    new_experiment_root = osp.join(trainSaveDir, versionDir)
    os.makedirs(new_experiment_root, exist_ok=True)

    # Copy the command for reproducibility
    basename = get_version_to_use(new_experiment_root, basename='command_',
                                  path_pattern=r'(command_)(\d+)(\.txt)$',
                                  extension="txt")
    fname = osp.join(new_experiment_root, basename)
    current_process = psutil.Process()
    # parent_process = current_process.parent()
    _command = " ".join(["'" + x + "'" if x.startswith('{"') else x for x in current_process.cmdline()])
    with open(fname, "w") as f:
        f.write(_command)

    # Copy the env vars
    basename = get_version_to_use(new_experiment_root, basename='envs_',
                                  path_pattern=r'(envs_)(\d+)(\.json)$',
                                  extension="json")
    fname = osp.join(new_experiment_root, basename)
    with open(fname, 'w') as f:
        json.dump(dict(os.environ), f)

    ##### COPY CODE FILES FOR REPRODUCIBILITY
    copycodedirBase = get_version_to_use(new_experiment_root, basename='code_',
                                         path_pattern=r'(code_)(\d+)$')
    copycodedir = osp.join(new_experiment_root, copycodedirBase)
    os.makedirs(copycodedir, exist_ok=True)

    modulePath = osp.abspath(sys.modules[__name__].__file__)
    rootPath = osp.dirname(osp.dirname(osp.dirname(modulePath)))
    _copyCode(rootPath, osp.join(copycodedir, "cryoSolver"))

    modulePath = osp.abspath(sys.modules["cryoUtils"].__file__)
    rootPath = osp.dirname(modulePath)
    _copyCode(rootPath, osp.join(copycodedir, "cryoUtils"))


    if finetuneCheckpointDir is not None:
        finetuneCheckpoinBase = get_version_to_use(new_experiment_root, basename='finetuneCheckpoin_',
                                                   path_pattern=r'(finetuneCheckpoin_)(\d+)(\.txt)$', extension="txt")
        with open(osp.join(new_experiment_root, finetuneCheckpoinBase), "w") as f:
            f.write(f"finetuneCheckpoin: {finetuneCheckpointDir}")

    if launch_tensorboard:
        subprocess.Popen(["tensorboard", "--logdir", trainSaveDir], stdout=sys.stdout)
        print("Use the url below to monitor training on tensorboard")

    with tempfile.TemporaryDirectory() as tmpdir:
        if split_halfs:
            inputStarFnames = dict(half1=[], half2=[])
            if projectionsStarFname is not None:  # TODO: implement this, by allowing multiple fnames to splitHalfsFromName
                raise ValueError("Error, projectionsStarFname not implemented for split_halfs")
            for fname in particlesStarFname:
                pset0, pset1 = splitHalfsFromName(fname, tmpdir, template_name="halfset_%d.star", count_from_zero=False)
                inputStarFnames["half1"].append(pset0)
                inputStarFnames["half2"].append(pset1)
        else:
            inputStarFnames = dict(allParticles=particlesStarFname)

        for workerIdx, partition in enumerate(inputStarFnames.keys()):
            if continueCheckpointDir is not None:
                continueCheckpointFname = osp.join(continueCheckpointDir, partition, "checkpoints", "last.ckpt")
                if not osp.isfile(continueCheckpointFname):
                    warnings.warn(f"continueCheckpointFname {continueCheckpointFname} not found, initializing from scratch")
                    continueCheckpointFname = None
            else:
                continueCheckpointFname = None
            if finetuneCheckpointDir is not None:
                finetuneCheckpointFname = get_best_checkpoint(osp.join(finetuneCheckpointDir, partition, "checkpoints"))
            else:
                finetuneCheckpointFname = None

            train_root_dir = osp.join(new_experiment_root, partition)
            os.makedirs(train_root_dir, exist_ok=True)

            ## Training with particlesStarFname

            _checkpoint = None
            done_dir = osp.join(new_experiment_root, partition, "checkpoints")
            done_fname = osp.join(done_dir, TRAINING_DONE_TEMPLATE)
            os.makedirs(done_dir, exist_ok=True)
            if not skip_training and not osp.isfile(done_fname):

                ## Training with simulated
                if mapFname_for_simulated_pretraining is not None:
                    assert len(mapFname_for_simulated_pretraining) == len(inputStarFnames[partition]), (
                        "Error, different number of "
                        "mapFname_for_simulated_pretraining")
                    simulated_checkpoint_fname = osp.join(done_dir, "simulated.cpkt")
                    if not osp.isfile(simulated_checkpoint_fname):
                        assert finetuneCheckpointFname is None and continueCheckpointFname is None, (
                            "Error, if mapFname_for_simulated_pretraining, finetuneCheckpointFname or "
                            "continueCheckpointFname cannot be provided")
                        from cryoUtils.emPackages.relion.relionProject import compute_projections
                        simulationRoot = osp.join(tmpdir, "simulations")
                        simulateStarList = []
                        simulateDirList = []
                        os.makedirs(simulationRoot, exist_ok=True)
                        for i, mapFname in enumerate(mapFname_for_simulated_pretraining):
                            outdirSimulationData = osp.join(simulationRoot, "data_"+partition, osp.basename(mapFname))
                            os.makedirs(outdirSimulationData, exist_ok=True)
                            simulatedStar = compute_projections(particlesStarFname=inputStarFnames[partition][i],
                                                                mapFname=mapFname, outdir=outdirSimulationData,
                                                                particlesDir=particlesDir[i] if particlesDir else None,
                                                                add_ctfCorruption=True, simulate=True, simulate_snr=2.,
                                                                # TODO: Move some stuff to config
                                                                outnamePattern="proj", randomize_alignment_params=True,
                                                                n_workers=dataConfig.N_DATAWORKERS, verbose=True)
                            simulateStarList.append(simulatedStar)
                            simulateDirList.append(outdirSimulationData)

                        outdirSimulatedCheckpoint = osp.join(simulationRoot, "train")
                        os.makedirs(outdirSimulatedCheckpoint, exist_ok=True)
                        _launch_train_command(symmetry, simulateStarList, partition,
                                              new_experiment_root=outdirSimulatedCheckpoint,
                                              particlesDir=simulateDirList,
                                              continueCheckpointFname=None,
                                              finetuneCheckpointFname=None,
                                              n_epochs=trainConfig.N_SIMULATION_EPOCHS, # TODO: for simulation, make this configurable
                                              find_lr=find_lr, compileModel=compileModel,
                                              val_check_interval=val_check_interval, overfit_batches=overfit_batches)
                        _simulated_checkpoint_fname = get_best_checkpoint(osp.join(outdirSimulatedCheckpoint, partition, "checkpoints"))
                        shutil.copyfile(_simulated_checkpoint_fname, simulated_checkpoint_fname)
                        # Cleanup-simulation (takes a lot of space)
                        shutil.rmtree(simulationRoot)
                    finetuneCheckpointFname = simulated_checkpoint_fname
                ## Training with particlesStarFname
                pid = _launch_train_command(symmetry, inputStarFnames[partition], partition,
                                            new_experiment_root=new_experiment_root,
                                            projectionsStarFname=projectionsStarFname,
                                            particlesDir=particlesDir, projectionsDir=projectionsDir,
                                            continueCheckpointFname=continueCheckpointFname,
                                            finetuneCheckpointFname=finetuneCheckpointFname,
                                            find_lr=find_lr, compileModel=compileModel,
                                            val_check_interval=val_check_interval, overfit_batches=overfit_batches)


            else:
                assert continueCheckpointFname is not None, "Error, if skip_training is provided, continueCheckpointFname is required"
                _checkpoint = continueCheckpointFname
            _checkpointDir = osp.dirname(done_fname)
            if _checkpoint is None:
                assert pid is not None, "Error, training was not completed"
                with open(done_fname) as f:
                    for line in f:
                        if line.startswith("Best checkpoint is:"):
                            _checkpoint = osp.join(_checkpointDir, line.split(":")[-1].strip())
                            break
            assert _checkpoint is not None and osp.exists(_checkpoint), f"Error, no valid _checkpoint ({_checkpoint}"

            from cryoSolver.supervisedAngles.trainWorkers.runDirectionalPercentiles import DONE_PERCENTILES
            if not osp.isfile(osp.join(_checkpointDir, DONE_PERCENTILES)):
                from cryoSolver.supervisedAngles.trainWorkers.runTrainOnePartition import DATA_SPLITS_BASENAME
                train_val_partition_dir = osp.join(osp.dirname(_checkpointDir), DATA_SPLITS_BASENAME)
                assert osp.isdir(train_val_partition_dir), "Error, no train_val_partition_dir"
                cmd = ["python", "-m", "cryoSolver.supervisedAngles.trainWorkers.runDirectionalPercentiles"]
                cmd += ["--best_model_path", _checkpoint]
                cmd += ["--train_val_partition_dir", train_val_partition_dir]
                cmd += ["--symmetry", args['main'].symmetry]
                if args['main'].particlesDir:
                    cmd += ["--particlesDir"]
                    for dirname in args['main'].particlesDir:
                        cmd += [dirname]
                print(" ".join(cmd))
                subprocess.check_call(cmd, env=os.environ.copy())
                # torch.cuda.empty_cache()
    print("Training script done!!!")


def _launch_train_command(symmetry: str, inputStarFnames: List[str], partition: str,
                          new_experiment_root: str, projectionsStarFname: Optional[List[str]] = None,
                          particlesDir: Optional[List[str]] = None, projectionsDir: Optional[List[str]] = None,
                          continueCheckpointFname: Optional[str] = None, finetuneCheckpointFname: Optional[str] = None,
                          n_epochs: Optional[int] = None,
                          find_lr: bool = False, compileModel: bool = False,
                          val_check_interval: Optional[float] = None, overfit_batches: Optional[int] = None):

    ## Remember, do not pass --CONFIG__PARAM arguemnts here, since they are not parsed
    cmd = ["python", "-m", "cryoSolver.supervisedAngles.trainWorkers.runTrainOnePartition",
           "--symmetry", symmetry,
           "--particlesStarFname", *inputStarFnames,
           "--partition", partition,
           "--trainSaveDir", new_experiment_root]
    if n_epochs:
        cmd += ["--n_epochs", n_epochs]
    if particlesDir is not None:
        cmd += ["--particlesDir", *particlesDir]
    if continueCheckpointFname is not None:
        cmd += ["--continueCheckpointFname", continueCheckpointFname]
    if finetuneCheckpointFname is not None:
        cmd += ["--finetuneCheckpointFname", finetuneCheckpointFname]
    if find_lr:
        cmd += ["--find_lr"]
    if compileModel:
        cmd += ["--compileModel"]
    if val_check_interval:
        cmd += ["--val_check_interval", str(val_check_interval)]
    if overfit_batches:
        cmd += ["--overfit_batches", str(overfit_batches)]

    cmd = [str(arg) for arg in cmd]
    print(f">> Training partition {partition}")
    print(" ".join(cmd))
    p = subprocess.Popen(cmd, env=os.environ.copy())
    pid = p.pid
    returncode = p.wait()
    assert returncode == 0, "Error, training died unexpectedly"
    return pid


if __name__ == "__main__":

    os.environ[constants.PROJECT_NAME + "__ENTRY_POINT"] = "train.py"

    print("---------------------------------------")
    print(" ".join(sys.argv))
    print("---------------------------------------")
    import configfile
    from packaging import version
    from argParseFromDoc import AutoArgumentParser, get_parser_from_function

    assert version.parse(configfile.__version__) >= version.parse("0.1.5")
    from cryoSolver.supervisedAngles.configs.train_config import trainConfig
    from cryoSolver.supervisedAngles.configs.augmentations_config import augmentationConf
    from cryoSolver.supervisedAngles.trainWorkers.checkpointUtils import get_best_checkpoint

    # UPDATE CONFIG ACCORDING CHECKPOINT. COMMAND LINE wins if different value in two places, as it will be changed after the config update
    architectureName = trainConfig.ARTCHITECTURE
    ARTCHITECTURE_PARAMNAME = "--TRAIN__ARTCHITECTURE"

    modelFname = None
    if ARTCHITECTURE_PARAMNAME in sys.argv:
        idx = sys.argv.index(ARTCHITECTURE_PARAMNAME)
        _architectureName = sys.argv[idx + 1]

    elif "--continueCheckpointDir" in sys.argv or "--finetuneCheckpointDir" in sys.argv:  # We do this ugly check before setting the parser to make it more dynamic
        if "--continueCheckpointDir" in sys.argv:
            idx = sys.argv.index("--continueCheckpointDir")
        else:
            idx = sys.argv.index("--finetuneCheckpointDir")
        modelDir = osp.join(osp.expanduser(sys.argv[idx + 1]), "half1", "checkpoints")
        modelFname = get_best_checkpoint(modelDir)
        _architectureName = None
    else:
        _architectureName = architectureName

    if modelFname is not None:
        from supervisedAngles.configs import updateConfigFromModelFname

        updateConfigFromModelFname(modelFname)
        architectureName = trainConfig.ARTCHITECTURE
        assert not _architectureName or architectureName == _architectureName, \
            f"Error, selected architectureName {_architectureName} not compatible with checkpoint {architectureName}"
    else:
        architectureName = _architectureName
        trainConfig.ARTCHITECTURE = architectureName

    if architectureName.split(".")[-1].startswith("InPlane"):  # TODO: Remove this legacy function
        augmentationConf.remove_fullPoseOnly()
    else:
        augmentationConf.remove_inplaneOnly()

    # UPDATE CONFING ACCORDIG COMMAND LINE

    from cryoSolver.supervisedAngles.supervisedAnglesConfig import config

    parser = AutoArgumentParser()
    mainGroup = parser.add_argument_group("main")
    get_parser_from_function(train, parser=mainGroup, args_to_ignore=["optuna_trial_for_prunner"])
    configGroup = parser.add_argument_group("config")
    config.add_args_to_argparse(configGroup)
    args = parser.parse_args_groups()
    confKwargs = vars(args["config"])
    mainKwargs = vars(args["main"])
    config.update(confKwargs)

    train(**mainKwargs)

"""
ulimit -n 16000

PYTHONPATH="../cryoUtils/" python -m cryoUtils.emPackages.relion.relionProject --particlesStarFname ~/cryo/myProjects/micSimulations/data/temSimulatorOutput/example1/particleImgs/particles.star --mapFname ~/cryo/myProjects/micSimulations/data/temSimulatorOutput/example1/reconstruction/relion_reconstruct_half2.mrc --outdir /tmp/relion_project/
SupervisedAnglesDATA___N_DATAWORKERS=4 PYTHONPATH="../cryoUtils/" python -m cryoSolver.supervisedAngles.trainWorkers.runTrainWithHalfs --particlesStarFname /home/sanchezg/cryo/myProjects/micSimulations/data/temSimulatorOutput/example1/particleImgs/particles.star --projectionsStarFname /tmp/relion_project/proj.star

SupervisedAngleslocalSE3Conf___BATCH_SIZE=32 SupervisedAnglesDATA___N_DATAWORKERS=4 PYTHONPATH="../cryoUtils/" python -m cryoSolver.supervisedAngles.trainWorkers.runTrainWithHalfs --symmetry d2 --particlesStarFname ~/cryo/data/supervisedAngles/BENCHMKARK/BGAL/apo_04882/good_04882/Refine3D/001_after2DclsEval/debug_run_data.star --particlesDir ~/cryo/data/supervisedAngles/BENCHMKARK/BGAL/apo_04882/good_04882 --overfit 40 --TRAIN__N_EPOCHS 2 --trainSaveDir /tmp2/manual_processing/trainCryosolver/ --CACHE__MODEL_CACHE_DIR /tmp/kk
"""
