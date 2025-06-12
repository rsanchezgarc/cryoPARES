import json
import os
import shutil
import warnings
import psutil
import torch
import sys
import os.path as osp
import tempfile
from typing import Optional, List

from cryoPARES import constants
from cryoPARES.configManager.inject_defaults import inject_defaults_from_config, CONFIG_PARAM
from cryoPARES.configs.mainConfig import main_config

class Trainer:
    @inject_defaults_from_config(main_config.train, update_config_with_args=True)
    def __init__(self, symmetry: str, particles_star_fname: List[str], train_save_dir: str,
                 particles_dir: Optional[List[str]] = None, n_epochs: int = CONFIG_PARAM(), split_halfs: bool = True,
                 continue_checkpoint_dir: Optional[str] = None, finetune_checkpoint_dir: Optional[str] = None,
                 compile_model: bool = False, val_check_interval: Optional[float] = None,
                 overfit_batches: Optional[int] = None,
                 map_fname_for_simulated_pretraining: Optional[List[str]] = None):
        """
        Trainer a model on particle data.

        Args:
            symmetry: The point symmetry of the reconstruction
            particles_star_fname: The starfile containing the pre-aligned particles
            train_save_dir: The root directory where models and logs are saved.
            particles_dir: The directory where the particles of the particlesStarFname are located. If not, it is assumed os.dirname(particlesStarFname)
            n_epochs: The number of epochs
            split_halfs: If True, it trains a model for each half of the data
            continue_checkpoint_dir: The path of a pre-trained model to continue training.
            finetune_checkpoint_dir: The path of a pre-trained model to do finetunning
            compile_model: If True, torch will try to compile the model to make training faster.
            val_check_interval: The fraction of an epoch after which the validation set will be evaluated
            overfit_batches: If provided, number of train and validation batches to use
            map_fname_for_simulated_pretraining: If provided, it will run a warmup training on simulations using this maps. They need to match the particlesStarFname order
        """
        self.symmetry = symmetry
        self.particles_star_fname = [osp.expanduser(fname) for fname in particles_star_fname]
        self.particles_dir = particles_dir
        self.n_epochs = n_epochs
        self.split_halfs = split_halfs
        self.continue_checkpoint_dir = continue_checkpoint_dir
        self.finetune_checkpoint_dir = finetune_checkpoint_dir
        self.compile_model = compile_model
        self.val_check_interval = val_check_interval
        self.overfit_batches = overfit_batches
        self.map_fname_for_simulated_pretraining = map_fname_for_simulated_pretraining

        self._validate_inputs()
        self._setup_training_dir(train_save_dir)

    def _validate_inputs(self):
        assert (0 <= int(bool(self.continue_checkpoint_dir)) +
                int(bool(self.finetune_checkpoint_dir)) <= 1), \
            "Error, continue_checkpoint_dir and finetune_checkpoint_dir are mutually exclusive"

        for fname in self.particles_star_fname:
            assert osp.isfile(fname), f"Error, fname {fname} not found"

    def _setup_training_dir(self, train_save_dir: str):
        from cryoPARES.utils.checkpointUtils import get_version_to_use

        assert os.path.isdir(train_save_dir), f"Error, training directory {train_save_dir} not found!"

        if self.continue_checkpoint_dir is not None:
            _train_save_dir, version_dir = osp.split(self.continue_checkpoint_dir)
            if train_save_dir is not None:
                assert _train_save_dir == train_save_dir, (
                    "Error, when continuing a checkpoint, please do not provide a train_save_dir")
            self.train_save_dir = _train_save_dir
        else:
            self.train_save_dir = os.path.expanduser(train_save_dir)
            version_dir = get_version_to_use(self.train_save_dir, "version_", dir_only=True)

        self.experiment_root = osp.join(self.train_save_dir, version_dir)
        os.makedirs(self.experiment_root, exist_ok=True)

        self._save_command_info()
        self._save_env_vars()
        self._copy_code_files()

    def _save_command_info(self):
        from cryoPARES.utils.checkpointUtils import get_version_to_use
        basename = get_version_to_use(
            self.experiment_root,
            basename='command_',
            path_pattern=r'(command_)(\d+)(\.txt)$',
            extension="txt"
        )
        fname = osp.join(self.experiment_root, basename + "_command.txt")
        current_process = psutil.Process()
        _command = " ".join(["'" + x + "'" if x.startswith('{"') else x
                             for x in current_process.cmdline()])
        with open(fname, "w") as f:
            f.write(_command)

    def _save_env_vars(self):
        from cryoPARES.utils.checkpointUtils import get_version_to_use
        basename = get_version_to_use(
            self.experiment_root,
            basename='envs_',
            path_pattern=r'(envs_)(\d+)(\.json)$',
            extension="json"
        )
        fname = osp.join(self.experiment_root, basename + "_env.json")
        with open(fname, 'w') as f:
            json.dump(dict(os.environ), f)

    def _copy_code_files(self):
        from cryoPARES.utils.reproducibility import _copyCode
        from cryoPARES.utils.checkpointUtils import get_version_to_use
        copycodedir_base = get_version_to_use(
            self.experiment_root,
            basename='code_',
            path_pattern=r'(code_)(\d+)$'
        )
        copycodedir = osp.join(self.experiment_root, copycodedir_base)
        os.makedirs(copycodedir, exist_ok=True)

        module_path = osp.abspath(sys.modules[__name__].__file__)
        root_path = osp.dirname(osp.dirname(osp.dirname(module_path)))
        _copyCode(root_path, osp.join(copycodedir, "cryoSolver"))

    def run(self):
        from cryoPARES.train.runTrainOnePartition import execute_trainOnePartition, check_if_training_partion_done
        torch.set_float32_matmul_precision(constants.float32_matmul_precision)
        print(main_config)

        if self.finetune_checkpoint_dir is not None:
            self._save_finetune_checkpoint_info()

        partitions = ["allParticles"] if not self.split_halfs else ["half1", "half2"]

        with tempfile.TemporaryDirectory() as tmpdir:
            for partition in partitions:
                if check_if_training_partion_done(self.experiment_root, partition):
                    continue
                if self.map_fname_for_simulated_pretraining:
                    # TODO: Implement simulate_particles
                    simulatedParticles = simulate_particles(self.map_fname_for_simulated_pretraining, tmpdir)
                    raise NotImplementedError()
                    execute_trainOnePartition()

                execute_trainOnePartition(
                    symmetry=self.symmetry,
                    particles_star_fname=self.particles_star_fname,
                    train_save_dir=self.experiment_root,
                    particles_dir=self.particles_dir,
                    partition=partition,
                    continue_checkpoint_fname=self.continue_checkpoint_dir,
                    finetune_checkpoint_fname=self.finetune_checkpoint_dir,
                    compile_model=self.compile_model,
                    val_check_interval=self.val_check_interval,
                    overfit_batches=self.overfit_batches
                )

        print("Training complete!")

    def _save_finetune_checkpoint_info(self):
        from cryoPARES.utils.checkpointUtils import get_version_to_use
        finetune_checkpoint_base = get_version_to_use(
            self.experiment_root,
            basename='finetuneCheckpoin_',
            path_pattern=r'(finetuneCheckpoin_)(\d+)(\.txt)$',
            extension="txt"
        )
        with open(osp.join(self.experiment_root, finetune_checkpoint_base), "w") as f:
            f.write(f"finetuneCheckpoin: {self.finetune_checkpoint_dir}")


if __name__ == "__main__":
    os.environ[constants.PROJECT_NAME + "__ENTRY_POINT"] = "train.py"
    print("---------------------------------------")
    print(" ".join(sys.argv))
    print("---------------------------------------")

    from argParseFromDoc import AutoArgumentParser

    parser = AutoArgumentParser(prog="train cryoPARES")
    parser.add_args_from_function(Trainer.__init__)
    args = parser.parse_args()
    Trainer(**vars(args)).run()