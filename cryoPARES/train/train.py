import glob
import json
import os
import shutil
import subprocess
import warnings

import psutil
import torch
import sys
import os.path as osp
import tempfile
from typing import Optional, List, Literal
from argParseFromDoc import generate_command_for_argparseFromDoc

from cryoPARES import constants
from cryoPARES.configManager.inject_defaults import inject_defaults_from_config, inject_docs_from_config_params, CONFIG_PARAM
from cryoPARES.configs.mainConfig import main_config
from cryoPARES.constants import DATA_SPLITS_BASENAME
from cryoPARES.scripts.gmm_hists import compare_prob_hists
from cryoPARES.utils.checkpointUtils import get_best_checkpoint
from cryoPARES.utils.paths import get_most_recent_file


class Trainer:
    @inject_docs_from_config_params
    @inject_defaults_from_config(main_config.train, update_config_with_args=True)
    def __init__(self, symmetry: str, particles_star_fname: List[str], train_save_dir: str,
                 particles_dir: Optional[List[str]] = None, n_epochs: int = CONFIG_PARAM(),
                 batch_size: int = CONFIG_PARAM(), #CONFIG_PARAM status with update_config_with_args gets updated in config directly
                 num_dataworkers: int = CONFIG_PARAM(config=main_config.datamanager), #CONFIG_PARAM status with update_config_with_args gets updated in config directly
                 image_size_px_for_nnet: int = CONFIG_PARAM(config=main_config.datamanager.particlesdataset), #CONFIG_PARAM status with update_config_with_args gets updated in config directly
                 sampling_rate_angs_for_nnet: float = CONFIG_PARAM(config=main_config.datamanager.particlesdataset), # CONFIG_PARAM status with update_config_with_args gets updated in config directly
                 mask_radius_angs: Optional[float] = CONFIG_PARAM(config=main_config.datamanager.particlesdataset), # CONFIG_PARAM status with update_config_with_args gets updated in config directly
                 split_halfs: bool = True,
                 continue_checkpoint_dir: Optional[str] = None, finetune_checkpoint_dir: Optional[str] = None,
                 compile_model: bool = False,
                 val_check_interval: Optional[float] = None,
                 overfit_batches: Optional[int] = None,
                 map_fname_for_simulated_pretraining: Optional[List[str]] = None,
                 junk_particles_star_fname: Optional[List[str]] = None,
                 junk_particles_dir: Optional[List[str]] = None,
                 ):
        """
        Train a model on particle data.

        Args:
            symmetry: {symmetry}
            particles_star_fname: {particles_star_fname}
            train_save_dir: {train_save_dir}
            particles_dir: {particles_dir}
            n_epochs: {n_epochs}
            batch_size: {batch_size}
            num_dataworkers: {num_dataworkers}
            image_size_px_for_nnet: {image_size_px_for_nnet}
            sampling_rate_angs_for_nnet: {sampling_rate_angs_for_nnet}
            mask_radius_angs: {mask_radius_angs}
            split_halfs: {split_halfs}
            continue_checkpoint_dir: {continue_checkpoint_dir}
            finetune_checkpoint_dir: {finetune_checkpoint_dir}
            compile_model: {compile_model}
            val_check_interval: {val_check_interval}
            overfit_batches: {overfit_batches}
            map_fname_for_simulated_pretraining: {map_fname_for_simulated_pretraining}
            junk_particles_star_fname: {junk_particles_star_fname}
            junk_particles_dir: {junk_particles_dir}
        """
        self.symmetry = symmetry
        self.particles_star_fname = [osp.expanduser(fname) for fname in particles_star_fname]
        self.particles_dir = particles_dir
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.num_dataworkers = num_dataworkers
        self.split_halfs = split_halfs
        self.continue_checkpoint_dir = continue_checkpoint_dir
        self.finetune_checkpoint_dir = finetune_checkpoint_dir
        self.compile_model = compile_model
        self.val_check_interval = val_check_interval
        self.overfit_batches = overfit_batches
        self.map_fname_for_simulated_pretraining = map_fname_for_simulated_pretraining
        self.junk_particles_star_fname = junk_particles_star_fname
        self.junk_particles_dir = junk_particles_dir
        if self.junk_particles_star_fname:
            if self.junk_particles_dir:
                assert len(self.junk_particles_star_fname) == len(self.junk_particles_dir), ("Error, the"
                                                                                             "number of star filenames needs to match the number of junk particles dirs")
        if map_fname_for_simulated_pretraining is not None:
            for fname in map_fname_for_simulated_pretraining:
                assert os.path.isfile(fname), f"Error, {fname} is not a file"
            assert len(map_fname_for_simulated_pretraining) == len(particles_star_fname), (f"Error,"
                                                                                           f"the number of particle star files and maps for simulation needs to be the same")
            assert not finetune_checkpoint_dir, ("Error, if using map_fname_for_simulated_pretraining, "
                                                 "finetune_checkpoint_dir is not a valid option")

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

        # assert os.path.isdir(train_save_dir), f"Error, training directory {train_save_dir} not found!"
        os.makedirs(train_save_dir, exist_ok=True)
        if self.continue_checkpoint_dir is not None:
            _train_save_dir, version_dir = osp.split(self.continue_checkpoint_dir.rstrip("/"))
            if train_save_dir is not None:
                assert os.path.abspath(_train_save_dir) == os.path.abspath(train_save_dir.rstrip("/")), (
                    f"Error, when continuing a checkpoint, _train_save_dir ({_train_save_dir}) =="
                    f" train_save_dir ({train_save_dir})")
            self.train_save_dir = _train_save_dir
        else:
            self.train_save_dir = os.path.expanduser(train_save_dir)
            version_dir = get_version_to_use(self.train_save_dir, "version_", dir_only=True)

        self.experiment_root = osp.join(self.train_save_dir, version_dir)
        os.makedirs(self.experiment_root, exist_ok=True)

        self._save_command_info()
        self._save_env_vars()
        self._copy_code_files()
        self._save_config_vals()

    def _save_command_info(self):
        from cryoPARES.utils.checkpointUtils import get_version_to_use
        basename = get_version_to_use(
            self.experiment_root,
            basename='command_',
            path_pattern=r'(command_)(\d+)(\.txt)$',
            extension="txt"
        )
        fname = osp.join(self.experiment_root, basename)
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
        fname = osp.join(self.experiment_root, basename)
        with open(fname, 'w') as f:
            json.dump(dict(os.environ), f)

    def _save_config_vals(self):
        from cryoPARES.utils.checkpointUtils import get_version_to_use
        if self.continue_checkpoint_dir:
            prefix = 'continueConfigs_'
        else:
            prefix = 'configs_'
        #TODO: How to deal with finetuning
        basename = get_version_to_use(
            self.experiment_root,
            basename=prefix,
            path_pattern=fr'({prefix}_)(\d+)(\.yml)$',
            extension="yml"
        )
        fname = osp.join(self.experiment_root, basename)
        from cryoPARES.configManager.configParser import export_config_to_yaml
        export_config_to_yaml(main_config, fname)

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
        _copyCode(root_path, osp.join(copycodedir, "cryoPARES"))

    def get_continue_checkpoint_fname(self, partition: Literal["allParticles", "half1", "half2"]):
        if self.continue_checkpoint_dir:
            fname = get_most_recent_file(os.path.join(self.continue_checkpoint_dir, partition, "checkpoints"), "*.ckpt")
            warnings.warn(f"Error, checkpoint not found at {self.continue_checkpoint_dir}")
            return fname
        else:
            return None

    def get_finetune_checkpoint_fname(self, partition: Literal["allParticles", "half1", "half2"]):
        if self.finetune_checkpoint_dir:
            fname = get_best_checkpoint(os.path.join(self.finetune_checkpoint_dir, partition, "checkpoints"), "*.ckpt")
            assert fname is not None, f"Error, checkpoint not found at {self.finetune_checkpoint_dir}"
            return fname
        else:
            return None

    def run(self, config_args):
        """

        :param config_args: The command line arguments provided to modify the config
        :return:
        """
        from cryoPARES.train.runTrainOnePartition import execute_trainOnePartition, check_if_training_partion_done
        torch.set_float32_matmul_precision(main_config.train.float32_matmul_precision)

        if self.finetune_checkpoint_dir is not None:
            self._save_finetune_checkpoint_info()

        partitions = ["allParticles"] if not self.split_halfs else ["half1", "half2"]

        with tempfile.TemporaryDirectory() as tmpdir:
            for partition in partitions:
                if check_if_training_partion_done(self.experiment_root, partition):
                    continue
                if self.map_fname_for_simulated_pretraining:
                    # TODO: Implement simulate_particles
                    from cryoPARES.simulation.simulateParticles import run_simulation
                    sim_star_fnames = []
                    sim_dirs = []
                    for sim_idx, fname in enumerate(self.map_fname_for_simulated_pretraining):
                        simulation_dir = os.path.join(tmpdir, "simulation%d" % sim_idx)
                        os.makedirs(simulation_dir, exist_ok=True)
                        simulation_kwargs = dict(
                            volume=self.map_fname_for_simulated_pretraining[sim_idx],
                            in_star=self.particles_star_fname[sim_idx],
                            output_dir=simulation_dir,
                            basename="projections",
                            images_per_file=10000,
                            batch_size=self.batch_size,
                            num_workers=self.num_dataworkers,
                            apply_ctf=True,
                            use_gpu=main_config.train.use_cuda,
                            gpus=",".join([str(x) for x in range(torch.cuda.device_count())]), #TODO: homogenize API to use list rather than str
                            simulation_mode="central_slice",  #"noise_additive",
                            snr=main_config.train.snr_for_simulation,  #None
                        )
                        cmd = generate_command_for_argparseFromDoc(
                            "cryoPARES.simulation.simulateParticles",
                            fun=run_simulation,
                            use_module=True,
                            python_executable=sys.executable,
                            **simulation_kwargs
                        )
                        print(cmd)
                        subprocess.run(
                            cmd.split(),
                            cwd=os.path.abspath(os.path.join(__file__, "..", "..", "..")), check=True
                        )
                        print(f"simulated data was generated at {simulation_dir}")
                        sim_star_fnames.append(os.path.join(simulation_dir, "projections.star"))
                        sim_dirs.append(simulation_dir)
                    simulation_train_dir = os.path.join(tmpdir, "simulation_train")
                    try:
                        idx_n_epochs = [x.split("=")[0] for x in config_args].index('train.n_epochs')
                        config_args[idx_n_epochs] = f'train.n_epochs={main_config.train.n_epochs_simulation}'
                    except ValueError:
                        pass
                    execute_trainOnePartition(
                        symmetry=self.symmetry,
                        particles_star_fname=sim_star_fnames,
                        train_save_dir=simulation_train_dir,
                        particles_dir=sim_dirs,
                        n_epochs=main_config.train.n_epochs_simulation,
                        partition=partition,
                        compile_model=self.compile_model,
                        val_check_interval=self.val_check_interval,
                        overfit_batches=self.overfit_batches,
                        config_args=config_args
                    )
                    self.finetune_checkpoint_dir = simulation_train_dir
                    config_args[idx_n_epochs] = f'train.n_epochs={main_config.train.n_epochs}'
                    config_fname = get_most_recent_file(self.experiment_root, "configs_*.yml")
                    shutil.copy(config_fname, simulation_train_dir)
                print(f"\nExecuting training for partition {partition}")
                execute_trainOnePartition(
                    symmetry=self.symmetry,
                    particles_star_fname=self.particles_star_fname,
                    train_save_dir=self.experiment_root,
                    particles_dir=self.particles_dir,
                    n_epochs=self.n_epochs,
                    partition=partition,
                    continue_checkpoint_fname=self.get_continue_checkpoint_fname(partition),
                    finetune_checkpoint_fname=self.get_finetune_checkpoint_fname(partition),
                    compile_model=self.compile_model,
                    val_check_interval=self.val_check_interval,
                    overfit_batches=self.overfit_batches,
                    config_args=config_args
                )
                if self.junk_particles_star_fname:
                    # if self.junk_particles_dir:
                    junk_stars = self._infer_particles(self.junk_particles_star_fname, self.junk_particles_dir,
                                                  partition, outdirbasename="junk")
                    val_stars = glob.glob(osp.join(self.experiment_root, partition, DATA_SPLITS_BASENAME, "val",
                                                   "*-particles.star"))
                    if self.particles_dir is None:
                        particles_dir = [os.path.dirname(x) for x in self.particles_star_fname]
                    else:
                        particles_dir = self.particles_dir
                    val_stars = self._infer_particles(val_stars,
                                                      particles_dir,
                                                      partition, outdirbasename="val")
                    compare_prob_hists(fname_good=val_stars, fname_bad=junk_stars, show_plots=False,
                                       plot_fname=osp.join(self.experiment_root, partition,"directional_threshold.png"),
                                       symmetry=self.symmetry, compute_gmm=True)
        print("Training complete!")

    def _save_finetune_checkpoint_info(self):
        from cryoPARES.utils.checkpointUtils import get_version_to_use
        finetune_checkpoint_base = get_version_to_use(
            self.experiment_root,
            basename='finetuneCheckpoint_',
            path_pattern=r'(finetuneCheckpoint_)(\d+)(\.txt)$',
            extension="txt"
        )
        with open(osp.join(self.experiment_root, finetune_checkpoint_base), "w") as f:
            f.write(f"finetuneCheckpoint: {self.finetune_checkpoint_dir}")

    def _infer_particles(self, particles_star_fnames, particles_dirs, partition, outdirbasename):
        from cryoPARES.inference.infer import distributed_inference
        results_dir = osp.join(self.experiment_root, partition, "inference", outdirbasename)
        if particles_dirs is None:
            particles_dirs = [None] * len(particles_star_fnames)
        for idx, (fname, dirname) in enumerate(zip(particles_star_fnames, particles_dirs)):
            infer_kwars = dict(
                particles_star_fname=fname,
                particles_dir=dirname,
                checkpoint_dir=self.experiment_root,
                results_dir=results_dir + f"_{idx}",
                data_halfset=partition,
                model_halfset=partition,
                skip_localrefinement=True,
                skip_reconstruction=True,
            )
            cmd = generate_command_for_argparseFromDoc(
                "cryoPARES.inference.infer",
                fun=distributed_inference,
                use_module=True,
                python_executable=sys.executable,
                **infer_kwars
            )
            print(cmd)
            subprocess.run(
                cmd.split(),
                cwd=os.path.abspath(os.path.join(__file__, "..", "..", "..")), check=True
            )
        star_fnames = glob.glob(osp.join(results_dir+"_*", f"particles*_{partition}.star"))
        return star_fnames
def main():
    os.environ[constants.PROJECT_NAME + "__ENTRY_POINT"] = "train.py"
    print("---------------------------------------")
    print(" ".join(sys.argv))
    print("---------------------------------------")
    from cryoPARES.configManager.configParser import ConfigArgumentParser, export_config_to_yaml

    parser = ConfigArgumentParser(prog="train_cryoPARES", config_obj=main_config, verbose=True)
    parser.add_args_from_function(Trainer.__init__)
    args, config_args = parser.parse_args()
    Trainer(**vars(args)).run(config_args)


if __name__ == "__main__":
    main()

    """

python -m cryoPARES.train.train  \
--symmetry C1 --particles_star_fname /home/sanchezg/cryo/data/preAlignedParticles/EMPIAR-10166/data/allparticles.star  --train_save_dir /tmp/cryoPARES_train/ --n_epochs 1 --overfit_batches 20 --batch_size 4 --config models.image2sphere.lmax=6 models.image2sphere.so3components.i2sprojector.hp_order=2 models.image2sphere.so3components.s2conv.hp_order=2 models.image2sphere.so3components.so3outputgrid.hp_order=2  models.image2sphere.imageencoder.encoderArtchitecture="resnet" models.image2sphere.imageencoder.resnet.resnetName="resnet18" datamanager.num_dataworkers=1
 
    """
