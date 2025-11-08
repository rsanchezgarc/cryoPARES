import glob
import json
import os
import shutil
import warnings

import psutil
import torch
import sys
import os.path as osp
import tempfile
from typing import Optional, List, Literal
from argParseFromDoc import generate_command_for_argparseFromDoc

from cryoPARES import constants
from cryoPARES.utils.subprocessUtils import run_subprocess_with_error_summary
from autoCLI_config import inject_defaults_from_config, inject_docs_from_config_params, CONFIG_PARAM
from cryoPARES.configs.mainConfig import main_config
from cryoPARES.constants import DATA_SPLITS_BASENAME
from cryoPARES.scripts.gmm_hists import compare_prob_hists
from cryoPARES.utils.checkpointUtils import get_best_checkpoint
from cryoPARES.utils.paths import get_most_recent_file, convert_config_args_to_absolute_paths


def _check_if_in_config_args(param_path: str) -> bool:
    """
    Check if a parameter will be provided via --config at parse time.

    This inspects sys.argv to see if --config is present and would provide the given
    parameter (either via key=value or in a YAML file).

    Args:
        param_path: The config path to check (e.g., 'datamanager.particlesdataset.image_size_px_for_nnet')

    Returns:
        True if --config appears to provide this parameter, False otherwise
    """
    import sys

    # If --config is not present, definitely not in config
    if '--config' not in sys.argv:
        return False

    # Find all --config values
    try:
        config_idx = sys.argv.index('--config')
        # Collect all config arguments until next flag or end
        config_args = []
        for i in range(config_idx + 1, len(sys.argv)):
            if sys.argv[i].startswith('--'):
                break
            config_args.append(sys.argv[i])

        # Check if any config arg is a key=value containing our parameter
        for arg in config_args:
            if '=' in arg and param_path in arg:
                return True
            # If it's a YAML file, we'd need to load it to check - too complex
            # for this pre-parse check, so we conservatively assume it might be there
            if arg.endswith('.yaml') or arg.endswith('.yml'):
                return True  # Assume config file might contain it

        return False
    except (ValueError, IndexError):
        return False


def estimate_particle_stacks_size(star_fnames: List[str], particles_dir: Optional[List[str]]) -> int:
    """
    Estimate the total disk space required for particle stacks by summing sizes of all unique .mrcs files.

    Args:
        star_fnames: List of paths to RELION .star files
        particles_dir: List of root directories for particle paths (one per star file, or None)

    Returns:
        Total size in bytes of all unique particle stack files
    """
    from starstack.particlesStar import ParticlesStarSet, split_particle_and_fname

    total_size = 0
    unique_stack_files = set()

    for idx, star_fname in enumerate(star_fnames):
        # Get the corresponding particles_dir for this star file
        pdir = particles_dir[idx] if particles_dir and idx < len(particles_dir) else None
        if pdir:
            pdir = osp.expanduser(pdir)

        # Load the star file
        ps = ParticlesStarSet(starFname=star_fname, particlesDir=pdir)

        # Extract unique stack filenames
        stack_fnames = ps.particles_md["rlnImageName"].map(
            lambda x: split_particle_and_fname(x)["basename"]
        )

        # For each unique stack file, get its full path and size
        for stack_fname in stack_fnames.unique():
            # Construct full path
            if pdir:
                full_path = osp.join(pdir, stack_fname)
            else:
                full_path = stack_fname

            # Add to set to avoid counting duplicates across star files
            if full_path not in unique_stack_files:
                unique_stack_files.add(full_path)
                if osp.exists(full_path):
                    total_size += osp.getsize(full_path)

    return total_size


def check_disk_space(target_dir: str, required_bytes: int, safety_factor: float = 2.0) -> None:
    """
    Check if target directory has sufficient disk space for simulation.

    Args:
        target_dir: Directory where temporary files will be created
        required_bytes: Estimated space needed in bytes
        safety_factor: Multiply required space by this factor (default: 2.0 for 2x safety margin)

    Raises:
        RuntimeError: If insufficient disk space is available
    """
    # Ensure target directory exists (if None, use system temp dir)
    if target_dir is None:
        target_dir = tempfile.gettempdir()

    # Get disk usage stats
    disk_stats = shutil.disk_usage(target_dir)
    available_bytes = disk_stats.free
    required_with_margin = required_bytes * safety_factor

    # Convert to human-readable units
    def bytes_to_human(n):
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if n < 1024.0:
                return f"{n:.2f} {unit}"
            n /= 1024.0
        return f"{n:.2f} PB"

    if available_bytes < required_with_margin:
        raise RuntimeError(
            f"\nInsufficient disk space for simulated particle generation!\n"
            f"  Target directory: {target_dir}\n"
            f"  Available space: {bytes_to_human(available_bytes)}\n"
            f"  Required space (with {safety_factor}x safety margin): {bytes_to_human(required_with_margin)}\n"
            f"  Estimated particle data size: {bytes_to_human(required_bytes)}\n\n"
            f"Solutions:\n"
            f"  1. Free up space in {target_dir}\n"
            f"  2. Use a different directory with more space via:\n"
            f"     --config train.simulation_tmp_dir=/path/to/large/disk\n"
        )


def create_simulation_config(config_args, tmpdir, n_epochs_simulation):
    """
    Create a temporary config file for simulation with patched n_epochs.

    This function takes the original config arguments (which may include YAML files
    and/or key=val pairs), applies them to a copy of main_config, patches train.n_epochs
    to use n_epochs_simulation, and writes the result to a temporary YAML file.

    Args:
        config_args: List of config arguments (YAML files and/or key=val pairs), or None
        tmpdir: Temporary directory to write config file
        n_epochs_simulation: Value to use for train.n_epochs in simulation

    Returns:
        str: Path to temporary config file
    """
    from copy import deepcopy
    from autoCLI_config import export_config_to_yaml, ConfigOverrideSystem

    # Create a deep copy of main_config to avoid modifying the global config
    config_copy = deepcopy(main_config)

    # Apply all config overrides (both YAML files and key=val pairs) if provided
    if config_args:
        # Separate YAML files from key=val pairs
        yaml_files = [arg for arg in config_args if arg.endswith(('.yaml', '.yml'))]
        key_val_pairs = [arg for arg in config_args if not arg.endswith(('.yaml', '.yml'))]

        # Apply YAML files first
        for yaml_file in yaml_files:
            ConfigOverrideSystem.update_config_from_file(config_copy, yaml_file)

        # Then apply key=val pairs (these can override values from YAML files)
        if key_val_pairs:
            ConfigOverrideSystem.update_config_from_configstrings(config_copy, key_val_pairs)

    # Patch train.n_epochs for simulation
    config_copy.train.n_epochs = n_epochs_simulation

    # Export to temporary YAML file
    temp_config_path = os.path.join(tmpdir, "simulation_config.yaml")
    export_config_to_yaml(config_copy, temp_config_path)

    return temp_config_path


class Trainer:
    @inject_docs_from_config_params
    @inject_defaults_from_config(main_config.train, update_config_with_args=True)
    def __init__(self, symmetry: str, particles_star_fname: List[str], train_save_dir: str,
                 # image_size_px_for_nnet is REQUIRED, but can be provided via --config or CLI.
                 # Uses is_required_arg_for_cli_fun to check sys.argv and determine if it
                 # should be marked as required for argparse.
                 image_size_px_for_nnet: Optional[int] = CONFIG_PARAM(
                     config=main_config.datamanager.particlesdataset,
                     is_required_arg_for_cli_fun=lambda: not _check_if_in_config_args('datamanager.particlesdataset.image_size_px_for_nnet')
                 ),
                 particles_dir: Optional[List[str]] = None, n_epochs: int = CONFIG_PARAM(),
                 batch_size: int = CONFIG_PARAM(), num_dataworkers: int = CONFIG_PARAM(config=main_config.datamanager),
                 sampling_rate_angs_for_nnet: float = CONFIG_PARAM(config=main_config.datamanager.particlesdataset),
                 mask_radius_angs: Optional[float] = CONFIG_PARAM(config=main_config.datamanager.particlesdataset),
                 split_halves: bool = True, continue_checkpoint_dir: Optional[str] = None,
                 finetune_checkpoint_dir: Optional[str] = None, compile_model: bool = False,
                 val_check_interval: Optional[float] = None, overfit_batches: Optional[int] = None,
                 map_fname_for_simulated_pretraining: Optional[List[str]] = None,
                 junk_particles_star_fname: Optional[List[str]] = None, junk_particles_dir: Optional[List[str]] = None):
        """
        Train a model on particle data.

        Args:
            symmetry: {symmetry}
            particles_star_fname: {particles_star_fname}
            train_save_dir: {train_save_dir}
            image_size_px_for_nnet: {image_size_px_for_nnet}
            particles_dir: {particles_dir}
            n_epochs: {n_epochs}
            batch_size: {batch_size}
            num_dataworkers: {num_dataworkers}
            sampling_rate_angs_for_nnet: {sampling_rate_angs_for_nnet}
            mask_radius_angs: {mask_radius_angs}
            split_halves: {split_halves}
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
        self.split_halves = split_halves
        self.continue_checkpoint_dir = continue_checkpoint_dir
        self.finetune_checkpoint_dir = finetune_checkpoint_dir
        self.compile_model = compile_model
        self.val_check_interval = val_check_interval
        self.overfit_batches = overfit_batches
        self.map_fname_for_simulated_pretraining = map_fname_for_simulated_pretraining
        self.junk_particles_star_fname = junk_particles_star_fname
        self.junk_particles_dir = junk_particles_dir

        # Validate and set image_size_px_for_nnet
        # Note: This is a safety check. The main validation happens in main() after
        # --config is processed. This check catches the case where Trainer() is called
        # directly (not via main()), or if something goes wrong with config injection.
        if image_size_px_for_nnet is not None:
            main_config.datamanager.particlesdataset.image_size_px_for_nnet = image_size_px_for_nnet
        else:
            raise ValueError(
                "image_size_px_for_nnet is required. Provide it via:\n"
                "  --image_size_px_for_nnet VALUE\n"
                "  --config datamanager.particlesdataset.image_size_px_for_nnet=VALUE\n"
                "  --config FILE.yaml (containing datamanager.particlesdataset.image_size_px_for_nnet)"
            )
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
        from autoCLI_config import export_config_to_yaml
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

        partitions = ["allParticles"] if not self.split_halves else ["half1", "half2"]

        # Check disk space before simulation if needed
        if self.map_fname_for_simulated_pretraining:
            required_space = estimate_particle_stacks_size(
                self.particles_star_fname,
                self.particles_dir
            )
            target_tmp_dir = main_config.train.simulation_tmp_dir or tempfile.gettempdir()
            check_disk_space(target_tmp_dir, required_space, safety_factor=2.0)

        with tempfile.TemporaryDirectory(dir=main_config.train.simulation_tmp_dir) as tmpdir:
            for partition in partitions:
                if check_if_training_partion_done(self.experiment_root, partition):
                    continue
                if self.map_fname_for_simulated_pretraining:
                    sim_star_fnames = []
                    sim_dirs = []
                    for sim_idx, fname in enumerate(self.map_fname_for_simulated_pretraining):
                        simulation_dir = os.path.join(tmpdir, "simulation%d" % sim_idx)
                        os.makedirs(simulation_dir, exist_ok=True)

                        # Determine GPU configuration for simulation
                        n_gpus = main_config.train.n_gpus_for_simulation
                        if not main_config.train.use_cuda or not torch.cuda.is_available():
                            n_gpus = 0
                        elif n_gpus == -1:
                            # Use all available GPUs
                            n_gpus = torch.cuda.device_count()

                        simulation_kwargs = dict(
                            volume=self.map_fname_for_simulated_pretraining[sim_idx],
                            in_star=self.particles_star_fname[sim_idx],
                            output_dir=simulation_dir,
                            basename="projections",
                            images_per_file=5000,
                            batch_size=self.batch_size,
                            num_dataworkers=self.num_dataworkers,
                            n_gpus_for_simulation=n_gpus,
                            apply_ctf=True,
                            simulation_mode="central_slice",
                            snr_for_simulation=main_config.train.snr_for_simulation,
                        )
                        # Import the CLI function to generate command
                        from cryoPARES.simulation.simulateParticles import simulate_particles_cli

                        cmd = generate_command_for_argparseFromDoc(
                            "cryoPARES.simulation.simulateParticles",
                            fun=simulate_particles_cli,
                            use_module=True,
                            python_executable=sys.executable,
                            **simulation_kwargs
                        )
                        # Propagate config args to simulation subprocess, but filter out
                        # datamanager configs that don't apply to simulation
                        if config_args:
                            # Filter out datamanager.particlesdataset configs that are training-specific
                            filtered_config = [arg for arg in config_args
                                             if not arg.startswith('datamanager.particlesdataset.')]
                            if filtered_config:
                                cmd += " --config " + " ".join(filtered_config)
                        print(cmd)
                        run_subprocess_with_error_summary(
                            cmd.split(),
                            cwd=os.path.abspath(os.path.join(__file__, "..", "..", "..")),
                            description="Simulating particles for pre-training"
                        )
                        print(f"simulated data was generated at {simulation_dir}")
                        sim_star_fnames.append(os.path.join(simulation_dir, "projections.star"))
                        sim_dirs.append(simulation_dir)
                    simulation_train_dir = os.path.join(tmpdir, "simulation_train")
                    # Create a temporary config file for simulation with patched n_epochs
                    sim_config_path = create_simulation_config(
                        config_args,
                        tmpdir,
                        main_config.train.n_epochs_simulation
                    )
                    sim_config_args = [sim_config_path]

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
                        config_args=sim_config_args
                    )
                    self.finetune_checkpoint_dir = simulation_train_dir
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
                print(f"\nFinished training for partition {partition}.")
                if self.junk_particles_star_fname:
                    junk_stars = self._infer_particles(self.junk_particles_star_fname, self.junk_particles_dir,
                                                  partition, outdirbasename="junk", config_args=config_args)

                    if self.particles_dir is None:
                        particles_dir = [os.path.dirname(x) for x in self.particles_star_fname]
                    else:
                        particles_dir = self.particles_dir

                    val_stars = glob.glob(osp.join(self.experiment_root, partition, DATA_SPLITS_BASENAME, "val",
                                                   "*-particles*.star"))
                    assert val_stars, (f"Error, no validation data found at "
                                       f"{os.path.join(self.experiment_root, partition, DATA_SPLITS_BASENAME, 'val')}.")
                    val_stars = self._infer_particles(val_stars,
                                                      particles_dir,
                                                      partition, outdirbasename="val",
                                                      config_args=config_args)
                    assert val_stars, (f"Error, no validation predictions found")
                    assert junk_stars, (f"Error, no junk_stars predictions found")
                    #TODO: compare_prob_hists breaks when multiple gpus are used
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

    def _infer_particles(self, particles_star_fnames, particles_dirs, partition, outdirbasename, config_args=None):
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
                n_first_particles=None if self.overfit_batches in [1, None] else self.overfit_batches*self.batch_size,
            )
            cmd = generate_command_for_argparseFromDoc(
                "cryoPARES.inference.infer",
                fun=distributed_inference,
                use_module=True,
                python_executable=sys.executable,
                **infer_kwars
            )
            if config_args is not None:
                cmd += " --config " + " ".join(config_args)
            print(cmd)
            run_subprocess_with_error_summary(
                cmd.split(),
                cwd=os.path.abspath(os.path.join(__file__, "..", "..", "..")),
                description=f"Running inference on {outdirbasename} particles"
            )
        if partition in ["half1", "half2"]:
            star_fnames = glob.glob(osp.join(results_dir+"_*", f"*_{partition}.star"))
        else:
            star_fnames = glob.glob(osp.join(results_dir+"_*", f"*.star"))
        return star_fnames
def main():
    os.environ[constants.PROJECT_NAME + "__ENTRY_POINT"] = "train.py"
    print("---------------------------------------")
    print(" ".join(sys.argv))
    print("---------------------------------------")
    from autoCLI_config import ConfigArgumentParser

    parser = ConfigArgumentParser(prog="train_cryoPARES", config_obj=main_config, verbose=True)
    parser.add_args_from_function(Trainer.__init__)

    # Parse arguments
    # Note: image_size_px_for_nnet uses is_required_arg_for_cli_fun in CONFIG_PARAM
    # to intelligently determine if it should be required based on --config presence
    args, config_args = parser.parse_args()

    # Convert any relative config file paths to absolute paths for subprocess propagation
    kwargs = vars(args)
    config_args = convert_config_args_to_absolute_paths(config_args)
    Trainer(**kwargs).run(config_args)


if __name__ == "__main__":
    main()

    """

python -m cryoPARES.train.train  \
--symmetry C1 --image_size_px_for_nnet 160 --particles_star_fname /home/sanchezg/cryo/data/preAlignedParticles/EMPIAR-10166/data/allparticles.star  --train_save_dir /tmp/cryoPARES_train/ --n_epochs 1 --overfit_batches 20 --batch_size 4 --config models.image2sphere.lmax=6 models.image2sphere.so3components.i2sprojector.hp_order=2 models.image2sphere.so3components.s2conv.hp_order=2 models.image2sphere.so3components.so3outputgrid.hp_order=2  models.image2sphere.imageencoder.encoderArtchitecture="resnet" models.image2sphere.imageencoder.resnet.resnetName="resnet18" datamanager.num_dataworkers=1
 
    """
