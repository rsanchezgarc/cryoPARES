import glob
import os
import os.path as osp
import shutil
import sys
import tempfile
import warnings
import re
import psutil
import yaml
from typing import Optional, List, Dict, Any
import subprocess

import numpy as np
import pandas as pd
import starfile
import torch

from cryoPARES import constants
from cryoPARES.configManager.inject_defaults import inject_defaults_from_config, CONFIG_PARAM
from cryoPARES.configs.mainConfig import main_config


class Inferencer:
    @inject_defaults_from_config(main_config.inference, update_config_with_args=True)
    def __init__(self,
                 particles_star_fname: str,
                 model_fnames: List[str],
                 out_dir: str,
                 skip_reconstruct: bool = False,
                 particles_dir: Optional[str] = None,
                 skip_neural_network: bool = False,
                 skip_local_refinement: bool = False,
                 keep_original_sphere: bool = False,
                 keep_original_inplane: bool = False,
                 keep_original_shifts: bool = False,
                 reference_map_fname: Optional[str] = None,
                 mask_or_pdb_fname: Optional[str] = None,
                 combining_mode: str = CONFIG_PARAM(),
                 limit_to_n_particles: Optional[int] = None,
                 rm_particles_lt_score: Optional[List[float]] = None):
        """Initialize inference pipeline.

        Args:
            particles_star_fname: Path to the starfile containing particles
            model_fnames: List of checkpoint directories. One or two for half-maps
            out_dir: Directory to save results
            skip_reconstruct: Skip reconstruction step if True
            particles_dir: Directory containing particles (defaults to starfile dir)
            skip_neural_network: Skip neural network inference if True
            skip_local_refinement: Skip local refinement step if True
            keep_original_sphere: Keep original sphere parameters if True
            keep_original_inplane: Keep original in-plane parameters if True
            keep_original_shifts: Keep original shift parameters if True
            reference_map_fname: Map for brute force in-plane alignment
            mask_or_pdb_fname: Mask or PDB for Relion post-processing
            combining_mode: How to combine sampled orientations ("copy" or "average")
            limit_to_n_particles: Only use first N particles if provided
            rm_particles_lt_score: Remove particles below score thresholds
        """
        self._validate_inputs(particles_star_fname, model_fnames, out_dir,
                              combining_mode, rm_particles_lt_score)

        self.particles_star_fname = osp.expanduser(particles_star_fname)
        self.model_fnames = [osp.expanduser(f) for f in model_fnames]
        self.out_dir = out_dir
        self.skip_reconstruct = skip_reconstruct
        self.particles_dir = particles_dir or osp.dirname(particles_star_fname)
        self.skip_neural_network = skip_neural_network
        self.skip_local_refinement = skip_local_refinement
        self.keep_original_sphere = keep_original_sphere
        self.keep_original_inplane = keep_original_inplane
        self.keep_original_shifts = keep_original_shifts
        self.reference_map_fname = osp.expanduser(reference_map_fname) if reference_map_fname else None
        self.mask_or_pdb_fname = mask_or_pdb_fname
        self.combining_mode = combining_mode
        self.limit_to_n_particles = limit_to_n_particles
        self.rm_particles_lt_score = rm_particles_lt_score

        self.model_fnames, self.use_halfs = self._resolve_model_fnames()
        self.symmetry = self._determine_symmetry()
        self._save_command_info()

    def _validate_inputs(self, particles_star_fname: str, model_fnames: List[str],
                         out_dir: str, combining_mode: str,
                         rm_particles_lt_score: Optional[List[float]]) -> None:
        """Validate input parameters."""
        assert osp.exists(particles_star_fname), f"Particles starfile {particles_star_fname} not found"
        assert osp.isdir(out_dir), f"Output directory {out_dir} does not exist"
        assert combining_mode.startswith(("copy", "average")), "Invalid combining_mode"
        assert 1 <= len(model_fnames) <= 2, "Error, you need to provide either one or two model_fnames"
        if rm_particles_lt_score:
            if len(rm_particles_lt_score) == 1:
                rm_particles_lt_score *= 2
            elif len(rm_particles_lt_score) != 2:
                raise ValueError("rm_particles_lt_score must be one or two numbers")

    def _resolve_model_fnames(self) -> tuple[List[str], bool]:
        """Resolve model filenames and determine if using half maps."""
        use_halfs = False
        model_fnames = self.model_fnames

        if len(model_fnames) == 1 and osp.isdir(model_fnames[0]):
            model_fnames = glob.glob(osp.join(model_fnames[0], f"**/{constants.BEST_CHECKPOINT_BASENAME}"),
                                     recursive=True)
            use_halfs = (len(model_fnames) == 2)
        elif any(osp.isdir(x) for x in model_fnames):
            raise ValueError("One of the model filenames is a directory")

        return model_fnames, use_halfs

    def _determine_symmetry(self) -> str:
        """Determine symmetry from models"""

        symmetries = set()
        for fname in self.model_fnames:
            if fname.endswith(constants.BEST_CHECKPOINT_BASENAME):
                conf = torch.load(fname, weights_only=False)["hyper_parameters"]
                sym = conf["symmetry"]
            else:
                sym = torch.jit.load(fname).symmetry
            symmetries.add(sym)

        assert len(symmetries) == 1, "Error, Models have different symmetries"
        return symmetries.pop()

    def _save_command_info(self) -> None:
        """Save command information to output directory."""
        fname = osp.join(self.out_dir, "command.txt")
        current_process = psutil.Process()
        # parent_process = current_process.parent()
        parent_command = " ".join(
            ["'" + x + "'" if x.startswith('{"') else x
             for x in current_process.cmdline()]
        )
        with open(fname, "w") as f:
            f.write(parent_command)

    def _setup_accelerator(self):
        """Set up compute accelerator (GPU/CPU)."""
        from cryoPARES.utils.torchUtils import accelerator_selector

        accel, dev_count = accelerator_selector(
            main_config.inference.use_cuda,
            n_cpus_torch=main_config.inference.n_cpus_if_no_cuda
        )
        print(f'devices={dev_count} accelerator={accel}', flush=True)
        return accel, dev_count

    def _run_neural_network(self, star_fnames: List[str], raw_preds_dir: str,
                            accel: str, dev_count: int) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """Run neural network inference on particles."""

        infer_config = main_config.inference
        df_list = []

        for i, (model_fname, particles_star_fname) in enumerate(zip(self.model_fnames, star_fnames)):
            star_out_fname = osp.join(raw_preds_dir, f'nnet_predictions_{i}.star')

            cmd = [
                "python", "-m", "cryoPARES.inference.neuralNetInference",
                "--particles_star_fname", particles_star_fname,
                "--model_fname", model_fname,
                "--star_out_fname", star_out_fname,
                "--n_dataworkers", str(infer_config.N_DATAWORKERS),
                "--batch_size", str(infer_config.BATCH_SIZE),
            ]

            if self.particles_dir:
                cmd.extend(["--particles_dir", self.particles_dir])
            if self.keep_original_sphere:
                cmd.append("--keep_original_sphere")
            if self.keep_original_inplane:
                cmd.append("--keep_original_inplane")
            if self.keep_original_shifts:
                cmd.append("--keep_original_shifts")
            if self.rm_particles_lt_score:
                cmd.extend(["--rm_particles_lt_score", str(self.rm_particles_lt_score[0])])
            if accel == "cpu":
                cmd.append("--NOT_use_gpu")

            print(" ".join(cmd))
            subprocess.run(cmd, check=True, env=os.environ.copy())

            star_data = starfile.read(star_out_fname)
            out_df = star_data["particles"]
            optics = star_data["optics"]

            if len(self.model_fnames) > 1 and "rlnRandomSubset" not in out_df:
                out_df["rlnRandomSubset"] = 1 + i
            df_list.append(out_df)

        return pd.concat(df_list), optics

    def _run_local_refinement(self, star_fname: str, out_dir: str,
                              accel: str, dev_count: int) -> str:
        """Run local refinement step."""

        star_out_fname = osp.join(out_dir, "local_refinement.star")

        if self.use_halfs and not self.reference_map_fname:
            reference_map_fnames = []
            for model_fname in self.model_fnames:
                ref_map = osp.abspath(osp.join(
                    osp.dirname(model_fname), "..", "data_splits", "train",
                    "reconstruct", "relion_reconstruct.mrc"
                ))
                assert osp.exists(ref_map), f"Reference map {ref_map} not found"
                reference_map_fnames.append(ref_map)
        else:
            assert self.reference_map_fname, "Local refinement requires a reference map"
            reference_map_fnames = [self.reference_map_fname]

        cmd = [
            "python", "-m", "cryoPARES.inference.localRefinement",
            "--particles_star_fname", star_fname,
            "--reference_map_fnames", *reference_map_fnames,
            "--symmetry", self.symmetry,
            "--star_out_fname", star_out_fname,
            "--n_dataworkers", str(infer_config.N_DATAWORKERS),
        ]

        if self.particles_dir:
            cmd.extend(["--particles_dir", self.particles_dir])
        if self.rm_particles_lt_score:
            cmd.extend(["--rm_particles_lt_score", str(self.rm_particles_lt_score[1])])

        model_dir = osp.dirname(self.model_fnames[0])
        cmd.extend(["--scoring_parameters_dir", model_dir])

        print(" ".join(cmd))
        subprocess.run(cmd, check=True, env=os.environ.copy())

        return star_out_fname

    def _run_reconstruction(self, star_fname: str, raw_preds_dir: str,
                            accel: str, dev_count: int) -> None:
        """Run Relion reconstruction."""

        reconstruction_dir = osp.join(self.out_dir, "reconstruction")
        os.makedirs(reconstruction_dir, exist_ok=True)

        to_reconstruct_dir = osp.join(raw_preds_dir, "toReconstruct")
        os.makedirs(to_reconstruct_dir, exist_ok=True)

        # Create symlinks for particle files
        for fname in os.listdir(self.particles_dir):
            os.symlink(
                osp.join(self.particles_dir, fname),
                osp.join(to_reconstruct_dir, fname)
            )

        # Prepare star file
        data = starfile.read(star_fname)
        to_reconstruct_star = osp.join(to_reconstruct_dir, "to_reconstruct_particles.star")
        starfile.write(data, to_reconstruct_star)

        cmd = [
            "python", "-m", "cryoPARES.reconstruction.relionReconstruct",
            "--particles_star_fname", to_reconstruct_star,
            "--outdir", reconstruction_dir,
            "--symmetry", self.symmetry,
            "--use_per_particle_weight",
            "--n_workers", str(
                infer_config.N_DATAWORKERS * dev_count
                if accel == "gpu" else infer_config.N_DATAWORKERS
            ),
        ]

        if self.mask_or_pdb_fname:
            cmd.extend(["--mask_or_pdb_fname", self.mask_or_pdb_fname])

        print(" ".join(cmd))
        subprocess.run(cmd, check=True, env=os.environ.copy())

        if self.mask_or_pdb_fname and self.mask_or_pdb_fname.endswith(".pdb"):
            self._calculate_phenix_fsc(reconstruction_dir)

        def _calculate_phenix_fsc(self, reconstruction_dir: str) -> None:
            """Calculate FSC using Phenix if PDB provided."""
            from cryoPARES.utils.phenix import calculate_fsc

            phenix_fscs = calculate_fsc(
                [
                    osp.join(reconstruction_dir, "relion_reconstruct_half1.mrc"),
                    osp.join(reconstruction_dir, "relion_reconstruct_half2.mrc")
                ],
                pdb_file=self.mask_or_pdb_fname,
                align_pdb=True
            )
            print("Phenix FSCs")
            print(yaml.dump(phenix_fscs))



    def run(self) -> None:
        """Run the complete inference pipeline."""
        torch.set_float32_matmul_precision(constants.float32_matmul_precision)

        accel, dev_count = self._setup_accelerator()
        print(f"Starting inference with symmetry {self.symmetry}")

        with tempfile.TemporaryDirectory() as raw_preds_dir:

            # Neural network inference
            if not self.skip_neural_network:
                out_df, optics = self._run_neural_network(
                    self.particles_star_fname, raw_preds_dir, accel, dev_count
                )
            else:
                data = starfile.read(self.particles_star_fname)
                out_df = data["particles"]
                optics = data["optics"]
                if "rlnParticleFigureOfMerit" not in out_df.columns:
                    out_df["rlnParticleFigureOfMerit"] = 1.0


            # Save neural network predictions
            current_star_fname = osp.join(self.out_dir, "nnet_predictions.star")
            starfile.write(
                {"optics": optics, "particles": out_df},
                current_star_fname,
                overwrite=True
            )

            # Local refinement
            if not self.skip_local_refinement:
                print("Running local refinement")
                current_star_fname = self._run_local_refinement(
                    current_star_fname, self.out_dir, accel, dev_count
                )
            else:
                warnings.warn("Local refinement skipped as requested.")

            # Reconstruction
            if not self.skip_reconstruct:
                print("Running Relion reconstruction")
                self._run_reconstruction(
                    current_star_fname, raw_preds_dir, accel, dev_count
                )
            else:
                warnings.warn("Reconstruction skipped as requested.")

            # Save final particles
            final_star_fname = osp.join(self.out_dir, "final_particles.star")
            shutil.copyfile(current_star_fname, final_star_fname)

        print("Inference completed successfully!")

def main():
    """Main entry point for inference."""
    from argParseFromDoc import AutoArgumentParser
    from lightning_fabric.utilities.seed import seed_everything

    seed_everything(main_config.inference.random_seed)
    parser = AutoArgumentParser(prog="inference cryoPARES")
    parser.add_args_from_function(Inferencer.__init__)

    # # Add configuration options
    # config_group = parser.add_argument_group("config")
    # infer_config.add_args_to_argparse(config_group)

    args = parser.parse_args()

    # Run inference
    inferencer = Inferencer(**vars(args))
    inferencer.run()

    if __name__ == "__main__":
        main()