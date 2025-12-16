import glob
import os
import sys
from time import time
from pathlib import Path

import torch
import yaml
from lightning import seed_everything
from typing import Optional, List, Literal

from cryoPARES import constants
from autoCLI_config import ConfigOverrideSystem
from autoCLI_config import inject_defaults_from_config, CONFIG_PARAM
from cryoPARES.configs.mainConfig import main_config
from cryoPARES.inference.daemon.queueManager import queue_connection, get_all_available_items, DEFAULT_IP, \
    DEFAULT_PORT, DEFAULT_AUTHKEY
from cryoPARES.inference.daemon.spoolingFiller import POISON_PILL
from cryoPARES.inference.inferencer import SingleInferencer
from cryoPARES.utils.paths import get_most_recent_file

class DaemonInferencer(SingleInferencer):

    @inject_defaults_from_config(main_config.inference, update_config_with_args=True)
    def __init__(self,
                 checkpoint_dir: str,
                 results_dir: str,
                 net_address:str = DEFAULT_IP,
                 net_port: int = DEFAULT_PORT,
                 net_authkey: Optional[str] = DEFAULT_AUTHKEY,
                 model_halfset: Literal["half1", "half2"] = "half1",
                 particles_dir: Optional[str] = None,
                 batch_size: int = CONFIG_PARAM(),
                 num_dataworkers: int = CONFIG_PARAM(config=main_config.datamanager),
                 use_cuda: bool = CONFIG_PARAM(),
                 n_cpus_if_no_cuda: int = CONFIG_PARAM(),
                 compile_model: bool = False,
                 top_k_poses_nnet: int = CONFIG_PARAM(),
                 top_k_poses_localref: int = CONFIG_PARAM(config=main_config.projmatching),
                 reference_map: Optional[str] = None,
                 reference_mask: Optional[str] = None,
                 directional_zscore_thr: Optional[float] = CONFIG_PARAM(),
                 skip_localrefinement: bool = CONFIG_PARAM(),
                 skip_reconstruction: bool = CONFIG_PARAM(),
                 show_debug_stats: bool = False,
                 secs_between_partial_results_written: int = 5,
                 resubmit_poison_pill: bool = True
                 ):
        """
        Initializes the DaemonInferencer for running inference on a set of particles from a queue.

        :param checkpoint_dir: Directory where the trained model checkpoints are stored.
        :param results_dir: Directory where the inference results will be saved.
        :param net_address: Network address of the queue manager.
        :param net_port: Network port of the queue manager.
        :param net_authkey: Network authentication key for the queue manager.
        :param model_halfset: Specifies which half-set of the model to use ("half1", "half2").
        :param particles_dir: Directory where the particle images are located. If None, paths in the STAR file are assumed to be absolute.
        :param batch_size: The number of particles to process in each batch.
        :param num_dataworkers: The number of worker processes to use for data loading.
        :param use_cuda: Whether to use a CUDA-enabled GPU for inference.
        :param n_cpus_if_no_cuda: The number of CPU cores to use if CUDA is not available.
        :param compile_model: Whether to compile the model using `torch.compile` for potential speed-up.
        :param top_k_poses_nnet: The number of top predictions to predict with the nn for each particle.
        :param top_k_poses_localref: The number of top predictions to return after local refinement.
        :param reference_map: Path to the reference map for local refinement. If not provided, it will be loaded from the checkpoint.
        :param reference_mask: Path to the mask of the reference map. Used only for FSC calculation.
        :param directional_zscore_thr: The threshold for the directional Z-score to filter particles.
        :param skip_localrefinement: Whether to skip local refinement of the particle poses.
        :param skip_reconstruction: Whether to skip 3D reconstruction from the inferred poses.
        :param show_debug_stats: Whether to print debug statistics, such as rotation errors if ground truth in the starfile.
        :param secs_between_partial_results_written: Partial results are saved from RAM to disk every few seconds
        :param resubmit_poison_pill: If True, posion pills are re-submitted, to ensure that all workers will die
        """

        super().__init__(particles_star_fname=None,
                         checkpoint_dir=checkpoint_dir,
                         results_dir=results_dir,
                         data_halfset="allParticles",
                         model_halfset=model_halfset,
                         particles_dir=particles_dir,
                         batch_size=batch_size,
                         num_dataworkers=num_dataworkers,
                         use_cuda=use_cuda,
                         n_cpus_if_no_cuda=n_cpus_if_no_cuda,
                         compile_model=compile_model,
                         top_k_poses_nnet=top_k_poses_nnet,
                         top_k_poses_localref=top_k_poses_localref,
                         reference_map=reference_map,
                         reference_mask=reference_mask,
                         directional_zscore_thr=directional_zscore_thr,
                         skip_localrefinement=skip_localrefinement,
                         skip_reconstruction=skip_reconstruction,
                         subset_idxs=None,
                         n_first_particles=None,
                         show_debug_stats=show_debug_stats)

        self.net_address = net_address
        self.net_port = net_port
        self.net_authkey = net_authkey
        self.secs_between_partial_results_written = secs_between_partial_results_written
        self.resubmit_poison_pill = resubmit_poison_pill
        self.particles_star_fname_list = []

    def _setup_reconstructor(self, symmetry: Optional[str]= None):
        print("Setting-up the reconstructor")
        self.particles_star_fname = self.particles_star_fname_list[0]
        reconstructor = super()._setup_reconstructor(symmetry)
        self.particles_star_fname = self.particles_star_fname_list
        return reconstructor

    def run(self):
        self._run()

    def resolve_data(self, starfname:Optional[str]):
        if starfname is POISON_PILL:
            return True
        self.particles_star_fname_list.append(starfname)
        return False

    def _setup_dataloader(self, rank: Optional[int] = None):
        if self.particles_star_fname_list:
            self.particles_star_fname = self.particles_star_fname_list
            return super()._setup_dataloader(rank)
        else:
            return None

    def _run(self):

        all_results_list = []
        datasets = []
        particles_md_list = []

        with queue_connection(ip=self.net_address, port=self.net_port, authkey=self.net_authkey) as q:
            # The first batch needs to be treated differently to set up the reconstructor
            data = q.get()
            print("First data item arrived")
            terminateJob = self.resolve_data(data)
            if terminateJob:
                if self.resubmit_poison_pill:
                    q.put(POISON_PILL)
                return
            model = self._setup_model()
            terminateJob = False
            last_save_timestamp = time()

            while not terminateJob:
                # 1) If we already have files pending, process them first
                if self.particles_star_fname_list:
                    print("processing:")
                    print("\n".join(self.particles_star_fname_list))
                    dataloader = self._setup_dataloader()
                    if dataloader:
                        all_results_list.append(
                            self._process_all_batches(model, dataloader, pbar=None)
                        )
                        datasets.append(dataloader.dataset)
                        self.particles_star_fname_list = []
                else:
                    # 2) Only block for new data when idle
                    print("waiting for new data")
                    data_list = get_all_available_items(q)
                    for elem in data_list:
                        _terminateJob = self.resolve_data(elem)
                        terminateJob = _terminateJob or terminateJob

                # Periodically save partial results
                current_time = time()
                if (
                    all_results_list
                    and current_time - last_save_timestamp >= self.secs_between_partial_results_written
                ):
                    for all_results, dataset in zip(all_results_list, datasets):
                        particles_md_list += self._save_particles_results(all_results, dataset)
                    all_results_list = []
                    datasets = []
                    if not self.skip_reconstruction:
                        self._save_reconstruction(materialize=False)
                    last_save_timestamp = time()  # Reset the timer

        # Final save for any remaining results
        if all_results_list:
            for all_results, dataset in zip(all_results_list, datasets):
                particles_md_list += self._save_particles_results(all_results, dataset)
            if not self.skip_reconstruction:
                self._save_reconstruction(materialize=False)

        if self.resubmit_poison_pill:
            q.put(POISON_PILL)


if __name__ == "__main__":

    seed_everything(111)
    os.environ[constants.SCRIPT_ENTRY_POINT] = "daemonInference.py"

    # Increase file descriptor limit to avoid "too many open files" errors
    from cryoPARES.utils.systemUtils import increase_file_descriptor_limit
    increase_file_descriptor_limit()

    print("---------------------------------------")
    print(" ".join(sys.argv))
    print("---------------------------------------")
    from autoCLI_config import ConfigArgumentParser

    parser = ConfigArgumentParser(prog="inferWorker_cryoPARES", description="Run inference with cryoPARES model",
                                  config_obj=main_config)
    parser.add_args_from_function(DaemonInferencer.__init__)
    args, config_args = parser.parse_args()
    assert os.path.isdir(args.checkpoint_dir), f"Error, checkpoint_dir {args.checkpoint_dir} not found"
    config_fname = get_most_recent_file(args.checkpoint_dir, "configs_*.yml")
    ConfigOverrideSystem.update_config_from_file(main_config, config_fname, drop_paths=["inference", "projmatching"])

    with DaemonInferencer(**vars(args)) as inferencer:
        inferencer.run()

    """
PYTHONPATH=. python cryoPARES/inference/daemonInference.py  --checkpoint_dir /tmp/cryoPARES_train/version_0/ --results_dir /tmp/cryoPARES_train/ontheflyInference/  
    """