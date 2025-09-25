import glob
import os
import sys
from time import time
from pathlib import Path

import torch
import yaml
from lightning import seed_everything
from tqdm import tqdm
from torch import  multiprocessing as mp
from typing import Optional, List, Literal, Dict, Any, Tuple

from cryoPARES.constants import RELION_ANGLES_NAMES, RELION_SHIFTS_NAMES, RELION_PRED_POSE_CONFIDENCE_NAME, \
    RELION_EULER_CONVENTION, DIRECTIONAL_ZSCORE_NAME

from cryoPARES import constants
from cryoPARES.configManager.configParser import ConfigOverrideSystem
from cryoPARES.configManager.inject_defaults import inject_defaults_from_config, CONFIG_PARAM
from cryoPARES.configs.mainConfig import main_config
from cryoPARES.constants import BEST_DIRECTIONAL_NORMALIZER, BEST_CHECKPOINT_BASENAME, BEST_MODEL_SCRIPT_BASENAME
from cryoPARES.geometry.convert_angles import matrix_to_euler_angles
from cryoPARES.inference.daemon.queueManager import queue_connection, get_all_available_items, DEFAULT_IP, \
    DEFAULT_PORT, DEFAULT_AUTHKEY
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
                 num_data_workers: int = CONFIG_PARAM(config=main_config.datamanager),
                 use_cuda: bool = CONFIG_PARAM(),
                 n_cpus_if_no_cuda: int = CONFIG_PARAM(),
                 compile_model: bool = False,
                 top_k_poses_nnet: int = CONFIG_PARAM(),
                 reference_map: Optional[str] = None,
                 directional_zscore_thr: Optional[float] = CONFIG_PARAM(),
                 skip_localrefinement: bool = CONFIG_PARAM(),
                 skip_reconstruction: bool = CONFIG_PARAM(),
                 update_progressbar_n_batches: int = CONFIG_PARAM(),
                 secs_between_partial_results_written: int = 5,
                 ):
        """

        :param checkpoint_dir:
        :param results_dir:
        :param net_address:
        :param net_port:
        :param net_authkey:
        :param model_halfset:
        :param particles_dir:
        :param batch_size:
        :param num_data_workers:
        :param use_cuda:
        :param n_cpus_if_no_cuda:
        :param compile_model:
        :param top_k_poses_nnet:
        :param reference_map: If not provided, it will be tried to load from the checkpointç
        :param directional_zscore_thr:
        :param skip_localrefinement:
        :param skip_reconstruction:
        :param update_progressbar_n_batches:
        :param secs_between_partial_results_written:
        """

        super().__init__(particles_star_fname=None,
                         checkpoint_dir =checkpoint_dir,
                         results_dir = results_dir,
                         data_halfset = "allParticles",
                         model_halfset = model_halfset,
                         particles_dir = particles_dir,
                         batch_size = batch_size,
                         num_data_workers = num_data_workers,
                         use_cuda = use_cuda,
                         n_cpus_if_no_cuda = n_cpus_if_no_cuda,
                         compile_model = compile_model,
                         top_k_poses_nnet = top_k_poses_nnet,
                         reference_map = reference_map,
                         directional_zscore_thr = directional_zscore_thr,
                         skip_localrefinement = skip_localrefinement,
                         skip_reconstruction = skip_reconstruction,
                         update_progressbar_n_batches=update_progressbar_n_batches,
                         subset_idxs=None)

        self.net_address = net_address
        self.net_port = net_port
        self.net_authkey = net_authkey
        self.secs_between_partial_results_written = secs_between_partial_results_written
        self.particles_star_fname_list = []

    def _setup_reconstructor(self, symmetry: Optional[str]= None):
        self.particles_star_fname = self.particles_star_fname_list[0]
        reconstructor = super()._setup_reconstructor(symmetry)
        self.particles_star_fname = self.particles_star_fname_list
        return reconstructor

    def run(self):
        self._run()

    def resolve_data(self, starfname:Optional[str]):
        if starfname is None:
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
            if terminateJob: return
            model = self._setup_model()
            terminateJob = False
            last_save_timestamp = time()
            while not terminateJob:
                print("wating for new data")
                data_list = get_all_available_items(q)
                for elem in data_list:
                    _terminateJob = self.resolve_data(elem)
                    terminateJob = _terminateJob or terminateJob
                if self.particles_star_fname_list:
                    print("processing:")
                    print("\n".join(self.particles_star_fname_list))
                dataloader = self._setup_dataloader()
                if not dataloader:
                    continue
                all_results_list += [self._process_all_batches(model, dataloader, pbar=None)]
                datasets += [dataloader.dataset]
                self.particles_star_fname_list = []

                current_time = time()
                if all_results_list and \
                   current_time - last_save_timestamp >= self.secs_between_partial_results_written:
                    for all_results, dataset in zip(all_results_list, datasets):
                        particles_md_list += self._save_particles_results(all_results, dataset)
                    all_results_list = []
                    datasets = []
                    if not self.skip_reconstruction:
                        self._save_reconstruction(materialize=False)
                    last_save_timestamp = time()  # Reset the timer



if __name__ == "__main__":

    seed_everything(111)
    os.environ[constants.SCRIPT_ENTRY_POINT] = "daemonInference.py"
    print("---------------------------------------")
    print(" ".join(sys.argv))
    print("---------------------------------------")
    from cryoPARES.configManager.configParser import ConfigArgumentParser
    torch.set_float32_matmul_precision(constants.float32_matmul_precision)

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