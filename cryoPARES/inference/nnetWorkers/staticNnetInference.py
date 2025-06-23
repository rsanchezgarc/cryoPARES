import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tqdm
from starstack.constants import RELION_ANGLES_NAMES, RELION_SHIFTS_NAMES, RELION_PRED_POSE_CONFIDENCE_NAME, \
    RELION_EULER_CONVENTION
from torch.nn.parallel import DistributedDataParallel as DDP
import os.path as osp
import socket
import contextlib
from typing import Optional, List, Literal, Dict, Any
from progressBarDistributed import SharedMemoryProgressBarWorker, SharedMemoryProgressBar

from cryoPARES.geometry.convert_angles import matrix_to_euler_angles
from cryoPARES.models.model import PlModel
from cryoPARES.datamanager.datamanager import DataManager


class SharedMemoryManager:
    """Manages shared memory tensors across processes."""

    def __init__(self, world_size: int, n_items: int, top_k: int):
        self.world_size = world_size
        self.n_items = n_items
        self.top_k = top_k
        self.arrays: List[Dict[str, torch.Tensor]] = [{} for _ in range(self.world_size)]

    def create_shared_arrays(self):
        """Create shared memory arrays based on sample outputs."""
        with torch.inference_mode():
            for worker in range(self.world_size):
                self.arrays[worker] = {
                                    'eulerdegs': torch.zeros(
                                        (self.n_items, self.top_k, 3),
                                        dtype=torch.float32),
                                    'rotprobs': torch.zeros(
                                        (self.n_items, self.top_k),
                                        dtype=torch.float32),
                                    'shifts': torch.zeros(
                                        (self.n_items, self.top_k, 2),
                                        dtype=torch.float32),
                                    'shiftprobs': torch.zeros(
                                        (self.n_items, self.top_k),
                                        dtype=torch.float32),
                                    'idxs': torch.zeros(
                                        (self.n_items,),
                                        dtype=torch.int64)
                }
                if self.world_size > 1:
                    for k in self.arrays[worker].keys():
                        self.arrays[worker][k] = self.arrays[worker][k].share_memory_()

    def cleanup(self):
        """Clear shared memory arrays."""
        for worker_arrays in self.arrays:
            worker_arrays.clear()
        self.arrays.clear()

class PartitionInferencer:
    UPDATE_PROGRESS_BAR_N_BATCHES = 10

    def __init__(self,
                 star_fnames: List[str],
                 checkpoint_path: str,
                 results_dir: str,
                 halfset: Literal["half1", "half2", "allParticles"] = None,
                 particles_dir: Optional[List[str]] = None,
                 batch_size: int = 32,
                 use_cuda: bool = True,
                 n_cpus_if_no_cuda: int = 4,
                 compile_model: bool = False,
                 top_k: int =1):
        """Initialize inference for a partition."""
        self.star_fnames = star_fnames
        self.checkpoint_path = checkpoint_path
        self.particles_dir = particles_dir
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.n_cpus_if_no_cuda = n_cpus_if_no_cuda
        self.results_dir = results_dir
        self.halfset = halfset
        self.compile_model = compile_model
        self.top_k = top_k
        os.makedirs(results_dir, exist_ok=True)

        self.accelerator, self.device_count = self._setup_accelerator()
        self.pbar_shm_name = None

    def _setup_accelerator(self):
        """Setup the computation device and count."""
        if self.use_cuda and torch.cuda.is_available():
            accelerator = "cuda"
            device_count = torch.cuda.device_count()
        else:
            accelerator = "cpu"
            device_count = self.n_cpus_if_no_cuda

        print(f'devices={device_count} accelerator={accelerator}', flush=True)
        torch.set_num_threads(min(1, mp.cpu_count() // device_count))
        return accelerator, device_count

    def _find_free_port(self):
        """Find a free port for distributed communication."""
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(('', 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]

    def _setup_model(self, rank: Optional[int] = None):
        """Setup the model for inference."""
        model = PlModel.load_from_checkpoint(
            self.checkpoint_path,
            map_location="cpu",
            symmetry=self.symmetry
        )
        device = rank if rank is not None else 'cuda:0'
        model = model.to(device)

        if rank is not None:  # Distributed mode
            model = DDP(model, device_ids=[rank])

        if self.compile_model:
            print("Compiling model")
            model = torch.compile(model)

        model.eval()
        return model

    def _setup_dataloader(self, rank: Optional[int] = None, world_size: Optional[int] = None):
        """Setup the dataloader for inference."""
        halfset = None
        if self.halfset == "half1":
            halfset = 1
        elif self.halfset == "half2":
            halfset = 2

        datamanager = DataManager(
            star_fnames=self.star_fnames,
            symmetry=self.symmetry,
            particles_dir=self.particles_dir,
            batch_size=self.batch_size,
            augment_train=False,  # No augmentation during inference
            halfset=halfset,
            is_global_zero=rank == 0 if rank is not None else True,
            save_train_val_partition_dir=None  # Not needed for inference
        )

        if rank is not None:
            datamanager.setup_distributed(world_size, rank)

        return datamanager.predict_dataloader()

    def _setup_shared_memory(self, world_size: int, n_items: int, top_k: int):
        """Setup shared memory arrays for results."""
        return SharedMemoryManager(world_size, n_items, top_k)

    def _process_batch(self, model: PlModel, batch: Dict[str, Any], batch_idx: int, device: torch.device):
        """Process a single batch of data."""
        batch = model.transfer_batch_to_device(batch, device, dataloader_idx=0)
        (idd, (pred_rotmats, maxprobs, all_angles_probs),
         (pred_shifts, shifts_probs), errors, metadata) = model.predict_step(batch, batch_idx=batch_idx)
        euler_degs = torch.rad2deg(matrix_to_euler_angles(pred_rotmats, RELION_EULER_CONVENTION))
        pred_shifts.fill_(0.)
        shifts_probs.fill_(1.)

        return idd, (euler_degs, maxprobs), (pred_shifts, shifts_probs), metadata


    def _run_inference_worker(self, rank: Optional[int], world_size: Optional[int],
                              port: Optional[int] = None, shared_mem: Optional[SharedMemoryManager] = None,
                              pbar_shm_name: Optional[str] = None):
        """Run inference on a single worker."""
        if rank is not None:
            dist.init_process_group(
                "nccl",
                init_method=f"tcp://localhost:{port}",
                world_size=world_size,
                rank=rank
            )
            torch.cuda.set_device(rank)

        try:
            # Setup for this worker
            model = self._setup_model(rank)
            dataloader = self._setup_dataloader(rank, world_size)

            if rank is None:
                shared_mem = shared_mem.arrays[0]
            else:
                shared_mem = shared_mem.arrays[rank]

            if pbar_shm_name is None:
                pbar = tqdm.tqdm(total=len(dataloader))
            else:
                pbar = SharedMemoryProgressBarWorker(worker_id=rank if rank is not None else 0,
                                               shm_name=pbar_shm_name)
                pbar.set_total_steps(len(dataloader))

            with (pbar):
                device = torch.device(rank if rank is not None else 'cuda:0') #TODO: enable cpu only

                # Ensure all processes have access to shared memory
                if rank is not None:
                    dist.barrier()
                with torch.inference_mode():
                    current_idx = 0
                    for batch_idx, batch in enumerate(dataloader):
                        # Process batch
                        ids, (euler_degs, maxprobs), \
                        (pred_shifts, shifts_probs), metadata = self._process_batch(model, batch, batch_idx, device)

                        # Write results directly to shared memory
                        batch_size = len(ids)
                        end_idx = current_idx + batch_size

                        shared_mem['eulerdegs'][current_idx:end_idx] = euler_degs.cpu()
                        shared_mem['rotprobs'][current_idx:end_idx] = maxprobs.cpu()
                        shared_mem['shifts'][current_idx:end_idx] = pred_shifts.cpu()
                        shared_mem['shiftprobs'][current_idx:end_idx] = shifts_probs.cpu()
                        shared_mem['idxs'][current_idx:end_idx] = torch.arange(current_idx, end_idx)
                        current_idx = end_idx

                        if (batch_idx + 1) % self.UPDATE_PROGRESS_BAR_N_BATCHES == 0:
                            pbar.update(self.UPDATE_PROGRESS_BAR_N_BATCHES)

                pbar.update(len(dataloader) - batch_idx - 1 )
            # Ensure all processes have finished writing
            if rank is not None:
                dist.barrier()
        finally:
            if rank is not None:
                dist.destroy_process_group()

    def run(self):
        """Main entry point for running inference."""

        try:
            dataloader = self._setup_dataloader(None, world_size=self.device_count)
            dataset = dataloader.dataset
            shared_mem = self._setup_shared_memory(self.device_count, len(dataset), top_k=self.top_k)
            shared_mem.create_shared_arrays()

            if self.device_count > 1:

                # Start distributed processes
                port = self._find_free_port()
                mp.spawn(
                    self._run_inference_worker,
                    args=(self.device_count, port, shared_mem, self.pbar_shm_name),
                    nprocs=self.device_count,
                    join=True
                )
            else:
                self._run_inference_worker(None, None, shared_mem=shared_mem,
                                           pbar_shm_name=self.pbar_shm_name)

            for _dataset in dataset.datasets:
                particlesSet = _dataset.particles
                particles_md = particlesSet.particles_md
                for k in range(self.top_k):
                    suffix = "" if k == 0 else f"_top{k}"
                    angles_names = [x + suffix for x in RELION_ANGLES_NAMES]
                    shifts_names = [x + suffix for x in RELION_SHIFTS_NAMES]
                    confide_name = RELION_PRED_POSE_CONFIDENCE_NAME + suffix
                    particles_md[angles_names] = 0.
                    particles_md[shifts_names] = 0.
                    particles_md[confide_name] = 0.
                    for worker in range(self.device_count):
                        _worker_data = shared_mem.arrays[worker]
                        idxs = shared_mem.arrays[worker]["idxs"]
                        ids = particles_md.index[idxs]
                        particles_md.loc[ids, angles_names] = _worker_data["eulerdegs"][..., k, :].numpy()
                        particles_md.loc[ids, shifts_names] = _worker_data["shifts"][..., k, :].numpy()
                        particles_md.loc[ids, confide_name] = (_worker_data["rotprobs"][..., k] *
                                                               _worker_data["shiftprobs"][..., k]).numpy()

                out_fname = os.path.join(self.results_dir,
                                         os.path.basename(particlesSet.starFname).removesuffix(".star")+"_nnet.star")
                print(f"Results were saved at {out_fname}")
                particlesSet.save(out_fname)
        finally:
            if hasattr(self, 'shared_mem'):
                self.shared_mem.cleanup()

    def create_shared_bar(self):
        if self.device_count == 1:
            return MokupProgressBar()
        else:
            shmPbar = SharedMemoryProgressBar(self.device_count)
            self.pbar_shm_name = shmPbar.shm_name
            return shmPbar

class MokupProgressBar():
    def __init__(self):
        pass
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Run inference with cryoPARES model")
    parser.add_argument("--star_fnames", type=str, nargs="+", required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--particles_dir", type=str, nargs="+")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--compile_model", action="store_true")
    parser.add_argument("--halfset", type=str, choices=["half1", "half2", "allParticles"])
    parser.add_argument("--top_k", type=int, default=1)


    args = parser.parse_args()

    inferencer = PartitionInferencer(
        star_fnames=args.star_fnames,
        checkpoint_path=args.checkpoint_path,
        results_dir=args.results_dir,
        particles_dir=args.particles_dir,
        batch_size=args.batch_size,
        compile_model=args.compile_model,
        halfset=args.halfset,
        top_k=args.top_k
    )
    with inferencer.create_shared_bar() as pbar:
        inferencer.run()

"""

python -m cryoPARES.inference.nnetWorkers.staticNnetInference --star_fnames /home/sanchezg/cryo/data/preAlignedParticles/EMPIAR-10166/data/1000particles.star --checkpoint_path /tmp/train_cryoPARES/version_0/half1/checkpoints/best.ckpt --results_dir /tmp/cryoParesInfer/

"""