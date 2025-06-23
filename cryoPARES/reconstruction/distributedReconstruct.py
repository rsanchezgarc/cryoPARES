import multiprocessing
import os

from more_itertools import batched
from progressBarDistributed import SharedMemoryProgressBar, SharedMemoryProgressBarWorker

from joblib import Parallel, delayed

from cryoPARES.reconstruction.reconstruction import Reconstructor

_RECONSTRUCTOR = None
def worker(resources_q, pbar_fname, n_particles, particles_idxs, reconstructor_init_kwargs, reconstructor_run_kwargs):

    # print("Launching worker!!!!!", os.getpid())

    global _RECONSTRUCTOR
    if _RECONSTRUCTOR is None:
        _RECONSTRUCTOR = Reconstructor(**reconstructor_init_kwargs)
    reconstructor_run_kwargs["subset_idxs"] = list(particles_idxs)

    resource = resources_q.get()

    with SharedMemoryProgressBarWorker(resource, pbar_fname) as pbar:
        for n_parts in _RECONSTRUCTOR._backproject_particles(**reconstructor_run_kwargs, verbose=False):
            pbar.update(n_parts)
            pass
        resources_q.put(resource)
        return _RECONSTRUCTOR.f_num,  _RECONSTRUCTOR.weights, _RECONSTRUCTOR.ctfs

def main():

    n_jobs = 2
    outname = "/tmp/reconstructed_vol.mrc"
    reconstructor_init_kwargs = dict(
        symmetry="C1", correct_ctf=True, eps=1e-3, min_denominator_value=1e-4,
        device="cuda"
    )
    reconstructor_run_kwargs = dict(
        # particles_star_fname="/home/sanchezg/cryo/data/preAlignedParticles/EMPIAR-10166/data/projections/1000proj_with_ctf.star",
        particles_star_fname="/home/sanchezg/cryo/data/preAlignedParticles/EMPIAR-10166/data/allparticles.star",
        particles_dir="/home/sanchezg/cryo/data/preAlignedParticles/EMPIAR-10166/data/",
        batch_size=32,
        num_dataworkers=2
    )


    # os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
    # os.environ["MKL_NUM_THREADS"] = str(threads_per_worker)
    # os.environ["OPENBLAS_NUM_THREADS"] = str(threads_per_worker)
    # os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads_per_worker)
    # os.environ["NUMEXPR_NUM_THREADS"] = str(threads_per_worker)

    reconstructor = Reconstructor(**reconstructor_init_kwargs)
    particles = reconstructor._get_reconstructionParticlesDataset(reconstructor_run_kwargs["particles_star_fname"],
                                                                  reconstructor_run_kwargs["particles_dir"]).particles
    n_particles = len(particles)

    batch_size = n_particles // (2 * n_jobs)

    with SharedMemoryProgressBar(n_jobs) as pbar:
        pbar_fname = pbar.shm_name
        with multiprocessing.Manager() as mg:
            q = mg.Queue()
            for i in range(n_jobs):
                q.put(i)
                pbar.set_total_steps(n_particles//n_jobs, i)

            outgen = Parallel(n_jobs=n_jobs,
                              return_as="generator", #"generator" #"list",
                              backend="loky",
                              batch_size=1)(delayed(worker)(q, pbar_fname, n_particles, particles_idxs,
                                              reconstructor_init_kwargs, reconstructor_run_kwargs)
                                                            for worker_id, particles_idxs
                                                                in enumerate(batched(range(n_particles), batch_size)))

        # args_list = [(worker_id, pbar_fname, n_particles, particles_idxs,
        #               reconstructor_init_kwargs, reconstructor_run_kwargs)
        #                                for worker_id, particles_idxs
        #                                         in enumerate(batched(range(n_particles), batch_size))]
        # processes = []
        # for i in range(n_jobs):
        #     p = multiprocessing.Process(target=worker, args=args_list[i])
        #     p.start()
        #     processes.append(p)
        #     # worker(*args_list[i])
        # for p in processes:
        #     p.join()

            for f_num, weights, ctfs in outgen:
                reconstructor.f_num += f_num
                reconstructor.weights += weights
                reconstructor.ctfs += ctfs

    print("Done")
    reconstructor.generate_volume(outname)


if __name__ == "__main__":
    main()