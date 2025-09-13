import random

import torch
from numba import types
from numba.typed.typeddict import Dict
from torch import multiprocessing as mp
from more_itertools import chunked


def compute_idxs(idxs, data_shape):  # This is a placeholder example, but it will be much more complex in reality
    idxs = torch.as_tensor(idxs, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1).expand(-1, *data_shape)
    return idxs


def worker_fun(worker_id, n_workers, batch_size, idxs, prefetch_batches, n_free_slots_sem, batch_idx_ready_q, device):
    # TODO: batch_idx_ready_q is not really needed, we just need to use a mp.Event to signal the end. But it is useful for debugging
    current_prefetch_idx = 0
    n_prefetch_batches = len(prefetch_batches)
    _, *data_shape = prefetch_batches[0].shape

    for batch_idx, idxs in enumerate(chunked(idxs, batch_size)):
        if batch_idx % n_workers == worker_id:  # Each worker will take care of one of the pre-fetched batches
            n_free_slots_sem.acquire()

            prefetch_batches[current_prefetch_idx][:] = compute_idxs(idxs, data_shape).to(device)
            batch_idx_ready_q.put((current_prefetch_idx, worker_id, idxs[0]))
            current_prefetch_idx += 1
            current_prefetch_idx %= n_prefetch_batches
    batch_idx_ready_q.put(None)



def _test_workers():

    n_items = 10_000
    n_classess = 1000
    n_workers = 2
    cache_size = 8
    data_shape = (1, 1)
    batch_size = 4
    n_prefetch_batches = 3
    device = "cuda:0"
    idxs = random.choices(range(n_classess), k=n_items)  # range(n_items)

    mp.set_start_method("spawn")
    prefetch_batches = [[torch.empty(batch_size, *data_shape, device=device).share_memory_()
                         for _ in range(n_prefetch_batches)] for _ in range(n_workers)]
    batch_idx_ready_queues = [mp.Queue(maxsize=n_prefetch_batches * batch_size + 1)
                              for _ in range(n_workers)]
    n_free_slots_semaphore_per_worker = [mp.Semaphore(n_prefetch_batches) for _ in range(n_workers)]
    workers = []
    for w in range(n_workers):
        producer_process = mp.Process(target=worker_fun, args=(w, n_workers, batch_size, idxs, prefetch_batches[w],
                                                               n_free_slots_semaphore_per_worker[w],
                                                               batch_idx_ready_queues[w], device))
        producer_process.start()
        workers.append((producer_process))

    finished = False
    while not finished:
        n_workers_done = 0
        for p in range(n_prefetch_batches):
            for w in range(n_workers):
                sem = n_free_slots_semaphore_per_worker[w]
                q = batch_idx_ready_queues[w]
                results_list = prefetch_batches[w][p]
                item = q.get()
                print(item)
                if item is None:
                    n_workers_done += 1
                sem.release()
        if n_workers_done == n_workers:
            finished = True

    # Wait for both processes to complete
    for producer_process in workers:
        producer_process.join()

    print("Main process exiting")


if __name__ == "__main__":
    # _test_workers()
    _test_online_stats()