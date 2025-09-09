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




class OnlineStatsRecorder:
    def __init__(self, total_length, dtype=torch.float32):

        self.n = torch.zeros(total_length, dtype=dtype)
        self.mean = torch.zeros(total_length, dtype=dtype)
        self.M2 = torch.zeros(total_length, dtype=dtype)

    def update(self, batch, indices):

        # Get unique indices and the first occurrence of each unique index
        unique_indices, inverse_indices = torch.unique(indices, return_inverse=True, sorted=True)

        # Calculate deltas before updating means
        delta = batch - self.mean[indices]

        # Calculate sum of delta for unique indices using index_add
        expanded_size = unique_indices.size(0)
        delta_sum = torch.zeros(expanded_size, dtype=self.mean.dtype, device=self.mean.device)
        delta_sum.index_add_(0, inverse_indices, delta)

        # Update counts
        counts = torch.zeros_like(delta_sum).index_add_(0, inverse_indices, torch.ones_like(delta, dtype=self.n.dtype))
        self.n[unique_indices] += counts

        # Update means
        self.mean[unique_indices] += delta_sum / self.n[unique_indices]

        # Calculate delta2 for variance calculation
        delta2 = batch - self.mean[indices]
        delta2_sum = torch.zeros(expanded_size, dtype=self.M2.dtype, device=self.M2.device)
        delta2_sum.index_add_(0, inverse_indices, delta * delta2)

        # Update M2
        self.M2[unique_indices] += delta2_sum

    def get_mean(self):
        return self.mean.float()

    def get_standard_deviation(self):
        _std = torch.sqrt(self.M2 / (self.n-1)).float()
        _std = torch.nan_to_num(_std, 0.)
        return _std.float()



def _test_online_stats(L=1000, C=200):
    import torch_scatter
    from lightning import seed_everything
    seed_everything(1111)
    xs = 10 * torch.rand(L)
    ids = torch.randint(0, C, size=(L,))
    mean = torch_scatter.scatter_mean(xs, ids)
    std = torch_scatter.scatter_std(xs, ids)
    # print(mean)
    # print(mean.shape)
    stat = OnlineStatsRecorder(C)
    for idxs in chunked(range(L), 32):
        stat.update(xs[idxs], ids[idxs])
    # print(torch.isclose(stat.get_mean(), mean))
    # print(stat.get_standard_deviation(), std)
    assert torch.isclose(stat.get_mean(), mean).all()
    assert torch.isclose(stat.get_standard_deviation(), std).all()

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