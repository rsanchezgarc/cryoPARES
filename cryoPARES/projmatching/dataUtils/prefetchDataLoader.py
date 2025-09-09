import functools
import os
import signal
import sys
import traceback
from types import GeneratorType

import numpy as np
import torch
from torch import multiprocessing as mp



class PrefetchProcessor:
    def __init__(self, batch_size, n_workers, n_prefetch_batches, data_shape, data_dtype, device):

        assert n_workers < (n_prefetch_batches + 1), "Error, n_workers < (n_prefetch_batches + 1) required"
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.n_prefetch_batches = n_prefetch_batches
        self.data_shape = data_shape
        self.device = device

        self.prefetch_batches = [torch.zeros(batch_size, *data_shape, device=device, dtype=data_dtype).share_memory_()
                                 for _ in range(n_prefetch_batches)]

        self.errorMessageQ = mp.Queue()

    def assing_idx_to_worker(self, idx):
        return idx % self.n_workers

    def _compute_helper(self, compute_idxs_fn, idxs_to_be_computed, to_be_computed_in_batch_idxs,
                        prefetch_batches_condition_and_counts, worker_id, n_last_batch=None, **kwargs):
        try:
            computed_data = compute_idxs_fn(idxs_to_be_computed, **kwargs).to(self.device)
            for prefetchId in range(self.n_prefetch_batches):
                computed_idxs = to_be_computed_in_batch_idxs[prefetchId]
                if computed_idxs:
                    true_idx, idx_in_computed, idxs_in_batch = zip(*computed_idxs)
                    # print(f"~Producer (prefetchID {prefetchId}) computed ids: {idx_in_computed} {idxs_in_batch}")
                    condition_prefetch, count = prefetch_batches_condition_and_counts[prefetchId]
                    with condition_prefetch:
                        condition_prefetch.wait_for(lambda: count.value < self.batch_size)
                        self.prefetch_batches[prefetchId][idxs_in_batch, ...] = computed_data[idx_in_computed, ...]
                        to_be_computed_in_batch_idxs[prefetchId] = []
                        with count.get_lock():
                            count.value += len(idx_in_computed)
                            # print(f"~Producer{prefetchId}: {count.value}  {n_last_batch}")
                            if count.value == n_last_batch if n_last_batch else self.batch_size:
                                condition_prefetch.notify() #notify instead notify_all because there is only one consumer
        except Exception as e:
            self.errorMessageQ.put(
                f"Exception occurred in worker {worker_id}. Printing stack trace:" +
                traceback.format_exc()
            )
            os.kill(os.getppid(), signal.SIGTERM)
            # sys.exit(1)

    def worker_fun(self, compute_idxs_fn, worker_init_fn, worker_id, idxs,
               workers_done, condition_idx_lastBatchToPrefetch,
               prefetch_batches_condition_and_counts):

        if worker_init_fn is not None:
            kwargs = worker_init_fn(worker_id, self.n_workers)
        else:
            kwargs = {}

        n_items = len(idxs)
        batch_size = self.batch_size
        prefetch_batches = self.prefetch_batches
        n_prefetch_batches = self.n_prefetch_batches
        _, *data_shape = prefetch_batches[0].shape

        to_be_computed_in_batch_idxs = [[] for _ in range(n_prefetch_batches)]
        total_to_be_computed = 0
        idxs_to_be_computed = []

        last_incomplete_batch = n_items % batch_size

        prev_batch_idx = 0
        batch_idx = 0
        for idx in idxs:
            batch_idx = idx // batch_size
            if self.assing_idx_to_worker(idx) == worker_id:
                if prev_batch_idx != batch_idx:
                    prev_batch_idx = batch_idx
                    condition_lastBatchToPrefetch, idx_lastBatchToPrefetch = condition_idx_lastBatchToPrefetch
                    with condition_lastBatchToPrefetch:
                        # print(f"~Producer is goint to sleep batch_idx {batch_idx}, idx_lastBatchToPrefetch {idx_lastBatchToPrefetch.value}")
                        condition_lastBatchToPrefetch.wait_for(lambda : batch_idx<idx_lastBatchToPrefetch.value)
                        # print(f"~Producer woke up. {to_be_computed_in_batch_idxs}")

                    self._compute_helper(compute_idxs_fn, idxs_to_be_computed, to_be_computed_in_batch_idxs,
                                    prefetch_batches_condition_and_counts, worker_id, n_last_batch=None, **kwargs)
                    total_to_be_computed = 0
                    idxs_to_be_computed = []

                batch_position_idx = idx % batch_size
                prefetch_slot_id = batch_idx % n_prefetch_batches
                to_be_computed_in_batch_idxs[prefetch_slot_id].append([idx, total_to_be_computed, batch_position_idx])
                idxs_to_be_computed.append(idx)
                total_to_be_computed += 1


        if idxs_to_be_computed: #TODO: #This code should be the same as in the look. Factor in in a function
            self._compute_helper(compute_idxs_fn, idxs_to_be_computed, to_be_computed_in_batch_idxs,
                                 prefetch_batches_condition_and_counts, worker_id, n_last_batch=last_incomplete_batch,
                                 **kwargs)

        # print(f"~Producer finished producing results! {batch_idx}")
        # Signal completion for this worker
        with workers_done.get_lock():
            workers_done.value += 1

    def signal_handler(self, signum, frame):
        print("Parent process received termination signal.")
        while not self.errorMessageQ.empty():
            err = self.errorMessageQ.get()
            print(err)
        sys.exit(1)

    def yield_batches(self, compute_idxs_fn, idxs, worker_init_fn=None):
        if isinstance(idxs, GeneratorType):
            idxs = list(idxs)

        signal.signal(signal.SIGTERM, self.signal_handler)

        n_items = len(idxs)
        batch_size = self.batch_size
        n_prefetch_batches = self.n_prefetch_batches
        last_incomplete_batch_size = n_items % batch_size
        n_batches = n_items // batch_size + int(bool(last_incomplete_batch_size))

        workers_done = mp.Value("i", 0)  # Track completed workers
        condition_lastBatchToPrefetch = mp.Condition()
        idx_lastBatchToPrefetch = mp.Value("i", self.n_prefetch_batches - 1)
        condition_idx_lastBatchToPrefetch = (condition_lastBatchToPrefetch, idx_lastBatchToPrefetch)
        prefetch_batches_condition_and_counts = [(
            mp.Condition(), #Contition to notify the consumer
            mp.Value("i", 0), #Number of items written in prefetcBatch[prefetchBatchIdxs]
                                                  )
                                                 for _ in range(self.n_prefetch_batches)]
        workers = []
        for w in range(self.n_workers):
            producer_process = mp.Process(target=self.worker_fun, args=(compute_idxs_fn, worker_init_fn, w, idxs,
                                                                        workers_done, condition_idx_lastBatchToPrefetch,
                                                                        prefetch_batches_condition_and_counts))
            producer_process.start()
            workers.append((producer_process))


        finished = False
        n_batches_consumed = 0
        while not finished:
            for p in range(n_prefetch_batches):
                # print("#Consumer prefetch_num", p)
                condition_prefetch, count_prefetch = prefetch_batches_condition_and_counts[p]
                with condition_prefetch:
                    # print(f"#Consumer (prefetch_buffer{p}) will wait!", count_prefetch.value, count_prefetch.value == batch_size)
                    if n_batches_consumed < n_batches - 1:
                        condition_prefetch.wait_for(lambda: count_prefetch.value == batch_size)
                    elif last_incomplete_batch_size:
                        condition_prefetch.wait_for(lambda: count_prefetch.value == last_incomplete_batch_size)
                    else:
                        raise ValueError()
                    # print(f"#Consumer (prefetch_buffer{p}) was woken up", count_prefetch.value, count_prefetch.value == batch_size)
                    batch = self.prefetch_batches[p].clone()

                    with count_prefetch.get_lock():
                        count_prefetch.value = 0  # The buffer has been used, make it available for the producers
                        # print(f"#Consumer (prefetch_buffer{p}) ", count_prefetch.value)
                    condition_prefetch.notify_all()

                n_batches_consumed += 1
                if n_batches_consumed == n_batches and last_incomplete_batch_size != 0:
                    batch = batch[:last_incomplete_batch_size, ...]

                yield batch

                with condition_lastBatchToPrefetch:
                    with idx_lastBatchToPrefetch.get_lock():
                        # if n_batches_consumed < n_batches:
                        idx_lastBatchToPrefetch.value += 1
                        condition_lastBatchToPrefetch.notify_all()

                # print("#Consumer found tensor ->", batch.squeeze(-1).squeeze(-1), "for batch", n_batches_consumed - 1)
                # print("#Consumer n_batches_consumed ->", n_batches_consumed,  n_batches, workers_done.value)
                if n_batches_consumed == n_batches:
                    finished = True
                    break

        # Wait for both processes to complete
        for producer_process in workers:
            producer_process.join()
        assert workers_done.value == self.n_workers, "Error, some of the workers didn't finish!"


def compute_idxs_fn_example(idxs, data_shape, device, **kwargs):
    """Placeholder for a more complex computation function."""
    import time, random
    time.sleep(0.01*random.random())
    idxs = torch.as_tensor(idxs, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1).expand(-1, *data_shape)
    return idxs.to(device)

if __name__ == "__main__":
    # Example usage
    n_items = 954
    batch_size = 5
    n_prefetch_batches = 3
    data_shape = (1, 1)
    device = "cuda:0" #"cpu"
    n_workers = 3
    data_dtype = torch.float32

    mp.set_start_method("spawn")
    _compute_idxs_fn_example = functools.partial(compute_idxs_fn_example, data_shape=data_shape, data_dtype=data_dtype,
                                                 device=device)

    processor = PrefetchProcessor(batch_size, n_workers, n_prefetch_batches, data_shape, data_dtype, device)
    all_elems = []
    for batch in processor.yield_batches(_compute_idxs_fn_example, range(n_items), worker_init_fn=None):
        fbatch = batch.flatten()
        print(fbatch)
        all_elems.append(fbatch)

    all_elems = torch.concatenate(all_elems)
    is_sorted = torch.all(all_elems[:-1] <= all_elems[1:])
    print(is_sorted) #This is to check if the array is sorted
    assert is_sorted