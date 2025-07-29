import functools
import itertools
from typing import Callable, Dict, List, Any

import torch
from torch import nn


class StreamingBuffer(nn.Module):

    def __init__(
        self,
        buffer_size: int,
        processing_fn: Callable[[Dict[str, torch.Tensor], Dict[str, List[Any]]], Any],
    ) -> None:

        super().__init__()
        self.buffer_size = buffer_size
        self.procesing_fn = processing_fn

        self.storage = []

    def add_batch(self, batch: Dict[str, torch.Tensor | List[Any]]) -> Any:

        self.storage.append(batch)
        return self._process()


    def _process(self):

        n_to_process = 0
        for batch_dict in self.storage:
            n_to_process += len(next(iter(batch_dict.values()), []))

        all_results = []
        while n_to_process >= self.buffer_size:
            for elem in self.storage:
                out = self.procesing_fn(**elem)
                all_results.append(out)
                if isinstance(out, (list, tuple)):
                    n_to_process -= len(out[0])
                elif isinstance(out, dict):
                    n_to_process -= len(next(iter(out.values()), []))
        if not all_results:
            return None
        else:
            self.storage = []
            all_results = list(zip(*all_results))
            out = [None]* len(all_results)
            for i, x in enumerate(all_results):
                if isinstance(x[0], list):
                    out[i] = list(itertools.chain.from_iterable(x))
                elif isinstance(x[0], torch.Tensor):
                    out[i] = torch.cat(x)
                else:
                    raise NotImplementedError()
            return out

    def flush(self):
        return self._process()

def _test_my_case():

    buffer_size = 8
    batch_size = 16

    def proc(**kwargs):
        return kwargs["ids"], kwargs["x"], ["caca"]*len(kwargs["ids"])

    sb = StreamingBuffer(buffer_size, proc)

    for i in range(10):
        ids = [i*batch_size + j for j in range(batch_size)]
        batch_to_add = {
            "ids": ids,
            "x": torch.as_tensor(ids)
        }
        out = sb.add_batch(batch_to_add)
        if out is not None:
            print(out[0])
            print(out[1])
            print("---------------")
            assert out[0] == out[1].tolist()
        else:
            print(f"batch {i} was not productive")


if __name__ == "__main__":
    _test_my_case()