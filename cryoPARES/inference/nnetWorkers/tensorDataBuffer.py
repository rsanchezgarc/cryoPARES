import functools
import itertools
from typing import Callable, Dict, List, Any

import torch
from torch import nn


class StreamingBuffer(nn.Module):
    """
    A buffer that accumulates batches and processes them once a size threshold is met.
    """

    def __init__(
            self,
            buffer_size: int,
            processing_fn: Callable[[Dict[str, torch.Tensor], Dict[str, List[Any]]], Any],
    ) -> None:
        super().__init__()
        self.buffer_size = buffer_size
        self.processing_fn = processing_fn
        self.storage: List[Dict[str, torch.Tensor | List[Any]]] = []

    def add_batch(self, batch: Dict[str, torch.Tensor | List[Any]]) -> Any:
        """Adds a new batch to the buffer and processes if the buffer is full."""
        self.storage.append(batch)
        return self._process(flush=False)

    def _process(self, flush: bool = False):
        """
        Processes the buffer if it has reached the required size or if flushing.
        """
        if not self.storage:
            return None

        # Calculate the total number of items currently in storage
        num_items_in_storage = sum(len(next(iter(b.values()))) for b in self.storage)

        # Only process if the buffer is full or if we are flushing
        if num_items_in_storage < self.buffer_size and not flush:
            return None

        # If there's nothing to process, return None
        if num_items_in_storage == 0:
            return None

        # Combine all stored batches into a single "meta-batch"
        # This is more efficient as it calls the processing function only once.
        first_item_keys = self.storage[0].keys()
        meta_batch = {}

        for key in first_item_keys:
            # Collect all items for the current key from all batches
            items_to_combine = [b[key] for b in self.storage]

            # Combine based on type
            if isinstance(items_to_combine[0], torch.Tensor):
                meta_batch[key] = torch.cat(items_to_combine, dim=0)
            elif isinstance(items_to_combine[0], list):
                meta_batch[key] = list(itertools.chain.from_iterable(items_to_combine))
            else:
                raise NotImplementedError(f"Combining type {type(items_to_combine[0])} is not supported.")

        # Now process the entire meta-batch at once
        results = self.processing_fn(**meta_batch)

        # Clear the storage after processing
        self.storage = []

        return results

    def flush(self):
        """Processes any remaining data in the buffer."""
        return self._process(flush=True)