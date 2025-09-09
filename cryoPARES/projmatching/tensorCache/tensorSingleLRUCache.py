from abc import abstractmethod

import torch

from .abstractTorchCache import AbstractTorchLRUCache


class TensorLRUCache(AbstractTorchLRUCache):
    def __init__(self, cache_size, tensor_shape, max_index, dtype, data_device, logic_device="cpu"):
        super().__init__(cache_size, max_index, data_device, logic_device)
        self.tensor_shape = tensor_shape
        self.dtype = dtype
        self.do_ping_memory = not str(self.data_device).startswith("cuda")
        self.cache = torch.zeros((self.cache_size, *self.tensor_shape), dtype=dtype, device=self.data_device,
                                 pin_memory=self.do_ping_memory)

    @abstractmethod
    def compute_idxs(self, idxs):
        raise NotImplementedError("Implement this method to use the torchCache object ")

    def _insert_data(self, tensor_data, lru_slots):
        self.cache[lru_slots.to(self.data_device)] = tensor_data.to(self.data_device)

    def _get_empty_output(self, n):
        return torch.empty((n,) + self.tensor_shape, dtype=self.dtype, device=self.data_device,
                           pin_memory=self.do_ping_memory)

    def _set_output_values(self, output_tensor, output_idxs, cache_idxs):
        if len(cache_idxs) > 0:
            output_tensor[output_idxs] = self.cache[cache_idxs]