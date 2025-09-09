from abc import abstractmethod

import torch

from .abstractTorchCache import AbstractTorchLRUCache


class TensorListLRUCache(AbstractTorchLRUCache):

    def __init__(self, cache_size, max_index, tensor_shapes, dtypes, data_device, logic_device="cpu"):
        super().__init__(cache_size, max_index, data_device, logic_device)
        self.tensor_shapes = tensor_shapes
        self.dtypes = dtypes
        # Initialize multiple caches
        self.caches = [torch.empty((self.cache_size, *shape), dtype=dtype, device=self.data_device)
                       for shape, dtype in zip(self.tensor_shapes, self.dtypes)]

    @abstractmethod
    def compute_idxs(self, idxs):
        raise NotImplementedError("Implement this method to use the TensorListLRUCache object")

    def _compute_idxs(self, idxs):
        return [x.to(self.data_device) for x in self.compute_idxs(idxs)]

    def _insert_data(self, tensor_data_list, lru_slots):
        for cache, tensor_data in zip(self.caches, tensor_data_list):
            cache[lru_slots.to(cache.device)] = tensor_data.to(cache.device)


    def _get_empty_output(self, n):
        return [torch.empty((n,) + shape, dtype=dtype, device=self.data_device)
                for shape, dtype in zip(self.tensor_shapes, self.dtypes)]


    def _set_output_values(self, output_tensors, output_idxs, cache_idxs): #This is quite ineficient
        for output_tensor, cache in zip(output_tensors, self.caches):
            if len(cache_idxs) > 0:
                output_tensor[output_idxs] = cache[cache_idxs]