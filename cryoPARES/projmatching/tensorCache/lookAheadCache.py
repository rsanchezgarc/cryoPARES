import torch

from .abstractTorchCache import AbstractTorchCache


class LookAheadTensorCache(AbstractTorchCache):
    def __init__(self, cache_size, tensor_shape, max_index, dtype, data_device, logic_device="cpu"):

        super().__init__(cache_size, max_index, data_device, logic_device)

        self.tensor_shape = tensor_shape
        self.dtype = dtype
        self.cache = torch.zeros((self.cache_size, *self.tensor_shape), dtype=dtype, device=self.data_device)
        self.current_min = -1

    def compute_idxs(self, idxs):
        # Placeholder: Implement this method based on specific computation for tensor values given idxs
        raise NotImplementedError()


    def _get_empty_output(self, n):
        return torch.empty((n,) + self.tensor_shape, dtype=self.dtype, device=self.data_device)

    def _set_output_values(self, output_tensor, output_idxs, cache_idxs):
        if len(cache_idxs) > 0:
            output_tensor[output_idxs] = self.cache[cache_idxs]


    def insert(self, index_tensors, min_index_to_keep):
        """

        :param index_tensors: unique indices
        :param tensor_data:   values associated to the unique indices
        :return:
        """

        num_unique_elements = len(index_tensors)
        # cache_idx_for_min = self.index_tracker[min_index_to_keep]
        # cache_idx_for_min = torch.where(cache_idx_for_min>=0, cache_idx_for_min, torch.inf)
        non_useful_cache_slots = torch.lt(self.reverse_index_tracker, min_index_to_keep)
        free_space = torch.count_nonzero(non_useful_cache_slots)
        # if free_space < num_unique_elements:
        #     print()
        assert free_space >= num_unique_elements, (f"Error, the cache size ({self.cache_size}) "
                                                   f"is too small to store all the elements of the batch "
                                                   f"(free:{free_space}, num_unique_elements:{num_unique_elements})")
        max_in_indices = torch.max(index_tensors).item()

        lookahead_index = min(max_in_indices+1+free_space-num_unique_elements, self.max_index)
        if max_in_indices+1 < lookahead_index:
            next_indices = torch.arange(max_in_indices+1, lookahead_index, device=self.logic_device)
            index_tensors_with_lookahead = torch.cat([index_tensors, next_indices])
        else:
            next_indices = torch.tensor([], dtype=torch.float)
            index_tensors_with_lookahead = index_tensors

        # Compute data for new indices
        if len(next_indices) > 0:
            # next_data = self.compute_idxs(next_indices)
            # tensor_data = torch.cat([tensor_data, next_data])
            tensor_data = self.compute_idxs(torch.cat([index_tensors, next_indices]))
        else:
            tensor_data = self.compute_idxs(index_tensors)

        new_cache_idxs = torch.where(non_useful_cache_slots)[0][:tensor_data.shape[0]]

        # non_useful_cache_slots[new_cache_idxs[:self.cache_size-free_space]] = False  # Adjust for the case in which we have less new examples than the number of available cache slots (when i==i_Max)
        # new_cache_idxs = new_cache_idxs[self.cache_size - free_space:]

        # Update the cache and index tracker
        self.cache[new_cache_idxs] = tensor_data
        self.index_tracker[self.reverse_index_tracker[new_cache_idxs]] = -1
        self.index_tracker[index_tensors_with_lookahead] = new_cache_idxs
        self.reverse_index_tracker[new_cache_idxs] = index_tensors_with_lookahead

    def retrieve(self, index_tensors):
        # Ensure index_tensors is sorted and consecutive

        output_tensor = self._get_empty_output(len(index_tensors))
        cache_slots = self.index_tracker[index_tensors]
        valid_slots = cache_slots != -1
        self._set_output_values(output_tensor, valid_slots, cache_slots[valid_slots])

        missing_indices = index_tensors[~valid_slots]

        if len(missing_indices) > 0:
            unique_missing_indices = torch.unique(missing_indices)

            min_index_to_keep =  index_tensors.min()
            min_index_to_keep_item = min_index_to_keep.item()
            assert self.current_min <= min_index_to_keep_item, "Error, this cache expects monotonically increasing values"
            self.current_min = min_index_to_keep_item
            self.insert(unique_missing_indices, min_index_to_keep)
            self._set_output_values(output_tensor, ~valid_slots, self.index_tracker[missing_indices])
        return output_tensor

    def __getitem__(self, item):
        return self.retrieve(item)

