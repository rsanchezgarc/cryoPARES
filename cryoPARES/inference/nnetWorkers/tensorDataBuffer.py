import functools
import itertools
from typing import Callable, Dict, List, Any, Union

import torch
from torch import nn


class StreamingBuffer(nn.Module):
    def __init__(
            self,
            buffer_size: int,
            processing_fn: Callable[[Dict[str, torch.Tensor]], Any],
            max_buffer_capacity: int = None,
    ) -> None:
        super().__init__()
        if max_buffer_capacity is not None and max_buffer_capacity < buffer_size:
            raise ValueError("max_buffer_capacity must be greater than or equal to buffer_size")

        self.buffer_size = buffer_size
        self.processing_fn = processing_fn
        self.max_capacity = max_buffer_capacity or (buffer_size * 2)

        self.tensor_buffers = {}
        self.list_buffers = {}
        self.total_items = 0

    def _ensure_buffer_exists(self, key: str, value: Union[torch.Tensor, List]):
        if isinstance(value, torch.Tensor):
            if key not in self.tensor_buffers:
                buffer_shape = (self.max_capacity,) + value.shape[1:]
                self.tensor_buffers[key] = torch.empty(
                    buffer_shape, dtype=value.dtype, device=value.device
                )
        elif isinstance(value, list):
            if key not in self.list_buffers:
                self.list_buffers[key] = [None] * self.max_capacity

    def add_batch(self, batch: Dict[str, Union[torch.Tensor, List[Any]]]) -> Any:
        new_batch_size = len(next(iter(batch.values()))) if batch else 0
        if new_batch_size == 0:
            return None

        for key, value in batch.items():
            self._ensure_buffer_exists(key, value)

        all_processed_results = []

        buffered_items_available = self.total_items
        new_batch_items_available = new_batch_size

        buffered_items_pos = 0
        new_batch_items_pos = 0

        while buffered_items_available + new_batch_items_available >= self.buffer_size:
            num_from_buffer = min(buffered_items_available, self.buffer_size)
            num_from_new = self.buffer_size - num_from_buffer

            chunk_to_process = {}
            for key, new_data in batch.items():
                if isinstance(new_data, torch.Tensor):
                    slice1 = self.tensor_buffers[key][buffered_items_pos: buffered_items_pos + num_from_buffer]
                    slice2 = new_data[new_batch_items_pos: new_batch_items_pos + num_from_new]
                    if num_from_buffer > 0 and num_from_new > 0:
                        chunk_to_process[key] = torch.cat((slice1, slice2), dim=0)
                    else:
                        chunk_to_process[key] = slice1 if num_from_new == 0 else slice2
                else:  # list
                    slice1 = self.list_buffers[key][buffered_items_pos: buffered_items_pos + num_from_buffer]
                    slice2 = new_data[new_batch_items_pos: new_batch_items_pos + num_from_new]
                    chunk_to_process[key] = slice1 + slice2

            result = self.processing_fn(**chunk_to_process)
            all_processed_results.append(result)

            buffered_items_pos += num_from_buffer
            buffered_items_available -= num_from_buffer
            new_batch_items_pos += num_from_new
            new_batch_items_available -= num_from_new

        # 3. Create a single batch of all leftover items.
        total_leftovers = buffered_items_available + new_batch_items_available
        self.total_items = total_leftovers

        if total_leftovers > 0:
            leftover_batch = {}
            for key, new_data in batch.items():
                if isinstance(new_data, torch.Tensor):
                    s1 = self.tensor_buffers[key][buffered_items_pos: buffered_items_pos + buffered_items_available]
                    s2 = new_data[new_batch_items_pos:]
                    leftover_batch[key] = torch.cat((s1, s2)) if buffered_items_available > 0 else s2
                else:  # list
                    s1 = self.list_buffers[key][buffered_items_pos: buffered_items_pos + buffered_items_available]
                    s2 = new_data[new_batch_items_pos:]
                    leftover_batch[key] = s1 + s2

            # 4. Store the single leftover_batch in the clean buffers
            for key, data in leftover_batch.items():
                if isinstance(data, torch.Tensor):
                    self.tensor_buffers[key][:total_leftovers] = data
                else:
                    self.list_buffers[key][:total_leftovers] = data

        if not all_processed_results:
            return None

        return self._combine_results(all_processed_results) if len(all_processed_results) > 1 else \
        all_processed_results[0]

    def flush(self):
        if self.total_items == 0:
            return None

        batch_to_process = {key: buf[:self.total_items] for key, buf in self.tensor_buffers.items()}
        batch_to_process.update({key: buf[:self.total_items] for key, buf in self.list_buffers.items()})

        result = self.processing_fn(**batch_to_process) if batch_to_process else None
        self.total_items = 0
        return result

    def _combine_results(self, all_results):
        all_results = [res for res in all_results if res is not None]
        if not all_results: return None

        first_item = all_results[0]

        if isinstance(first_item, torch.Tensor):
            return torch.cat(all_results, dim=0)

        if isinstance(first_item, (list, tuple)):
            combined_result = []
            num_outputs = len(first_item)
            for i in range(num_outputs):
                output_parts = [result[i] for result in all_results]
                if not output_parts: continue

                if isinstance(output_parts[0], torch.Tensor):
                    combined_result.append(torch.cat(output_parts, dim=0))
                elif isinstance(output_parts[0], list):
                    combined_result.append(list(itertools.chain.from_iterable(output_parts)))
                else:
                    combined_result.append(output_parts)
            return combined_result

        else:  # Handle scalar results like int, float
            return all_results


# ==================================================================
#                           TEST FUNCTIONS
# ==================================================================

def test_comprehensive_chunking_scenarios():
    """Tests multiple scenarios for the buffer's chunking logic."""
    print("\n=== Comprehensive Chunking and Buffering Test ===")

    scenarios = [
        ("Small batch, no processing", 5, 4, [], 9),
        ("Small batch, triggers processing", 7, 8, [10], 5),
        ("Large batch, multiple chunks, exact", 5, 15, [10, 10], 0),
        ("Large batch, multiple chunks, with leftovers", 5, 18, [10, 10], 3),
        ("Empty buffer, large batch with leftovers", 0, 23, [10, 10], 3),
        ("Empty buffer, small batch", 0, 7, [], 7),
    ]

    for name, initial_items, new_batch_size, expected_chunks, expected_leftovers in scenarios:
        print(f"\n--- Testing Scenario: {name} ---")

        proc_results = []

        def proc_fn(**kwargs):
            chunk_size = kwargs['data'].shape[0]
            proc_results.append(chunk_size)
            return chunk_size

        buffer = StreamingBuffer(buffer_size=10, processing_fn=proc_fn, max_buffer_capacity=20)

        if initial_items > 0:
            buffer.add_batch({'data': torch.randn(initial_items, 2)})
        print(f"Initial state: {buffer.total_items} items in buffer.")
        assert buffer.total_items == initial_items

        proc_results.clear()
        buffer.add_batch({'data': torch.randn(new_batch_size, 2)})

        print(f"add_batch({new_batch_size}) -> Processed chunks: {proc_results}, Leftovers: {buffer.total_items}")

        assert proc_results == expected_chunks
        assert buffer.total_items == expected_leftovers

        if buffer.total_items > 0:
            flushed_size = buffer.flush()
            print(f"flush() -> Flushed size: {flushed_size}, Final leftovers: {buffer.total_items}")
            assert flushed_size == expected_leftovers
            assert buffer.total_items == 0

        print(f"SCENARIO PASSED.")


def test_mixed_data_types():
    """Test with mixed tensors and lists."""
    print("\n=== Mixed Data Types Test ===")

    def proc_fn(**kwargs):
        return (kwargs['names'], kwargs['data'])

    buffer = StreamingBuffer(buffer_size=3, processing_fn=proc_fn, max_buffer_capacity=5)

    buffer.add_batch({'names': ['A', 'B'], 'data': torch.ones(2, 1)})
    assert buffer.total_items == 2

    results = buffer.add_batch({'names': ['C', 'D', 'E'], 'data': torch.ones(3, 1)})
    print(f"Processed names: {results[0]}")
    print(f"Leftover items: {buffer.total_items}")

    assert results[0] == ['A', 'B', 'C']
    assert results[1].shape[0] == 3
    assert buffer.total_items == 2

    flushed = buffer.flush()
    print(f"Flushed names: {flushed[0]}")
    assert flushed[0] == ['D', 'E']
    assert flushed[1].shape[0] == 2
    assert buffer.total_items == 0
    print("SCENARIO PASSED.")


if __name__ == "__main__":
    test_comprehensive_chunking_scenarios()
    print("-" * 50)
    test_mixed_data_types()