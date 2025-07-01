import torch
from typing import List, Dict, Any, Callable, Optional

from torch import nn


class StreamingBuffer(nn.Module):
    """
    A generic buffer for streaming and batching data for processing.
    """

    def __init__(
            self,
            buffer_size: int,
            processing_fn: Callable[[Dict[str, torch.Tensor], List[Dict[str, Any]]], List[Any]],
            device: str = 'cpu'
    ):
        super().__init__()
        if buffer_size <= 0:
            raise ValueError("buffer_size must be a positive integer.")
        self.buffer_size = buffer_size
        self.processing_fn = processing_fn
        self.device = device
        self.current_fill = 0
        self.initialized = False
        self.tensor_buffers: Dict[str, torch.Tensor] = {}
        self.metadata_buffer: List[Optional[Dict[str, Any]]] = [None] * buffer_size

    def _initialize(self, sample_item: Dict[str, Any]):
        """Initialize buffers based on the shape and type of the first item."""
        for key, value in sample_item.items():
            if isinstance(value, torch.Tensor):
                value_on_device = value.to(self.device)
                self.tensor_buffers[key] = torch.empty(
                    (self.buffer_size,) + value_on_device.shape,
                    dtype=value_on_device.dtype,
                    device=self.device
                )
        self.initialized = True

    def _process(self) -> List[Any]:
        """Processes the current buffer contents and resets the buffer."""
        if self.current_fill == 0:
            return []
        active_tensors = {k: v[:self.current_fill].clone() for k, v in self.tensor_buffers.items()}
        active_metadata = self.metadata_buffer[:self.current_fill]
        processed_results = self.processing_fn(active_tensors, active_metadata)
        self.current_fill = 0
        return processed_results

    def add(self, item: Dict[str, Any]) -> Optional[List[Any]]:
        """Adds a single item. Less efficient than add_batch."""
        # This can be seen as a special case of add_batch with a batch size of 1
        batch_dict = {key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else [value]
                      for key, value in item.items()}
        return self.add_batch(batch_dict)

    def add_batch(self, batch_dict: Dict[str, Any]) -> Optional[List[Any]]:
        """
        Adds a batch of items using optimized, vectorized memory copies.
        """
        if not batch_dict: return None

        # --- Determine Batch Size ---
        batch_size = 0
        tensor_keys = [k for k, v in batch_dict.items() if isinstance(v, torch.Tensor)]
        if not tensor_keys:  # Handle metadata-only case
            batch_size = len(next(iter(batch_dict.values()), []))
        else:
            batch_size = batch_dict[tensor_keys[0]].shape[0]

        if batch_size == 0: return None

        # --- Lazily Initialize Buffer ---
        if not self.initialized:
            # Reconstruct a single sample item for initialization
            sample_item = {key: value[0] for key, value in batch_dict.items()}
            self._initialize(sample_item)

        # --- Main Processing Loop (per chunk, not per item) ---
        all_processed_results = []
        items_added_from_batch = 0
        while items_added_from_batch < batch_size:
            space_available = self.buffer_size - self.current_fill

            if space_available == 0:
                results = self._process()
                if results: all_processed_results.extend(results)
                space_available = self.buffer_size

            num_to_add = min(batch_size - items_added_from_batch, space_available)

            # Define source and destination slices
            src_start = items_added_from_batch
            src_end = items_added_from_batch + num_to_add
            dest_start = self.current_fill
            dest_end = self.current_fill + num_to_add

            # Perform batched memory copy for all tensors
            for key in tensor_keys:
                self.tensor_buffers[key][dest_start:dest_end] = batch_dict[key][src_start:src_end]

            # Handle metadata chunk
            non_tensor_keys = [k for k in batch_dict if k not in tensor_keys]
            for i in range(num_to_add):
                item_idx_in_batch = src_start + i
                buffer_idx = dest_start + i
                self.metadata_buffer[buffer_idx] = {key: batch_dict[key][item_idx_in_batch] for key in non_tensor_keys}

            # Update counters
            self.current_fill += num_to_add
            items_added_from_batch += num_to_add

        # If the last addition made the buffer exactly full, process it
        if self.current_fill >= self.buffer_size:
            results = self._process()
            if results: all_processed_results.extend(results)

        return all_processed_results if all_processed_results else None

    def flush(self) -> List[Any]:
        """Process any remaining items in the buffer."""
        return self._process()

    def to(self, *args, **kwargs):
        """
        Overrides the nn.Module's .to() method to also handle the buffer.
        """
        # 1. First, call the original .to() method from the parent class
        # This will move all registered sub-modules (stage1_model, stage2_model)
        # and registered parameters/buffers.
        super().to(*args, **kwargs)

        # 2. Extract the target device from the arguments
        # This handles calls like .to('cuda'), .to(torch.device('cuda')), etc.
        device, _, _, _ = torch.nn.modules.module._global_forward_hooks.values()
        if device is None:
            # Handle cases where device is not in the hook
            if 'device' in kwargs:
                device = kwargs['device']
            elif len(args) > 0:
                device = args[0]

        # 3. Update the device for the model and the buffer
        self.device = device
        self.buffer.device = device

        # 4. If the buffer has already been initialized and contains tensors,
        # move them to the new device.
        if self.buffer.initialized:
            for key, tensor_buffer in self.buffer.tensor_buffers.items():
                self.buffer.tensor_buffers[key] = tensor_buffer.to(device)

        # 5. Return self to allow for chaining (e.g., model = Model().to(device))
        return self


# ============================================================================
# Test Battery for StreamingBuffer
# ============================================================================

class TestStreamingBuffer:
    def _setup(self):
        """Creates a mock processing function for testing."""
        self.mock_call_log = []

        def mock_processing_fn(tensors, metadata):
            # Log the call with the data it received
            self.mock_call_log.append({
                'tensors': tensors,
                'metadata': metadata
            })
            # Return something predictable, like the IDs of processed items
            return [m['id'] for m in metadata]

        return mock_processing_fn

    def _test_initialization(self):
        print("Running: _test_initialization")
        processing_fn = self._setup()
        buffer = StreamingBuffer(buffer_size=10, processing_fn=processing_fn)

        assert buffer.buffer_size == 10
        assert buffer.current_fill == 0
        assert not buffer.initialized
        assert not buffer.tensor_buffers

        # Test for invalid buffer size
        try:
            StreamingBuffer(buffer_size=0, processing_fn=processing_fn)
            assert False, "Should have raised ValueError for buffer_size=0"
        except ValueError:
            pass  # Expected
        print("✓ PASSED\n")

    def _test_lazy_initialization_on_first_add(self):
        print("Running: _test_lazy_initialization_on_first_add")
        processing_fn = self._setup()
        buffer = StreamingBuffer(buffer_size=5, processing_fn=processing_fn)

        assert not buffer.initialized

        item = {'id': 1, 'data': torch.randn(3, 32)}
        buffer.add(item)

        assert buffer.initialized
        assert buffer.current_fill == 1
        assert 'data' in buffer.tensor_buffers
        assert buffer.tensor_buffers['data'].shape == (5, 3, 32)
        assert len(self.mock_call_log) == 0, "Processing should not be called"
        print("✓ PASSED\n")

    def _test_add_items_to_exact_capacity(self):
        print("Running: _test_add_items_to_exact_capacity")
        processing_fn = self._setup()
        buffer = StreamingBuffer(buffer_size=3, processing_fn=processing_fn)

        # Helper to create items, as add() now wraps add_batch()
        def item_creator(item_id):
            return {'id': item_id, 'data': torch.ones(1, device=buffer.device)}

        # Add 2 items, buffer should not process
        results1 = buffer.add(item_creator(1))
        results2 = buffer.add(item_creator(2))
        assert results1 is None
        assert results2 is None
        assert buffer.current_fill == 2

        results3 = buffer.add(item_creator(3))
        assert results3 == [1, 2, 3], "The 3rd item should make the buffer full and trigger processing."
        assert buffer.current_fill == 0, "Buffer should be empty after processing."

        # Adding a 4th item now just adds to the empty buffer
        results4 = buffer.add(item_creator(4))
        assert results4 is None, "Adding to a freshly emptied buffer should not return results."
        assert buffer.current_fill == 1
        print("✓ PASSED\n")

    def _test_flush_non_full_buffer(self):
        print("Running: _test_flush_non_full_buffer")
        processing_fn = self._setup()
        buffer = StreamingBuffer(buffer_size=10, processing_fn=processing_fn)

        for i in range(4):
            buffer.add({'id': i, 'data': torch.tensor([float(i)])})

        assert buffer.current_fill == 4
        assert len(self.mock_call_log) == 0

        # Flush the buffer
        results = buffer.flush()

        assert results == [0, 1, 2, 3], "Flush should process the 4 items"
        assert len(self.mock_call_log) == 1
        assert buffer.current_fill == 0, "Buffer should be empty after flush"
        print("✓ PASSED\n")

    def _test_flush_empty_buffer(self):
        print("Running: _test_flush_empty_buffer")
        processing_fn = self._setup()
        buffer = StreamingBuffer(buffer_size=5, processing_fn=processing_fn)

        results = buffer.flush()

        assert results == [], "Flushing an empty buffer should return an empty list"
        assert len(self.mock_call_log) == 0, "Processing function should not be called"
        print("✓ PASSED\n")

    def _test_data_correctness_in_processing_fn(self):
        print("Running: _test_data_correctness_in_processing_fn")
        processing_fn = self._setup()
        buffer = StreamingBuffer(buffer_size=2, processing_fn=processing_fn)

        item1 = {'id': 10, 'name': 'A', 'data': torch.tensor([1.0, 1.0])}
        item2 = {'id': 20, 'name': 'B', 'data': torch.tensor([2.0, 2.0])}
        buffer.add(item1)
        buffer.add(item2)
        # Add a 3rd item to trigger processing of the first two
        buffer.add({'id': 30, 'name': 'C', 'data': torch.tensor([3.0, 3.0])})

        assert len(self.mock_call_log) == 1
        call_data = self.mock_call_log[0]

        # Check tensors
        assert 'data' in call_data['tensors']
        batched_tensor = call_data['tensors']['data']
        assert batched_tensor.shape == (2, 2)
        assert torch.equal(batched_tensor[0], torch.tensor([1.0, 1.0], device=buffer.device))
        assert torch.equal(batched_tensor[1], torch.tensor([2.0, 2.0], device=buffer.device))

        # Check metadata
        metadata = call_data['metadata']
        assert len(metadata) == 2
        assert metadata[0] == {'id': 10, 'name': 'A'}
        assert metadata[1] == {'id': 20, 'name': 'B'}
        print("✓ PASSED\n")

    def _test_no_tensor_data(self):
        print("Running: _test_no_tensor_data")
        processing_fn = self._setup()

        # The processing function needs to be adapted as it expects tensors
        def no_tensor_fn(tensors, metadata):
            self.mock_call_log.append(metadata)
            return [m['id'] for m in metadata]

        buffer = StreamingBuffer(buffer_size=3, processing_fn=no_tensor_fn)

        buffer.add({'id': 1})
        buffer.add({'id': 2})

        results = buffer.flush()

        assert results == [1, 2]
        assert len(self.mock_call_log) == 1
        assert self.mock_call_log[0] == [{'id': 1}, {'id': 2}]
        assert not buffer.tensor_buffers, "Tensor buffers should be empty"
        print("✓ PASSED\n")

    def _test_large_input_causing_multiple_overflows(self):
        print("Running: _test_large_input_causing_multiple_overflows")
        processing_fn = self._setup()
        buffer = StreamingBuffer(buffer_size=3, processing_fn=processing_fn)

        total_results = []
        # Simulate adding 10 items from a large batch
        for i in range(10):
            results = buffer.add({'id': i, 'data': torch.tensor([float(i)], device=buffer.device)})
            if results:
                total_results.extend(results)

        # Check intermittent results
        assert len(total_results) == 6, f"Expected 6 processed items from 2 overflows, got {len(total_results)}"
        assert total_results == [0, 1, 2, 3, 4, 5]

        # Check state before flush
        assert buffer.current_fill == 1, "Buffer should have 1 item left before flush (item with id 9 was added last)"
        assert len(self.mock_call_log) == 2, "Processing fn should have been called twice"

        # Flush the remainder
        final_results = buffer.flush()
        assert final_results == [9], "Flush should process the last item (with id 9)"
        total_results.extend(final_results)

        assert len(total_results) == 7, "Total items processed after flush is wrong"
        assert buffer.current_fill == 0
        print("✓ PASSED\n")

    def _test_reuse_buffer_after_flush(self):
        print("Running: _test_reuse_buffer_after_flush")
        processing_fn = self._setup()
        buffer = StreamingBuffer(buffer_size=5, processing_fn=processing_fn)

        # First stream of data
        for i in range(7):
            buffer.add({'id': i})

        results1 = buffer.flush()
        assert results1 == [5, 6], "First flush should process remaining items"
        assert buffer.current_fill == 0, "Buffer should be empty"
        assert len(self.mock_call_log) == 2, "Should be 2 calls from first stream"

        # Second stream of data
        for i in range(100, 103):
            buffer.add({'id': i})

        assert buffer.current_fill == 3, "Buffer should have 3 new items"
        results2 = buffer.flush()
        assert results2 == [100, 101, 102], "Second flush should process the new items"
        assert len(self.mock_call_log) == 3, "A 3rd call should have happened for the second stream"
        assert buffer.current_fill == 0
        print("✓ PASSED\n")

    def _test_large_input_causing_multiple_overflows(self):
        print("Running: _test_large_input_causing_multiple_overflows")
        processing_fn = self._setup()
        buffer = StreamingBuffer(buffer_size=3, processing_fn=processing_fn)

        total_results = []
        # Simulate adding 10 items from a large batch
        for i in range(10):
            results = buffer.add({'id': i, 'data': torch.tensor([float(i)], device=buffer.device)})
            if results:
                total_results.extend(results)

        # ✅ FIX: Corrected assertions based on the proper trace.
        # Check intermittent results from 3 overflows, not 2.
        assert len(total_results) == 9, f"Expected 9 processed items from 3 overflows, got {len(total_results)}"
        assert total_results == [0, 1, 2, 3, 4, 5, 6, 7, 8]

        # Check state before flush
        assert buffer.current_fill == 1, "Buffer should have 1 item left (item with id 9)"
        assert len(self.mock_call_log) == 3, "Processing fn should have been called three times"

        # Flush the remainder
        final_results = buffer.flush()
        assert final_results == [9], "Flush should process the last item (with id 9)"
        total_results.extend(final_results)

        assert len(total_results) == 10, "Total items processed after flush should be 10"
        assert buffer.current_fill == 0
        print("✓ PASSED\n")

    def _test_reuse_buffer_after_flush(self):
        print("Running: _test_reuse_buffer_after_flush")
        processing_fn = self._setup()
        buffer = StreamingBuffer(buffer_size=5, processing_fn=processing_fn)

        # First stream of data
        for i in range(7):
            buffer.add({'id': i})

        results1 = buffer.flush()
        assert results1 == [5, 6], "First flush should process remaining items"
        assert buffer.current_fill == 0, "Buffer should be empty"
        assert len(self.mock_call_log) == 2, "Should be 2 calls from first stream"

        # Second stream of data
        for i in range(100, 103):
            buffer.add({'id': i})

        assert buffer.current_fill == 3, "Buffer should have 3 new items"
        results2 = buffer.flush()
        assert results2 == [100, 101, 102], "Second flush should process the new items"
        assert len(self.mock_call_log) == 3, "A 3rd call should have happened for the second stream"
        assert buffer.current_fill == 0
        print("✓ PASSED\n")

    def _test_error_on_mismatched_tensor_shape(self):
        print("Running: _test_error_on_mismatched_tensor_shape")
        processing_fn = self._setup()
        buffer = StreamingBuffer(buffer_size=5, processing_fn=processing_fn)

        # Initialize with a tensor of shape (2,)
        buffer.add({'id': 1, 'data': torch.randn(2)})

        # Try to add a tensor with a different shape
        try:
            buffer.add({'id': 2, 'data': torch.randn(3)})  # Shape (3,) is different
            assert False, "Should have raised RuntimeError for mismatched shapes"
        except RuntimeError as e:
            # This is the expected error from PyTorch when shapes don't align
            assert " size of" in str(e) or "shape mismatch" in str(e)

        print("✓ PASSED\n")

    def _test_device_placement(self):
        print("Running: _test_device_placement")
        if not torch.cuda.is_available():
            print("! SKIPPED: CUDA not available.")
            return

        processing_fn = self._setup()
        buffer = StreamingBuffer(buffer_size=2, processing_fn=processing_fn, device='cuda')

        # Add CPU tensors to the CUDA-configured buffer
        buffer.add({'id': 1, 'data': torch.randn(2)})
        buffer.add({'id': 2, 'data': torch.randn(2)})

        # Flush to trigger processing
        buffer.flush()

        assert len(self.mock_call_log) == 1
        processed_tensors = self.mock_call_log[0]['tensors']

        # Verify the tensor received by the processing function is on the correct device
        assert 'data' in processed_tensors
        assert 'cuda' in str(processed_tensors['data'].device)
        print("✓ PASSED\n")

    def test_adding_from_a_batched_output(self):
        print("Running: test_adding_from_a_batched_output")
        processing_fn = self._setup()
        buffer = StreamingBuffer(buffer_size=4, processing_fn=processing_fn)

        # 1. Simulate a single batch of 5 items produced by a model
        batch_output = {
            'id': [f'img_{i}' for i in range(5)],
            'data': torch.randn(5, 10),
        }

        # 2. Add the entire batch in one call
        results = buffer.add_batch(batch_output)

        # This single call should have processed the first 4 items
        assert results == ['img_0', 'img_1', 'img_2', 'img_3']
        assert buffer.current_fill == 1, "One item should be left in the buffer"

        # 3. Flush the remaining item
        final_results = buffer.flush()
        assert final_results == ['img_4']
        print("✓ PASSED\n")

    def test_batch_add_to_exact_capacity(self):
        print("Running: test_batch_add_to_exact_capacity")
        processing_fn = self._setup()
        buffer = StreamingBuffer(buffer_size=5, processing_fn=processing_fn)

        # Add a batch of 3, should not process
        batch1 = {'id': [1, 2, 3], 'data': torch.randn(3, 2)}
        results1 = buffer.add_batch(batch1)
        assert results1 is None
        assert buffer.current_fill == 3

        # Add a batch of 2, making the buffer exactly full and triggering processing
        batch2 = {'id': [4, 5], 'data': torch.randn(2, 2)}
        results2 = buffer.add_batch(batch2)
        assert results2 == [1, 2, 3, 4, 5]
        assert buffer.current_fill == 0
        print("✓ PASSED\n")

    def test_batch_add_causing_overflow(self):
        print("Running: test_batch_add_causing_overflow")
        processing_fn = self._setup()
        buffer = StreamingBuffer(buffer_size=5, processing_fn=processing_fn)

        # 1. Add a batch of 3, leaving 2 spots available
        batch1 = {'id': [0, 1, 2], 'data': torch.randn(3, 2)}
        buffer.add_batch(batch1)
        assert buffer.current_fill == 3

        # 2. Add a batch of 4. This will:
        #    - Fill the last 2 spots
        #    - Trigger processing of the full buffer [0,1,2,3,4]
        #    - Add the remaining 2 items to the now-empty buffer
        batch2 = {'id': [3, 4, 5, 6], 'data': torch.randn(4, 2)}
        results = buffer.add_batch(batch2)

        assert results == [0, 1, 2, 3, 4]
        assert buffer.current_fill == 2, "Should have 2 items left in the buffer"

        # 3. Flush the remaining items
        final = buffer.flush()
        assert final == [5, 6]
        assert buffer.current_fill == 0
        print("✓ PASSED\n")

    def test_batch_overflow_with_different_buffer_size(self):
        print("Running: test_batch_overflow_with_different_buffer_size (buffer_size=7)")
        processing_fn = self._setup()
        buffer = StreamingBuffer(buffer_size=7, processing_fn=processing_fn)

        # 1. Add a batch of 4, leaving 3 spots available
        batch1 = {'id': [0, 1, 2, 3], 'data': torch.randn(4, 2)}
        buffer.add_batch(batch1)
        assert buffer.current_fill == 4

        # 2. Add a batch of 5. This will:
        #    - Fill the last 3 spots with items [4, 5, 6]
        #    - Trigger processing of the full buffer [0,1,2,3,4,5,6]
        #    - Add the remaining 2 items [7, 8] to the now-empty buffer
        batch2 = {'id': [4, 5, 6, 7, 8], 'data': torch.randn(5, 2)}
        results = buffer.add_batch(batch2)

        assert results == [0, 1, 2, 3, 4, 5, 6]
        assert buffer.current_fill == 2, "Should have 2 items left in the buffer"

        # 3. Flush the remaining items
        final = buffer.flush()
        assert final == [7, 8]
        assert buffer.current_fill == 0
        print("✓ PASSED\n")

    def run_all(self):
        """Runs all test cases for the StreamingBuffer."""
        print("--- Starting Test Battery for StreamingBuffer ---")
        self._test_initialization()
        self._test_lazy_initialization_on_first_add()
        self._test_add_items_to_exact_capacity()
        self._test_flush_non_full_buffer()
        self._test_flush_empty_buffer()
        self._test_data_correctness_in_processing_fn()
        self._test_no_tensor_data()
        self._test_large_input_causing_multiple_overflows()
        self._test_reuse_buffer_after_flush()
        self._test_error_on_mismatched_tensor_shape()
        self._test_device_placement()
        self.test_adding_from_a_batched_output()
        self.test_batch_add_to_exact_capacity()
        self.test_batch_add_causing_overflow()
        self.test_batch_overflow_with_different_buffer_size()
        print("--- All Buffer Tests Completed Successfully ---")

if __name__ == "__main__":
    tester = TestStreamingBuffer()
    tester.run_all()