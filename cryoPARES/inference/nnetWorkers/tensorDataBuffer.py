import torch
from typing import List, Dict, Any, Callable, Optional

from torch import nn


class StreamingBuffer:
    def __init__(
            self,
            buffer_size: int,
            processing_fn: Callable[[Dict[str, torch.Tensor], Dict[str, List]], List[Any]],
            device: str = 'cpu'
    ):
        if buffer_size <= 0:
            raise ValueError("buffer_size must be a positive integer.")
        self.buffer_size = buffer_size
        self.processing_fn = processing_fn
        self.device = device
        self.current_fill = 0
        self.initialized = False
        self.tensor_buffers: Dict[str, torch.Tensor] = {}
        # Stores metadata in a column-oriented format
        self.metadata_buffers: Dict[str, List] = {}

    def _initialize(self, sample_item: Dict[str, Any]):
        for key, value in sample_item.items():
            if isinstance(value, torch.Tensor):
                self.tensor_buffers[key] = torch.empty(
                    (self.buffer_size,) + value.shape, dtype=value.dtype, device=self.device
                )
            else:  # For metadata, create a pre-allocated list
                self.metadata_buffers[key] = [None] * self.buffer_size
        self.initialized = True

    def _process(self) -> List[Any]:
        if self.current_fill == 0:
            return None
        active_tensors = {k: v[:self.current_fill].clone() for k, v in self.tensor_buffers.items()}
        # Create a dictionary of metadata columns
        active_metadata = {k: v[:self.current_fill] for k, v in self.metadata_buffers.items()}
        processed_results = self.processing_fn(active_tensors, active_metadata)
        self.current_fill = 0
        return processed_results

    def add_batch(self, batch_dict: Dict[str, Any]) -> Optional[List[Any]]:
        if not batch_dict: return None
        batch_size = len(next(iter(batch_dict.values()), []))
        if batch_size == 0: return None
        if not self.initialized:
            self._initialize({key: value[0] for key, value in batch_dict.items()})

        all_processed_results = []
        items_added = 0
        while items_added < batch_size:
            space = self.buffer_size - self.current_fill
            if space == 0:
                results = self._process()
                if results: all_processed_results.extend(results)
                space = self.buffer_size

            num_to_add = min(batch_size - items_added, space)
            src, dest = slice(items_added, items_added + num_to_add), slice(self.current_fill,
                                                                            self.current_fill + num_to_add)

            for key, value in batch_dict.items():
                if key in self.tensor_buffers:
                    self.tensor_buffers[key][dest] = value[src]
                elif key in self.metadata_buffers:
                    self.metadata_buffers[key][dest] = value[src]

            self.current_fill += num_to_add
            items_added += num_to_add

        if self.current_fill >= self.buffer_size:
            results = self._process()
            if results: all_processed_results.extend(results)

        return all_processed_results if all_processed_results else None

    def flush(self) -> List[Any]:
        return self._process()

# ============================================================================
# Test Battery for StreamingBuffer
# ============================================================================
class TestStreamingBuffer:
    def _setup(self):
        """Creates a mock processing function that logs calls."""
        self.mock_call_log = []

        def mock_processing_fn(tensors, metadata):
            self.mock_call_log.append({'tensors': tensors, 'metadata': metadata})
            # Return a predictable value, like the IDs
            return metadata.get('id', [])

        return mock_processing_fn

    def test_initialization(self):
        print("Running: test_initialization")
        processing_fn = self._setup()
        buffer = StreamingBuffer(buffer_size=10, processing_fn=processing_fn)
        assert buffer.buffer_size == 10 and not buffer.initialized
        try:
            StreamingBuffer(buffer_size=0, processing_fn=processing_fn)
            assert False, "Should raise ValueError for non-positive buffer size"
        except ValueError:
            pass  # Expected
        print("✓ PASSED\n")

    def test_flush_empty_buffer(self):
        print("Running: test_flush_empty_buffer")
        processing_fn = self._setup()
        buffer = StreamingBuffer(buffer_size=5, processing_fn=processing_fn)
        results = buffer.flush()
        assert results == []
        assert len(self.mock_call_log) == 0
        print("✓ PASSED\n")

    def test_flush_non_full_buffer(self):
        print("Running: test_flush_non_full_buffer")
        processing_fn = self._setup()
        buffer = StreamingBuffer(buffer_size=10, processing_fn=processing_fn)
        batch = {'id': [0, 1, 2], 'data': torch.randn(3, 2)}
        buffer.add_batch(batch)
        assert buffer.current_fill == 3

        results = buffer.flush()
        assert results == [0, 1, 2]
        assert buffer.current_fill == 0
        print("✓ PASSED\n")

    def test_data_correctness_in_processing_fn(self):
        print("Running: test_data_correctness_in_processing_fn")
        processing_fn = self._setup()
        buffer = StreamingBuffer(buffer_size=2, processing_fn=processing_fn)

        batch = {
            'id': [10, 20],
            'name': ['A', 'B'],
            'data': torch.tensor([[1., 1.], [2., 2.]], device=buffer.device)
        }
        buffer.add_batch(batch)

        assert len(self.mock_call_log) == 1
        call_data = self.mock_call_log[0]

        assert torch.equal(call_data['tensors']['data'], batch['data'])
        assert call_data['metadata']['id'] == [10, 20]
        assert call_data['metadata']['name'] == ['A', 'B']
        print("✓ PASSED\n")

    def test_no_tensor_data(self):
        print("Running: test_no_tensor_data")
        processing_fn = self._setup()
        buffer = StreamingBuffer(buffer_size=3, processing_fn=processing_fn)

        batch = {'id': [101, 102]}
        buffer.add_batch(batch)
        results = buffer.flush()

        assert results == [101, 102]
        assert not self.mock_call_log[0]['tensors']  # Tensors dict should be empty
        print("✓ PASSED\n")

    def test_reuse_buffer_after_flush(self):
        print("Running: test_reuse_buffer_after_flush")
        processing_fn = self._setup()
        buffer = StreamingBuffer(buffer_size=5, processing_fn=processing_fn)

        buffer.add_batch({'id': list(range(7))})  # Causes one overflow
        buffer.flush()
        assert len(self.mock_call_log) == 2

        buffer.add_batch({'id': list(range(100, 103))})
        results = buffer.flush()
        assert results == [100, 101, 102]
        assert len(self.mock_call_log) == 3
        print("✓ PASSED\n")

    def test_error_on_mismatched_tensor_shape(self):
        print("Running: test_error_on_mismatched_tensor_shape")
        processing_fn = self._setup()
        buffer = StreamingBuffer(buffer_size=5, processing_fn=processing_fn)
        buffer.add_batch({'id': [1], 'data': torch.randn(1, 2)})  # Initializes with shape (2,)

        try:
            buffer.add_batch({'id': [2], 'data': torch.randn(1, 3)})  # Mismatched shape (3,)
            assert False, "Should have raised RuntimeError for mismatched shapes"
        except RuntimeError:
            pass  # Expected
        print("✓ PASSED\n")

    def test_device_placement(self):
        print("Running: test_device_placement")
        if not torch.cuda.is_available():
            print("! SKIPPED: CUDA not available.")
            return

        processing_fn = self._setup()
        buffer = StreamingBuffer(buffer_size=2, processing_fn=processing_fn, device='cuda')

        # Add a batch of CPU tensors
        buffer.add_batch({'id': [1], 'data': torch.randn(1, 2)})
        buffer.flush()

        processed_tensors = self.mock_call_log[0]['tensors']
        assert 'cuda' in str(processed_tensors['data'].device)
        print("✓ PASSED\n")

    def run_all(self):
        """Runs all test cases."""
        print("--- Starting Comprehensive Test Battery for StreamingBuffer ---")
        self.test_initialization()
        self.test_flush_empty_buffer()
        self.test_flush_non_full_buffer()
        self.test_data_correctness_in_processing_fn()
        self.test_no_tensor_data()
        self.test_reuse_buffer_after_flush()
        self.test_error_on_mismatched_tensor_shape()
        self.test_device_placement()
        print("--- All Buffer Tests Completed Successfully ---")


if __name__ == "__main__":
    tester = TestStreamingBuffer()
    tester.run_all()