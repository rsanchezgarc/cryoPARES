# tests/test_tensorDataBuffer.py
import unittest
import torch
from cryoPARES.inference.nnetWorkers.tensorDataBuffer import StreamingBuffer


class TestStreamingBuffer(unittest.TestCase):
    """
    Tests for the StreamingBuffer class, which accumulates batches and processes
    them once a size threshold is met.
    """

    def setUp(self):
        """Set up test fixtures."""
        self.processing_calls = []

    def _create_mock_processing_fn(self, return_value="processed"):
        """
        Creates a mock processing function that records calls and returns a value.

        Args:
            return_value: The value to return when called

        Returns:
            A callable that records arguments and returns the specified value
        """
        def mock_fn(**kwargs):
            self.processing_calls.append(kwargs)
            return return_value
        return mock_fn

    def test_basic_buffer_accumulation(self):
        """Test that buffer accumulates batches without processing until threshold."""
        buffer = StreamingBuffer(
            buffer_size=10,
            processing_fn=self._create_mock_processing_fn()
        )

        # Add small batch (3 items)
        batch1 = {'data': torch.tensor([1, 2, 3]), 'labels': [0, 1, 2]}
        result = buffer.add_batch(batch1)

        # Should not process yet (3 < 10)
        self.assertIsNone(result)
        self.assertEqual(len(self.processing_calls), 0)
        self.assertEqual(len(buffer.storage), 1)

    def test_buffer_processes_when_threshold_reached(self):
        """Test that buffer processes when accumulated size reaches threshold."""
        buffer = StreamingBuffer(
            buffer_size=10,
            processing_fn=self._create_mock_processing_fn("result")
        )

        # Add batches that total to >= 10
        batch1 = {'data': torch.tensor([1, 2, 3, 4, 5]), 'labels': ['a', 'b', 'c', 'd', 'e']}
        batch2 = {'data': torch.tensor([6, 7, 8, 9, 10]), 'labels': ['f', 'g', 'h', 'i', 'j']}

        result1 = buffer.add_batch(batch1)
        self.assertIsNone(result1)  # 5 < 10

        result2 = buffer.add_batch(batch2)
        self.assertEqual(result2, "result")  # 5 + 5 >= 10, should process

        # Check processing was called
        self.assertEqual(len(self.processing_calls), 1)

        # Verify data was combined correctly
        call_args = self.processing_calls[0]
        self.assertTrue(torch.equal(call_args['data'], torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])))
        self.assertEqual(call_args['labels'], ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])

        # Storage should be cleared after processing
        self.assertEqual(len(buffer.storage), 0)

    def test_tensor_concatenation(self):
        """Test that tensors are correctly concatenated along dimension 0."""
        buffer = StreamingBuffer(
            buffer_size=6,
            processing_fn=self._create_mock_processing_fn()
        )

        batch1 = {'tensor': torch.tensor([[1, 2], [3, 4]])}  # 2 items
        batch2 = {'tensor': torch.tensor([[5, 6], [7, 8]])}  # 2 items
        batch3 = {'tensor': torch.tensor([[9, 10], [11, 12]])}  # 2 items

        buffer.add_batch(batch1)
        buffer.add_batch(batch2)
        buffer.add_batch(batch3)  # Total: 6, should process

        self.assertEqual(len(self.processing_calls), 1)

        expected_tensor = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
        self.assertTrue(torch.equal(self.processing_calls[0]['tensor'], expected_tensor))

    def test_list_concatenation(self):
        """Test that lists are correctly combined using itertools.chain."""
        buffer = StreamingBuffer(
            buffer_size=5,
            processing_fn=self._create_mock_processing_fn()
        )

        batch1 = {'ids': [0, 1], 'dummy_tensor': torch.zeros(2)}
        batch2 = {'ids': [2, 3, 4], 'dummy_tensor': torch.zeros(3)}

        buffer.add_batch(batch1)
        buffer.add_batch(batch2)  # Total: 5

        self.assertEqual(self.processing_calls[0]['ids'], [0, 1, 2, 3, 4])

    def test_mixed_types_combination(self):
        """Test combining batches with both tensors and lists."""
        buffer = StreamingBuffer(
            buffer_size=4,
            processing_fn=self._create_mock_processing_fn()
        )

        batch1 = {
            'images': torch.randn(2, 3, 64, 64),
            'ids': ['img1', 'img2'],
            'scores': torch.tensor([0.9, 0.8])
        }
        batch2 = {
            'images': torch.randn(2, 3, 64, 64),
            'ids': ['img3', 'img4'],
            'scores': torch.tensor([0.7, 0.6])
        }

        buffer.add_batch(batch1)
        buffer.add_batch(batch2)  # Total: 4

        call_args = self.processing_calls[0]

        # Check shapes
        self.assertEqual(call_args['images'].shape, (4, 3, 64, 64))
        self.assertEqual(call_args['scores'].shape, (4,))

        # Check list combination
        self.assertEqual(call_args['ids'], ['img1', 'img2', 'img3', 'img4'])

    def test_flush_processes_remaining_data(self):
        """Test that flush() processes data even if buffer not full."""
        buffer = StreamingBuffer(
            buffer_size=100,
            processing_fn=self._create_mock_processing_fn("flushed")
        )

        # Add small batch
        batch = {'data': torch.tensor([1, 2, 3]), 'labels': ['a', 'b', 'c']}
        buffer.add_batch(batch)

        # Should not process yet
        self.assertEqual(len(self.processing_calls), 0)

        # Flush should process
        result = buffer.flush()
        self.assertEqual(result, "flushed")
        self.assertEqual(len(self.processing_calls), 1)

        # Storage should be cleared
        self.assertEqual(len(buffer.storage), 0)

    def test_flush_empty_buffer_returns_none(self):
        """Test that flushing an empty buffer returns None."""
        buffer = StreamingBuffer(
            buffer_size=10,
            processing_fn=self._create_mock_processing_fn()
        )

        result = buffer.flush()
        self.assertIsNone(result)
        self.assertEqual(len(self.processing_calls), 0)

    def test_multiple_process_cycles(self):
        """Test that buffer can process multiple times in sequence."""
        buffer = StreamingBuffer(
            buffer_size=3,
            processing_fn=self._create_mock_processing_fn("cycle")
        )

        # First cycle
        buffer.add_batch({'x': torch.tensor([1, 2, 3]), 'ids': [1, 2, 3]})
        self.assertEqual(len(self.processing_calls), 1)

        # Buffer should be empty now
        self.assertEqual(len(buffer.storage), 0)

        # Second cycle
        buffer.add_batch({'x': torch.tensor([4]), 'ids': [4]})
        buffer.add_batch({'x': torch.tensor([5, 6]), 'ids': [5, 6]})
        self.assertEqual(len(self.processing_calls), 2)

        # Check second call
        self.assertTrue(torch.equal(self.processing_calls[1]['x'], torch.tensor([4, 5, 6])))

    def test_exact_threshold_triggers_processing(self):
        """Test that reaching exactly the threshold triggers processing."""
        buffer = StreamingBuffer(
            buffer_size=5,
            processing_fn=self._create_mock_processing_fn()
        )

        buffer.add_batch({'x': torch.tensor([1, 2]), 'ids': [1, 2]})
        buffer.add_batch({'x': torch.tensor([3, 4, 5]), 'ids': [3, 4, 5]})

        # Exactly 5 items, should process
        self.assertEqual(len(self.processing_calls), 1)

    def test_exceeding_threshold_triggers_processing(self):
        """Test that exceeding threshold triggers processing with all accumulated data."""
        buffer = StreamingBuffer(
            buffer_size=5,
            processing_fn=self._create_mock_processing_fn()
        )

        buffer.add_batch({'x': torch.tensor([1, 2, 3]), 'ids': [1, 2, 3]})
        buffer.add_batch({'x': torch.tensor([4, 5, 6]), 'ids': [4, 5, 6]})

        # 3 + 3 = 6 > 5, should process all 6
        self.assertEqual(len(self.processing_calls), 1)
        self.assertEqual(len(self.processing_calls[0]['ids']), 6)

    def test_empty_batch_handling(self):
        """Test handling of batches with zero items."""
        buffer = StreamingBuffer(
            buffer_size=5,
            processing_fn=self._create_mock_processing_fn()
        )

        # Add batch with 0 items
        buffer.add_batch({'x': torch.tensor([]), 'ids': []})

        # Should not process (0 < 5)
        self.assertIsNone(buffer.flush())
        self.assertEqual(len(self.processing_calls), 0)

    def test_processing_fn_receives_kwargs(self):
        """Test that processing function receives data as keyword arguments."""
        received_kwargs = {}

        def capture_kwargs(**kwargs):
            received_kwargs.update(kwargs)
            return "captured"

        buffer = StreamingBuffer(buffer_size=2, processing_fn=capture_kwargs)

        batch = {
            'rotmats': torch.randn(2, 3, 3),
            'maxprobs': torch.tensor([0.9, 0.8]),
            'ids': ['p1', 'p2']
        }
        buffer.add_batch(batch)

        # Verify all keys were passed as kwargs
        self.assertIn('rotmats', received_kwargs)
        self.assertIn('maxprobs', received_kwargs)
        self.assertIn('ids', received_kwargs)

    def test_buffer_size_one(self):
        """Test buffer with size=1 processes immediately."""
        buffer = StreamingBuffer(
            buffer_size=1,
            processing_fn=self._create_mock_processing_fn()
        )

        result = buffer.add_batch({'x': torch.tensor([1]), 'ids': [1]})

        # Should process immediately
        self.assertIsNotNone(result)
        self.assertEqual(len(self.processing_calls), 1)

    def test_large_buffer_accumulation(self):
        """Test buffer can accumulate many batches before processing."""
        buffer = StreamingBuffer(
            buffer_size=1000,
            processing_fn=self._create_mock_processing_fn()
        )

        # Add 100 batches of 5 items each
        for i in range(100):
            batch = {'x': torch.ones(5), 'ids': list(range(i*5, (i+1)*5))}
            buffer.add_batch(batch)

        # Should not process yet (500 < 1000)
        self.assertEqual(len(self.processing_calls), 0)
        self.assertEqual(len(buffer.storage), 100)

        # Add one more to exceed
        buffer.add_batch({'x': torch.ones(500), 'ids': list(range(500, 1000))})

        # Now should process (1000 >= 1000)
        self.assertEqual(len(self.processing_calls), 1)
        self.assertEqual(len(self.processing_calls[0]['ids']), 1000)

    def test_unsupported_type_raises_error(self):
        """Test that combining unsupported types raises NotImplementedError."""
        buffer = StreamingBuffer(
            buffer_size=3,
            processing_fn=self._create_mock_processing_fn()
        )

        # Add batch with unsupported type (numpy array, not torch tensor)
        import numpy as np
        batch1 = {'data': np.array([1])}  # 1 item, won't trigger processing
        batch2 = {'data': np.array([2, 3])}  # 2 more items, total=3, triggers processing

        buffer.add_batch(batch1)  # Should succeed (1 < 3)

        # Adding second batch should trigger processing and raise error
        with self.assertRaises(NotImplementedError):
            buffer.add_batch(batch2)

    def test_realistic_inference_scenario(self):
        """
        Test a realistic scenario mimicking the inference pipeline:
        - Neural network produces batches of varying sizes (after z-score filtering)
        - Buffer accumulates until threshold
        - Expensive local refinement called only once with large batch
        """
        local_refinement_calls = 0

        def expensive_local_refinement(imgs, ctfs, rotmats, maxprobs, norm_nn_score, ids):
            nonlocal local_refinement_calls
            local_refinement_calls += 1
            # Simulate expensive operation
            return (ids, rotmats, torch.zeros_like(maxprobs), maxprobs, norm_nn_score)

        buffer = StreamingBuffer(
            buffer_size=64,
            processing_fn=expensive_local_refinement
        )

        # Simulate 10 batches from neural network, each with ~5-10 particles passing filter
        torch.manual_seed(42)
        total_particles = 0
        batch_sizes = [5, 7, 3, 8, 6, 9, 4, 10, 7, 5]

        for i, n in enumerate(batch_sizes):
            batch = {
                'imgs': torch.randn(n, 3, 64, 64),
                'ctfs': torch.randn(n, 64, 64),
                'rotmats': torch.randn(n, 3, 3),
                'maxprobs': torch.rand(n),
                'norm_nn_score': torch.rand(n),
                'ids': [f'particle_{total_particles + j}' for j in range(n)]
            }
            buffer.add_batch(batch)
            total_particles += n

        # Buffer should have processed once when crossing 64 threshold
        # batch_sizes cumsum: [5, 12, 15, 23, 29, 38, 42, 52, 59, 64]
        # Should process at batch 10 (total=64)
        self.assertEqual(local_refinement_calls, 1)

        # Flush remaining
        buffer.flush()

        # No remaining particles (exactly 64)
        self.assertEqual(local_refinement_calls, 1)

    def test_multidimensional_tensor_concatenation(self):
        """Test that multi-dimensional tensors are concatenated correctly along dim 0."""
        buffer = StreamingBuffer(
            buffer_size=3,
            processing_fn=self._create_mock_processing_fn()
        )

        # 3D tensors (batch, height, width)
        batch1 = {'img': torch.ones(1, 10, 10), 'ids': [1]}
        batch2 = {'img': torch.ones(2, 10, 10) * 2, 'ids': [2, 3]}

        buffer.add_batch(batch1)
        buffer.add_batch(batch2)

        result_img = self.processing_calls[0]['img']

        # Should have shape (3, 10, 10)
        self.assertEqual(result_img.shape, (3, 10, 10))

        # Check values
        self.assertTrue(torch.equal(result_img[0], torch.ones(10, 10)))
        self.assertTrue(torch.equal(result_img[1], torch.ones(10, 10) * 2))
        self.assertTrue(torch.equal(result_img[2], torch.ones(10, 10) * 2))


if __name__ == '__main__':
    unittest.main()
