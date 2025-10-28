"""
Unit tests for error formatting utilities.
"""

import sys
import unittest
from io import StringIO

from cryoPARES.utils.errorFormatting import (
    extract_relevant_frames,
    extract_exception_info,
    format_exception_summary,
    print_error_summary,
    extract_traceback_from_text,
    ExceptionInfo,
    FrameInfo
)


class TestExtractRelevantFrames(unittest.TestCase):
    """Test frame extraction from tracebacks."""

    def test_extract_frames_from_simple_exception(self):
        """Test extracting frames from a simple exception."""
        try:
            # Create a stack trace by calling nested functions
            def inner():
                raise ValueError("Test error")

            def outer():
                inner()

            outer()
        except ValueError as e:
            frames = extract_relevant_frames(e.__traceback__, filter_to_cryopares=False)
            self.assertGreater(len(frames), 0)
            # Check that we got the inner function
            self.assertTrue(any(f.function == 'inner' for f in frames))

    def test_filter_to_cryopares(self):
        """Test filtering frames to only cryoPARES code."""
        try:
            # This error should have test file in the trace
            raise RuntimeError("Test error")
        except RuntimeError as e:
            # Without filter
            frames_unfiltered = extract_relevant_frames(e.__traceback__, filter_to_cryopares=False)
            # With filter
            frames_filtered = extract_relevant_frames(e.__traceback__, filter_to_cryopares=True)

            # Should have at least one frame (the last one is always kept)
            self.assertGreater(len(frames_filtered), 0)

    def test_max_frames_limit(self):
        """Test that max_frames limit is respected."""
        try:
            # Create deep call stack
            def recursive_call(depth):
                if depth <= 0:
                    raise ValueError("Deep error")
                return recursive_call(depth - 1)

            recursive_call(20)
        except ValueError as e:
            frames = extract_relevant_frames(e.__traceback__, max_frames=5, filter_to_cryopares=False)
            self.assertLessEqual(len(frames), 5)


class TestExtractExceptionInfo(unittest.TestCase):
    """Test exception information extraction."""

    def test_extract_basic_exception_info(self):
        """Test extracting information from a basic exception."""
        try:
            raise ValueError("Test message")
        except ValueError as e:
            info = extract_exception_info(e, filter_to_cryopares=False)

            self.assertEqual(info.exception_type, "ValueError")
            self.assertEqual(info.exception_message, "Test message")
            self.assertIn("ValueError: Test message", info.full_traceback_text)
            self.assertGreater(len(info.relevant_frames), 0)

    def test_extract_chained_exception_info(self):
        """Test extracting information from chained exceptions."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise RuntimeError("Wrapped error") from e
        except RuntimeError as e:
            info = extract_exception_info(e, filter_to_cryopares=False)

            self.assertEqual(info.exception_type, "RuntimeError")
            self.assertEqual(info.exception_message, "Wrapped error")


class TestFormatExceptionSummary(unittest.TestCase):
    """Test exception summary formatting."""

    def test_format_basic_summary(self):
        """Test formatting a basic exception summary."""
        exc_info = ExceptionInfo(
            exception_type="ValueError",
            exception_message="Invalid data format",
            relevant_frames=[
                FrameInfo(
                    filename="/path/to/cryoPARES/module.py",
                    lineno=100,
                    function="process_data",
                    code_line="result = compute(data)"
                )
            ],
            full_traceback_text="Full traceback here"
        )

        summary = format_exception_summary(exc_info)

        self.assertIn("ERROR SUMMARY", summary)
        self.assertIn("ROOT CAUSE:", summary)
        self.assertIn("ValueError: Invalid data format", summary)
        self.assertIn("Relevant traceback:", summary)
        self.assertIn("line 100", summary)
        self.assertIn("process_data", summary)

    def test_format_summary_with_worker_id(self):
        """Test formatting summary with worker ID."""
        exc_info = ExceptionInfo(
            exception_type="RuntimeError",
            exception_message="Worker failed",
            relevant_frames=[],
            full_traceback_text=""
        )

        summary = format_exception_summary(exc_info, worker_id=5)

        self.assertIn("WORKER ERROR SUMMARY (Worker 5)", summary)

    def test_format_summary_with_description(self):
        """Test formatting summary with description."""
        exc_info = ExceptionInfo(
            exception_type="IOError",
            exception_message="File not found",
            relevant_frames=[],
            full_traceback_text=""
        )

        summary = format_exception_summary(exc_info, description="Loading particles")

        self.assertIn("Description: Loading particles", summary)


class TestPrintErrorSummary(unittest.TestCase):
    """Test error summary printing."""

    def test_print_error_summary(self):
        """Test that print_error_summary writes to stderr."""
        try:
            raise ValueError("Test error for printing")
        except ValueError as e:
            # Capture stderr
            captured_output = StringIO()
            print_error_summary(e, description="Test operation", file=captured_output)

            output = captured_output.getvalue()

            self.assertIn("ERROR SUMMARY", output)
            self.assertIn("ValueError: Test error for printing", output)
            self.assertIn("Description: Test operation", output)


class TestExtractTracebackFromText(unittest.TestCase):
    """Test extracting traceback from text output."""

    def test_extract_simple_traceback(self):
        """Test extracting a simple traceback from text."""
        text = """
Some output here
Traceback (most recent call last):
  File "test.py", line 10, in <module>
    raise ValueError("Test error")
ValueError: Test error
More output here
"""

        result = extract_traceback_from_text(text)

        self.assertIsNotNone(result)
        exception_type, exception_message, traceback_lines = result
        self.assertEqual(exception_type, "ValueError")
        self.assertEqual(exception_message, "Test error")
        self.assertGreater(len(traceback_lines), 0)

    def test_extract_no_traceback(self):
        """Test text with no traceback returns None."""
        text = "Just some normal output\nNo errors here\n"

        result = extract_traceback_from_text(text)

        self.assertIsNone(result)

    def test_extract_complex_traceback(self):
        """Test extracting a multi-frame traceback."""
        text = """
Running command...
Traceback (most recent call last):
  File "cryoPARES/train/train.py", line 100, in run
    self.execute()
  File "cryoPARES/train/train.py", line 200, in execute
    process_data()
  File "cryoPARES/datamanager/dataset.py", line 50, in process_data
    raise RuntimeError("Processing failed")
RuntimeError: Processing failed
Command failed with exit code 1
"""

        result = extract_traceback_from_text(text)

        self.assertIsNotNone(result)
        exception_type, exception_message, traceback_lines = result
        self.assertEqual(exception_type, "RuntimeError")
        self.assertEqual(exception_message, "Processing failed")
        # Should have multiple frames
        self.assertGreater(len(traceback_lines), 3)


if __name__ == '__main__':
    unittest.main()
