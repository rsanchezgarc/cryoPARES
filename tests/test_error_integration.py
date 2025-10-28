"""
Integration tests for error formatting in real scenarios.

This creates test scripts that fail in various ways and verifies
that error summaries are printed correctly.
"""

import os
import sys
import tempfile
import unittest

from cryoPARES.utils.subprocessUtils import run_subprocess_with_error_summary, SubprocessErrorWithSummary


class TestErrorIntegration(unittest.TestCase):
    """Integration tests for error reporting."""

    def test_subprocess_python_error_shows_summary(self):
        """Test that a Python error in subprocess shows error summary."""
        # Create a test script that fails
        script = """
import sys

def process_data():
    raise ValueError("Invalid data format: expected shape (128, 128), got (64, 64)")

def main():
    process_data()

if __name__ == '__main__':
    main()
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script)
            script_path = f.name

        try:
            # Run the script and expect error summary
            with self.assertRaises(SubprocessErrorWithSummary) as cm:
                run_subprocess_with_error_summary(
                    [sys.executable, script_path],
                    description="Test data processing",
                    capture_output=True,
                    exit_on_error=False  # Allow tests to catch exceptions
                )

            exc = cm.exception
            self.assertIsNotNone(exc.traceback_info)
            exception_type, exception_message, _ = exc.traceback_info
            self.assertEqual(exception_type, "ValueError")
            self.assertIn("Invalid data format", exception_message)

        finally:
            os.unlink(script_path)

    def test_subprocess_nested_error_shows_relevant_frames(self):
        """Test that nested errors show relevant stack frames."""
        script = """
import sys

def inner_function():
    raise RuntimeError("Deep error in processing")

def middle_function():
    inner_function()

def outer_function():
    middle_function()

def main():
    try:
        outer_function()
    except RuntimeError as e:
        # Re-raise to test error propagation
        raise ValueError("Processing pipeline failed") from e

if __name__ == '__main__':
    main()
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script)
            script_path = f.name

        try:
            with self.assertRaises(SubprocessErrorWithSummary) as cm:
                run_subprocess_with_error_summary(
                    [sys.executable, script_path],
                    description="Test nested errors",
                    capture_output=True,
                    exit_on_error=False  # Allow tests to catch exceptions
                )

            exc = cm.exception
            self.assertIsNotNone(exc.traceback_info)
            # Should capture the first traceback (which is the root cause)
            exception_type, _, _ = exc.traceback_info
            # The parser finds the first/root exception (RuntimeError)
            self.assertEqual(exception_type, "RuntimeError")

        finally:
            os.unlink(script_path)

    def test_worker_error_format(self):
        """Test error formatting in worker context (manual test)."""
        # This test demonstrates the worker error format
        # In real usage, this would be called from a multiprocessing worker

        from cryoPARES.utils.errorFormatting import print_error_summary
        from io import StringIO

        try:
            def worker_task():
                raise ValueError("Worker failed to process batch")

            worker_task()
        except Exception as e:
            # Capture output
            captured = StringIO()
            print_error_summary(
                e,
                description="Processing batch 42",
                worker_id=5,
                file=captured
            )

            output = captured.getvalue()

            # Verify format
            self.assertIn("WORKER ERROR SUMMARY (Worker 5)", output)
            self.assertIn("Description: Processing batch 42", output)
            self.assertIn("ValueError: Worker failed to process batch", output)
            self.assertIn("ROOT CAUSE:", output)

    def test_subprocess_without_python_error(self):
        """Test subprocess that fails without Python traceback."""
        with self.assertRaises(SubprocessErrorWithSummary) as cm:
            run_subprocess_with_error_summary(
                ["false"],  # Command that simply exits with code 1
                description="Test non-Python error",
                capture_output=True,
                exit_on_error=False  # Allow tests to catch exceptions
            )

        exc = cm.exception
        self.assertEqual(exc.returncode, 1)
        # No Python traceback should be found
        self.assertIsNone(exc.traceback_info)

    def test_successful_subprocess_no_error(self):
        """Test that successful subprocess doesn't trigger error handling."""
        result = run_subprocess_with_error_summary(
            [sys.executable, "-c", "print('Success!')"],
            description="Test successful operation",
            capture_output=True
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("Success!", result.stdout)


if __name__ == '__main__':
    unittest.main()
