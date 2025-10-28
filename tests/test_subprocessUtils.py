"""
Unit tests for subprocess utilities.
"""

import os
import sys
import tempfile
import unittest

from cryoPARES.utils.subprocessUtils import (
    run_subprocess_with_error_summary,
    SubprocessErrorWithSummary
)


class TestRunSubprocessWithErrorSummary(unittest.TestCase):
    """Test the subprocess wrapper with error summary."""

    def test_successful_command(self):
        """Test that successful commands work normally."""
        result = run_subprocess_with_error_summary(
            [sys.executable, "-c", "print('Hello, World!')"],
            capture_output=True,
            description="Test successful command"
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("Hello, World!", result.stdout)

    def test_successful_command_with_streaming(self):
        """Test successful command with real-time streaming."""
        result = run_subprocess_with_error_summary(
            [sys.executable, "-c", "print('Streaming output')"],
            capture_output=False,
            description="Test streaming"
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("Streaming output", result.stdout)

    def test_command_with_python_error(self):
        """Test command that fails with Python exception."""
        with self.assertRaises(SubprocessErrorWithSummary) as cm:
            run_subprocess_with_error_summary(
                [sys.executable, "-c", "raise ValueError('Test error message')"],
                capture_output=True,
                description="Test Python error",
                exit_on_error=False  # Allow tests to catch exceptions
            )

        exc = cm.exception
        self.assertIsNotNone(exc.traceback_info)
        exception_type, exception_message, _ = exc.traceback_info
        self.assertEqual(exception_type, "ValueError")
        self.assertIn("Test error message", exception_message)

    def test_command_with_non_python_error(self):
        """Test command that fails without Python traceback."""
        with self.assertRaises(SubprocessErrorWithSummary) as cm:
            run_subprocess_with_error_summary(
                [sys.executable, "-c", "import sys; sys.exit(42)"],
                capture_output=True,
                description="Test non-Python error",
                exit_on_error=False  # Allow tests to catch exceptions
            )

        exc = cm.exception
        self.assertEqual(exc.returncode, 42)
        # Should be None since there's no Python traceback
        self.assertIsNone(exc.traceback_info)

    def test_command_without_check(self):
        """Test that check=False doesn't raise exception."""
        result = run_subprocess_with_error_summary(
            [sys.executable, "-c", "import sys; sys.exit(1)"],
            check=False,
            capture_output=True
        )

        self.assertEqual(result.returncode, 1)
        # Should not raise

    def test_command_with_working_directory(self):
        """Test command execution with specific working directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_subprocess_with_error_summary(
                [sys.executable, "-c", "import os; print(os.getcwd())"],
                cwd=tmpdir,
                capture_output=True
            )

            self.assertEqual(result.returncode, 0)
            self.assertIn(tmpdir, result.stdout)

    def test_error_summary_includes_description(self):
        """Test that error summary includes the description."""
        # We can't easily capture stderr in the test, but we can verify
        # the exception is raised with the right attributes
        with self.assertRaises(SubprocessErrorWithSummary) as cm:
            run_subprocess_with_error_summary(
                [sys.executable, "-c", "raise RuntimeError('fail')"],
                capture_output=True,
                description="Custom description for test",
                exit_on_error=False  # Allow tests to catch exceptions
            )

        exc = cm.exception
        self.assertEqual(exc.description, "Custom description for test")

    def test_complex_python_traceback(self):
        """Test parsing of multi-frame Python traceback."""
        # Create a script with nested function calls
        script = """
def inner():
    raise ValueError('Inner error')

def middle():
    inner()

def outer():
    middle()

outer()
"""

        with self.assertRaises(SubprocessErrorWithSummary) as cm:
            run_subprocess_with_error_summary(
                [sys.executable, "-c", script],
                capture_output=True,
                description="Test complex traceback",
                exit_on_error=False  # Allow tests to catch exceptions
            )

        exc = cm.exception
        self.assertIsNotNone(exc.traceback_info)
        exception_type, exception_message, traceback_lines = exc.traceback_info
        self.assertEqual(exception_type, "ValueError")
        self.assertIn("Inner error", exception_message)
        # Should have multiple frames
        self.assertGreater(len(traceback_lines), 1)

    def test_command_as_string_with_shell(self):
        """Test passing command as string with shell=True."""
        result = run_subprocess_with_error_summary(
            "echo 'Shell command'",
            shell=True,
            capture_output=True
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("Shell command", result.stdout)

    def test_importerror_in_subprocess(self):
        """Test handling of ImportError in subprocess."""
        with self.assertRaises(SubprocessErrorWithSummary) as cm:
            run_subprocess_with_error_summary(
                [sys.executable, "-c", "import nonexistent_module"],
                capture_output=True,
                description="Test ImportError",
                exit_on_error=False  # Allow tests to catch exceptions
            )

        exc = cm.exception
        self.assertIsNotNone(exc.traceback_info)
        exception_type, _, _ = exc.traceback_info
        self.assertEqual(exception_type, "ModuleNotFoundError")


class TestSubprocessErrorWithSummary(unittest.TestCase):
    """Test the SubprocessErrorWithSummary exception class."""

    def test_exception_attributes(self):
        """Test that exception preserves all attributes."""
        import subprocess

        # Create a fake CalledProcessError
        original = subprocess.CalledProcessError(
            returncode=1,
            cmd=['python', 'test.py'],
            output="Some output",
            stderr="Some error"
        )

        traceback_info = ("ValueError", "Test message", ["line 1", "line 2"])
        description = "Test description"

        enhanced = SubprocessErrorWithSummary(original, traceback_info, description)

        self.assertEqual(enhanced.returncode, 1)
        self.assertEqual(enhanced.cmd, ['python', 'test.py'])
        self.assertEqual(enhanced.output, "Some output")
        self.assertEqual(enhanced.stderr, "Some error")
        self.assertEqual(enhanced.traceback_info, traceback_info)
        self.assertEqual(enhanced.description, description)


if __name__ == '__main__':
    unittest.main()
