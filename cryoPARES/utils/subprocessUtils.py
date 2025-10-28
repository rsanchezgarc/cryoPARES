"""
Subprocess utilities with enhanced error reporting.

Provides a drop-in replacement for subprocess.run() that automatically prints
helpful error summaries when subprocesses fail, while maintaining all original output.
"""

import os
import pty
import select
import subprocess
import sys
from typing import Union, List, Optional, Any

from cryoPARES.utils.errorFormatting import extract_traceback_from_text, print_error_summary


class SubprocessErrorWithSummary(subprocess.CalledProcessError):
    """
    Enhanced CalledProcessError that includes parsed traceback information.

    This exception is raised when a subprocess fails and we've successfully
    parsed error information from its output.
    """

    def __init__(self, original_error, traceback_info=None, description=None):
        """
        Initialize with original error and optional parsed info.

        Args:
            original_error: The original CalledProcessError
            traceback_info: Tuple of (exception_type, exception_message, traceback_lines)
            description: Optional description of what was being executed
        """
        super().__init__(
            original_error.returncode,
            original_error.cmd,
            original_error.output,
            original_error.stderr
        )
        self.traceback_info = traceback_info
        self.description = description


def run_subprocess_with_error_summary(
    cmd: Union[str, List[str]],
    cwd: Optional[str] = None,
    check: bool = True,
    description: Optional[str] = None,
    shell: bool = False,
    capture_output: bool = False,
    text: bool = True,
    exit_on_error: bool = True,
    **kwargs
) -> subprocess.CompletedProcess:
    """
    Run a subprocess with enhanced error reporting.

    This is a drop-in replacement for subprocess.run() that:
    1. Runs the command normally, streaming output in real-time
    2. Captures all output for error analysis
    3. On failure, parses the output for Python tracebacks
    4. Prints a clean error summary at the end
    5. Either exits cleanly (exit_on_error=True) or raises exception

    Args:
        cmd: Command to execute (string or list)
        cwd: Working directory
        check: Whether to check for non-zero exit code
        description: Human-readable description of what's being executed
        shell: Whether to execute through shell
        capture_output: If True, capture stdout/stderr instead of streaming
        text: If True, decode output as text
        exit_on_error: If True, call sys.exit() after error summary instead of
                      raising exception. This prevents the parent process traceback
                      from printing after the error summary, keeping output clean.
        **kwargs: Additional arguments passed to subprocess.Popen

    Returns:
        CompletedProcess object (only if successful)

    Raises:
        SubprocessErrorWithSummary: If subprocess fails, check=True, and exit_on_error=False

    Example:
        >>> run_subprocess_with_error_summary(
        ...     ['python', '-m', 'cryoPARES.train.train', '--help'],
        ...     description="Training model"
        ... )
    """
    # If capture_output is requested, use standard subprocess.run with error handling
    if capture_output:
        return _run_with_capture(cmd, cwd, check, description, shell, text, exit_on_error, **kwargs)

    # Otherwise, stream output in real-time while capturing for analysis
    return _run_with_streaming(cmd, cwd, check, description, shell, text, exit_on_error, **kwargs)


def _run_with_streaming(
    cmd: Union[str, List[str]],
    cwd: Optional[str],
    check: bool,
    description: Optional[str],
    shell: bool,
    text: bool,
    exit_on_error: bool,
    **kwargs
) -> subprocess.CompletedProcess:
    """
    Run subprocess with real-time output streaming using a pseudo-TTY.

    IMPORTANT DESIGN DECISIONS:

    1. Progress Bar Preservation:
       - Uses a pseudo-TTY (pty) so subprocess thinks stdout is a terminal
       - This makes progress bars with \r work correctly (they update in place)
       - Without pty, subprocesses detect pipes and disable \r-based updates

    2. Real-time Streaming:
       - Reads from master side of pty in small chunks
       - Writes immediately to stdout and flushes
       - Fully responsive to subprocess output

    3. Error Handling:
       - Captures all output for error analysis
       - Uses sys.exit() instead of raising exception to prevent parent traceback
       - Keeps error summary clean at the end

    Args:
        exit_on_error: If True, call sys.exit() on error instead of raising exception
    """
    # Prepare command
    if isinstance(cmd, str) and not shell:
        import shlex
        cmd_list = shlex.split(cmd)
    else:
        cmd_list = cmd if isinstance(cmd, list) else [cmd]

    # Create a pseudo-TTY for subprocess
    # This makes the subprocess think it's connected to a terminal
    master_fd, slave_fd = pty.openpty()

    try:
        # Start process with pty slave as stdout/stderr
        process = subprocess.Popen(
            cmd_list if not shell else cmd,
            cwd=cwd,
            stdout=slave_fd,
            stderr=slave_fd,
            stdin=subprocess.DEVNULL,
            shell=shell,
            **kwargs
        )

        # Close slave fd in parent process (subprocess has its own copy)
        os.close(slave_fd)

        # Read from master side of pty and stream to stdout
        output_chunks = []

        # Set master fd to non-blocking mode
        import fcntl
        flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
        fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

        while True:
            # Use select to wait for data (with timeout)
            readable, _, _ = select.select([master_fd], [], [], 0.1)

            if readable:
                try:
                    # Read available data
                    chunk = os.read(master_fd, 4096)
                    if not chunk:
                        break

                    # Decode if text mode
                    if text:
                        chunk = chunk.decode('utf-8', errors='replace')

                    # Write to stdout and flush immediately
                    sys.stdout.write(chunk)
                    sys.stdout.flush()

                    # Capture for error analysis
                    output_chunks.append(chunk)

                except OSError:
                    # End of file or error
                    break

            # Check if process has finished
            if process.poll() is not None:
                # Read any remaining data
                try:
                    while True:
                        chunk = os.read(master_fd, 4096)
                        if not chunk:
                            break
                        if text:
                            chunk = chunk.decode('utf-8', errors='replace')
                        sys.stdout.write(chunk)
                        sys.stdout.flush()
                        output_chunks.append(chunk)
                except OSError:
                    pass
                break

        # Final flush
        sys.stdout.flush()

        # Get return code
        returncode = process.wait()

    finally:
        # Always close master fd
        try:
            os.close(master_fd)
        except OSError:
            pass

    # Combine captured output
    full_output = ''.join(output_chunks) if text else b''.join(output_chunks)

    # Create CompletedProcess object
    result = subprocess.CompletedProcess(
        cmd_list if not shell else cmd,
        returncode,
        full_output,
        None  # stderr (we merged it into stdout)
    )

    # Handle non-zero exit code
    if check and returncode != 0:
        _handle_subprocess_error(result, description, full_output, exit_on_error)

    return result


def _run_with_capture(
    cmd: Union[str, List[str]],
    cwd: Optional[str],
    check: bool,
    description: Optional[str],
    shell: bool,
    text: bool,
    exit_on_error: bool,
    **kwargs
) -> subprocess.CompletedProcess:
    """
    Run subprocess with output capture (no real-time streaming).

    This is used when capture_output=True, and provides a fallback
    for compatibility with standard subprocess.run() behavior.
    """
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=check,
            capture_output=True,
            text=text,
            shell=shell,
            **kwargs
        )
        return result
    except subprocess.CalledProcessError as e:
        # Get combined output
        full_output = (e.output or '') + (e.stderr or '')
        _handle_subprocess_error(e, description, full_output, exit_on_error)
        raise  # Should not reach here if exit_on_error=True


def _handle_subprocess_error(
    error: Union[subprocess.CalledProcessError, subprocess.CompletedProcess],
    description: Optional[str],
    full_output: str,
    exit_on_error: bool = True
):
    """
    Handle subprocess error by printing summary and either exiting or raising.

    IMPORTANT: By default, this calls sys.exit() instead of raising an exception.
    This prevents the parent process from printing its own traceback after the
    error summary, which would bury the useful information.

    Args:
        error: CalledProcessError or CompletedProcess with non-zero returncode
        description: Optional description of what failed
        full_output: Captured output from the subprocess
        exit_on_error: If True (default), call sys.exit() to prevent parent traceback.
                      If False, raise SubprocessErrorWithSummary exception.
    """
    # Try to extract Python traceback from output
    traceback_info = None
    if full_output:
        traceback_info = extract_traceback_from_text(full_output)

    # Print error summary
    _print_subprocess_error_summary(error, traceback_info, description, full_output)

    # Get the return code
    returncode = error.returncode if hasattr(error, 'returncode') else 1

    if exit_on_error:
        # Exit cleanly without printing parent traceback
        # This keeps the error summary as the last thing the user sees
        sys.exit(returncode)
    else:
        # Raise enhanced exception (for tests or when exception handling is needed)
        if isinstance(error, subprocess.CalledProcessError):
            raise SubprocessErrorWithSummary(error, traceback_info, description)
        else:
            # Create CalledProcessError from CompletedProcess
            exc = subprocess.CalledProcessError(
                returncode,
                error.args,
                output=error.stdout,
                stderr=error.stderr
            )
            raise SubprocessErrorWithSummary(exc, traceback_info, description)


def _print_subprocess_error_summary(
    error: Union[subprocess.CalledProcessError, subprocess.CompletedProcess],
    traceback_info: Optional[tuple],
    description: Optional[str],
    full_output: str
):
    """
    Print a formatted error summary for a failed subprocess.

    Args:
        error: The error or failed process
        traceback_info: Parsed traceback info if available
        description: Optional description
        full_output: Full captured output
    """
    box_width = 80
    sep_line = "=" * box_width
    sub_sep_line = "-" * box_width

    print("\n" + sep_line, file=sys.stderr)
    print("SUBPROCESS FAILED", file=sys.stderr)
    print(sep_line, file=sys.stderr)
    print("", file=sys.stderr)

    # Description
    if description:
        print(f"Description: {description}", file=sys.stderr)
        print("", file=sys.stderr)

    # Command
    cmd = error.cmd if hasattr(error, 'cmd') else error.args
    if isinstance(cmd, list):
        # Format command nicely with line continuation for long commands
        cmd_str = " ".join(cmd)
        if len(cmd_str) > box_width - 4:
            # Split long commands
            print("Command:", file=sys.stderr)
            current_line = "  "
            for part in cmd:
                if len(current_line) + len(part) + 3 > box_width:
                    print(current_line + " \\", file=sys.stderr)
                    current_line = "    " + part
                else:
                    if current_line == "  ":
                        current_line += part
                    else:
                        current_line += " " + part
            print(current_line, file=sys.stderr)
        else:
            print(f"Command: {cmd_str}", file=sys.stderr)
    else:
        print(f"Command: {cmd}", file=sys.stderr)

    print("", file=sys.stderr)

    # Exit code
    returncode = error.returncode if hasattr(error, 'returncode') else error.returncode
    print(f"Exit code: {returncode}", file=sys.stderr)
    print("", file=sys.stderr)

    # Root cause (if we found a Python traceback)
    if traceback_info:
        exception_type, exception_message, traceback_lines = traceback_info

        print(sub_sep_line, file=sys.stderr)
        print("ROOT CAUSE:", file=sys.stderr)
        print(sub_sep_line, file=sys.stderr)
        print("", file=sys.stderr)
        print(f"{exception_type}: {exception_message}", file=sys.stderr)
        print("", file=sys.stderr)

        # Show all traceback frames (no filtering - user needs to see full context)
        print("Traceback:", file=sys.stderr)
        for line in traceback_lines:
            # Print all traceback lines with proper indentation
            print("  " + line.rstrip(), file=sys.stderr)

        print("", file=sys.stderr)

    # Footer
    print(sub_sep_line, file=sys.stderr)
    if traceback_info:
        print("For full output, see above.", file=sys.stderr)
    else:
        print("No Python traceback found. The error may be from bash or another tool.", file=sys.stderr)
        print("Check the full output above for details.", file=sys.stderr)
    print(sep_line, file=sys.stderr)
    print("", file=sys.stderr, flush=True)
