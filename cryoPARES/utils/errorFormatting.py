"""
Error formatting utilities for better error reporting in subprocess and worker failures.

This module provides functions to extract and format error information from exceptions,
making it easier to identify the root cause of failures in complex multi-process applications.
"""

import sys
import traceback
import types
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class FrameInfo:
    """Information about a single stack frame."""
    filename: str
    lineno: int
    function: str
    code_line: Optional[str] = None


@dataclass
class ExceptionInfo:
    """Structured information extracted from an exception."""
    exception_type: str
    exception_message: str
    relevant_frames: List[FrameInfo]
    full_traceback_text: str


def extract_relevant_frames(
    tb: Optional[types.TracebackType],
    max_frames: int = 10,
    filter_to_cryopares: bool = True
) -> List[FrameInfo]:
    """
    Extract relevant stack frames from a traceback object.

    Args:
        tb: Traceback object from exception
        max_frames: Maximum number of frames to include
        filter_to_cryopares: If True, only show frames from cryoPARES code

    Returns:
        List of FrameInfo objects with relevant frames
    """
    if tb is None:
        return []

    frames = []
    current_tb = tb

    # Walk the traceback chain
    while current_tb is not None:
        frame = current_tb.tb_frame
        filename = frame.f_code.co_filename
        lineno = current_tb.tb_lineno
        function = frame.f_code.co_name

        # Read the code line if possible
        code_line = None
        try:
            import linecache
            code_line = linecache.getline(filename, lineno).strip()
        except Exception:
            pass

        frames.append(FrameInfo(
            filename=filename,
            lineno=lineno,
            function=function,
            code_line=code_line
        ))

        current_tb = current_tb.tb_next

    # Filter frames if requested
    if filter_to_cryopares:
        cryopares_frames = [f for f in frames if 'cryoPARES' in f.filename]
        # Always keep the last frame even if not from cryoPARES (shows where error occurred)
        if cryopares_frames:
            frames = cryopares_frames
        elif frames:
            # No cryoPARES frames, keep last frame at minimum
            frames = [frames[-1]]

    # Limit number of frames
    if len(frames) > max_frames:
        frames = frames[-max_frames:]

    return frames


def extract_exception_info(
    exc: BaseException,
    max_frames: int = 10,
    filter_to_cryopares: bool = True
) -> ExceptionInfo:
    """
    Extract structured information from an exception.

    Args:
        exc: The exception object
        max_frames: Maximum number of frames to include
        filter_to_cryopares: If True, filter to show only cryoPARES code

    Returns:
        ExceptionInfo object with structured error data
    """
    exception_type = type(exc).__name__
    exception_message = str(exc)

    # Get full traceback as text
    full_traceback_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    full_traceback_text = ''.join(full_traceback_lines)

    # Extract relevant frames
    relevant_frames = extract_relevant_frames(
        exc.__traceback__,
        max_frames=max_frames,
        filter_to_cryopares=filter_to_cryopares
    )

    return ExceptionInfo(
        exception_type=exception_type,
        exception_message=exception_message,
        relevant_frames=relevant_frames,
        full_traceback_text=full_traceback_text
    )


def format_exception_summary(
    exc_info: ExceptionInfo,
    description: Optional[str] = None,
    worker_id: Optional[int] = None,
    box_width: int = 80
) -> str:
    """
    Format exception information into a clean summary box.

    Args:
        exc_info: ExceptionInfo object with error details
        description: Optional description of what was being done
        worker_id: Optional worker ID for multiprocessing errors
        box_width: Width of the summary box

    Returns:
        Formatted error summary as a string
    """
    lines = []
    sep_line = "=" * box_width
    sub_sep_line = "-" * box_width

    # Header
    lines.append(sep_line)
    if worker_id is not None:
        lines.append(f"WORKER ERROR SUMMARY (Worker {worker_id})")
    else:
        lines.append("ERROR SUMMARY")
    lines.append(sep_line)
    lines.append("")

    # Description
    if description:
        lines.append(f"Description: {description}")
        lines.append("")

    # Root cause
    lines.append(sub_sep_line)
    lines.append("ROOT CAUSE:")
    lines.append(sub_sep_line)
    lines.append("")
    lines.append(f"{exc_info.exception_type}: {exc_info.exception_message}")
    lines.append("")

    # Relevant traceback
    if exc_info.relevant_frames:
        lines.append("Relevant traceback:")
        for frame in exc_info.relevant_frames:
            # Shorten long paths for readability
            filename = frame.filename
            if 'cryoPARES' in filename:
                # Show path relative to cryoPARES
                parts = filename.split('cryoPARES')
                if len(parts) > 1:
                    filename = 'cryoPARES' + parts[-1]

            lines.append(f'  File "{filename}", line {frame.lineno}, in {frame.function}')
            if frame.code_line:
                lines.append(f"    {frame.code_line}")
        lines.append("")

    # Footer
    lines.append(sub_sep_line)
    lines.append("For full traceback, see above.")
    lines.append(sep_line)

    return '\n'.join(lines)


def print_error_summary(
    exc: BaseException,
    description: Optional[str] = None,
    worker_id: Optional[int] = None,
    max_frames: int = 10,
    filter_to_cryopares: bool = True,
    box_width: int = 80,
    file=None
):
    """
    Print a formatted error summary for an exception.

    Args:
        exc: The exception object
        description: Optional description of what was being done
        worker_id: Optional worker ID for multiprocessing errors
        max_frames: Maximum number of frames to show
        filter_to_cryopares: If True, filter to show only cryoPARES code
        box_width: Width of the summary box
        file: File object to print to (default: stderr)
    """
    if file is None:
        file = sys.stderr

    # Extract exception information
    exc_info = extract_exception_info(
        exc,
        max_frames=max_frames,
        filter_to_cryopares=filter_to_cryopares
    )

    # Format and print summary
    summary = format_exception_summary(
        exc_info,
        description=description,
        worker_id=worker_id,
        box_width=box_width
    )

    print("\n" + summary, file=file, flush=True)


def extract_traceback_from_text(text: str) -> Optional[Tuple[str, str, List[str]]]:
    """
    Extract Python traceback information from text output.

    This is useful for parsing subprocess output to find error information.

    IMPORTANT: Looks for the LAST traceback in the output, as this is usually
    the root cause. For example, PyTorch DataLoader errors have a wrapper
    traceback followed by "Original Traceback (most recent call last):" which
    contains the actual error location.

    Args:
        text: Text that may contain a Python traceback

    Returns:
        Tuple of (exception_type, exception_message, traceback_lines) or None if no traceback found
    """
    lines = text.split('\n')

    # Find ALL "Traceback (most recent call last):" or "Original Traceback" occurrences
    # We want the LAST one as it's usually the root cause
    traceback_starts = []
    for i, line in enumerate(lines):
        if 'Traceback (most recent call last):' in line or 'Original Traceback' in line:
            traceback_starts.append(i)

    if not traceback_starts:
        return None

    # Use the LAST traceback found (usually the root cause)
    traceback_start = traceback_starts[-1]

    # Find the exception line (last non-empty line of traceback)
    traceback_lines = []
    exception_line = None

    for i in range(traceback_start + 1, len(lines)):
        line = lines[i]
        if line.strip():
            # Check if this looks like an exception line (no leading spaces or starts with exception name)
            if i > traceback_start + 1 and not line.startswith(' ') and not line.startswith('\t'):
                exception_line = line
                break
            traceback_lines.append(line)

    if not exception_line:
        return None

    # Parse exception line: "ExceptionType: message"
    parts = exception_line.split(':', 1)
    if len(parts) >= 2:
        exception_type = parts[0].strip()
        exception_message = parts[1].strip()
    else:
        exception_type = exception_line.strip()
        exception_message = ""

    return exception_type, exception_message, traceback_lines
