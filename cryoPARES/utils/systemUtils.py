"""
System utility functions for cryoPARES.

This module provides utility functions for system-level operations like
managing resource limits.
"""

import resource
import warnings


def increase_file_descriptor_limit():
    """
    Attempt to maximize the file descriptor limit to avoid 'too many open files' errors.

    This sets the soft limit to match the hard limit (maximum allowed by the system).
    This is better than using a fixed value like 65536 since different systems have
    different hard limits.

    CryoPARES opens file handlers for each .mrcs file in RELION .star files, which can
    quickly exceed default system limits. This function automatically increases the limit
    to the maximum allowed, eliminating the need for users to manually run 'ulimit -n'
    before training or inference.

    Raises a warning if the final limit is very small (<=1024), which may cause issues
    when working with many .mrcs files.

    Returns:
        int: The actual limit that was set

    Example:
        >>> from cryoPARES.utils.systemUtils import increase_file_descriptor_limit
        >>> limit = increase_file_descriptor_limit()
        File descriptor limit increased from 1024 to 65536
    """
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)

        # If already at maximum, no need to change
        if soft >= hard:
            final_limit = soft
        else:
            # Set soft limit to hard limit (maximum allowed)
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
            print(f"File descriptor limit increased from {soft} to {hard}")
            final_limit = hard

    except (ValueError, OSError) as e:
        # Permission denied or other error - continue with current limit
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        warnings.warn(
            f"Could not increase file descriptor limit to maximum ({hard}): {e}. "
            f"Current limit: {soft}. You may need to increase system limits or run "
            f"'ulimit -n {hard}' manually if you encounter 'too many open files' errors."
        )
        final_limit = soft

    # Warn if the final limit is very small
    if final_limit <= 1024:
        warnings.warn(
            f"File descriptor limit is very low ({final_limit}). This may cause "
            f"'too many open files' errors when working with many .mrcs files. "
            f"Consider increasing system limits (e.g., edit /etc/security/limits.conf "
            f"or /etc/sysctl.conf) and restarting your session."
        )

    return final_limit