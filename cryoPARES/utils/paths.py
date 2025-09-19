import functools
import glob
import os
from os import PathLike
from pathlib import Path
from typing import Union

import numpy as np
import torch


def _find_directory_with_marker(marker_file):
    """Find the project root by looking for setup.py"""
    current = Path(__file__).resolve()
    while current.parent != current:  # While we haven't hit the root
        if (current / marker_file).exists():
            return current
        current = current.parent
    raise FileNotFoundError(f"Could not find {marker_file} in any parent directory")

@functools.cache
def find_configs_root() -> Path:
    """Find the project root by looking for setup.py"""
    return _find_directory_with_marker('constants.py')

@functools.cache
def find_project_root() -> Path:
    """Find the project root by looking for setup.py"""
    return _find_directory_with_marker('setup.py')


def get_most_recent_file(folder_path: str, template: str) -> str | None:
    """
    Finds the most recent file in a directory that matches a given template.

    Args:
        folder_path: The path to the directory to search.
        template: The filename pattern to match (e.g., 'file_*.txt').

    Returns:
        The path to the most recent file, or None if no matching files are found.
    """
    try:
        # Construct the full search path
        search_path = os.path.join(folder_path, template)

        # Find all files matching the template
        matching_files = glob.glob(search_path)

        # If no files match, return None
        if not matching_files:
            return None

        # Get the most recent file based on modification time
        most_recent_file = max(matching_files, key=os.path.getmtime)

        return most_recent_file

    except FileNotFoundError:
        print(f"Error: The folder '{folder_path}' was not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


FNAME_TYPE = Union[PathLike, str]
MAP_AS_ARRAY_OR_FNAME_TYPE = Union[FNAME_TYPE, torch.Tensor, np.ndarray]
