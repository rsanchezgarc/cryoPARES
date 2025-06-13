import functools
from os import PathLike
from pathlib import Path
from typing import Union


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


FnameType = Union[PathLike, str]
