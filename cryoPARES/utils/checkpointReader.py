"""
Helper class to read from either a directory or a ZIP archive checkpoint.
Provides unified interface for file access.
"""

import os
import zipfile
import io
import tempfile
import torch
from typing import List


class CheckpointReader:
    """
    Helper class to read from either a directory or a ZIP archive checkpoint.
    Provides unified interface for file access.
    """

    def __init__(self, checkpoint_path: str):
        """
        Initialize checkpoint reader.

        Args:
            checkpoint_path: Path to checkpoint directory or ZIP file
        """
        self.checkpoint_path = checkpoint_path
        self.is_zip = checkpoint_path.endswith('.zip') and os.path.isfile(checkpoint_path)
        self._zipfile = None
        self._temp_dir = None

        if self.is_zip:
            self._zipfile = zipfile.ZipFile(checkpoint_path, 'r')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Clean up resources."""
        if self._zipfile is not None:
            self._zipfile.close()
            self._zipfile = None
        if self._temp_dir is not None:
            import shutil
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None

    def exists(self, relative_path: str) -> bool:
        """Check if a file exists in the checkpoint."""
        if self.is_zip:
            try:
                self._zipfile.getinfo(relative_path)
                return True
            except KeyError:
                return False
        else:
            return os.path.exists(os.path.join(self.checkpoint_path, relative_path))

    def read_bytes(self, relative_path: str) -> bytes:
        """Read file as bytes."""
        if self.is_zip:
            return self._zipfile.read(relative_path)
        else:
            full_path = os.path.join(self.checkpoint_path, relative_path)
            with open(full_path, 'rb') as f:
                return f.read()

    def read_text(self, relative_path: str) -> str:
        """Read file as text."""
        return self.read_bytes(relative_path).decode('utf-8')

    def load_torch(self, relative_path: str, **kwargs):
        """Load PyTorch model/tensor from checkpoint."""
        if self.is_zip:
            # PyTorch can load from BytesIO
            file_bytes = self.read_bytes(relative_path)
            buffer = io.BytesIO(file_bytes)
            return torch.load(buffer, **kwargs)
        else:
            full_path = os.path.join(self.checkpoint_path, relative_path)
            return torch.load(full_path, **kwargs)

    def load_jit(self, relative_path: str):
        """Load TorchScript model from checkpoint."""
        if self.is_zip:
            file_bytes = self.read_bytes(relative_path)
            buffer = io.BytesIO(file_bytes)
            return torch.jit.load(buffer)
        else:
            full_path = os.path.join(self.checkpoint_path, relative_path)
            return torch.jit.load(full_path)

    def get_real_path(self, relative_path: str) -> str:
        """
        Get real filesystem path. For ZIP, extracts to temp directory.
        Use this only when absolutely necessary (e.g., for mrcfile which needs real paths).

        Args:
            relative_path: Path relative to checkpoint root

        Returns:
            Absolute filesystem path
        """
        if self.is_zip:
            if self._temp_dir is None:
                self._temp_dir = tempfile.mkdtemp(prefix='cryopares_checkpoint_')

            # Extract file to temp dir
            dest_path = os.path.join(self._temp_dir, relative_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            with open(dest_path, 'wb') as f:
                f.write(self.read_bytes(relative_path))

            return dest_path
        else:
            return os.path.join(self.checkpoint_path, relative_path)

    def glob(self, pattern: str) -> List[str]:
        """
        Find files matching pattern (relative paths).

        Args:
            pattern: Glob pattern (e.g., "configs_*.yml")

        Returns:
            List of matching file paths (relative to checkpoint root)
        """
        if self.is_zip:
            import fnmatch
            return [name for name in self._zipfile.namelist() if fnmatch.fnmatch(name, pattern)]
        else:
            import glob
            full_pattern = os.path.join(self.checkpoint_path, pattern)
            matches = glob.glob(full_pattern)
            # Convert to relative paths
            return [os.path.relpath(m, self.checkpoint_path) for m in matches]
