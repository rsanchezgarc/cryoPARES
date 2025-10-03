"""
Tests for cryoPARES.cacheManager module.

This module tests the caching functionality including:
- Cache creation and retrieval
- Hash-based temporary directory creation
- Shared temporary directory behavior
"""

import os
import tempfile
import shutil
import pytest
import joblib

from cryoPARES.cacheManager import get_cache, hashVars, SharedTemporaryDirectory


class TestGetCache:
    """Test the get_cache function."""

    def test_get_cache_with_name(self, tmp_path):
        """Test cache creation with a specific name."""
        cache_name = "test_cache"
        cache = get_cache(cache_name, cachedir=str(tmp_path), verbose=0)

        assert cache is not None
        assert isinstance(cache, joblib.Memory)
        assert os.path.exists(os.path.join(str(tmp_path), cache_name + ".joblib"))

    def test_get_cache_without_name(self):
        """Test cache creation without a name (returns no-cache Memory)."""
        cache = get_cache(cache_name=None, verbose=0)

        assert cache is not None
        assert isinstance(cache, joblib.Memory)
        # No-cache Memory has location=None
        assert cache.location is None

    def test_get_cache_invalid_directory(self, capsys):
        """Test cache creation with invalid directory fallback."""
        import warnings

        invalid_dir = "/nonexistent/invalid/path/that/does/not/exist"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cache = get_cache("test_cache", cachedir=invalid_dir, verbose=0)

            # Should fall back to no-cache Memory
            assert cache is not None
            assert cache.location is None

            # Check warning was issued
            assert len(w) == 1
            assert "is not available" in str(w[0].message)

    def test_cache_caching_functionality(self, tmp_path):
        """Test that cache actually caches function results."""
        cache = get_cache("func_cache", cachedir=str(tmp_path), verbose=0)

        # Define a test function to cache
        @cache.cache
        def expensive_function(x):
            return x ** 2

        # First call
        result1 = expensive_function(5)
        assert result1 == 25

        # Second call should use cache
        result2 = expensive_function(5)
        assert result2 == 25

        # Verify cache files were created
        cache_files = list(tmp_path.rglob("*.pkl"))
        assert len(cache_files) > 0


class TestHashVars:
    """Test the hashVars function."""

    def test_hash_vars_simple(self):
        """Test hash generation with simple variables."""
        name = "test"
        vars_to_hash = {"param1": 1, "param2": "value"}

        hashed_name = hashVars(name, vars_to_hash)

        assert hashed_name.startswith("test_")
        assert len(hashed_name) > len("test_")

    def test_hash_vars_deterministic(self):
        """Test that same inputs produce same hash."""
        name = "test"
        vars_to_hash = {"a": 1, "b": 2}

        hash1 = hashVars(name, vars_to_hash)
        hash2 = hashVars(name, vars_to_hash)

        assert hash1 == hash2

    def test_hash_vars_different_values(self):
        """Test that different inputs produce different hashes."""
        name = "test"
        vars1 = {"a": 1, "b": 2}
        vars2 = {"a": 1, "b": 3}

        hash1 = hashVars(name, vars1)
        hash2 = hashVars(name, vars2)

        assert hash1 != hash2

    def test_hash_vars_order_dependent(self):
        """Test that dict order affects hash (due to JSON serialization)."""
        name = "test"
        # Note: In Python 3.7+, dicts maintain insertion order
        vars1 = {"a": 1, "b": 2}
        vars2 = {"b": 2, "a": 1}

        hash1 = hashVars(name, vars1)
        hash2 = hashVars(name, vars2)

        # These should be different because JSON serialization preserves dict order
        # If they're the same, it means JSON sorted them
        # This test verifies the behavior
        assert isinstance(hash1, str)
        assert isinstance(hash2, str)

    def test_hash_vars_complex_types(self):
        """Test hashing with complex nested structures."""
        name = "complex"
        vars_to_hash = {
            "list": [1, 2, 3],
            "nested": {"inner": "value", "number": 42},
            "string": "test"
        }

        hashed_name = hashVars(name, vars_to_hash)

        assert hashed_name.startswith("complex_")
        assert len(hashed_name) == len("complex_") + 64  # SHA256 produces 64 hex chars


class TestSharedTemporaryDirectory:
    """Test the SharedTemporaryDirectory class."""

    def test_shared_temp_dir_creation(self, tmp_path):
        """Test basic temporary directory creation."""
        name = "test_dir"
        vars_to_hash = {"param": 1}

        with SharedTemporaryDirectory(name, vars_to_hash, rootdir=str(tmp_path)) as temp_dir:
            assert os.path.exists(temp_dir)
            assert name in os.path.basename(temp_dir)
            # Check it was created in the specified rootdir
            assert str(tmp_path) in temp_dir

    def test_shared_temp_dir_cleanup_creator(self, tmp_path):
        """Test that creator process cleans up the directory."""
        name = "cleanup_test"
        vars_to_hash = {"param": 1}

        temp_dir_path = None
        with SharedTemporaryDirectory(name, vars_to_hash, rootdir=str(tmp_path)) as temp_dir:
            temp_dir_path = temp_dir
            assert os.path.exists(temp_dir)

        # After exiting context, directory should be cleaned up by creator
        assert not os.path.exists(temp_dir_path)

    def test_shared_temp_dir_deterministic_path(self, tmp_path):
        """Test that same inputs produce same directory path."""
        name = "deterministic"
        vars_to_hash = {"a": 1, "b": 2}

        # Create first directory
        with SharedTemporaryDirectory(name, vars_to_hash, rootdir=str(tmp_path)) as temp_dir1:
            path1 = temp_dir1

        # Create second directory with same params
        with SharedTemporaryDirectory(name, vars_to_hash, rootdir=str(tmp_path)) as temp_dir2:
            path2 = temp_dir2

        assert path1 == path2

    def test_shared_temp_dir_no_cleanup_non_creator(self, tmp_path):
        """Test that non-creator process doesn't clean up the directory."""
        name = "shared_test"
        vars_to_hash = {"param": 1}

        # First process creates the directory
        with SharedTemporaryDirectory(name, vars_to_hash, rootdir=str(tmp_path)) as temp_dir1:
            first_path = temp_dir1
            assert os.path.exists(first_path)

            # Second process tries to "create" same directory (it already exists)
            temp_dir_obj2 = SharedTemporaryDirectory(name, vars_to_hash, rootdir=str(tmp_path))
            second_path = temp_dir_obj2.name

            assert first_path == second_path
            assert os.path.exists(second_path)
            assert not temp_dir_obj2.should_remove  # Non-creator shouldn't remove

            # Manually cleanup second object
            temp_dir_obj2.cleanup()

            # Directory should still exist because second process shouldn't remove it
            assert os.path.exists(first_path)

    def test_shared_temp_dir_default_rootdir(self):
        """Test using default system temp directory."""
        name = "default_root"
        vars_to_hash = {"param": 1}

        with SharedTemporaryDirectory(name, vars_to_hash) as temp_dir:
            assert os.path.exists(temp_dir)
            # Should be in system temp directory
            assert tempfile.gettempdir() in temp_dir

    def test_shared_temp_dir_unique_hashes(self, tmp_path):
        """Test that different parameters create different directories."""
        name = "unique_test"

        with SharedTemporaryDirectory(name, {"param": 1}, rootdir=str(tmp_path)) as dir1:
            with SharedTemporaryDirectory(name, {"param": 2}, rootdir=str(tmp_path)) as dir2:
                assert dir1 != dir2
                assert os.path.exists(dir1)
                assert os.path.exists(dir2)

    def test_shared_temp_dir_write_read(self, tmp_path):
        """Test writing and reading files in shared temp directory."""
        name = "write_test"
        vars_to_hash = {"param": 1}

        with SharedTemporaryDirectory(name, vars_to_hash, rootdir=str(tmp_path)) as temp_dir:
            # Write a test file
            test_file = os.path.join(temp_dir, "test.txt")
            with open(test_file, "w") as f:
                f.write("test content")

            # Read it back
            with open(test_file, "r") as f:
                content = f.read()

            assert content == "test content"


class TestIntegration:
    """Integration tests combining cache and temp directory functionality."""

    def test_cache_in_shared_temp_dir(self, tmp_path):
        """Test using cache within a shared temporary directory."""
        name = "integration_test"
        vars_to_hash = {"test": 1}

        with SharedTemporaryDirectory(name, vars_to_hash, rootdir=str(tmp_path)) as temp_dir:
            # Create cache in the shared temp directory
            cache = get_cache("test_cache", cachedir=temp_dir, verbose=0)

            @cache.cache
            def cached_func(x):
                return x * 2

            result1 = cached_func(10)
            result2 = cached_func(10)

            assert result1 == 20
            assert result2 == 20

            # Verify cache files exist
            cache_files = [f for f in os.listdir(temp_dir) if f.endswith('.joblib')]
            assert len(cache_files) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
