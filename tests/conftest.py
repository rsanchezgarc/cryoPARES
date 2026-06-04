import pytest
import torch


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="requires CUDA GPU")
    for item in items:
        if item.get_closest_marker("gpu"):
            item.add_marker(skip)
