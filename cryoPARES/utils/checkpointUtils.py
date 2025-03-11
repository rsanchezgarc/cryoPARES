import os
import re

import numpy as np
import torch


def get_best_checkpoint(checkpoint_dir, best_is_less=True):
    """
    Get the best checkpoint from a directory.

    Parameters:
    checkpoint_dir (str): Path to the directory containing the checkp
    Returns:
    str: The path to the best checkpoint.
    """
    if best_is_less:
        condition = lambda a,b: a < b
    else:
        condition = lambda a,b: a >= b

    best_checkpoint = None
    best_value = np.inf
    for filename in sorted(os.listdir(checkpoint_dir),
                           key=lambda x:os.stat(os.path.join(checkpoint_dir, x)).st_mtime, reverse=True):
        if filename.endswith(".ckpt"):
            checkpoint_path = os.path.join(checkpoint_dir, filename)
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            for cback_name in checkpoint['callbacks']:
                if cback_name.startswith("ModelCheckpoint"):
                    value = checkpoint["callbacks"][cback_name]["current_score"]
                    if value is None:
                        continue
                    value = value.item()
                    # if value < best_value:
                    if condition(value, best_value):
                        best_checkpoint = checkpoint_path
                        best_value = value
    if best_checkpoint is None:
        raise RuntimeError(f"best_cback_name not found in checkpoint_dir {checkpoint_dir} ")
    return best_checkpoint


def increment_generic(generic_string, pattern, count_group_idx):
    match = re.match(pattern, generic_string)
    if match:
        out = ""
        for i, g in enumerate(match.groups()):
            if i == count_group_idx:
                g = str(int(g) + 1)
            out += g
        return out
    else:
        raise ValueError(f"The string does not match the pattern '{pattern}'")

VERSION_PATTERN = r'(.*version_)(\d+)$'
def increment_version(fname_string, path_basename=VERSION_PATTERN, extension=None):
    pattern = r'(.*' + path_basename + r')(\d+)'
    if extension is not None:
        pattern += r"(\." + extension + r')$'
    else:
        pattern += r'$'

    return increment_generic(fname_string, pattern, 1)

def find_last_version(rootdir, count_group_idx=1, path_pattern=VERSION_PATTERN, dir_only=False):
    items = os.listdir(rootdir)
    if dir_only:
        items = [item for item in items if os.path.isdir(os.path.join(rootdir, item))]
    version_items = [item for item in items if re.match(path_pattern, item)]
    if not version_items:
        return None
    sorted_items = sorted(version_items, key=lambda x: int(re.search(path_pattern, x).group(count_group_idx+1)))
    return sorted_items[-1]


def get_version_to_use(rootdir, basename, path_pattern=VERSION_PATTERN, extension=None, dir_only=False ):
    last_version = find_last_version(rootdir, path_pattern=path_pattern, dir_only=dir_only)
    # print("last_version", last_version)
    if last_version is None:
        out =  basename + "0"
        if extension is not None:
            return out + "." + extension
        return  out
    return increment_version(last_version, basename, extension=extension)



# Tests
import pytest
@pytest.fixture
def tmpdir(tmp_path):
    return tmp_path

def test_increment_version():
    assert increment_version("version_0", "version_") == "version_1"
    assert increment_version("test_version_9", "version_") == "test_version_10"
    try:
        increment_version("invalid_string", "version_")
        raise RuntimeError("Error, invalid pattern was not picked")
    except ValueError:
        pass
    increment_version("invalid_string0", "invalid_string")


def test_find_last_version_directories(tmpdir):
    os.makedirs(os.path.join(tmpdir, "version_0"))
    os.makedirs(os.path.join(tmpdir, "version_1"))
    os.makedirs(os.path.join(tmpdir, "version_2"))
    assert find_last_version(tmpdir, dir_only=True) == "version_2"

def test_find_last_version_files(tmpdir):
    open(os.path.join(tmpdir, "file_v0.txt"), 'a').close()
    open(os.path.join(tmpdir, "file_v1.txt"), 'a').close()
    open(os.path.join(tmpdir, "file_v2.txt"), 'a').close()
    assert find_last_version(tmpdir, path_pattern=r'(file_v)(\d+)(\.txt)$') == "file_v2.txt"
def test_find_last_version_mixed(tmpdir):
    os.makedirs(os.path.join(tmpdir, "version_0"))
    open(os.path.join(tmpdir, "version_1"), 'a').close()
    os.makedirs(os.path.join(tmpdir, "version_2"))
    assert find_last_version(tmpdir) == "version_2"

def test_find_last_version_empty(tmpdir):
    assert find_last_version(tmpdir) is None

def test_get_version_to_use_existing_file(tmpdir):
    open(os.path.join(tmpdir, "file_v0.txt"), 'a').close()
    open(os.path.join(tmpdir, "file_v1.txt"), 'a').close()
    assert get_version_to_use(tmpdir, basename="file_v", path_pattern=r'(file_v)(\d+)(\.txt)$',
                              extension="txt") == "file_v2.txt"

def test_get_version_to_use_existing_dir(tmpdir):
    os.makedirs(os.path.join(tmpdir, "version_0"))
    os.makedirs(os.path.join(tmpdir, "version_1"))
    print(os.listdir(tmpdir))
    assert get_version_to_use(tmpdir, basename="version_", dir_only=True) == "version_2"


def test_get_version_to_use_empty(tmpdir):
    assert get_version_to_use(tmpdir, basename="version_") == "version_0"


def test_custom_pattern_with_dir(tmpdir):
    custom_pattern = r'(.*_v)(\d+)$'
    os.makedirs(os.path.join(tmpdir, "test_v0"))
    os.makedirs(os.path.join(tmpdir, "test_v1"))
    assert find_last_version(tmpdir, path_pattern=custom_pattern, dir_only=True) == "test_v1"
    assert get_version_to_use(tmpdir,basename="test_v",  path_pattern=custom_pattern, dir_only=True) ==  "test_v2"

def test_get_version_to_use_custom_pattern_empty(tmpdir):
    custom_pattern = r'(myfile_)(\d+)(\.txt)$'
    assert get_version_to_use(tmpdir, basename="myfile_", extension="txt", path_pattern=custom_pattern) == "myfile_0.txt"

if __name__ == "__main__":
    pytest.main([__file__])
