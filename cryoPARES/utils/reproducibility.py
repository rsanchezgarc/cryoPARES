import os
import shutil
from os import path as osp


def _copyCode(rootPath, copycodedir):
    os.makedirs(copycodedir, exist_ok=True)
    for root, dirs, files in os.walk(rootPath):
        # Iterate through all folders
        for directory in dirs:
            # Create the corresponding directory in the target path
            source_folder = osp.join(root, directory)
            target_folder = source_folder.replace(rootPath, copycodedir)
            os.makedirs(target_folder, exist_ok=True)
        # Iterate through all Python files
        for file in files:
            if file.endswith(".py") or file.endswith(".yaml") or file.endswith(".yml"):
                # Copy the Python file to the corresponding directory in the target path
                source_file = osp.join(root, file)
                target_file = source_file.replace(rootPath, copycodedir)
                shutil.copy2(source_file, target_file)
