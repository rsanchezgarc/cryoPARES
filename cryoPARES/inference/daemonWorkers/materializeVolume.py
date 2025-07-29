#!/usr/bin/env python

import glob
import os
import sys
from typing import List

import numpy as np
import torch

# Import the new parsing library
from argParseFromDoc import parse_function_and_call

# Ensure the cryoPARES package is in the Python path
# You might need to adjust this depending on your project structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cryoPARES.reconstruction.reconstruction import Reconstructor

def materialize_volume(
    input_files: List[str],
    output_mrc: str,
    eps: float = 1e-3
):
    """
    Combines partial reconstruction results from .npy/.npz files into a final .mrc volume.

    :param input_files: Path(s) to the input .npy/.npz files. Supports glob patterns (e.g., 'mapcomponents_*.npz').
    :param output_mrc: Path for the output .mrc file.
    :param eps: The Tikhonov regularization constant for the reconstruction.
    """
    all_files = []
    for pattern in input_files:
        found_files = glob.glob(pattern)
        if not found_files:
            print(f"Warning: No files found for pattern '{pattern}'")
        all_files.extend(found_files)

    if not all_files:
        print("Error: No input files were found. Aborting.")
        sys.exit(1)

    print(f"Found {len(all_files)} files to combine.")

    total_numerator, total_weights, total_ctfsq = None, None, None
    sampling_rate, box_size = None, None

    for f_path in all_files:
        print(f"Loading {f_path}...")
        try:
            data = np.load(f_path, allow_pickle=True)
        except Exception as e:
            print(f"Error loading {f_path}: {e}")
            continue
        try:
            numerator = torch.from_numpy(data['numerator'])
            weights = torch.from_numpy(data['weights'])
            ctfsq = torch.from_numpy(data['ctfsq'])

            if total_numerator is None:
                total_numerator, total_weights, total_ctfsq = numerator, weights, ctfsq
                sampling_rate = data['sampling_rate'].item()
                box_size = total_numerator.shape[1]
            else:
                total_numerator += numerator
                total_weights += weights
                total_ctfsq += ctfsq
        finally:
            data.close()

    if total_numerator is None:
        print("Error: Could not load data from any input files. Aborting.")
        sys.exit(1)

    print("\nAll partial results loaded and aggregated.")
    print(f"Reconstructing with sampling rate {sampling_rate:.4f} Å/px.")

    reconstructor = Reconstructor(
        symmetry="C1", #We use C1 because the partial components should be already symmetry expanded
        eps=eps,
        numerator=total_numerator,
        weights=total_weights,
        ctfsq=total_ctfsq
    )

    reconstructor.box_size = box_size
    reconstructor.sampling_rate = sampling_rate

    print(f"Generating final volume and saving to {output_mrc} ...")
    reconstructor.generate_volume(fname=output_mrc, overwrite_fname=True)

    print("Done!")

if __name__ == "__main__":
    parse_function_and_call(materialize_volume)