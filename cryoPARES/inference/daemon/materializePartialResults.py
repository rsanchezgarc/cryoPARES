#!/usr/bin/env python

import glob
import os
import sys
from typing import List, Optional

import numpy as np
import torch
import pandas as pd
import starfile

from cryoPARES.reconstruction.reconstructor import Reconstructor


def materialize_partial_results(
    partial_outputs_dirs: List[str],
    output_mrc: str,
    output_star: Optional[str] = None,
    eps: float = 1e-3
):
    """
    Combines partial reconstruction results and optionally merges star files from a directory.

    :param partial_outputs_dirs: List of directories containing the partial output files (.npz and .star), or glob patterns for files within those directories.
    :param output_mrc: Path for the output .mrc file.
    :param output_star: Optional. Path for the output merged .star file. If provided, all .star files in the partial_outputs_dir will be merged.
    :param eps: The Tikhonov regularization constant for the reconstruction.
    """
    dirs_to_search = set()
    for path in partial_outputs_dirs:
        # Path could be a directory, a file, or a glob pattern
        # To handle shell expansion of globs, we check if it's a file or dir first
        if os.path.isdir(path):
            dirs_to_search.add(path)
        elif os.path.isfile(path):
            dirs_to_search.add(os.path.dirname(path))
        else:
            # If not a file or dir, it might be a glob pattern that needs expansion
            expanded_paths = glob.glob(path)
            if not expanded_paths:
                print(f"Warning: Input path '{path}' is not an existing file or directory and did not match any paths.")
                continue
            
            for p in expanded_paths:
                if os.path.isdir(p):
                    dirs_to_search.add(p)
                elif os.path.isfile(p):
                    dirs_to_search.add(os.path.dirname(p))

    if not dirs_to_search:
        print("Error: No valid directories could be determined from the input paths. Aborting.")
        sys.exit(1)
        
    print(f"Searching for partial results in the following directories: {list(dirs_to_search)}")

    # Find partial volume files (.npz)
    all_files = []
    valid_npz_suffixes = ("_half1.npz", "_half2.npz", "_allParticles.npz")
    for d in dirs_to_search:
        npz_pattern = os.path.join(d, "mapcomponents*.npz")
        glob_files = glob.glob(npz_pattern)
        for f in glob_files:
            if f.endswith(valid_npz_suffixes):
                all_files.append(f)

    if not all_files:
        print(f"Error: No partial volume files (mapcomponents*.npz with valid suffixes) found in the provided directories. Aborting.")
        sys.exit(1)

    print(f"Found {len(all_files)} partial volume files to combine.")

    if output_star:
        # Find star files
        star_files_to_merge = []
        valid_star_suffixes = ("_half1.star", "_half2.star", "_allParticles.star")
        for d in dirs_to_search:
            star_pattern = os.path.join(d, "*.star")
            glob_files = glob.glob(star_pattern)
            for f in glob_files:
                if f.endswith(valid_star_suffixes):
                    star_files_to_merge.append(f)

        if not star_files_to_merge:
            print(f"Warning: No .star files with valid suffixes found in the provided directories to merge.")
        else:
            print(f"Found {len(star_files_to_merge)} star files to merge.")

            all_particles = []
            all_optics = []

            for sf_path in star_files_to_merge:
                try:
                    star_data = starfile.read(sf_path, always_dict=True)
                    for key, value in star_data.items():
                        if key == 'optics':
                            all_optics.append(value)
                        elif 'particles' in key:
                            all_particles.append(value)
                except Exception as e:
                    print(f"Warning: Could not read star file {sf_path}: {e}")

            if all_particles:
                merged_particles = pd.concat(all_particles, ignore_index=True)

                merged_star_data = {'particles': merged_particles}

                if all_optics:
                    merged_optics = pd.concat(all_optics).drop_duplicates().reset_index(drop=True)
                    merged_star_data['optics'] = merged_optics

                starfile.write(merged_star_data, output_star, overwrite=True)
                print(f"Merged star file saved to {output_star}")
            else:
                print("Warning: No particle data found in star files to merge.")

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
    from argParseFromDoc import parse_function_and_call
    parse_function_and_call(materialize_partial_results)