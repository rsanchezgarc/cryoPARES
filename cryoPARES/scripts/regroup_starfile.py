#!/usr/bin/env python3
"""
Regroup particles from multiple .mrcs stacks into fewer consolidated stacks.
Maintains micrograph integrity and optical group consistency.
Optimized for performance and memory efficiency.
"""

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

try:
    import starfile
    import mrcfile
except ImportError:
    print("Error: Required packages not found.")
    print("Install with: pip install starfile mrcfile")
    sys.exit(1)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Regroup Relion starfile particles into consolidated .mrcs stacks"
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input starfile path"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output starfile path"
    )
    parser.add_argument(
        "-d", "--output-dir",
        required=True,
        help="Output directory for new .mrcs files"
    )
    parser.add_argument(
        "-n", "--target-particles",
        type=int,
        default=10000,
        help="Target number of particles per output stack (default: 10000)"
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Use absolute paths for output .mrcs files in starfile"
    )
    parser.add_argument(
        "--prefix",
        default="regroup",
        help="Prefix for output .mrcs files (default: regroup)"
    )
    
    return parser.parse_args()


def load_starfile(filepath):
    """Load Relion 3.1+ starfile."""
    data = starfile.read(filepath)
    
    # Check if it's Relion 3.1+ format
    if not isinstance(data, dict):
        # Relion 3.0 or single-table format
        if isinstance(data, pd.DataFrame):
            return {'particles': data}
        print("Error: Unknown starfile format.")
        sys.exit(1)
    
    # Support common RELION block names
    if 'particles' in data:
        return data
    elif 'data_particles' in data:
        data['particles'] = data.pop('data_particles')
        return data
    else:
        # Fallback to the first block that looks like a particle table
        for key, df in data.items():
            if isinstance(df, pd.DataFrame) and 'rlnImageName' in df.columns:
                print(f"Warning: Using block '{key}' as particles table.")
                data['particles'] = data.pop(key)
                return data
                
    print("Error: Could not find particles table in starfile.")
    sys.exit(1)


def parse_image_name(image_name):
    """Parse Relion image name format: particle_index@stack_path"""
    parts = image_name.split('@')
    if len(parts) != 2:
        raise ValueError(f"Invalid image name format: {image_name}")
    
    particle_idx = int(parts[0]) - 1  # Convert to 0-based
    stack_path = parts[1]
    
    return particle_idx, stack_path


def group_particles_by_micrograph_and_optics(particles_df):
    """
    Group particles by micrograph and optical group using pandas for performance.
    """
    # Ensure rlnOpticsGroup exists for grouping
    cols = ['rlnMicrographName']
    if 'rlnOpticsGroup' in particles_df.columns:
        cols.insert(0, 'rlnOpticsGroup')
    
    # Use pandas groupby for much faster performance than iterrows
    return particles_df.groupby(cols).groups


def consolidate_stacks(particles_df, micrograph_groups, target_particles, 
                       output_dir, prefix, input_starfile_dir):
    """
    Consolidate particles into new stacks.
    Returns updated particles dataframe.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    current_stack_particles = []
    current_stack_idx = 1
    
    # Work on a copy to avoid SettingWithCopy warnings and preserve original indices
    updated_df = particles_df.copy()
    
    # Sort groups to process them predictably
    sorted_group_keys = sorted(micrograph_groups.keys())
    
    total_groups = len(sorted_group_keys)
    for i, group_key in enumerate(sorted_group_keys):
        particle_indices = micrograph_groups[group_key]
        n_particles = len(particle_indices)
        
        # Check if adding this micrograph exceeds target
        if (current_stack_particles and 
            len(current_stack_particles) + n_particles > target_particles):
            
            # Write current stack
            write_stack(current_stack_particles, particles_df, updated_df,
                       current_stack_idx, output_dir, prefix, 
                       input_starfile_dir)
            current_stack_idx += 1
            current_stack_particles = []
        
        current_stack_particles.extend(particle_indices)
        
        if (i + 1) % 100 == 0 or i == total_groups - 1:
            print(f"  Processed {i+1}/{total_groups} micrograph groups...", end='\r')
    print()
    
    # Write remaining particles
    if current_stack_particles:
        write_stack(current_stack_particles, particles_df, updated_df,
                   current_stack_idx, output_dir, prefix, 
                   input_starfile_dir)
        current_stack_idx += 1
    
    print(f"\nCreated {current_stack_idx - 1} output stacks")
    return updated_df


def write_stack(particle_indices, particles_df, updated_df, 
                stack_idx, output_dir, prefix, input_starfile_dir):
    """Write a consolidated stack and update dataframe efficiently."""
    
    output_filename = f"{prefix}_{stack_idx:06d}.mrcs"
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"  Writing {output_filename} with {len(particle_indices)} particles...")
    
    # Collect metadata to group by source stack
    source_info = []
    for df_idx in particle_indices:
        image_name = particles_df.at[df_idx, 'rlnImageName']
        p_idx, s_path = parse_image_name(image_name)
        if not os.path.isabs(s_path):
            s_path = os.path.join(input_starfile_dir, s_path)
        source_info.append({'df_idx': df_idx, 'p_idx': p_idx, 's_path': s_path})
    
    source_df = pd.DataFrame(source_info)
    
    # Pre-allocate buffer if we know the shape (need to peek first image)
    first_path = source_info[0]['s_path']
    first_idx = source_info[0]['p_idx']
    with mrcfile.open(first_path, mode='r', permissive=True, mmap=True) as mrc:
        sample_shape = mrc.data.shape[1:] if mrc.data.ndim == 3 else mrc.data.shape
        dtype = mrc.data.dtype
    
    stacked_data = np.empty((len(particle_indices), *sample_shape), dtype=dtype)
    
    # Iterate by source stack to minimize file opening overhead
    # Use mmap=True to avoid loading the whole stack into memory
    current_pos = 0
    new_image_names = {}
    
    for s_path, group in source_df.groupby('s_path', sort=False):
        with mrcfile.open(s_path, mode='r', permissive=True, mmap=True) as mrc:
            # Handle both single-image MRC and stacks
            data_ref = mrc.data
            for row in group.itertuples():
                if data_ref.ndim == 3:
                    stacked_data[current_pos] = data_ref[row.p_idx]
                else:
                    stacked_data[current_pos] = data_ref
                
                # Store the new image name mapped to original dataframe index
                new_image_names[row.df_idx] = f"{current_pos+1:06d}@{output_filename}"
                current_pos += 1
    
    # Write new stack
    with mrcfile.new(output_path, overwrite=True) as mrc:
        mrc.set_data(stacked_data)
    
    # Update updated_df using the mapping
    # Faster to update in bulk if possible, but .at in loop is okay for 10k items
    for df_idx, new_name in new_image_names.items():
        updated_df.at[df_idx, 'rlnImageName'] = new_name


def make_paths_absolute_or_relative(particles_df, output_dir, use_absolute):
    """Update image paths to be absolute or relative to output starfile."""
    if use_absolute:
        abs_output_dir = os.path.abspath(output_dir)
        def fix_path(img):
            p_num, s_name = img.split('@')
            return f"{p_num}@{os.path.join(abs_output_dir, os.path.basename(s_name))}"
    else:
        def fix_path(img):
            p_num, s_name = img.split('@')
            return f"{p_num}@{os.path.basename(s_name)}"
            
    particles_df['rlnImageName'] = particles_df['rlnImageName'].apply(fix_path)
    return particles_df


def main():
    args = parse_arguments()
    
    print(f"Reading starfile: {args.input}")
    data = load_starfile(args.input)
    
    particles_df = data['particles']
    optics_df = data.get('optics', None)
    
    print(f"Total particles: {len(particles_df)}")
    
    # Get input starfile directory for resolving relative paths
    input_starfile_dir = os.path.dirname(os.path.abspath(args.input))
    
    # Group particles by micrograph and optics
    print("Grouping particles by micrograph and optical group...")
    micrograph_groups = group_particles_by_micrograph_and_optics(particles_df)
    
    print(f"Found {len(micrograph_groups)} unique micrograph-optics combinations")
    
    # Consolidate stacks
    print(f"\nConsolidating stacks (target: {args.target_particles} particles/stack)...")
    updated_particles_df = consolidate_stacks(
        particles_df, 
        micrograph_groups,
        args.target_particles,
        args.output_dir,
        args.prefix,
        input_starfile_dir
    )
    
    # Update paths
    print("\nUpdating paths in starfile...")
    updated_particles_df = make_paths_absolute_or_relative(
        updated_particles_df,
        args.output_dir,
        args.absolute_paths
    )
    
    # Prepare output data
    output_data = {'particles': updated_particles_df}
    if optics_df is not None:
        output_data['optics'] = optics_df
    
    # Write output starfile
    print(f"\nWriting output starfile: {args.output}")
    starfile.write(output_data, args.output, overwrite=True)
    
    print("\nDone!")
    print(f"Output stacks written to: {args.output_dir}")
    print(f"Output starfile: {args.output}")


if __name__ == "__main__":
    main()
