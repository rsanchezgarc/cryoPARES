#!/usr/bin/env python3
"""
Regroup particles from multiple .mrcs stacks into fewer consolidated stacks.
Maintains micrograph integrity and optical group consistency.
"""

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

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
    if not isinstance(data, dict) or 'particles' not in data:
        print("Error: Starfile must be in Relion 3.1+ format with 'particles' block")
        sys.exit(1)
    
    return data


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
    Group particles by micrograph and optical group.
    Returns dict: {(optics_group, micrograph): list of particle indices}
    """
    groups = defaultdict(list)
    
    has_optics = 'rlnOpticsGroup' in particles_df.columns
    
    for idx, row in particles_df.iterrows():
        micrograph = row['rlnMicrographName']
        optics_group = row['rlnOpticsGroup'] if has_optics else 1
        
        groups[(optics_group, micrograph)].append(idx)
    
    return groups


def consolidate_stacks(particles_df, micrograph_groups, target_particles, 
                       output_dir, prefix, input_starfile_dir):
    """
    Consolidate particles into new stacks.
    Returns updated particles dataframe.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Track current stack being written
    current_stack_particles = []
    current_stack_idx = 1
    current_optics_group = None
    
    # Copy particles dataframe
    updated_df = particles_df.copy()
    
    # Group micrographs by optical group to process them together
    optics_micrographs = defaultdict(list)
    for (optics_group, micrograph), indices in micrograph_groups.items():
        optics_micrographs[optics_group].append((micrograph, indices))
    
    # Process each optical group separately
    for optics_group in sorted(optics_micrographs.keys()):
        micrograph_list = optics_micrographs[optics_group]
        
        print(f"\nProcessing optical group {optics_group} "
              f"({len(micrograph_list)} micrographs)...")
        
        for micrograph, particle_indices in micrograph_list:
            n_particles = len(particle_indices)
            
            # Check if adding this micrograph exceeds target
            if (current_stack_particles and 
                len(current_stack_particles) + n_particles > target_particles):
                
                # Write current stack and start new one
                write_stack(current_stack_particles, particles_df, updated_df,
                           current_stack_idx, output_dir, prefix, 
                           input_starfile_dir)
                current_stack_idx += 1
                current_stack_particles = []
            
            # Add all particles from this micrograph
            current_stack_particles.extend(particle_indices)
        
        # Write remaining particles from this optical group
        if current_stack_particles:
            write_stack(current_stack_particles, particles_df, updated_df,
                       current_stack_idx, output_dir, prefix, 
                       input_starfile_dir)
            current_stack_idx += 1
            current_stack_particles = []
    
    print(f"\nCreated {current_stack_idx - 1} output stacks")
    
    return updated_df


def write_stack(particle_indices, particles_df, updated_df, 
                stack_idx, output_dir, prefix, input_starfile_dir):
    """Write a consolidated stack and update dataframe."""
    
    output_filename = f"{prefix}_{stack_idx:06d}.mrcs"
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"  Writing {output_filename} with {len(particle_indices)} particles...")
    
    # Collect all particle data
    all_particles = []
    
    for df_idx in particle_indices:
        row = particles_df.iloc[df_idx]
        image_name = row['rlnImageName']
        
        # Parse source stack
        particle_idx, stack_path = parse_image_name(image_name)
        
        # Handle relative paths
        if not os.path.isabs(stack_path):
            stack_path = os.path.join(input_starfile_dir, stack_path)
        
        # Read particle from source stack
        with mrcfile.open(stack_path, mode='r', permissive=True) as mrc:
            if mrc.data.ndim == 2:
                # Single particle file
                particle_data = mrc.data
            else:
                # Stack file
                particle_data = mrc.data[particle_idx]
        
        all_particles.append(particle_data)
    
    # Stack all particles
    stacked_data = np.stack(all_particles, axis=0)
    
    # Write new stack
    with mrcfile.new(output_path, overwrite=True) as mrc:
        mrc.set_data(stacked_data.astype(np.float32))
    
    # Update dataframe with new image names
    for new_idx, df_idx in enumerate(particle_indices, start=1):
        new_image_name = f"{new_idx:06d}@{output_filename}"
        updated_df.at[df_idx, 'rlnImageName'] = new_image_name


def make_paths_absolute_or_relative(particles_df, output_dir, use_absolute):
    """Update image paths to be absolute or relative to output starfile."""
    
    updated_df = particles_df.copy()
    
    for idx, row in updated_df.iterrows():
        image_name = row['rlnImageName']
        particle_num, stack_filename = image_name.split('@')
        
        # Get full path to stack
        stack_path = os.path.join(output_dir, stack_filename)
        
        if use_absolute:
            new_path = os.path.abspath(stack_path)
        else:
            new_path = stack_filename
        
        updated_df.at[idx, 'rlnImageName'] = f"{particle_num}@{new_path}"
    
    return updated_df


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
    starfile.write(output_data, args.output)
    
    print("\nDone!")
    print(f"Output stacks written to: {args.output_dir}")
    print(f"Output starfile: {args.output}")


if __name__ == "__main__":
    main()
