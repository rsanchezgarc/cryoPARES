#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import starfile
from scipy.spatial.transform import Rotation as R

def get_symmetry_rotations(sym_group):
    """Generates a list of scipy Rotation objects for C and D point groups."""
    sym_group = sym_group.upper()
    ops = [R.identity()]
    
    if sym_group == 'C1':
        return ops
        
    group_type = sym_group[0]
    try:
        n = int(sym_group[1:])
    except ValueError:
        raise ValueError(f"Unsupported symmetry group format: {sym_group}. Use C<n> or D<n>.")

    # Generate C_n rotations around Z-axis
    c_ops = []
    for i in range(n):
        angle = 2 * np.pi * i / n
        # Z-axis rotation
        rot = R.from_rotvec(np.array([0, 0, angle]))
        c_ops.append(rot)
    
    if group_type == 'C':
        return c_ops
        
    elif group_type == 'D':
        # D_n adds 2-fold axes perpendicular to Z (typically along X or bisecting)
        d_ops = []
        flip_x = R.from_rotvec(np.array([np.pi, 0, 0])) # 180 deg around X
        for c_op in c_ops:
            d_ops.append(c_op)
            d_ops.append(c_op * flip_x)
        return d_ops
    else:
        raise NotImplementedError(f"Symmetry {sym_group} not implemented in this script.")

def angles_to_vector(rot, tilt):
    """Converts RELION Rot/Tilt (in degrees) to a 3D viewing vector."""
    rot_rad = np.radians(rot)
    tilt_rad = np.radians(tilt)
    x = np.sin(tilt_rad) * np.cos(rot_rad)
    y = np.sin(tilt_rad) * np.sin(rot_rad)
    z = np.cos(tilt_rad)
    return np.array([x, y, z])

def main():
    parser = argparse.ArgumentParser(description="Simulate preferred orientation in cryo-EM star files.")
    parser.add_argument("-i", "--input", required=True, help="Input RELION .star file")
    parser.add_argument("-o", "--output", required=True, help="Output .star file")
    parser.add_argument("--sym", default="C1", help="Symmetry group (e.g., C1, C3, D7)")
    parser.add_argument("--target_tilt", type=float, required=True, help="Tilt angle of the view to deplete (degrees)")
    parser.add_argument("--target_rot", type=float, default=0.0, help="Rot angle of the view to deplete (degrees)")
    parser.add_argument("--tolerance", type=float, default=15.0, help="Radius of the depletion cone (degrees)")
    parser.add_argument("--fraction", type=float, default=0.8, help="Fraction of particles to REMOVE in the cone (0.0 to 1.0)")
    
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    ds = starfile.read(args.input)
    block_name = 'particles' if 'particles' in ds else list(ds.keys())[0]
    df = ds[block_name]

    # 1. Define the base target vector
    base_target_vec = angles_to_vector(args.target_rot, args.target_tilt)
    
    # 2. Generate all symmetric equivalents of the target vector
    sym_ops = get_symmetry_rotations(args.sym)
    target_vectors = []
    for op in sym_ops:
        # Apply symmetry rotation to the target vector
        sym_vec = op.apply(base_target_vec)
        target_vectors.append(sym_vec)
    target_vectors = np.array(target_vectors)
    
    print(f"Generated {len(target_vectors)} equivalent target viewing directions for {args.sym} symmetry.")

    # 3. Calculate viewing vectors for all particles
    # Using numpy vectorization for speed
    rots = df['rlnAngleRot'].values
    tilts = df['rlnAngleTilt'].values
    
    rot_rads = np.radians(rots)
    tilt_rads = np.radians(tilts)
    
    px = np.sin(tilt_rads) * np.cos(rot_rads)
    py = np.sin(tilt_rads) * np.sin(rot_rads)
    pz = np.cos(tilt_rads)
    particle_vectors = np.vstack((px, py, pz)).T # Shape: (N, 3)

    # 4. Find particles within the tolerance cone of ANY target vector
    # Dot product of unit vectors = cosine of angle between them
    # We want max dot product (minimum angle) across all symmetric targets
    dot_products = np.dot(particle_vectors, target_vectors.T) # Shape: (N, num_sym_ops)
    max_dots = np.max(dot_products, axis=1) # Shape: (N,)
    
    # Convert dot product back to angle in degrees, clip to avoid floating point errors > 1.0
    min_angles = np.degrees(np.arccos(np.clip(max_dots, -1.0, 1.0)))
    
    mask_in_cone = min_angles <= args.tolerance
    
    df_safe = df[~mask_in_cone]
    df_target = df[mask_in_cone]
    
    print(f"Total particles: {len(df)}")
    print(f"Particles falling within {args.tolerance} degrees of target view(s): {len(df_target)}")
    
    # 5. Subsample the target group
    keep_fraction = 1.0 - args.fraction
    df_target_reduced = df_target.sample(frac=keep_fraction, random_state=42)
    
    df_final = pd.concat([df_safe, df_target_reduced]).sort_index()
    
    print(f"Removed {len(df_target) - len(df_target_reduced)} particles.")
    print(f"Particles remaining: {len(df_final)}")
    
    # 6. Save
    ds[block_name] = df_final
    starfile.write(ds, args.output, overwrite=True)
    print(f"Done. Saved to {args.output}")

if __name__ == "__main__":
    main()
