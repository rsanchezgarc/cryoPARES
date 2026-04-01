#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import starfile
from scipy.spatial.transform import Rotation as R
import os

def get_symmetry_rotations(sym_group):
    """Generates a list of scipy Rotation objects for the given symmetry group."""
    sym_group = sym_group.upper()
    if sym_group == 'C1':
        return [R.identity()]
    
    try:
        # scipy >= 1.8.0
        return list(R.create_group(sym_group))
    except (ValueError, AttributeError):
        # Fallback for older scipy or custom handling
        ops = [R.identity()]
        group_type = sym_group[0]
        try:
            n = int(sym_group[1:])
        except ValueError:
            raise ValueError(f"Unsupported symmetry group format: {sym_group}. Use C<n> or D<n>.")

        # Generate C_n rotations around Z-axis
        c_ops = []
        for i in range(n):
            angle = 2 * np.pi * i / n
            rot = R.from_rotvec(np.array([0, 0, angle]))
            c_ops.append(rot)
        
        if group_type == 'C':
            return c_ops
        elif group_type == 'D':
            d_ops = []
            flip_x = R.from_rotvec(np.array([np.pi, 0, 0]))
            for c_op in c_ops:
                d_ops.append(c_op)
                d_ops.append(c_op * flip_x)
            return d_ops
        else:
            raise ValueError(f"Symmetry {sym_group} not supported by fallback. Upgrade scipy to 1.8.0+.")

def normalize_relion_euler(rot, tilt, psi):
    """Normalize ZYZ Euler angles to RELION convention: Tilt in [0, 180]."""
    rot %= 360
    tilt %= 360
    psi %= 360
    
    if tilt > 180:
        tilt = 360 - tilt
        rot = (rot + 180) % 360
        psi = (psi + 180) % 360
    return rot, tilt, psi

def fold_to_asymmetric_unit(rotations, sym_ops):
    """
    Given a list of rotations and symmetry operators, 
    returns rotations folded into a 'canonical' asymmetric unit.
    """
    folded_rots = []
    for rot in rotations:
        # Particle symmetry S is applied in the particle's local frame: R_new = R * S
        equivalent_rots = [rot * op for op in sym_ops]
        
        euler_equivs = [r.as_euler('ZYZ', degrees=True) for r in equivalent_rots]
        
        norm_eulers = []
        for e in euler_equivs:
            norm_eulers.append(normalize_relion_euler(e[0], e[1], e[2]))
        
        # Sort and pick the first one as canonical
        norm_eulers.sort()
        folded_rots.append(norm_eulers[0])
        
    return np.array(folded_rots)

def main():
    parser = argparse.ArgumentParser(description="Randomize particle orientations and shifts in a STAR file.")
    parser.add_argument("-i", "--input", required=True, help="Input RELION .star file")
    parser.add_argument("-o", "--output", required=True, help="Output .star file")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of particles to randomize (0.0 to 1.0)")
    parser.add_argument("--max_shift", type=float, default=0.0, help="Maximum shift in pixels (uniform distribution [-max, max])")
    parser.add_argument("--sym", default="C1", help="Symmetry group (e.g., C1, C3, D7)")
    parser.add_argument("--fold_au", action="store_true", help="Fold randomized orientations into the asymmetric unit")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    print(f"Loading {args.input}...")
    ds = starfile.read(args.input)
    
    # Robust block selection
    if isinstance(ds, pd.DataFrame):
        df = ds
        block_name = None
    else:
        # Priority: 'particles', 'data_particles', then first block with Euler columns
        block_name = None
        for cand in ['particles', 'data_particles']:
            if cand in ds:
                block_name = cand
                break
        
        if block_name is None:
            for name, block_df in ds.items():
                if isinstance(block_df, pd.DataFrame) and 'rlnAngleRot' in block_df.columns:
                    block_name = name
                    break
        
        if block_name is None:
            block_name = list(ds.keys())[0]
            print(f"Warning: Could not identify particle block. Using '{block_name}'.")
            
        df = ds[block_name]

    num_particles = len(df)
    num_to_randomize = int(round(num_particles * args.fraction))
    
    print(f"Total particles: {num_particles}")
    if num_to_randomize == 0:
        print("Nothing to randomize. Saving input to output.")
        starfile.write(ds, args.output, overwrite=True)
        return

    print(f"Randomizing {num_to_randomize} particles ({args.fraction*100:.1f}%)...")

    # Select random indices
    all_indices = np.arange(num_particles)
    random_indices = np.random.choice(all_indices, size=num_to_randomize, replace=False)
    
    # 1. Randomize Orientations
    random_rots = R.random(num_to_randomize)
    
    if args.fold_au and args.sym.upper() != "C1":
        print(f"Folding orientations to {args.sym} asymmetric unit...")
        sym_ops = get_symmetry_rotations(args.sym)
        euler_angles = fold_to_asymmetric_unit(random_rots, sym_ops)
    else:
        euler_angles_raw = random_rots.as_euler('ZYZ', degrees=True)
        euler_angles = np.array([normalize_relion_euler(e[0], e[1], e[2]) for e in euler_angles_raw])

    # Update DataFrame
    df.loc[df.index[random_indices], 'rlnAngleRot'] = euler_angles[:, 0]
    df.loc[df.index[random_indices], 'rlnAngleTilt'] = euler_angles[:, 1]
    df.loc[df.index[random_indices], 'rlnAnglePsi'] = euler_angles[:, 2]

    # 2. Randomize Shifts
    if args.max_shift > 0:
        print(f"Applying random shifts up to {args.max_shift} pixels...")
        shift_x = np.random.uniform(-args.max_shift, args.max_shift, num_to_randomize)
        shift_y = np.random.uniform(-args.max_shift, args.max_shift, num_to_randomize)
        
        # Maintain consistency across all potential shift columns
        found_shift_col = False
        for col_x, col_y in [('rlnOriginX', 'rlnOriginY'), ('rlnOriginXAngst', 'rlnOriginYAngst')]:
            if col_x in df.columns:
                df.loc[df.index[random_indices], col_x] = shift_x
                df.loc[df.index[random_indices], col_y] = shift_y
                found_shift_col = True
        
        if not found_shift_col:
            print("Warning: No shift columns found. Adding rlnOriginX/Y.")
            df.loc[df.index[random_indices], 'rlnOriginX'] = shift_x
            df.loc[df.index[random_indices], 'rlnOriginY'] = shift_y

    # Save
    if block_name:
        ds[block_name] = df
    else:
        ds = df
        
    starfile.write(ds, args.output, overwrite=True)
    print(f"Done. Saved to {args.output}")

if __name__ == "__main__":
    main()
