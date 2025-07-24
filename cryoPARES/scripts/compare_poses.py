import argparse
import numpy as np
import torch
import starfile
import warnings
from cryoPARES.geometry.convert_angles import euler_angles_to_matrix
from cryoPARES.geometry.symmetry import getSymmetryGroup

def get_angular_distance(r_pred, r_true, symmetry_ops):
    r_true_sym = torch.matmul(symmetry_ops, r_true)
    
    # Broadcasting the prediction matrix for comparison with all symmetry variants
    r_pred_broadcast = r_pred.unsqueeze(0).expand_as(r_true_sym)
    
    # Calculate the trace for all comparisons at once
    # The formula for angular distance is arccos((trace(R) - 1) / 2)
    # where R = r_pred @ r_true_sym.T
    # trace(A @ B.T) is the same as sum(A * B) element-wise
    trace = torch.sum(r_pred_broadcast * r_true_sym, dim=(1, 2))
    
    # Clip the trace to avoid numerical errors with arccos
    trace_clipped = torch.clamp(trace, -1.0, 3.0)
    
    distances = torch.acos((trace_clipped - 1) / 2)
    return torch.min(distances).item()

def main():
    parser = argparse.ArgumentParser(description='Compare poses of a results starfile with a ground truth starfile.')
    parser.add_argument('-i', '--input', required=True, help='Path to the results starfile.')
    parser.add_argument('--gt', required=True, help='Path to the ground truth starfile.')
    parser.add_argument('--symmetry', default='C1', help='Symmetry group of the particle.')
    args = parser.parse_args()

    # Read starfiles using the starfile library
    results_df = starfile.read(args.input)["particles"]
    gt_df = starfile.read(args.gt)["particles"]

    # Merge dataframes on image name
    merged_df = results_df.merge(gt_df, on='rlnImageName', suffixes=['_pred', '_gt'])

    if len(merged_df) < len(results_df) or len(merged_df) < len(gt_df):
        warnings.warn("Warning: The number of matching particles is less than the total number of particles in the input files. Statistics will be computed on the matching subset.")

    if len(merged_df) == 0:
        print("Error: No matching particles found between the two starfiles.")
        return

    # Get symmetry operations
    symmetry_ops = getSymmetryGroup(args.symmetry, as_matrix=True)

    angular_errors = []
    shift_errors = []

    for _, row in merged_df.iterrows():
        # Angular error
        angles_pred = torch.deg2rad(torch.tensor([row['rlnAngleRot_pred'], row['rlnAngleTilt_pred'], row['rlnAnglePsi_pred']], dtype=torch.float32))
        angles_true = torch.deg2rad(torch.tensor([row['rlnAngleRot_gt'], row['rlnAngleTilt_gt'], row['rlnAnglePsi_gt']], dtype=torch.float32))

        r_pred = euler_angles_to_matrix(angles_pred, convention='ZYZ')
        r_true = euler_angles_to_matrix(angles_true, convention='ZYZ')

        angular_errors.append(np.rad2deg(get_angular_distance(r_pred, r_true, symmetry_ops)))

        # Shift error
        shift_pred = np.array([row['rlnOriginXAngst_pred'], row['rlnOriginYAngst_pred']])
        shift_true = np.array([row['rlnOriginXAngst_gt'], row['rlnOriginYAngst_gt']])
        shift_errors.append(np.linalg.norm(shift_pred - shift_true))

    angular_errors = np.array(angular_errors)
    shift_errors = np.array(shift_errors)

    print(f"Found {len(merged_df)} matching particles.")
    print("\nAngular errors (degrees):")
    print(f"  Mean: {np.mean(angular_errors):.2f}")
    print(f"  Median: {np.median(angular_errors):.2f}")
    print(f"  Std: {np.std(angular_errors):.2f}")
    print(f"  IQR: {np.quantile(angular_errors, 0.75) - np.quantile(angular_errors, 0.25):.2f}")

    print("\nShift errors (A):")
    print(f"  Mean: {np.mean(shift_errors):.2f}")
    print(f"  Median: {np.median(shift_errors):.2f}")
    print(f"  Std: {np.std(shift_errors):.2f}")
    print(f"  IQR: {np.quantile(shift_errors, 0.75) - np.quantile(shift_errors, 0.25):.2f}")

if __name__ == '__main__':
    main()