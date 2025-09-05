import argparse
import numpy as np
import torch
import starfile
import warnings
import matplotlib.pyplot as plt
from cesped.constants import RELION_EULER_CONVENTION

from cryoPARES.geometry.convert_angles import euler_angles_to_matrix
from cryoPARES.geometry.symmetry import getSymmetryGroup
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize


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


def apply_transformation(angles, transform_matrix):
    """Apply a transformation matrix to a set of Euler angles."""
    R = euler_angles_to_matrix(angles, convention=RELION_EULER_CONVENTION)
    R_transformed = torch.matmul(torch.from_numpy(transform_matrix).float(), R)
    # Convert back to angles using scipy
    R_scipy = Rotation.from_matrix(R_transformed.numpy())
    return torch.from_numpy(R_scipy.as_euler(RELION_EULER_CONVENTION)).float()


def optimize_transformation(angles_pred_list, angles_true_list, symmetry_ops):
    """Find the optimal transformation matrix that aligns two sets of angles."""

    def objective(params):
        R_transform = torch.from_numpy(Rotation.from_euler('xyz', params, degrees=True).as_matrix()).float()

        total_error = 0
        for angles_pred, angles_true in zip(angles_pred_list, angles_true_list):
            r_pred = euler_angles_to_matrix(angles_pred, convention=RELION_EULER_CONVENTION)
            r_true = euler_angles_to_matrix(angles_true, convention=RELION_EULER_CONVENTION)
            r_true_transformed = torch.matmul(R_transform, r_true)
            error = get_angular_distance(r_pred, r_true_transformed, symmetry_ops)
            total_error += error

        return total_error / len(angles_pred_list)

    initial_guess = [0, 0, 0]
    result = minimize(objective, initial_guess, method='Nelder-Mead')
    optimal_transform = Rotation.from_euler('xyz', result.x, degrees=True).as_matrix()

    return optimal_transform, result.fun


def plot_angular_errors(angular_errors, output_prefix=None):
    """Create scatter plot of angular errors vs particle index"""
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(angular_errors)), angular_errors, alpha=0.6, s=20)
    plt.xlabel('Particle Index')
    plt.ylabel('Angular Error (degrees)')
    plt.title('Angular Errors per Particle')
    plt.grid(True, alpha=0.3)

    # Add horizontal lines for mean and median
    mean_error = np.mean(angular_errors)
    median_error = np.median(angular_errors)
    plt.axhline(y=mean_error, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_error:.2f}°')
    plt.axhline(y=median_error, color='orange', linestyle='--', alpha=0.7, label=f'Median: {median_error:.2f}°')
    plt.legend()

    if output_prefix:
        plt.savefig(f'{output_prefix}_angular_errors.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_shift_errors(shift_errors, shift_x_errors, shift_y_errors, output_prefix=None):
    """Create scatter plots for shift errors"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Shift magnitude vs particle index
    axes[0].scatter(range(len(shift_errors)), shift_errors, alpha=0.6, s=20, color='blue')
    axes[0].set_xlabel('Particle Index')
    axes[0].set_ylabel('Shift Error Magnitude (Å)')
    axes[0].set_title('Shift Error Magnitude per Particle')
    axes[0].grid(True, alpha=0.3)

    # Add mean and median lines
    mean_shift = np.mean(shift_errors)
    median_shift = np.median(shift_errors)
    axes[0].axhline(y=mean_shift, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_shift:.2f}Å')
    axes[0].axhline(y=median_shift, color='orange', linestyle='--', alpha=0.7, label=f'Median: {median_shift:.2f}Å')
    axes[0].legend()

    # Plot 2: X vs Y shift errors
    axes[1].scatter(shift_x_errors, shift_y_errors, alpha=0.6, s=20, color='green')
    axes[1].set_xlabel('X Shift Error (Å)')
    axes[1].set_ylabel('Y Shift Error (Å)')
    axes[1].set_title('X vs Y Shift Errors')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1].axvline(x=0, color='black', linestyle='-', alpha=0.3)

    # Plot 3: Vector field representation (arrows showing shift direction)
    # Sample every nth particle to avoid overcrowding
    n_arrows = min(100, len(shift_x_errors))  # Max 100 arrows
    step = max(1, len(shift_x_errors) // n_arrows)
    indices = range(0, len(shift_x_errors), step)

    x_pos = [i for i in indices]
    y_pos = [0] * len(x_pos)  # All at y=0 baseline
    u = [shift_x_errors[i] for i in indices]
    v = [shift_y_errors[i] for i in indices]

    axes[2].quiver(x_pos, y_pos, u, v, angles='xy', scale_units='xy', scale=1, alpha=0.7, width=0.003)
    axes[2].set_xlabel('Particle Index')
    axes[2].set_ylabel('Shift Error (Å)')
    axes[2].set_title('Shift Error Vectors (sampled)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_prefix:
        plt.savefig(f'{output_prefix}_shift_errors.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_combined_errors(angular_errors, shift_errors, output_prefix=None):
    """Create scatter plot of angular vs shift errors"""
    plt.figure(figsize=(8, 6))
    plt.scatter(angular_errors, shift_errors, alpha=0.6, s=20, color='purple')
    plt.xlabel('Angular Error (degrees)')
    plt.ylabel('Shift Error Magnitude (Å)')
    plt.title('Angular vs Shift Errors')
    plt.grid(True, alpha=0.3)

    # Calculate and display correlation
    correlation = np.corrcoef(angular_errors, shift_errors)[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
             transform=plt.gca().transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    if output_prefix:
        plt.savefig(f'{output_prefix}_combined_errors.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Compare poses of a results starfile with a ground truth starfile.')
    parser.add_argument('-i', '--input', required=True, help='Path to the results starfile.')
    parser.add_argument('--gt', required=True, help='Path to the ground truth starfile.')
    parser.add_argument('--symmetry', default='C1', help='Symmetry group of the particle.')

    # Visualization options
    parser.add_argument('--plot-angular', action='store_true',
                        help='Show scatter plot of angular errors vs particle index.')
    parser.add_argument('--plot-shifts', action='store_true', help='Show scatter plots of shift errors.')
    parser.add_argument('--plot-combined', action='store_true', help='Show scatter plot of angular vs shift errors.')
    parser.add_argument('--plot-all', action='store_true', help='Show all scatter plots.')
    parser.add_argument('--save-plots', type=str,
                        help='Save plots with this prefix (e.g., "experiment1" -> "experiment1_angular_errors.png")')
    parser.add_argument('--skip-frame-alignment', action='store_true',
                        help='Skip trying to find optimal alignment between reference frames')
    args = parser.parse_args()

    # Read starfiles using the starfile library
    results_df = starfile.read(args.input)["particles"]
    gt_df = starfile.read(args.gt)["particles"]

    # Merge dataframes on image name
    merged_df = results_df.merge(gt_df, on='rlnImageName', suffixes=['_pred', '_gt'])
    n_merged = len(merged_df)
    n_results_df = len(results_df)
    n_gt_df = len(gt_df)
    if len(merged_df) < len(results_df) or len(merged_df) < len(gt_df):
        warnings.warn(f"Warning: The number of matching particles {n_merged} is less than the"
                      f" total number of particles in the input files ({n_results_df};{n_gt_df}). "
                      f"Statistics will be computed on the matching subset.")

    if len(merged_df) == 0:
        print("Error: No matching particles found between the two starfiles.")
        return

    # Get symmetry operations
    symmetry_ops = getSymmetryGroup(args.symmetry, as_matrix=True)

    # First pass: collect all angles and shifts
    angles_pred_list = []
    angles_true_list = []
    shifts_pred_list = []
    shifts_true_list = []

    for _, row in merged_df.iterrows():
        angles_pred = torch.deg2rad(
            torch.tensor([row['rlnAngleRot_pred'], row['rlnAngleTilt_pred'], row['rlnAnglePsi_pred']],
                         dtype=torch.float32))
        angles_true = torch.deg2rad(
            torch.tensor([row['rlnAngleRot_gt'], row['rlnAngleTilt_gt'], row['rlnAnglePsi_gt']], dtype=torch.float32))

        shift_pred = np.array([row['rlnOriginXAngst_pred'], row['rlnOriginYAngst_pred']])
        shift_true = np.array([row['rlnOriginXAngst_gt'], row['rlnOriginYAngst_gt']])

        angles_pred_list.append(angles_pred)
        angles_true_list.append(angles_true)
        shifts_pred_list.append(shift_pred)
        shifts_true_list.append(shift_true)

    # Apply frame alignment if requested
    transform_matrix = np.eye(3)
    if not args.skip_frame_alignment:
        print("Optimizing reference frame alignment...")
        transform_matrix, min_error = optimize_transformation(angles_pred_list, angles_true_list, symmetry_ops)
        print(f"Optimal alignment found with mean error: {min_error:.2f}°\n{transform_matrix.round(2)}\n-----------------")

    # Second pass: calculate errors with aligned frames
    angular_errors = []
    shift_errors = []
    shift_x_errors = []
    shift_y_errors = []

    for i, (angles_pred, angles_true, shift_pred, shift_true) in enumerate(
            zip(angles_pred_list, angles_true_list, shifts_pred_list, shifts_true_list)):
        # Angular error
        if not args.skip_frame_alignment:
            # Apply transformation to true angles
            angles_true = apply_transformation(angles_true, transform_matrix)

        r_pred = euler_angles_to_matrix(angles_pred, convention=RELION_EULER_CONVENTION)
        r_true = euler_angles_to_matrix(angles_true, convention=RELION_EULER_CONVENTION)

        angular_errors.append(np.rad2deg(get_angular_distance(r_pred, r_true, symmetry_ops)))

        # Shift error
        shift_diff = shift_pred - shift_true
        shift_x_errors.append(shift_diff[0])
        shift_y_errors.append(shift_diff[1])
        shift_errors.append(np.linalg.norm(shift_diff))

    angular_errors = np.array(angular_errors)
    shift_errors = np.array(shift_errors)
    shift_x_errors = np.array(shift_x_errors)
    shift_y_errors = np.array(shift_y_errors)

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

    # Generate plots based on user request
    if args.plot_all or args.plot_angular:
        plot_angular_errors(angular_errors, args.save_plots)

    if args.plot_all or args.plot_shifts:
        plot_shift_errors(shift_errors, shift_x_errors, shift_y_errors, args.save_plots)

    if args.plot_all or args.plot_combined:
        plot_combined_errors(angular_errors, shift_errors, args.save_plots)


if __name__ == '__main__':
    main()