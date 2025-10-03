import starfile
import numpy as np
from cryoPARES.constants import RELION_EULER_CONVENTION
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
import os
import warnings
import matplotlib.pyplot as plt
from typing import Optional


def load_starfile(filename, particles_key="particles"):
    """Load a RELION starfile and return the particles dataframe."""
    try:
        data = starfile.read(filename)
        if particles_key is None:
            # Get the particles table (usually the last table in the file)
            particles_key = list(data.keys())[-1]
        return data[particles_key]
    except Exception as e:
        raise Exception(f"Error reading starfile {filename}: {str(e)}")


def extract_particle_id(image_name):
    """Extract particle number from RELION image name (e.g., '0001@/path/stack.mrcs')."""
    num, stackname = image_name.split('@')
    return str(int(num)) + "_" + os.path.basename(stackname)


def match_particles(df1, df2, allow_partial=True):
    """
    Match particles between two dataframes and sort them accordingly.
    Returns sorted dataframes if matching is possible, otherwise raises an exception.
    If allow_partial=True, allows partial matching and returns only matching particles.
    """
    # Extract particle IDs
    df1['particle_id'] = df1['rlnImageName'].apply(extract_particle_id)
    df2['particle_id'] = df2['rlnImageName'].apply(extract_particle_id)

    # Check if particle sets are identical
    particles1 = set(df1['particle_id'])
    particles2 = set(df2['particle_id'])

    if allow_partial:
        # Find common particles
        common_particles = particles1.intersection(particles2)
        if len(common_particles) == 0:
            raise ValueError("No matching particles found between the two starfiles.")

        # Filter dataframes to only include common particles
        df1_filtered = df1[df1['particle_id'].isin(common_particles)]
        df2_filtered = df2[df2['particle_id'].isin(common_particles)]

        # Sort both dataframes by particle ID
        df1_sorted = df1_filtered.sort_values('particle_id').reset_index(drop=True)
        df2_sorted = df2_filtered.sort_values('particle_id').reset_index(drop=True)

        # Warn about partial matching
        n_merged = len(common_particles)
        n_df1 = len(particles1)
        n_df2 = len(particles2)
        if n_merged < n_df1 or n_merged < n_df2:
            warnings.warn(f"Warning: The number of matching particles {n_merged} is less than the"
                          f" total number of particles in the input files ({n_df1};{n_df2}). "
                          f"Statistics will be computed on the matching subset.")

    else:
        # Original strict matching logic
        if len(particles1) != len(particles2):
            raise ValueError(
                f"Starfiles contain different particles.\n"
                f"Particles only in first file: {len(particles1 - particles2)}\n"
                f"Particles only in second file: {len(particles2 - particles1)}"
            )
        sorted_particles1 = sorted(particles1)
        sorted_particles2 = sorted(particles2)

        if sorted_particles1 != sorted_particles2:
            for i in range(len(sorted_particles1)):
                raise ValueError(
                    f"Starfiles contain different particles.\n"
                    f"Particles first vs second:\n{sorted_particles1[i]}\n{sorted_particles2[i]}\n"
                )

        # Sort both dataframes by particle ID
        df1_sorted = df1.sort_values('particle_id').reset_index(drop=True)
        df2_sorted = df2.sort_values('particle_id').reset_index(drop=True)

    # Final verification
    if not (df1_sorted['particle_id'] == df2_sorted['particle_id']).all():
        raise ValueError("Failed to match particles even after sorting")

    return df1_sorted, df2_sorted


def euler_to_matrix(angles):
    """Convert RELION Euler angles (rot, tilt, psi) to rotation matrix."""
    return Rotation.from_euler(RELION_EULER_CONVENTION, angles, degrees=True).as_matrix()


def matrix_to_euler(matrix):
    """Convert rotation matrix to RELION Euler angles (rot, tilt, psi)."""
    return Rotation.from_matrix(matrix).as_euler(RELION_EULER_CONVENTION, degrees=True)


def get_symmetry_matrices(sym_group):
    """Get symmetry matrices using scipy's rotation groups."""
    if sym_group.upper() == 'C1':
        return [np.eye(3)]

    # Parse symmetry group
    if sym_group[0].upper() in ['C', 'D']:
        group = f"{sym_group[0].upper()}{sym_group[1:]}"
    else:
        group = sym_group.upper()

    try:
        rot_group = Rotation.create_group(group)
        return rot_group.as_matrix()
    except ValueError as e:
        raise ValueError(f"Invalid symmetry group: {sym_group}. Error: {str(e)}")


def calculate_angular_difference(R1, R2, sym_matrices):
    """Calculate the minimum angular difference between two rotation matrices considering symmetry."""
    R_diffs = np.einsum('ijk,kl->ijl', sym_matrices, np.dot(R2, R1.T))
    traces = np.einsum('ijj->i', R_diffs)
    angles = np.arccos(np.clip((traces - 1) / 2, -1.0, 1.0))
    return np.degrees(np.min(angles))


def apply_transformation(angles, transform_matrix):
    """Apply a transformation matrix to a set of Euler angles."""
    R = euler_to_matrix(angles)
    R_transformed = np.dot(transform_matrix, R)
    return matrix_to_euler(R_transformed)


def optimize_transformation(angles1, angles2, sym_matrices):
    """Find the optimal transformation matrix that aligns two sets of angles."""

    def objective(params):
        # Convert optimization parameters to rotation matrix
        R_transform = Rotation.from_euler('xyz', params, degrees=True).as_matrix()

        # Apply transformation to all angles in set 2
        total_error = 0
        for i in range(len(angles1)):
            R1 = euler_to_matrix(angles1[i])
            R2 = euler_to_matrix(angles2[i])
            R2_transformed = np.dot(R_transform, R2)
            error = calculate_angular_difference(R1, R2_transformed, sym_matrices)
            total_error += error

        return total_error / len(angles1)

    # Initial guess: identity transformation
    initial_guess = [0, 0, 0]

    # Optimize
    result = minimize(objective, initial_guess, method='Nelder-Mead')

    # Convert optimal parameters to rotation matrix
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


def analyze_angular_differences(starfile1: str,
                                starfile2: str,
                                sym: str,
                                align_frames: bool = False,
                                plot_angular: bool = False,
                                plot_shifts: bool = False,
                                plot_combined: bool = False,
                                plot_all: bool = False,
                                save_plots: Optional[str] = None):
    """
    Compare angular assignments between two RELION starfiles.

    :param starfile1: First starfile
    :param starfile2: Second starfile
    :param sym: Symmetry group (C1, C2, C3, ..., D2, D3, ..., T, O, I)
    :param align_frames: Try to find optimal alignment between reference frames
    :param plot_angular: Show scatter plot of angular errors vs particle index
    :param plot_shifts: Show scatter plots of shift errors
    :param plot_combined: Show scatter plot of angular vs shift errors
    :param plot_all: Show all scatter plots
    :param save_plots: Save plots with this prefix (e.g., "experiment1" -> "experiment1_angular_errors.png")
    """
    symmetry = sym.upper()
    # Load starfiles
    df1 = load_starfile(starfile1)
    df2 = load_starfile(starfile2)

    # Match and sort particles (allowing partial matching)
    df1, df2 = match_particles(df1, df2, allow_partial=True)

    # Get symmetry matrices
    sym_matrices = get_symmetry_matrices(symmetry)

    # Extract Euler angles
    angles1 = df1[['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']].values
    angles2 = df2[['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']].values

    # Extract shifts if available
    shifts_available = all(col in df1.columns and col in df2.columns
                           for col in ['rlnOriginXAngst', 'rlnOriginYAngst'])

    shift_x_errors = []
    shift_y_errors = []
    shift_errors = []

    if shifts_available:
        shifts1 = df1[['rlnOriginXAngst', 'rlnOriginYAngst']].values
        shifts2 = df2[['rlnOriginXAngst', 'rlnOriginYAngst']].values

        # Calculate shift errors
        shift_diffs = shifts1 - shifts2
        shift_x_errors = shift_diffs[:, 0]
        shift_y_errors = shift_diffs[:, 1]
        shift_errors = np.linalg.norm(shift_diffs, axis=1)

    # If requested, try to align reference frames
    transform_matrix = np.eye(3)
    if align_frames:
        print("Optimizing reference frame alignment...")
        transform_matrix, min_error = optimize_transformation(angles1, angles2, sym_matrices)
        print(f"Optimal alignment found with mean error: {min_error:.2f}°")

        # Apply transformation to second set of angles
        angles2 = np.array([apply_transformation(ang, transform_matrix) for ang in angles2])

    # Calculate angular differences
    angular_diffs = []
    for i in range(len(angles1)):
        R1 = euler_to_matrix(angles1[i])
        R2 = euler_to_matrix(angles2[i])
        diff = calculate_angular_difference(R1, R2, sym_matrices)
        angular_diffs.append(diff)

    angular_diffs = np.array(angular_diffs)

    # Calculate statistics
    stats = {
        'mean': np.mean(angular_diffs),
        'std': np.std(angular_diffs),
        'median': np.median(angular_diffs),
        'iqr': np.percentile(angular_diffs, 75) - np.percentile(angular_diffs, 25),
        'percent_below_5': np.mean(angular_diffs < 5) * 100,
        'percent_below_10': np.mean(angular_diffs < 10) * 100,
        'transform_matrix': transform_matrix if align_frames else None,
        'n_particles': len(angular_diffs),
        'shifts_available': shifts_available
    }

    if shifts_available:
        stats.update({
            'shift_mean': np.mean(shift_errors),
            'shift_std': np.std(shift_errors),
            'shift_median': np.median(shift_errors),
            'shift_iqr': np.percentile(shift_errors, 75) - np.percentile(shift_errors, 25),
        })

    # Generate plots based on user request
    if plot_all or plot_angular:
        plot_angular_errors(angular_diffs, save_plots)

    if shifts_available and (plot_all or plot_shifts):
        plot_shift_errors(shift_errors, shift_x_errors, shift_y_errors, save_plots)

    if shifts_available and (plot_all or plot_combined):
        plot_combined_errors(angular_diffs, shift_errors, save_plots)

    # Print statistics
    print(f"Found {stats['n_particles']} matching particles.")
    print(f"\nAnalyzing angular differences with {symmetry} symmetry:")
    print(f"Mean: {stats['mean']:.2f}°")
    print(f"Standard Deviation: {stats['std']:.2f}°")
    print(f"Median: {stats['median']:.2f}°")
    print(f"IQR: {stats['iqr']:.2f}°")
    print(f"\nPercentage of particles with angular error:")
    print(f"< 5°: {stats['percent_below_5']:.1f}%")
    print(f"< 10°: {stats['percent_below_10']:.1f}%")

    if stats['shifts_available']:
        print(f"\nShift errors (Å):")
        print(f"Mean: {stats['shift_mean']:.2f}")
        print(f"Standard Deviation: {stats['shift_std']:.2f}")
        print(f"Median: {stats['shift_median']:.2f}")
        print(f"IQR: {stats['shift_iqr']:.2f}")
    else:
        print("\nShift information not available in both starfiles.")

    if align_frames and stats['transform_matrix'] is not None:
        print("\nOptimal transformation matrix:")
        print(stats['transform_matrix'])

    return stats


def main():
    from argParseFromDoc import parse_function_and_call
    parse_function_and_call(analyze_angular_differences)


if __name__ == "__main__":
    main()