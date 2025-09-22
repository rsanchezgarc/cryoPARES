import numpy as np
import mrcfile
from numpy import fft
import matplotlib.pyplot as plt
import argparse
import os
from scipy.ndimage import zoom  # Added for map resizing


def compute_fsc(vol1, vol2, voxel_size, mask=None, allow_resize=False, resolution_from_A=15):
    """
    Calculates the Fourier Shell Correlation (FSC) between two 3D volumes.

    It operates by binning Fourier space voxels into concentric shells based on
    their integer pixel distance from the origin and calculating the correlation within each shell.

    Args:
        vol1 (numpy.ndarray): 3D numpy array for the first map.
        vol2 (numpy.ndarray): 3D numpy array for the second map.
        voxel_size (float): The voxel size of the maps in Angstroms (Å).
        mask (numpy.ndarray, optional): A 3D mask to apply to the volumes. Defaults to None.
        allow_resize (bool, optional): If True, resizes vol2 to match vol1's shape if they differ. Defaults to False.
        resolution_from_A (int, optional): Ignore resolution oscilations at resolution worse than resolution_from_A

    Returns:
        tuple: A tuple containing three 1D numpy arrays:
               - fsc (np.ndarray): The FSC value for each shell.
               - spatial_freq (np.ndarray): The spatial frequency for each shell (in 1/Å).
               - resolution_A (np.ndarray): The resolution in Angstroms for each shell.
               - resolutions (tuple): Resolutions at FSC=0.5 and FSC=0.143.
    """
    # --- Handle potential shape mismatch ---
    if vol1.shape != vol2.shape:
        if allow_resize:
            print(f"Map shapes differ: {vol1.shape} vs {vol2.shape}.")
            print(f"Resizing map 2 to match map 1 using cubic spline interpolation...")
            zoom_factors = np.array(vol1.shape) / np.array(vol2.shape)
            vol2 = zoom(vol2, zoom_factors, order=3, prefilter=True)
            print(f"Resizing complete. New map 2 shape: {vol2.shape}")
        else:
            raise ValueError(
                "Input maps must have the same shape. Use the --resize_maps flag to enable automatic resizing.")

    assert vol1.shape == vol2.shape, "Map resizing failed. Shapes still do not match."
    D = vol1.shape[0]

    if mask is not None:
        assert vol1.shape == mask.shape, "Mask must have the same shape as the maps."
        vol1 = np.multiply(vol1, mask)
        vol2 = np.multiply(vol2, mask)

    # --- 1. Compute Fourier Transforms and shift origin to center ---
    ft1 = fft.fftshift(fft.fftn(vol1))
    ft2 = fft.fftshift(fft.fftn(vol2))

    # --- 2. Generate Fourier space coordinates in integer pixels ---
    coords = np.arange(-D // 2, D // 2)
    kx, ky, kz = np.meshgrid(coords, coords, coords, indexing='ij')
    k_dist = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2).flatten()

    # --- 3. Bin values into shells and calculate FSC using histograms ---
    num_shells = D // 2
    bin_edges = np.arange(0.5, num_shells, 1.0)
    shell_radii = (bin_edges[:-1] + bin_edges[1:]) / 2

    C_terms = (ft1 * np.conj(ft2)).flatten().real
    A_terms = (np.abs(ft1) ** 2).flatten().real
    B_terms = (np.abs(ft2) ** 2).flatten().real

    C_sum = np.histogram(k_dist, bins=bin_edges, weights=C_terms)[0]
    A_sum = np.histogram(k_dist, bins=bin_edges, weights=A_terms)[0]
    B_sum = np.histogram(k_dist, bins=bin_edges, weights=B_terms)[0]

    denominator = np.sqrt(np.maximum(A_sum * B_sum, 1e-9))
    fsc = C_sum / denominator

    # --- 4. Calculate spatial frequency and resolution axes ---
    spatial_freq = shell_radii / (D * voxel_size)
    resolution_A = 1 / spatial_freq

    # --- 5. Apply resolution_from_A filtering ---
    # Find the index where resolution becomes better than resolution_from_A
    valid_resolution_mask = resolution_A <= resolution_from_A

    if np.any(valid_resolution_mask):
        # Start checking from the first resolution better than resolution_from_A
        start_idx = np.where(valid_resolution_mask)[0][0]
    else:
        # If no resolution is better than resolution_from_A, use all data
        start_idx = 0

    # Find resolution where FSC drops below common thresholds, but only from start_idx onwards
    try:
        # Look for FSC < 0.5 only in the valid resolution range
        valid_indices = np.where((fsc[start_idx:] < 0.5))[0]
        if len(valid_indices) > 0:
            res_05 = resolution_A[start_idx + valid_indices[0]]
        else:
            res_05 = np.nan
    except IndexError:
        res_05 = np.nan

    try:
        # Look for FSC < 0.143 only in the valid resolution range
        valid_indices = np.where((fsc[start_idx:] < 0.143))[0]
        if len(valid_indices) > 0:
            res_0143 = resolution_A[start_idx + valid_indices[0]]
        else:
            res_0143 = np.nan
    except IndexError:
        res_0143 = np.nan

    return fsc, spatial_freq, resolution_A, (res_05, res_0143)


def cli():
    """
    Command-line interface to run the FSC calculation.
    """
    parser = argparse.ArgumentParser(
        description="Calculate Fourier Shell Correlation (FSC) between two MRC maps. ⚛️",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("map1", help="Path to the first input .mrc file (reference map for shape).")
    parser.add_argument("map2", help="Path to the second input .mrc file.")
    parser.add_argument("--mask", default=None, help="Optional path to a .mrc mask file.")
    parser.add_argument("--resize_maps", action="store_true",
                        help="If maps have different box sizes, automatically resize map 2 to match map 1 using spline interpolation.")
    parser.add_argument("--show_plot", action="store_true", help="Display the FSC plot interactively.")
    parser.add_argument("--save_plot", default=None, help="Path to save the FSC plot image (e.g., fsc_plot.png).")
    parser.add_argument("--save_csv", default=None,
                        help="Path to save the FSC data as a CSV file (e.g., fsc_data.csv).")
    parser.add_argument("--resolution_from_A", type=float, default=15.0,
                        help="Ignore resolution oscillations at resolution worse than this value (in Angstroms).")

    args = parser.parse_args()

    # --- Load Data ---
    print(f"Loading map 1: {args.map1}")
    try:
        with mrcfile.open(args.map1) as mrc:
            vol1 = mrc.data.copy().astype(np.float32)
            voxel_size1 = mrc.voxel_size.x.item()
            if voxel_size1 == 0:
                raise ValueError("Voxel size in map 1 header is 0. Please set it to a correct non-zero value.")
    except Exception as e:
        print(f"Error loading {args.map1}: {e}")
        return

    print(f"Loading map 2: {args.map2}")
    try:
        with mrcfile.open(args.map2) as mrc:
            vol2 = mrc.data.copy().astype(np.float32)
            voxel_size2 = mrc.voxel_size.x.item()
    except Exception as e:
        print(f"Error loading {args.map2}: {e}")
        return

    # Use map 1's voxel size as the reference and warn if different
    voxel_size = voxel_size1
    if not np.isclose(voxel_size1, voxel_size2):
        print(
            f"⚠️ Warning: Voxel sizes differ ({voxel_size1:.3f} Å vs {voxel_size2:.3f} Å). Using map 1's voxel size as reference.")

    mask_data = None
    if args.mask:
        print(f"Loading mask: {args.mask}")
        try:
            with mrcfile.open(args.mask) as mrc:
                mask_data = mrc.data.copy().astype(np.float32)
        except Exception as e:
            print(f"Error loading mask {args.mask}: {e}")
            return

    # --- Computation ---
    print("Calculating FSC...")
    try:
        fsc, spatial_freq, resolution_A, (res_05, res_0143) = compute_fsc(
            vol1, vol2, voxel_size, mask=mask_data, allow_resize=args.resize_maps,
            resolution_from_A=args.resolution_from_A
        )
    except ValueError as e:
        print(f"❌ Error: {e}")
        return

    print("Calculation complete.")

    # --- Report Resolution ---
    print(f"Resolution threshold for oscillation filtering: {args.resolution_from_A:.1f} Å")
    print(f"Resolution at FSC=0.143 ('gold-standard'): {res_0143:.3f} Å")
    print(f"Resolution at FSC=0.5                    : {res_05:.3f} Å")

    # --- Save CSV Data ---
    if args.save_csv:
        print(f"Saving data to {args.save_csv}...")
        data_to_save = np.vstack((spatial_freq, resolution_A, fsc)).T
        np.savetxt(
            args.save_csv, data_to_save, delimiter=',', header='SpatialFrequency_invA,Resolution_A,FSC', comments=''
        )
        print("Data saved successfully.")

    # --- Plotting ---
    if args.show_plot or args.save_plot:
        print("Generating plot...")
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = 'tab:blue'
        ax1.set_xlabel("Spatial Frequency (1/Å)")
        ax1.set_ylabel("FSC", color=color)
        ax1.plot(spatial_freq, fsc, label='FSC Curve', color=color, linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim((-0.1, 1.1))
        ax1.set_xlim(left=0)

        ax1.axhline(y=0.5, color='gray', linestyle='--', linewidth=1)
        ax1.axhline(y=0.143, color='red', linestyle='--', linewidth=1)
        ax1.text(spatial_freq[len(spatial_freq) // 2], 0.5, ' FSC = 0.5', va='bottom', ha='center', color='gray',
                 backgroundcolor='white')
        ax1.text(spatial_freq[len(spatial_freq) // 2], 0.143, ' FSC = 0.143', va='bottom', ha='center', color='red',
                 backgroundcolor='white')

        # Add vertical line to show resolution threshold
        resolution_threshold_freq = 1 / args.resolution_from_A
        if resolution_threshold_freq <= max(spatial_freq):
            ax1.axvline(x=resolution_threshold_freq, color='orange', linestyle=':', linewidth=1, alpha=0.7)
            ax1.text(resolution_threshold_freq, 0.9, f' {args.resolution_from_A:.1f} Å threshold',
                     va='center', ha='left', color='orange', backgroundcolor='white', rotation=90, fontsize=9)

        ax2 = ax1.twiny()
        ax2.set_xlabel("Resolution (Å)")
        ticks = ax1.get_xticks()
        tick_labels = [f"{1 / t:.1f}" if t > 0 else "∞" for t in ticks]
        ax2.set_xticks(ticks)
        ax2.set_xticklabels(tick_labels)
        ax2.set_xlim(ax1.get_xlim())

        fig.suptitle("Fourier Shell Correlation", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        if args.save_plot:
            print(f"Saving plot to {args.save_plot}...")
            plt.savefig(args.save_plot, dpi=300, bbox_inches='tight')
            print("Plot saved successfully.")

        if args.show_plot:
            print("Displaying plot...")
            plt.show()


if __name__ == "__main__":
    cli()