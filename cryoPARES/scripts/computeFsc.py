import numpy as np
import mrcfile
from numpy import fft
import matplotlib.pyplot as plt
import argparse
from scipy.ndimage import zoom as rs_zoom, gaussian_filter, zoom


# ----------------------------- Crossing logic ----------------------------- #

def first_crossing_with_bounce_check(
    fsc: np.ndarray,
    resolution_A: np.ndarray,
    *,
    threshold: float,
    cutoff_res_A: float,
    persistence: int = 3,
    rebound_window: int = 6,
    eps: float = 1e-3
):
    """
    Return (res_A, idx) for the first FSC crossing below `threshold`.

    Logic:
      1) Scan from low to high frequency (left->right).
      2) The first index i where FSC goes from >= threshold to < threshold is a candidate.
      3) If its resolution is <= cutoff_res_A (e.g., 15 Å), accept immediately.
      4) If its resolution is >  cutoff_res_A (worse), treat it as 'noisy region':
         - accept only if it stays below threshold for at least `persistence` bins, AND
         - does not rebound above threshold within `rebound_window` bins.
      5) Otherwise, ignore as a bounce and keep searching.
    """
    f = np.asarray(fsc)
    th = float(threshold)

    # candidate crossings: previous >= th, current < th (left→right)
    cand = np.where((f[1:] < th - eps) & (f[:-1] >= th - eps))[0] + 1
    if cand.size == 0:
        return np.nan, None

    for i in cand:
        res_i = float(resolution_A[i])

        # Already at/better than cutoff? accept immediately
        if res_i <= cutoff_res_A:
            return res_i, int(i)

        # Otherwise, apply bounce check in the noisy (worse-than-cutoff) region
        end = min(i + rebound_window, len(f))
        seg = f[i:end]

        has_persistence = (end - i) >= persistence and np.all(seg[:persistence] < th - eps)
        # A "rebound" is popping back above threshold within the look-ahead window
        rebound_idxs = np.where(seg >= th + eps)[0]
        rebounds_soon = (rebound_idxs.size > 0) and (rebound_idxs[0] < persistence)

        if has_persistence and not rebounds_soon:
            return res_i, int(i)
        # else: ignore and continue

    return np.nan, None


# --------------------------- Resampling utilities ------------------------- #

def fourier_pad_crop_to_shape(vol: np.ndarray, target_shape):
    """
    Change array shape without changing sampling rate by cropping/padding
    the centered Fourier transform (band-limited resize of box size).
    """
    F = fft.fftshift(fft.fftn(vol))
    outF = np.zeros(target_shape, dtype=F.dtype)

    src_slices = []
    dst_slices = []
    for o, n in zip(F.shape, target_shape):
        if n <= o:  # crop
            start_o = (o - n) // 2
            src_slices.append(slice(start_o, start_o + n))
            dst_slices.append(slice(0, n))
        else:       # pad
            start_n = (n - o) // 2
            src_slices.append(slice(0, o))
            dst_slices.append(slice(start_n, start_n + o))

    outF[tuple(dst_slices)] = F[tuple(src_slices)]
    out = fft.ifftn(fft.ifftshift(outF)).real
    return out


def resample_to_voxel_size(vol: np.ndarray, voxel_size_from: float, voxel_size_to: float,
                           order: int = 3, anti_alias: bool = True):
    """
    Resample volume so that its voxel size becomes voxel_size_to (Å/px).
    Uses real-space cubic interpolation with an anti-aliasing Gaussian
    prefilter when downsampling. Practical, dependency-light approach.

    scale = voxel_size_from / voxel_size_to
      - scale > 1: upsample (more voxels, finer grid)
      - scale < 1: downsample (fewer voxels, coarser grid) -> anti-alias filter
    """
    if voxel_size_from <= 0 or voxel_size_to <= 0:
        raise ValueError("Voxel sizes must be positive non-zero numbers.")

    scale = float(voxel_size_from) / float(voxel_size_to)
    new_shape = tuple(np.maximum(1, np.round(np.array(vol.shape) * scale).astype(int)))

    v = vol
    if anti_alias and scale < 1.0:
        # Simple heuristic sigma in voxels to soften content before decimation
        sigma = max(0.0, 0.5 * (1.0 / scale - 1.0))
        if sigma > 0:
            v = gaussian_filter(v, sigma=sigma, mode="nearest")

    factors = np.array(new_shape) / np.array(vol.shape)
    v = rs_zoom(v, factors, order=order, prefilter=True)
    return v


def match_grid_to_reference(vol2: np.ndarray,
                            voxel_size2: float,
                            ref_shape,
                            ref_voxel_size: float,
                            mode: str = "auto",
                            interp_order: int = 3,
                            anti_alias: bool = True):
    """
    Make vol2 match the reference grid (ref_shape, ref_voxel_size).
    Steps:
      1) If voxel sizes differ, resample vol2 -> ref_voxel_size using real-space cubic
         with anti-aliasing when downsampling.
      2) If shapes still differ, band-limited Fourier pad/crop to ref_shape.

    mode: "auto" | "fourier-first" | "realspace-only"
      - "auto": voxel-size first (if needed), then fourier pad/crop for shape.
      - "fourier-first": try shape via Fourier first (only meaningful if voxel sizes equal),
                         else fall back to voxel-size resample then Fourier.
      - "realspace-only": use real-space zoom for shape as well (not recommended).
    """
    v = vol2.copy()
    vs = float(voxel_size2)

    # 1) Voxel size alignment
    if not np.isclose(vs, ref_voxel_size, rtol=1e-6, atol=1e-6):
        print("  Warning: Voxel sizes differ "
              f"({vs:.4f} Å vs {ref_voxel_size:.4f} Å).")
        print("    Resampling map 2 to match map 1's sampling rate "
              f"({ref_voxel_size:.4f} Å/px) using real-space cubic interpolation.")
        if anti_alias and vs < ref_voxel_size:
            print("    (Downsampling detected) Applying Gaussian anti-aliasing prefilter "
                  "to reduce high-frequency aliasing.")
        v = resample_to_voxel_size(v, voxel_size_from=vs, voxel_size_to=ref_voxel_size,
                                   order=interp_order, anti_alias=anti_alias)
        vs = ref_voxel_size
        print("      Voxel-size resampling complete.")

    # 2) Shape alignment
    if v.shape != tuple(ref_shape):
        if mode in ("auto", "fourier-first"):
            print("    Warning: Box sizes (array shapes) differ "
                  f"({v.shape} vs {tuple(ref_shape)}).")
            print("    Adjusting box size via Fourier pad/crop so sampling rate is preserved "
                  "(band-limited change of shape).")
            v = fourier_pad_crop_to_shape(v, tuple(ref_shape))
            print("      Fourier pad/crop complete.")
        elif mode == "realspace-only":
            print("   Warning: Using real-space interpolation to change shape "
                  "(not band-limited). Prefer 'auto' for FSC computations.")
            factors = np.array(ref_shape) / np.array(v.shape)
            v = rs_zoom(v, factors, order=interp_order, prefilter=True)
            print("    ✅ Real-space shape interpolation complete.")
        else:
            raise ValueError(f"Unknown resize mode: {mode}")

    return v


# ------------------------------ FSC function ------------------------------ #

def compute_fsc(
    vol1: np.ndarray,
    vol2: np.ndarray,
    voxel_size: float,
    mask: np.ndarray = None,
    allow_resize: bool = False,
    resolution_from_A: float = 15.0,
    persistence_bins: int = 3,
    rebound_window: int = 6,
    threshold_eps: float = 1e-3
):
    """
    Calculates the Fourier Shell Correlation (FSC) between two 3D volumes.

    Returns:
        tuple: (fsc, spatial_freq, resolution_A, (res_05, res_0143))
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
            raise ValueError( "Input maps must have the same shape. Use the --resize_maps flag to enable automatic resizing.") assert vol1.shape == vol2.shape, "Map resizing failed. Shapes still do not match." D = vol1.shape[0] if mask is not None: assert vol1.shape == mask.shape, "Mask must have the same shape as the maps."

    if mask is not None:
        assert vol1.shape == mask.shape, "Mask must have the same shape as the maps."
        vol1 = np.multiply(vol1, mask)
        vol2 = np.multiply(vol2, mask)
        print("Volumes were masked!")

    # --- 1. Compute Fourier Transforms and shift origin to center ---
    ft1 = fft.fftshift(fft.fftn(vol1))
    ft2 = fft.fftshift(fft.fftn(vol2))

    # --- 2. Generate Fourier space coordinates in integer pixels ---
    D = vol1.shape[0]
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
    resolution_A = 1.0 / spatial_freq

    # --- 5. First-crossing with conditional bounce check relative to cutoff ---
    res_05, _ = first_crossing_with_bounce_check(
        fsc, resolution_A,
        threshold=0.5,
        cutoff_res_A=resolution_from_A,
        persistence=persistence_bins,
        rebound_window=rebound_window,
        eps=threshold_eps
    )
    res_0143, _ = first_crossing_with_bounce_check(
        fsc, resolution_A,
        threshold=0.143,
        cutoff_res_A=resolution_from_A,
        persistence=persistence_bins,
        rebound_window=rebound_window,
        eps=threshold_eps
    )

    return fsc, spatial_freq, resolution_A, (res_05, res_0143)


# ------------------------------- CLI wrapper ------------------------------ #

def cli():
    parser = argparse.ArgumentParser(
        description="Calculate Fourier Shell Correlation (FSC) between two MRC maps. ⚛️",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("map1", help="Path to the first input .mrc file (reference map for shape & sampling).")
    parser.add_argument("map2", help="Path to the second input .mrc file.")
    parser.add_argument("--mask", default=None, help="Optional path to a .mrc mask file (same grid as map1).")

    # Resizing / resampling options
    parser.add_argument("--resize_maps", action="store_true",
                        help="If maps differ in voxel size and/or shape, align map 2 to map 1.")
    parser.add_argument("--resize_mode", choices=["auto", "fourier-first", "realspace-only"],
                        default="auto",
                        help="Strategy to make map 2 match map 1 (see help text).")
    parser.add_argument("--interp_order", type=int, default=3,
                        help="Spline order for real-space interpolation (0..5).")
    parser.add_argument("--no_anti_alias", action="store_true",
                        help="Disable Gaussian anti-alias prefilter when downsampling.")

    # Plot / outputs
    parser.add_argument("--show_plot", action="store_true", help="Display the FSC plot interactively.")
    parser.add_argument("--save_plot", default=None, help="Path to save the FSC plot image (e.g., fsc_plot.png).")
    parser.add_argument("--save_csv", default=None,
                        help="Path to save the FSC data as CSV (e.g., fsc_data.csv).")

    # Crossing logic
    parser.add_argument("--resolution_from_A", type=float, default=15.0,
                        help="Gate for bounce checking: if the first crossing is worse than this, "
                             "it must pass the anti-bounce rule to be accepted.")
    parser.add_argument("--persistence_bins", type=int, default=3,
                        help="Require this many consecutive bins below the threshold when crossing occurs "
                             "worse than the gate.")
    parser.add_argument("--rebound_window", type=int, default=6,
                        help="Look-ahead window (bins) to ensure no immediate rebound above the threshold.")
    parser.add_argument("--threshold_eps", type=float, default=1e-3,
                        help="Small epsilon around FSC threshold to avoid jittery misclassification.")

    args = parser.parse_args()

    # --- Load Data ---
    print(f"Loading map 1: {args.map1}")
    try:
        with mrcfile.open(args.map1) as mrc:
            vol1 = mrc.data.copy().astype(np.float32)
            voxel_size1 = float(mrc.voxel_size.x.item())
            if voxel_size1 == 0:
                raise ValueError("Voxel size in map 1 header is 0. Please set it to a correct non-zero value.")
    except Exception as e:
        print(f"Error loading {args.map1}: {e}")
        return

    print(f"Loading map 2: {args.map2}")
    try:
        with mrcfile.open(args.map2) as mrc:
            vol2 = mrc.data.copy().astype(np.float32)
            voxel_size2 = float(mrc.voxel_size.x.item())
            if voxel_size2 == 0:
                print("   Warning: Voxel size in map 2 header is 0. Assuming map 1's voxel size for comparisons.")
                voxel_size2 = voxel_size1
    except Exception as e:
        print(f"Error loading {args.map2}: {e}")
        return

    mask_data = None
    if args.mask:
        print(f"Loading mask: {args.mask}")
        try:
            with mrcfile.open(args.mask) as mrc:
                mask_data = mrc.data.copy().astype(np.float32)
        except Exception as e:
            print(f"Error loading mask {args.mask}: {e}")
            return

    # --- Grid alignment (if requested) ---
    if args.resize_maps:
        if (vol1.shape != vol2.shape) or (not np.isclose(voxel_size1, voxel_size2, rtol=1e-6, atol=1e-6)):
            print("Aligning map 2 to map 1 grid (sampling rate and box size)...")
            vol2 = match_grid_to_reference(
                vol2,
                voxel_size2,
                ref_shape=vol1.shape,
                ref_voxel_size=voxel_size1,
                mode=args.resize_mode,
                interp_order=args.interp_order,
                anti_alias=(not args.no_anti_alias)
            )
            print(f"New map 2 shape: {vol2.shape}; voxel size now {voxel_size1:.4f} Å/px")
        else:
            print("Maps already share the same grid; no resizing performed.")
    else:
        # If not allowed to resize, enforce equality
        if vol1.shape != vol2.shape:
            print("   Error: Input maps must have the same shape. Use --resize_maps to align.")
            return
        if not np.isclose(voxel_size1, voxel_size2, rtol=1e-6, atol=1e-6):
            print("   Error: Input maps must have the same voxel size. Use --resize_maps to resample.")
            return

    # Use map 1's voxel size as the common reference for FSC axes
    voxel_size = voxel_size1

    # --- Computation ---
    print("Calculating FSC...")
    try:
        fsc, spatial_freq, resolution_A, (res_05, res_0143) = compute_fsc(
            vol1,
            vol2,
            voxel_size,
            mask=mask_data,
            allow_resize=False,  # grid is already aligned above
            resolution_from_A=args.resolution_from_A,
            persistence_bins=args.persistence_bins,
            rebound_window=args.rebound_window,
            threshold_eps=args.threshold_eps
        )
    except ValueError as e:
        print(f"   Error during FSC computation: {e}")
        return

    print("Calculation complete.")

    # --- Report Resolution ---
    print(f"Bounce-check gate (resolution_from_A): {args.resolution_from_A:.2f} Å")
    print(f"Resolution at FSC=0.143 ('gold-standard'): {res_0143:.3f} Å")
    print(f"Resolution at FSC=0.5                    : {res_05:.3f} Å")

    # --- Save CSV Data ---
    if args.save_csv:
        print(f"Saving data to {args.save_csv}...")
        data_to_save = np.vstack((spatial_freq, resolution_A, fsc)).T
        np.savetxt(
            args.save_csv, data_to_save, delimiter=',',
            header='SpatialFrequency_invA,Resolution_A,FSC', comments=''
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

        # Threshold lines
        ax1.axhline(y=0.5, color='gray', linestyle='--', linewidth=1)
        ax1.axhline(y=0.143, color='red', linestyle='--', linewidth=1)
        ax1.text(
            0.01, 0.143, "FSC = 0.143",
            transform=ax1.get_yaxis_transform(),  # x in axes coords, y in data coords
            ha="left", va="bottom", color="red",
            bbox=dict(facecolor="white", edgecolor="none", pad=2)
        )
        ax1.text(
            0.01, 0.5, "FSC = 0.5",
            transform=ax1.get_yaxis_transform(),  # x in axes coords, y in data coords
            ha="left", va="bottom", color="gray",
            bbox=dict(facecolor="white", edgecolor="none", pad=2)
        )
        # Vertical line to show cutoff/gate resolution
        resolution_threshold_freq = 1.0 / args.resolution_from_A
        # if resolution_threshold_freq <= np.max(spatial_freq):
            # ax1.axvline(x=resolution_threshold_freq, color='orange', linestyle=':', linewidth=1, alpha=0.7)
            # ax1.text(resolution_threshold_freq, 0.92, f' {args.resolution_from_A:.1f} Å gate',
            #          va='center', ha='left', color='orange', backgroundcolor='white', rotation=90, fontsize=9)

        # Annotate accepted crossings
        res05_plot, idx05 = first_crossing_with_bounce_check(
            fsc, resolution_A,
            threshold=0.5,
            cutoff_res_A=args.resolution_from_A,
            persistence=args.persistence_bins,
            rebound_window=args.rebound_window,
            eps=args.threshold_eps
        )
        res0143_plot, idx0143 = first_crossing_with_bounce_check(
            fsc, resolution_A,
            threshold=0.143,
            cutoff_res_A=args.resolution_from_A,
            persistence=args.persistence_bins,
            rebound_window=args.rebound_window,
            eps=args.threshold_eps
        )

        def mark(idx, txt, y=0.06):
            if idx is not None:
                ax1.axvline(x=spatial_freq[idx], color='black', linestyle='-', linewidth=1, alpha=0.6)
                ax1.text(spatial_freq[idx], y, f' {txt}', va='bottom', ha='left', color='black',
                         backgroundcolor='white', rotation=90, fontsize=9)

        mark(idx05,   f'{res05_plot:.2f} Å', y=0.8)
        mark(idx0143, f'{res0143_plot:.2f} Å', y=0.8)

        # Top axis: resolution (Å)
        ax2 = ax1.twiny()
        ax2.set_xlabel("Resolution (Å)")
        ticks = ax1.get_xticks()
        tick_labels = [f"{1.0 / t:.1f}" if t > 0 else "∞" for t in ticks]
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
