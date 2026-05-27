#!/usr/bin/env python3
# TODO: Not experimentally validated. Reconstruct maps before/after and verify visually.
import argparse
import os

import numpy as np
import pandas as pd
import starfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R


def get_symmetry_rotations(sym_group):
    """
    Generate scipy Rotation objects for C<n> and D<n> point groups.
    Returns:
        ops, group_type, n
    """
    sym_group = sym_group.upper().strip()

    if sym_group == "C1":
        return [R.identity()], "C", 1

    group_type = sym_group[0]
    try:
        n = int(sym_group[1:])
    except ValueError:
        raise ValueError(f"Unsupported symmetry group format: {sym_group}. Use C<n> or D<n>.")

    if n < 1:
        raise ValueError(f"Symmetry order must be >= 1, got {n} in {sym_group}.")

    c_ops = []
    for i in range(n):
        angle = 2.0 * np.pi * i / n
        c_ops.append(R.from_rotvec([0.0, 0.0, angle]))

    if group_type == "C":
        return c_ops, "C", n

    if group_type == "D":
        flip_x = R.from_rotvec([np.pi, 0.0, 0.0])
        d_ops = []
        for c_op in c_ops:
            d_ops.append(c_op)
            d_ops.append(c_op * flip_x)
        return d_ops, "D", n

    raise NotImplementedError(f"Symmetry {sym_group} not implemented. Supported: C<n>, D<n>.")


def angles_to_vector(rot_deg, tilt_deg):
    """
    Convert RELION Rot/Tilt (degrees) into a 3D unit viewing vector.
    """
    rot = np.radians(rot_deg)
    tilt = np.radians(tilt_deg)

    x = np.sin(tilt) * np.cos(rot)
    y = np.sin(tilt) * np.sin(rot)
    z = np.cos(tilt)

    v = np.array([x, y, z], dtype=float)
    nrm = np.linalg.norm(v)
    if nrm == 0:
        raise ValueError("Zero-length vector generated from angles.")
    return v / nrm


def vector_to_angles(v):
    """
    Convert a 3D unit vector into (rot_deg, tilt_deg).
    rot in [0, 360), tilt in [0, 180].
    """
    v = np.asarray(v, dtype=float)
    v = v / np.linalg.norm(v)

    x, y, z = v
    z = np.clip(z, -1.0, 1.0)

    tilt = np.degrees(np.arccos(z))
    rot = np.degrees(np.arctan2(y, x)) % 360.0
    return rot, tilt


def canonicalize_to_asu(v, sym_ops, group_type, n):
    """
    Map a viewing direction to a canonical representative in the symmetry ASU.
    """
    sector = 2.0 * np.pi / n
    half_sector = 0.5 * sector

    copies = np.array([op.apply(v) for op in sym_ops], dtype=float)

    xs = copies[:, 0]
    ys = copies[:, 1]
    zs = np.clip(copies[:, 2], -1.0, 1.0)

    phis = np.arctan2(ys, xs)
    phi_centered = (phis + half_sector) % sector - half_sector

    order = np.lexsort((np.abs(phis), np.abs(phi_centered), -zs))
    best = order[0]

    return copies[best] / np.linalg.norm(copies[best])


def canonicalize_many_to_asu(vectors, sym_ops, group_type, n):
    """
    Apply canonicalize_to_asu to an array of vectors, shape (N, 3).
    """
    out = np.empty_like(vectors, dtype=float)
    for i, v in enumerate(vectors):
        out[i] = canonicalize_to_asu(v, sym_ops, group_type, n)
    return out


def symmetry_aware_angular_distance(vectors, target, sym_ops):
    """
    Compute the minimum angular distance in degrees between each vector 
    and all symmetry copies of the target.
    
    vectors: shape (N, 3)
    target: shape (3,)
    sym_ops: list of scipy Rotation objects
    """
    target_copies = np.array([op.apply(target) for op in sym_ops], dtype=float)
    dots = np.clip(vectors @ target_copies.T, -1.0, 1.0)
    max_dots = np.max(dots, axis=1)
    return np.degrees(np.arccos(max_dots))


def make_relion_like_angular_plot(vectors_asu, outfile, title="", bin_deg=5.0):
    """
    Create a RELION-like angular distribution plot from ASU-reduced vectors.
    """
    rots = []
    tilts = []
    for v in vectors_asu:
        rot, tilt = vector_to_angles(v)
        rots.append(rot)
        tilts.append(tilt)

    rots = np.asarray(rots, dtype=float)
    tilts = np.asarray(tilts, dtype=float)

    # Shift rotations to [-180, 180) for a continuous plot across the 0 boundary
    rots = np.where(rots >= 180.0, rots - 360.0, rots)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="polar")

    if len(rots) == 0:
        ax.set_title(title + "\n(no particles)")
        fig.tight_layout()
        fig.savefig(outfile, dpi=200)
        plt.close(fig)
        return

    rot_bins = np.arange(-180.0, 180.0 + bin_deg, bin_deg)
    tilt_max = max(90.0, np.ceil(np.max(tilts) / bin_deg) * bin_deg)
    tilt_bins = np.arange(0.0, tilt_max + bin_deg, bin_deg)

    H, rot_edges, tilt_edges = np.histogram2d(rots, tilts, bins=[rot_bins, tilt_bins])

    rot_centers = 0.5 * (rot_edges[:-1] + rot_edges[1:])
    tilt_centers = 0.5 * (tilt_edges[:-1] + tilt_edges[1:])

    rr, tt = np.meshgrid(np.radians(rot_centers), tilt_centers, indexing="ij")
    counts = H.ravel()

    mask = counts > 0
    rr = rr.ravel()[mask]
    tt = tt.ravel()[mask]
    counts = counts[mask]

    sizes = 20.0 + 280.0 * counts / counts.max()
    sc = ax.scatter(rr, tt, c=counts, s=sizes, alpha=0.85)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(135)
    ax.set_ylim(0, tilt_max)
    ax.set_title(title, pad=20)

    cbar = fig.colorbar(sc, ax=ax, pad=0.08)
    cbar.set_label("Particles per angular bin")

    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)


def load_star_as_dataframe(star_path):
    """
    Read a STAR file and return: ds, block_name, df
    """
    ds = starfile.read(star_path)

    if isinstance(ds, pd.DataFrame):
        return ds, None, ds.copy()

    if isinstance(ds, dict):
        block_name = "particles" if "particles" in ds else list(ds.keys())[0]
        return ds, block_name, ds[block_name].copy()

    raise TypeError(f"Unsupported object returned by starfile.read(): {type(ds)}")


def save_star(ds, block_name, df_final, output_path):
    """
    Save updated STAR content.
    """
    if block_name is None:
        starfile.write(df_final, output_path, overwrite=True)
    else:
        ds[block_name] = df_final
        starfile.write(ds, output_path, overwrite=True)


def dataframe_to_vectors(df):
    """
    Convert DataFrame columns rlnAngleRot / rlnAngleTilt into shape (N, 3) vectors.
    """
    rots = df["rlnAngleRot"].to_numpy(dtype=float)
    tilts = df["rlnAngleTilt"].to_numpy(dtype=float)

    rot_rads = np.radians(rots)
    tilt_rads = np.radians(tilts)

    px = np.sin(tilt_rads) * np.cos(rot_rads)
    py = np.sin(tilt_rads) * np.sin(rot_rads)
    pz = np.cos(tilt_rads)

    return np.column_stack((px, py, pz))


def main():
    parser = argparse.ArgumentParser(
        description="Simulate preferred orientation in RELION STAR files by depleting views."
    )
    parser.add_argument("-i", "--input", required=True, help="Input RELION .star file")
    parser.add_argument("-o", "--output", required=True, help="Output RELION .star file")
    parser.add_argument("--sym", default="C1", help="Symmetry group, e.g. C1, C3, D3, D7")
    parser.add_argument("--target_tilt", type=float, required=True,
                        help="Tilt angle of the target view (degrees)")
    parser.add_argument("--target_rot", type=float, default=0.0,
                        help="Rot angle of the target view (degrees)")
    parser.add_argument("--tolerance", type=float, default=15.0,
                        help="Angular radius of depletion cone in ASU (degrees)")
    parser.add_argument("--fraction", type=float, default=0.8,
                        help="Fraction of particles to REMOVE inside cone (0.0 to 1.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible subsampling")
    parser.add_argument("--plot", action="store_true",
                        help="Also create RELION-like angular distribution plots before and after")
    parser.add_argument("--plot_bin", type=float, default=5.0,
                        help="Angular bin size in degrees for the plots")

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input STAR file not found: {args.input}")

    if os.path.isdir(args.output):
        raise ValueError(
            f"--output must be a file path, not a directory: {args.output}\n"
            "Example: -o /path/to/output/depleted_particles.star"
        )

    if not (0.0 <= args.fraction <= 1.0):
        raise ValueError("--fraction must be between 0.0 and 1.0")

    if not (0.0 <= args.tolerance <= 180.0):
        raise ValueError("--tolerance must be between 0 and 180 degrees")

    if args.plot_bin <= 0:
        raise ValueError("--plot_bin must be > 0")

    outdir = os.path.dirname(os.path.abspath(args.output))
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    print(f"Loading {args.input}...")
    ds, block_name, df = load_star_as_dataframe(args.input)

    required_cols = ["rlnAngleRot", "rlnAngleTilt"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in STAR file: {missing}")

    sym_ops, group_type, n = get_symmetry_rotations(args.sym)

    print(f"Using symmetry = {args.sym}")
    print(f"Using target view: rot = {args.target_rot:.2f} deg, tilt = {args.target_tilt:.2f} deg")
    print(f"Using tolerance = {args.tolerance:.2f} deg")
    print(f"Removing fraction = {args.fraction:.4f}")

    particle_vectors = dataframe_to_vectors(df)

    if args.plot:
        print("Reducing particle directions to the symmetry asymmetric unit for plotting...")
        particle_vectors_asu = canonicalize_many_to_asu(particle_vectors, sym_ops, group_type, n)
        stem = os.path.splitext(args.output)[0]
        plot_before = stem + "_angular_dist_before.png"
        print(f"Writing angular distribution plot before depletion to {plot_before}")
        make_relion_like_angular_plot(
            particle_vectors_asu,
            plot_before,
            title=f"Angular distribution before depletion ({args.sym} ASU)",
            bin_deg=args.plot_bin,
        )

    target_vec = angles_to_vector(args.target_rot, args.target_tilt)
    
    print("Calculating symmetry-aware angular distances...")
    min_angles = symmetry_aware_angular_distance(particle_vectors, target_vec, sym_ops)
    mask_in_cone = min_angles <= args.tolerance

    df_safe = df.loc[~mask_in_cone]
    df_target = df.loc[mask_in_cone]

    print(f"Total particles: {len(df)}")
    print(f"Particles falling within {args.tolerance:.1f} degrees of target views: {len(df_target)}")

    keep_fraction = 1.0 - args.fraction
    if len(df_target) == 0 or keep_fraction == 1.0:
        df_target_reduced = df_target
    elif keep_fraction == 0.0:
        df_target_reduced = df_target.iloc[0:0]
    else:
        df_target_reduced = df_target.sample(frac=keep_fraction, random_state=args.seed)

    df_final = pd.concat([df_safe, df_target_reduced]).sort_index()

    print(f"Removed {len(df_target) - len(df_target_reduced)} particles.")
    print(f"Particles remaining: {len(df_final)}")

    save_star(ds, block_name, df_final, args.output)
    print(f"Done. Saved to {args.output}")

    if args.plot:
        particle_vectors_after = dataframe_to_vectors(df_final)
        particle_vectors_after_asu = canonicalize_many_to_asu(
            particle_vectors_after, sym_ops, group_type, n
        )

        stem = os.path.splitext(args.output)[0]
        plot_after = stem + "_angular_dist_after.png"
        print(f"Writing angular distribution plot after depletion to {plot_after}")
        make_relion_like_angular_plot(
            particle_vectors_after_asu,
            plot_after,
            title=f"Angular distribution after depletion ({args.sym} ASU)",
            bin_deg=args.plot_bin,
        )

if __name__ == "__main__":
    main()
