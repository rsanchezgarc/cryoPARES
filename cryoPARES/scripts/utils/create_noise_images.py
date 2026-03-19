#!/usr/bin/env python3
"""
RELION 3.1+ STAR: copy all metadata, but replace rlnImageName images with
random Gaussian N(mu, std) particles, mirroring the original .mrcs stack layout.

Key points:
- Does NOT read input stack files. Box size is taken from the STAR optics table.
- Only edits the particles block column rlnImageName.
- For each unique input stack referenced in rlnImageName, creates a matching
  output stack under --out-dir, preserving relative path structure (if original
  paths are relative). Absolute input paths are flattened to basenames.
- Output stack length per input stack is max(index) seen for that stack.
- Optics groups are preserved and used to determine box size.
- Seed and stats are settable via CLI.
- Option to write absolute paths in the out_star.

Deps:
  pip install numpy pandas starfile mrcfile
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any, Iterable

import numpy as np

try:
    import starfile  # https://github.com/teamtomo/starfile
except ImportError as e:
    raise SystemExit("Missing dependency 'starfile'. Install with: pip install starfile") from e

try:
    import mrcfile
except ImportError as e:
    raise SystemExit("Missing dependency 'mrcfile'. Install with: pip install mrcfile") from e


IMG_RE = re.compile(r"^\s*(\d+)\s*@\s*(.+?)\s*$")


def parse_relion_image_name(val: str) -> Tuple[int, str]:
    """Parse RELION rlnImageName entries like '000123@path/to/stack.mrcs'."""
    if not isinstance(val, str):
        raise ValueError(f"rlnImageName value is not a string: {val!r}")
    m = IMG_RE.match(val)
    if not m:
        raise ValueError(f"Could not parse rlnImageName entry: {val!r} (expected '####@stack.mrcs')")
    return int(m.group(1)), m.group(2).strip()


def output_stack_path(stack_path: str, out_dir: str) -> str:
    """
    Mirror original layout under out_dir:
      - if original path is relative: keep its relative subdirectories under out_dir
      - if original path is absolute: flatten to basename under out_dir
    """
    if os.path.isabs(stack_path):
        rel = os.path.basename(stack_path)
    else:
        rel = os.path.normpath(stack_path)
    return os.path.normpath(os.path.join(out_dir, rel))


def ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


@dataclass(frozen=True)
class StackPlan:
    out_path: str
    n_images: int
    ny: int
    nx: int


def find_block_with_column(blocks: Dict[str, Any], col: str) -> str:
    for name, df in blocks.items():
        if hasattr(df, "columns") and col in df.columns:
            return name
    raise SystemExit(f"Could not find a STAR block containing column '{col}'.")


def infer_box_from_optics(optics_df, optics_group: int) -> Tuple[int, int]:
    """
    Determine (ny, nx) for a given optics group.
    Supported:
      - rlnImageSizeX + rlnImageSizeY
      - rlnImageSize (assumed square)
    """
    # optics_group is usually int; be robust if it's float-like in pandas
    og = int(optics_group)

    if "rlnOpticsGroup" not in optics_df.columns:
        raise ValueError("Optics table missing rlnOpticsGroup column.")

    row = optics_df.loc[optics_df["rlnOpticsGroup"].astype(int) == og]
    if len(row) != 1:
        raise ValueError(f"Expected exactly 1 optics row for rlnOpticsGroup={og}, found {len(row)}")

    row = row.iloc[0]

    if "rlnImageSizeX" in optics_df.columns and "rlnImageSizeY" in optics_df.columns:
        nx = int(row["rlnImageSizeX"])
        ny = int(row["rlnImageSizeY"])
        return ny, nx

    if "rlnImageSize" in optics_df.columns:
        n = int(row["rlnImageSize"])
        return n, n

    raise ValueError(
        "Optics table must contain either (rlnImageSizeX,rlnImageSizeY) or rlnImageSize."
    )


def write_gaussian_stack(
    out_path: str,
    n: int,
    ny: int,
    nx: int,
    rng: np.random.Generator,
    mu: float,
    std: float,
    overwrite: bool,
) -> None:
    if std <= 0:
        raise ValueError(f"--std must be > 0 (got {std})")
    if (not overwrite) and os.path.exists(out_path):
        raise FileExistsError(f"Refusing to overwrite existing file: {out_path} (use --overwrite)")

    ensure_parent_dir(out_path)

    data = rng.normal(loc=mu, scale=std, size=(n, ny, nx)).astype(np.float32)

    with mrcfile.new(out_path, overwrite=True) as m:
        m.set_data(data)
        m.update_header_from_data()
        m.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("in_star", help="Input RELION 3.1+ .star")
    ap.add_argument("out_star", help="Output .star (metadata copied; rlnImageName rewritten)")
    ap.add_argument("--out-dir", required=True, help="Directory root where mirrored Gaussian .mrcs stacks are written")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed (optional, for reproducibility)")
    ap.add_argument("--mu", type=float, default=0.0, help="Gaussian mean (default 0)")
    ap.add_argument("--std", type=float, default=1.0, help="Gaussian stddev (default 1)")
    ap.add_argument("--overwrite", action="store_true", help="Allow overwriting output .mrcs files if they exist")
    ap.add_argument(
        "--abs-paths",
        action="store_true",
        help="Write absolute stack paths in out_star (default: paths relative to out_star location)",
    )
    args = ap.parse_args()

    in_star = os.path.abspath(args.in_star)
    out_star = os.path.abspath(args.out_star)
    out_dir = os.path.abspath(args.out_dir)
    out_star_dir = os.path.dirname(out_star)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_star), exist_ok=True)

    star_obj = starfile.read(in_star)

    # Normalize to dict-of-blocks
    if hasattr(star_obj, "columns"):  # single block
        blocks: Dict[str, Any] = {"data": star_obj}
    else:
        blocks = dict(star_obj)

    optics_block = find_block_with_column(blocks, "rlnOpticsGroup")
    # But particles also has rlnOpticsGroup; we specifically want data_optics.
    # Heuristic: prefer a block whose name contains "optics" and has rlnImageSize*.
    candidate_optics = []
    for name, df in blocks.items():
        if hasattr(df, "columns") and "rlnOpticsGroup" in df.columns:
            if ("rlnImageSize" in df.columns) or (("rlnImageSizeX" in df.columns) and ("rlnImageSizeY" in df.columns)):
                if "optics" in name.lower():
                    candidate_optics.append(name)
    if candidate_optics:
        optics_block = candidate_optics[0]

    optics_df = blocks[optics_block].copy()

    particles_block = find_block_with_column(blocks, "rlnImageName")
    particles_df = blocks[particles_block].copy()

    if "rlnOpticsGroup" not in particles_df.columns:
        raise SystemExit("Particles table missing rlnOpticsGroup column (needed to infer image size from optics).")

    # Plan per input stack: output path + n_images + (ny,nx)
    # n_images is max index seen for that stack.
    image_entries = particles_df["rlnImageName"].tolist()

    # Build stats for each stack
    stack_max_index: Dict[str, int] = {}
    stack_optics_groups: Dict[str, set[int]] = {}

    for v in image_entries:
        idx, stack_path = parse_relion_image_name(v)
        stack_max_index[stack_path] = max(stack_max_index.get(stack_path, 0), idx)

    # Associate each particle row's stack with its optics group, to infer size.
    for v, og in zip(image_entries, particles_df["rlnOpticsGroup"].tolist()):
        _, stack_path = parse_relion_image_name(v)
        stack_optics_groups.setdefault(stack_path, set()).add(int(og))

    plans: Dict[str, StackPlan] = {}
    for stack_path, max_idx in stack_max_index.items():
        ogs = stack_optics_groups.get(stack_path, set())
        if len(ogs) != 1:
            raise SystemExit(
                f"Stack {stack_path!r} is associated with multiple optics groups {sorted(ogs)}. "
                "Refusing because image size may be ambiguous."
            )
        og = next(iter(ogs))
        ny, nx = infer_box_from_optics(optics_df, og)
        out_path = output_stack_path(stack_path, out_dir)
        plans[stack_path] = StackPlan(out_path=out_path, n_images=max_idx, ny=ny, nx=nx)

    # Generate Gaussian stacks
    rng = np.random.default_rng(args.seed)
    for stack_path, plan in plans.items():
        write_gaussian_stack(
            plan.out_path,
            plan.n_images,
            plan.ny,
            plan.nx,
            rng=rng,
            mu=args.mu,
            std=args.std,
            overwrite=args.overwrite,
        )

    # Rewrite rlnImageName to point at the new stacks
    def rewrite_entry(val: str) -> str:
        idx, old_stack = parse_relion_image_name(val)
        new_abs = plans[old_stack].out_path

        if args.abs_paths:
            new_path_in_star = new_abs
        else:
            new_path_in_star = os.path.relpath(new_abs, out_star_dir)

        return f"{idx:06d}@{new_path_in_star}"

    particles_df["rlnImageName"] = [rewrite_entry(v) for v in image_entries]
    blocks[particles_block] = particles_df

    # Write output STAR (optics block preserved)
    starfile.write(blocks, out_star, overwrite=True)

    print("OK")
    print(f"  in_star:   {in_star}")
    print(f"  out_star:  {out_star}")
    print(f"  out_dir:   {out_dir}")
    print(f"  optics block:    {optics_block}")
    print(f"  particles block: {particles_block}")
    print(f"  stacks written:  {len(plans)}")
    if args.seed is not None:
        print(f"  seed: {args.seed}")
    print(f"  mu/std: {args.mu}/{args.std}")
    print(f"  paths in out_star: {'ABSOLUTE' if args.abs_paths else 'RELATIVE'}")


if __name__ == "__main__":
    main()

