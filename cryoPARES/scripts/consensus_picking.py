#!/usr/bin/env python3
"""
Compare two RELION 3.1+ STAR files by micrograph name + particle coordinates.

Default operation:
  intersection  -> keep rows from star_fname1 that have a match in star_fname2

Other operation:
  difference    -> keep rows from star_fname1 that do NOT have a match in star_fname2

Matching rule:
  - same micrograph basename
  - |x1*scale1 - x2*scale2| <= margin   (in a common coordinate space)
  - |y1*scale1 - y2*scale2| <= margin

--scale1 / --scale2 let you bring both files into the same pixel space.
Example: if star_fname1 is at bin1 and star_fname2 is at bin4, pass
  --scale1 1 --scale2 4
so that star_fname2 coordinates are multiplied by 4 before comparison.
"""

from __future__ import annotations

import argparse
import os
import warnings
from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import starfile


MICROGRAPH_COL_CANDIDATES = ["rlnMicrographName", "rlnImageName"]
COORD_COL_CANDIDATES = [
    ("rlnCoordinateX", "rlnCoordinateY"),
    ("rlnMicrographCoordinatesX", "rlnMicrographCoordinatesY"),
]


def find_particle_table(star_obj) -> Tuple[pd.DataFrame, Optional[str]]:
    if isinstance(star_obj, pd.DataFrame):
        return star_obj.reset_index(drop=True), None

    if isinstance(star_obj, dict):
        for key, df in star_obj.items():
            if isinstance(df, pd.DataFrame) and has_required_columns(df):
                return df.reset_index(drop=True), key

    raise ValueError(
        "Could not find a particle table with micrograph and coordinate columns."
    )


def has_required_columns(df: pd.DataFrame) -> bool:
    micrograph_ok = any(col in df.columns for col in MICROGRAPH_COL_CANDIDATES)
    coord_ok = any(
        x_col in df.columns and y_col in df.columns
        for x_col, y_col in COORD_COL_CANDIDATES
    )
    return micrograph_ok and coord_ok


def detect_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    micrograph_col = next(
        (c for c in MICROGRAPH_COL_CANDIDATES if c in df.columns), None
    )
    if micrograph_col is None:
        raise ValueError(
            f"Could not find a micrograph-name column. Tried: {MICROGRAPH_COL_CANDIDATES}"
        )

    coord_pair = next(
        (
            (x, y)
            for x, y in COORD_COL_CANDIDATES
            if x in df.columns and y in df.columns
        ),
        None,
    )
    if coord_pair is None:
        raise ValueError(
            f"Could not find coordinate columns. Tried: {COORD_COL_CANDIDATES}"
        )

    return micrograph_col, coord_pair[0], coord_pair[1]


def extract_micrograph_basename(series: pd.Series) -> pd.Series:
    """
    Normalise micrograph identifiers to a bare filename stem, handling:
      - full paths:          /data/.../Foo_0001.mrc        -> Foo_0001.mrc
      - relative paths:      raw_data/.../Foo_0001.mrc     -> Foo_0001.mrc
      - bare names:          Foo_0001.mrc                  -> Foo_0001.mrc
      - stack references:    000006@raw_data/.../Foo.mrcs  -> Foo.mrcs
        (the @-prefix particle index is stripped first)
    """
    def _normalise(val: str) -> str:
        s = str(val)
        # Strip leading stack index (e.g. "000006@some/path/file.mrcs")
        if "@" in s:
            s = s.split("@", 1)[1]
        return os.path.basename(s)

    return series.map(_normalise)


def prepare_for_matching(
    df: pd.DataFrame,
    micrograph_col: str,
    x_col: str,
    y_col: str,
    scale: float,
    bin_size: float,
) -> pd.DataFrame:
    out = df[[micrograph_col, x_col, y_col]].copy()
    out = out.rename(
        columns={
            micrograph_col: "__micrograph",
            x_col: "__x_raw",
            y_col: "__y_raw",
        }
    )

    out["__micrograph_key"] = extract_micrograph_basename(out["__micrograph"])

    out["__x_raw"] = pd.to_numeric(out["__x_raw"], errors="coerce")
    out["__y_raw"] = pd.to_numeric(out["__y_raw"], errors="coerce")

    if out["__x_raw"].isna().any() or out["__y_raw"].isna().any():
        raise ValueError("Found non-numeric coordinate values in one of the STAR files.")

    # Scale into the common coordinate space.
    out["__x"] = out["__x_raw"] * scale
    out["__y"] = out["__y_raw"] * scale

    out["__row_id"] = np.arange(len(out), dtype=np.int64)
    out["__bin_x"] = np.floor(out["__x"] / bin_size).astype(np.int64)
    out["__bin_y"] = np.floor(out["__y"] / bin_size).astype(np.int64)
    return out


def match_rows(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    margin: float,
    operation: str,
    scale1: float,
    scale2: float,
) -> pd.DataFrame:
    mg1, x1, y1 = detect_columns(df1)
    mg2, x2, y2 = detect_columns(df2)

    bin_size = max(1.0, float(margin))

    a = prepare_for_matching(df1, mg1, x1, y1, scale1, bin_size)
    b = prepare_for_matching(df2, mg2, x2, y2, scale2, bin_size)

    # Warn if basename collision is detected in either file.
    for label, frame in (("star_fname1", a), ("star_fname2", b)):
        n_full = frame["__micrograph"].nunique()
        n_base = frame["__micrograph_key"].nunique()
        if n_base < n_full:
            warnings.warn(
                f"{label}: {n_full} unique micrograph paths collapsed to "
                f"{n_base} unique basenames. Matching may be ambiguous.",
                stacklevel=2,
            )

    # Print a sample of resolved keys to help the user sanity-check.
    sample_a = a["__micrograph_key"].iloc[:3].tolist()
    sample_b = b["__micrograph_key"].iloc[:3].tolist()
    print(f"  star_fname1 micrograph keys (first 3): {sample_a}")
    print(f"  star_fname2 micrograph keys (first 3): {sample_b}")

    common_keys = set(a["__micrograph_key"].unique()) & set(b["__micrograph_key"].unique())
    print(f"  Micrographs in common (by basename): {len(common_keys)}")
    if not common_keys:
        warnings.warn(
            "No micrograph basenames are shared between the two files. "
            "The result will be empty. Check that both files refer to the same dataset.",
            stacklevel=2,
        )

    # Expand b into neighbouring bins so pairs within margin can still meet.
    expanded_b_parts = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            tmp = b.copy()
            tmp["__bin_x"] += dx
            tmp["__bin_y"] += dy
            expanded_b_parts.append(tmp)

    b_expanded = pd.concat(expanded_b_parts, ignore_index=True)

    merged = a.merge(
        b_expanded,
        on=["__micrograph_key", "__bin_x", "__bin_y"],
        how="inner",
        suffixes=("_1", "_2"),
    )

    if merged.empty:
        matched_ids = np.array([], dtype=np.int64)
    else:
        close_enough = (
            (merged["__x_1"] - merged["__x_2"]).abs() <= margin
        ) & (
            (merged["__y_1"] - merged["__y_2"]).abs() <= margin
        )
        matched_ids = pd.unique(merged.loc[close_enough, "__row_id_1"])

    keep_mask = df1.reset_index(drop=True).index.isin(matched_ids)

    if operation == "intersection":
        result = df1.reset_index(drop=True).loc[keep_mask].copy()
    elif operation == "difference":
        result = df1.reset_index(drop=True).loc[~keep_mask].copy()
    else:
        raise ValueError("operation must be either 'intersection' or 'difference'.")

    return result.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Intersect or subtract RELION 3.1+ STAR files using micrograph "
            "basename and particle coordinates, with optional rescaling."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Scale factors
-------------
Use --scale1 / --scale2 to bring both files into the same pixel space before
comparing coordinates.  The comparison is done in units of (raw_coord * scale).

Example: star_fname1 is full-resolution (bin1), star_fname2 is 4x downsampled.
  --scale1 1 --scale2 4

Example: both files are at bin2 relative to the raw micrographs.
  --scale1 1 --scale2 1   (default, no rescaling needed)

The --margin is applied in the scaled (common) coordinate space.
""",
    )
    parser.add_argument("--star_fname1", required=True, help="First input STAR file")
    parser.add_argument("--star_fname2", required=True, help="Second input STAR file")
    parser.add_argument(
        "--out_fname",
        default=None,
        help="Optional output STAR file to write the result",
    )
    parser.add_argument(
        "--operation",
        choices=["intersection", "difference"],
        default="intersection",
        help="Result to compute relative to star_fname1 (default: intersection)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help=(
            "Allowed coordinate difference (in the common/scaled pixel space) "
            "in both x and y (default: 0)"
        ),
    )
    parser.add_argument(
        "--scale1",
        type=float,
        default=1.0,
        help="Multiply star_fname1 coordinates by this factor (default: 1.0)",
    )
    parser.add_argument(
        "--scale2",
        type=float,
        default=1.0,
        help="Multiply star_fname2 coordinates by this factor (default: 1.0)",
    )
    args = parser.parse_args()

    if args.scale1 <= 0 or args.scale2 <= 0:
        raise ValueError("--scale1 and --scale2 must be positive.")

    star1 = starfile.read(args.star_fname1)
    star2 = starfile.read(args.star_fname2)

    df1, key1 = find_particle_table(star1)
    df2, key2 = find_particle_table(star2)

    print(f"Particles in {args.star_fname1}: {len(df1)}")
    print(f"Particles in {args.star_fname2}: {len(df2)}")
    print(f"Coordinate scale factors: star_fname1 x{args.scale1}, star_fname2 x{args.scale2}")

    result_df = match_rows(
        df1, df2,
        margin=args.margin,
        operation=args.operation,
        scale1=args.scale1,
        scale2=args.scale2,
    )

    print(f"Result size ({args.operation}): {len(result_df)}")

    if args.out_fname is not None:
        if isinstance(star1, pd.DataFrame):
            out_obj = result_df
        else:
            out_obj = deepcopy(star1)
            if key1 is None:
                raise RuntimeError("Internal error: expected a particle-table key.")
            out_obj[key1] = result_df

        starfile.write(out_obj, args.out_fname, overwrite=True)
        print(f"Wrote: {args.out_fname}")


if __name__ == "__main__":
    main()
