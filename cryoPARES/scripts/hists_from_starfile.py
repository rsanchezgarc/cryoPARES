#!/usr/bin/env python3

import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import starfile
from typing import Optional, List


def plot_star_histograms(
        input: str,
        cols: List[str],
        output: Optional[str] = None,
        nbins: str = 'auto',
        clip_percentile: Optional[List[float]] = None,
        clip_value: Optional[List[float]] = None
):
    """
    Load a STAR file and display histograms of selected metadata columns.

    :param input: Path to the input STAR file
    :param cols: One or more metadata column names to plot
    :param output: Optional path to save the output plot image
    :param nbins: Number of bins for the histogram (e.g., 500). Default: 'auto'
    :param clip_percentile: Clip data to percentile range. Provide pairs of (lower, upper) values. Example for one column: 1,99. Example for two columns: 1,99,5,95
    :param clip_value: Clip data to an absolute value range. Provide pairs of (lower, upper) values. Example: -1000,1000
    """
    # Validation: ensure mutually exclusive options
    if clip_percentile is not None and clip_value is not None:
        print(f"❌ Error: Cannot use both --clip_percentile and --clip_value simultaneously", file=sys.stderr)
        sys.exit(1)

    # Validate clipping arguments match number of columns
    num_cols = len(cols)
    if clip_percentile and len(clip_percentile) != 2 * num_cols:
        print(f"❌ Error: --clip_percentile requires 2 values per column. You provided {len(clip_percentile)} values for {num_cols} column(s).", file=sys.stderr)
        sys.exit(1)
    if clip_value and len(clip_value) != 2 * num_cols:
        print(f"❌ Error: --clip_value requires 2 values per column. You provided {len(clip_value)} values for {num_cols} column(s).", file=sys.stderr)
        sys.exit(1)

    star_file_path = input
    column_names = cols
    output_path = output
    # --- File Loading ---
    try:
        data = starfile.read(star_file_path, always_dict=True)
        key_found = 'particles' if 'particles' in data else list(data.keys())[0]
        df = data[key_found]
    except Exception as e:
        print(f"❌ Error reading STAR file '{star_file_path}': {e}", file=sys.stderr)
        sys.exit(1)

    # --- Column Validation ---
    missing_cols = [col for col in column_names if col not in df.columns]
    if missing_cols:
        print(f"❌ Error: Columns not found: {', '.join(missing_cols)}", file=sys.stderr)
        sys.exit(1)

    # --- Plotting Setup ---
    num_plots = len(column_names)
    ncols = min(3, num_plots)
    nrows = math.ceil(num_plots / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)
    axes = axes.flatten()

    try:
        plot_bins = int(nbins)
    except ValueError:
        plot_bins = nbins

    # --- Main Loop for Plotting Each Column ---
    for i, col_name in enumerate(column_names):
        ax = axes[i]
        column_data = df[col_name].dropna()

        # 1. Print statistics on the original data
        print(f"{col_name}:")
        print(f"  mean: {np.mean(column_data)}")
        print(f"  median: {np.median(column_data)}")
        print(f"  min: {np.min(column_data)}")
        print(f"  max: {np.max(column_data)}")
        print(f"  std: {np.std(column_data)}")
        print("-------------------------")

        plot_data = column_data.copy()
        plot_title = col_name

        # 2. Apply data clipping using np.clip if requested
        if clip_percentile:
            low_p, high_p = clip_percentile[2 * i], clip_percentile[2 * i + 1]
            low_val = plot_data.quantile(low_p / 100)
            high_val = plot_data.quantile(high_p / 100)
            plot_data = np.clip(plot_data, low_val, high_val)
            plot_title += f'\n(Clipped to {low_p}-{high_p} percentile)'

        elif clip_value:
            low_v, high_v = clip_value[2 * i], clip_value[2 * i + 1]
            plot_data = np.clip(plot_data, low_v, high_v)
            plot_title += f'\n(Clipped to [{low_v}, {high_v}])'

        # 3. Plot the histogram
        total_count = len(plot_data)
        ax.hist(plot_data, bins=plot_bins, color='skyblue', edgecolor='black')

        # 4. Create and configure the secondary y-axis for percentages
        ax_right = ax.twinx()
        ax_right.set_ylim(ax.get_ylim())  # Ensure ticks are aligned
        ax_right.yaxis.set_major_formatter(PercentFormatter(xmax=total_count))
        ax_right.set_ylabel('Percentage (%)')

        # Configure main plot labels
        ax.set_xlabel("Value")
        ax.set_ylabel("Counts")
        ax.set_title(plot_title, fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.5)

    # --- Finalize and Show/Save Plot ---
    for j in range(num_plots, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('STAR File Metadata Distribution', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"✅ Plot saved to '{output_path}'")
    else:
        plt.show()


def main():
    from argParseFromDoc import parse_function_and_call
    parse_function_and_call(plot_star_histograms)


if __name__ == "__main__":
    main()