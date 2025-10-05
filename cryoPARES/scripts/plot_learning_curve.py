import os.path
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def calculate_robust_ylim(data, percentile_low=5, percentile_high=95):
    """
    Calculate y-limits based on percentiles to avoid spike influence
    """
    if len(data) == 0:
        return None, None

    # For log scale, ensure we don't have negative or zero values
    if np.any(data <= 0):
        data = data[data > 0]
        if len(data) == 0:
            return None, None

    lower = np.percentile(data, percentile_low)
    upper = np.percentile(data, percentile_high)

    # Add some padding
    if upper > lower:
        padding = (upper - lower) * 0.1
        return lower - padding, upper + padding
    else:
        return None, None


def filter_by_steps(df, step_column, skip_steps):
    """
    Filter dataframe to skip the first skip_steps steps
    """
    if skip_steps <= 0 or step_column not in df.columns:
        return df

    min_step = df[step_column].min()
    threshold_step = min_step + skip_steps
    return df[df[step_column] >= threshold_step]


def visualize_training_metrics(
    csv_file: str,
    skip_steps: int = 0,
    log_scale: bool = False,
    percentile_low: float = 5.0,
    percentile_high: float = 95.0
):
    """
    Visualize training metrics from CSV file.

    :param csv_file: Path to the CSV file containing training metrics
    :param skip_steps: Skip the first S steps for plotting
    :param log_scale: Use logarithmic scale for y-axis
    :param percentile_low: Lower percentile for robust y-limits
    :param percentile_high: Upper percentile for robust y-limits
    """
    fname = os.path.expanduser(csv_file)
    df_clean = pd.read_csv(fname)

    # Filter by skip_steps if specified
    if skip_steps > 0:
        original_len = len(df_clean)
        df_clean = filter_by_steps(df_clean, 'step', skip_steps)
        filtered_len = len(df_clean)
        print(f"Filtered data: {original_len} -> {filtered_len} rows (skipped first {skip_steps} steps)")

    select_columns = ['step', 'val_geo_degs', 'geo_degs_epoch', 'val_loss', 'val_median_geo_degs', 'epoch']
    select_columns = [x for x in select_columns if x in df_clean.columns]
    df_val = df_clean[select_columns]

    print("Available columns:", df_clean.columns.tolist())
    if skip_steps > 0:
        print(f"Plotting data from step {df_clean['step'].min()} onwards")

    plt.figure(figsize=(12, 8))

    # First subplot: Geo Degrees
    plt.subplot(2, 1, 1)

    geo_values = []

    # Plot geo_degs_epoch
    if 'geo_degs_epoch' in df_clean.columns:
        df_geo = df_clean[['step', 'geo_degs_epoch']].dropna()
        if not df_geo.empty:
            plt.plot(df_geo['step'], df_geo['geo_degs_epoch'], 'g.-', label='geo_degs_epoch')
            geo_values.extend(df_geo['geo_degs_epoch'].values)

    # Plot val_geo_degs
    if 'val_geo_degs' in df_clean.columns:
        df_val_geo = df_clean[['step', 'val_geo_degs']].dropna()
        if not df_val_geo.empty:
            plt.plot(df_val_geo['step'], df_val_geo['val_geo_degs'], 'y-', label='val_geo_degs')
            geo_values.extend(df_val_geo['val_geo_degs'].values)

    # Set robust y-limits for geo degrees
    if geo_values:
        if log_scale and np.all(np.array(geo_values) > 0):
            plt.yscale('log')
            # For log scale, use geometric mean-based limits
            geo_values_pos = [x for x in geo_values if x > 0]
            if geo_values_pos:
                y_min, y_max = calculate_robust_ylim(np.array(geo_values_pos),
                                                     percentile_low, percentile_high)
                if y_min is not None and y_max is not None and y_min > 0:
                    plt.ylim(y_min, y_max)
        else:
            y_min, y_max = calculate_robust_ylim(np.array(geo_values),
                                                 percentile_low, percentile_high)
            if y_min is not None and y_max is not None:
                plt.ylim(y_min, y_max)

    plt.ylabel('Geo Degrees' + (' (log scale)' if log_scale else ''))
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Second subplot: Loss with spike-resistant y-limits
    if "loss" in df_clean.columns:
        plt.subplot(2, 1, 2)

        loss_values = []

        # Plot training loss
        df_loss = df_clean[['step', 'loss']].dropna()
        if not df_loss.empty:
            plt.plot(df_loss['step'], df_loss['loss'], 'r-', label='loss')
            loss_values.extend(df_loss['loss'].values)

        # Plot validation loss
        if 'val_loss' in df_clean.columns:
            df_val_loss = df_clean[['step', 'val_loss']].dropna()
            if not df_val_loss.empty:
                plt.plot(df_val_loss['step'], df_val_loss['val_loss'], 'b-', label='val_loss')
                loss_values.extend(df_val_loss['val_loss'].values)

        # Set robust y-limits to avoid spike influence
        if loss_values:
            if log_scale and np.all(np.array(loss_values) > 0):
                plt.yscale('log')
                # For log scale, filter out non-positive values
                loss_values_pos = [x for x in loss_values if x > 0]
                if loss_values_pos:
                    y_min, y_max = calculate_robust_ylim(np.array(loss_values_pos),
                                                         percentile_low, percentile_high)
                    if y_min is not None and y_max is not None and y_min > 0:
                        plt.ylim(y_min, y_max)
            else:
                y_min, y_max = calculate_robust_ylim(np.array(loss_values),
                                                     percentile_low, percentile_high)
                if y_min is not None and y_max is not None:
                    plt.ylim(y_min, y_max)

        plt.xlabel('Step')
        plt.ylabel('Loss' + (' (log scale)' if log_scale else ''))
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    """Command-line interface for training metrics visualization."""
    from argParseFromDoc import parse_function_and_call
    parse_function_and_call(visualize_training_metrics)


if __name__ == "__main__":
    main()