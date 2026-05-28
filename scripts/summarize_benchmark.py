"""
Parse benchmark output directory and print a markdown summary table.

Usage:
    python scripts/summarize_benchmark.py /path/to/benchmark_localref_YYYYMMDD_HHMMSS
"""
import re
import sys
from pathlib import Path


def parse_time(run_dir: Path) -> int | None:
    t = run_dir / "time.txt"
    if t.exists():
        try:
            return int(t.read_text().strip())
        except ValueError:
            pass
    return None


def parse_compare_poses_log(log: Path) -> dict:
    """Return stats from a compare_poses log."""
    stats = {}
    text = log.read_text()

    patterns = {
        "n":           r"Found\s+(\d+)\s+matching\s+particles",
        "mean":        r"Analyzing angular differences.*?\nMean:\s*([\d.]+)",
        "median":      r"Analyzing angular differences.*?\nMean:.*?\nStandard Deviation:.*?\nMedian:\s*([\d.]+)",
        "pct5":        r"<\s*5°:\s*([\d.]+)%",
        "pct10":       r"<\s*10°:\s*([\d.]+)%",
        "shift_mean":  r"Shift errors.*?\nMean:\s*([\d.]+)",
        "shift_median":r"Shift errors.*?\nMean:.*?\nStandard Deviation:.*?\nMedian:\s*([\d.]+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
        if m:
            val = m.group(1)
            stats[key] = int(val) if key == "n" else float(val)
    return stats


def collect_run(run_dir: Path) -> dict:
    result = {"label": run_dir.name}
    result["time_s"] = parse_time(run_dir)

    all_n = 0
    mean_vals, median_vals, pct5_vals, pct10_vals = [], [], [], []
    shift_mean_vals, shift_median_vals = [], []

    for log in sorted(run_dir.glob("compare_poses_half*.log")):
        s = parse_compare_poses_log(log)
        if "n" in s:
            all_n += s["n"]
        if "mean" in s:
            mean_vals.append(s["mean"])
        if "median" in s:
            median_vals.append(s["median"])
        if "pct5" in s:
            pct5_vals.append(s["pct5"])
        if "pct10" in s:
            pct10_vals.append(s["pct10"])
        if "shift_mean" in s:
            shift_mean_vals.append(s["shift_mean"])
        if "shift_median" in s:
            shift_median_vals.append(s["shift_median"])

    def avg(lst):
        return sum(lst) / len(lst) if lst else None

    result["n_particles"]   = all_n if all_n else None
    result["mean_deg"]      = avg(mean_vals)
    result["median_deg"]    = avg(median_vals)
    result["pct5"]          = avg(pct5_vals)
    result["pct10"]         = avg(pct10_vals)
    result["shift_mean"]    = avg(shift_mean_vals)
    result["shift_median"]  = avg(shift_median_vals)
    return result


def fmt(v, fmt_str=".2f", missing="—"):
    if v is None:
        return missing
    return format(v, fmt_str)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    base = Path(sys.argv[1])
    if not base.is_dir():
        print(f"ERROR: {base} is not a directory")
        sys.exit(1)

    run_dirs = sorted(d for d in base.iterdir() if d.is_dir())
    if not run_dirs:
        print(f"No run directories found in {base}")
        sys.exit(1)

    rows = [collect_run(d) for d in run_dirs]

    print(f"\n## Benchmark results  —  {base.name}\n")

    print("### Compute time\n")
    header = f"{'Run':<40} {'Time (s)':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['label']:<40} {fmt(r['time_s'], '.0f'):>10}")

    print("\n### Accuracy (averaged over half1+half2)\n")
    header2 = f"{'Run':<40} {'N':>6} {'Mean°':>7} {'Median°':>8} {'%<5°':>7} {'%<10°':>7} {'Med.Shift(Å)':>13} {'Mean.Shift(Å)':>14}"
    print(header2)
    print("-" * len(header2))
    for r in rows:
        n_s = fmt(r["n_particles"], ".0f")
        print(
            f"{r['label']:<40} {n_s:>6}"
            f" {fmt(r['mean_deg']):>7}"
            f" {fmt(r['median_deg']):>8}"
            f" {fmt(r['pct5']):>7}"
            f" {fmt(r['pct10']):>7}"
            f" {fmt(r['shift_median']):>13}"
            f" {fmt(r['shift_mean']):>14}"
        )
    print()


if __name__ == "__main__":
    main()
