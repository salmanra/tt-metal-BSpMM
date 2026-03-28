#!/usr/bin/env python3
"""Plot CDA algorithm scaling behavior across N, K, block size, and density.

Reads data from registries SweepN (reg 6), SweepK (reg 7),
SweepBlockSize (reg 8), SweepDensity (reg 9). Only host code 2 (CDA).

Produces a 2x2 subplot figure saved to scripts/figures/scaling.png.

Usage:
    python scripts/plot_scaling.py
    python scripts/plot_scaling.py --data-dir /path/to/csvs
"""

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

CDA_HOST_CODE = "bsr_spmm_multicore_snfin0_cdain1"

# Fixed base parameters
BASE_M, BASE_N, BASE_K = 8192, 8192, 8192
BASE_R, BASE_C = 256, 256
BASE_DENSITY = 10

LINE_COLOR = "#70AD47"
MARKER = "o"

# Sweep configurations
SWEEPS = [
    {
        "title": "Sweep N",
        "registry": "SweepN",
        "xlabel": "N",
        "values": [512, 1024, 2048, 4096, 8192],
        # For SweepN: M=8192, N=varied, K=8192, R=C=256, d=10%
        "make_test_name": lambda val: f"parametric_M{BASE_M}_N{val}_K{BASE_K}_R{BASE_R}_C{BASE_C}_d{BASE_DENSITY}",
    },
    {
        "title": "Sweep K",
        "registry": "SweepK",
        "xlabel": "K",
        "values": [512, 1024, 2048, 4096, 8192],
        # For SweepK: M=8192, N=8192, K=varied, R=C=256, d=10%
        "make_test_name": lambda val: f"parametric_M{BASE_M}_N{BASE_N}_K{val}_R{BASE_R}_C{BASE_C}_d{BASE_DENSITY}",
    },
    {
        "title": "Sweep Block Size",
        "registry": "SweepBlockSize",
        "xlabel": "Block Size (R=C)",
        "values": [32, 64, 128, 256],
        # For SweepBlockSize: M=N=K=8192, R=C=varied, d=10%
        "make_test_name": lambda val: f"parametric_M{BASE_M}_N{BASE_N}_K{BASE_K}_R{val}_C{val}_d{BASE_DENSITY}",
    },
    {
        "title": "Sweep Density",
        "registry": "SweepDensity",
        "xlabel": "Density (%)",
        "values": [5, 10, 25, 50, 75],
        # For SweepDensity: M=N=K=8192, R=C=256, d=varied
        "make_test_name": lambda val: f"parametric_M{BASE_M}_N{BASE_N}_K{BASE_K}_R{BASE_R}_C{BASE_C}_d{val}",
    },
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def get_sparse_log_path(data_dir, registry, host_code_name, test_name):
    """Return the path to the _sparse.log file."""
    return Path(data_dir) / registry / host_code_name / f"{test_name}_sparse.log"


def extract_tflops(log_path):
    """Extract TFLOPs/s value from a sparse log file."""
    if not log_path.exists():
        return None

    try:
        with open(log_path) as f:
            content = f.read()
    except Exception as e:
        print(f"  WARNING: Could not read {log_path}: {e}", file=sys.stderr)
        return None

    for pat in [
        r"(?:Device\s+)?TFLOP/?s:\s*([\d.]+)",
        r"TFLOP/s:\s*([\d.]+)",
        r"TFLOPs/s:\s*([\d.]+)",
    ]:
        match = re.search(pat, content, re.IGNORECASE)
        if match:
            return float(match.group(1))

    return None


def load_sweep_data(data_dir, sweep_cfg):
    """Load throughput data for one sweep dimension.

    Returns:
        list of (x_value, tflops) tuples (only for values with valid data)
    """
    results = []
    registry = sweep_cfg["registry"]

    for val in sweep_cfg["values"]:
        test_name = sweep_cfg["make_test_name"](val)
        log_path = get_sparse_log_path(data_dir, registry, CDA_HOST_CODE, test_name)
        tflops = extract_tflops(log_path)
        if tflops is not None:
            results.append((val, tflops))
        else:
            print(f"  WARNING: Missing throughput for {registry}/{CDA_HOST_CODE}/{test_name} "
                  f"(looked at {log_path})", file=sys.stderr)

    return results


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def plot_scaling(all_data, output_path):
    """Create 2x2 subplot figure with scaling curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for idx, (sweep_cfg, sweep_data) in enumerate(zip(SWEEPS, all_data)):
        ax = axes[idx]

        if not sweep_data:
            ax.text(0.5, 0.5, "No data available",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=12, color="gray")
            ax.set_title(sweep_cfg["title"], fontsize=12)
            continue

        x_vals = [d[0] for d in sweep_data]
        y_vals = [d[1] for d in sweep_data]

        ax.plot(x_vals, y_vals,
                marker=MARKER, color=LINE_COLOR, linewidth=2, markersize=7,
                markeredgecolor="white", markeredgewidth=1.0,
                label="SnF in0 CDA in1")

        # Annotate points
        for xv, yv in zip(x_vals, y_vals):
            ax.annotate(f"{yv:.2f}",
                        (xv, yv),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha="center", fontsize=8)

        ax.set_xlabel(sweep_cfg["xlabel"], fontsize=11)
        ax.set_ylabel("Throughput (TFLOPs/s)", fontsize=11)
        ax.set_title(sweep_cfg["title"], fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        # Use string x-tick labels for clearer spacing
        ax.set_xticks(x_vals)
        ax.set_xticklabels([str(v) for v in x_vals], fontsize=9)

        if idx == 0:
            ax.legend(fontsize=9, loc="best", framealpha=0.9)

    fig.suptitle("CDA Algorithm Scaling Behavior (d=10%, R=C=256 unless varied)",
                 fontsize=14, y=1.01)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Plot CDA algorithm scaling across N, K, block size, and density")
    parser.add_argument(
        "--data-dir",
        default="/home/user/tt-metal/profiles_sc26/csvs",
        help="Root directory containing registry/host_code/test CSV files")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(script_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    print("Loading scaling data...")
    all_data = []
    for sweep_cfg in SWEEPS:
        print(f"  {sweep_cfg['title']} (registry: {sweep_cfg['registry']})")
        sweep_data = load_sweep_data(args.data_dir, sweep_cfg)
        all_data.append(sweep_data)

    # Check if we have any data at all
    has_data = any(len(d) > 0 for d in all_data)
    if not has_data:
        print("WARNING: No valid scaling data found. Skipping plot.", file=sys.stderr)
        return

    output_path = os.path.join(figures_dir, "scaling.png")
    plot_scaling(all_data, output_path)


if __name__ == "__main__":
    main()
