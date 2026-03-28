#!/usr/bin/env python3
"""Plot throughput (TFLOPs/s) across sparsity patterns and density levels.

Reads data from registries PatternD5 (reg 2), PatternD10 (reg 3),
PatternD25 (reg 4), PatternD50 (reg 5). Only full algorithms (host codes 0-2).

Produces a single figure with 4 subplots (one per density) showing grouped
bars for each sparsity pattern and algorithm.

Usage:
    python scripts/plot_throughput.py
    python scripts/plot_throughput.py --data-dir /path/to/csvs
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

ALGORITHMS = [
    "bsr_spmm_multicore_load_balanced_new_DM",
    "bsr_spmm_multicore_snf",
    "bsr_spmm_multicore_snfin0_cdain1",
]

ALGO_PAPER_NAMES = {
    "bsr_spmm_multicore_load_balanced_new_DM": "Naive",
    "bsr_spmm_multicore_snf": "SnF in0 naive in1",
    "bsr_spmm_multicore_snfin0_cdain1": "SnF in0 CDA in1",
}

ALGO_COLORS = {
    "bsr_spmm_multicore_load_balanced_new_DM": "#4472C4",
    "bsr_spmm_multicore_snf": "#ED7D31",
    "bsr_spmm_multicore_snfin0_cdain1": "#70AD47",
}

# Pattern registries: each has 4 test cases in this order
PATTERNS = ["row", "col", "multi_diag", "random"]
PATTERN_LABELS = ["Row", "Column", "Multi-Diag", "Random"]

# Fixed problem parameters for pattern registries
M, N, K, R, C = 8192, 8192, 8192, 256, 256

DENSITY_REGISTRIES = [
    {"registry": "PatternD5",  "density_pct": 5},
    {"registry": "PatternD10", "density_pct": 10},
    {"registry": "PatternD25", "density_pct": 25},
    {"registry": "PatternD50", "density_pct": 50},
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def get_test_name(pattern, density_pct):
    """Build the test name for a given pattern and density."""
    if pattern == "random":
        return f"parametric_M{M}_N{N}_K{K}_R{R}_C{C}_d{density_pct}"
    else:
        return f"parametric_{pattern}_M{M}_N{N}_K{K}_R{R}_C{C}_d{density_pct}"


def get_sparse_log_path(data_dir, registry, host_code_name, test_name):
    """Return the path to the _sparse.log file."""
    return Path(data_dir) / registry / host_code_name / f"{test_name}_sparse.log"


def extract_tflops(log_path):
    """Extract TFLOPs/s value from a sparse log file.

    Searches for lines matching patterns like:
        TFLOPs/s: 12.34
        Device TFLOP/s: 12.34
    """
    if not log_path.exists():
        return None

    try:
        with open(log_path) as f:
            content = f.read()
    except Exception as e:
        print(f"  WARNING: Could not read {log_path}: {e}", file=sys.stderr)
        return None

    # Try multiple patterns
    for pat in [
        r"(?:Device\s+)?TFLOP/?s:\s*([\d.]+)",
        r"TFLOP/s:\s*([\d.]+)",
        r"TFLOPs/s:\s*([\d.]+)",
    ]:
        match = re.search(pat, content, re.IGNORECASE)
        if match:
            return float(match.group(1))

    return None


def load_throughput_data(data_dir):
    """Load throughput data for all density levels, patterns, and algorithms.

    Returns:
        dict: {density_pct: {pattern: {algo: tflops}}}
    """
    data = {}
    for cfg in DENSITY_REGISTRIES:
        registry = cfg["registry"]
        dpct = cfg["density_pct"]
        data[dpct] = {}

        for pat in PATTERNS:
            data[dpct][pat] = {}
            test_name = get_test_name(pat, dpct)
            for algo in ALGORITHMS:
                log_path = get_sparse_log_path(data_dir, registry, algo, test_name)
                tflops = extract_tflops(log_path)
                if tflops is not None:
                    data[dpct][pat][algo] = tflops
                else:
                    print(f"  WARNING: Missing throughput for {algo} / {test_name} "
                          f"(looked at {log_path})", file=sys.stderr)
    return data


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def plot_throughput(data, output_path):
    """Create a 2x2 figure with grouped bar charts for each density level."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=False)
    axes = axes.flatten()

    n_patterns = len(PATTERNS)
    n_algos = len(ALGORITHMS)
    bar_width = 0.22

    for subplot_idx, cfg in enumerate(DENSITY_REGISTRIES):
        ax = axes[subplot_idx]
        dpct = cfg["density_pct"]

        x = np.arange(n_patterns)

        for algo_idx, algo in enumerate(ALGORITHMS):
            values = []
            for pat in PATTERNS:
                values.append(data.get(dpct, {}).get(pat, {}).get(algo, 0.0))
            values = np.array(values)

            offset = (algo_idx - (n_algos - 1) / 2) * bar_width
            bars = ax.bar(
                x + offset,
                values,
                bar_width,
                color=ALGO_COLORS[algo],
                edgecolor="white",
                linewidth=0.5,
                label=ALGO_PAPER_NAMES[algo],
            )

            # Highlight where CDA loses to other algorithms
            if algo == "bsr_spmm_multicore_snfin0_cdain1":
                for pi, pat in enumerate(PATTERNS):
                    cda_val = data.get(dpct, {}).get(pat, {}).get(algo, 0.0)
                    other_vals = [
                        data.get(dpct, {}).get(pat, {}).get(a, 0.0)
                        for a in ALGORITHMS if a != algo
                    ]
                    if cda_val > 0 and any(ov > cda_val for ov in other_vals):
                        ax.plot(
                            x[pi] + offset, cda_val + 0.02 * ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else cda_val + 0.1,
                            marker="v", color="red", markersize=8, zorder=5,
                        )

        ax.set_xticks(x)
        ax.set_xticklabels(PATTERN_LABELS, fontsize=10)
        ax.set_ylabel("Throughput (TFLOPs/s)", fontsize=10)
        ax.set_title(f"Density = {dpct}%", fontsize=12)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

        if subplot_idx == 0:
            ax.legend(fontsize=9, loc="upper right", framealpha=0.9)

    # Second pass: mark CDA losses after y-limits are set
    for subplot_idx, cfg in enumerate(DENSITY_REGISTRIES):
        ax = axes[subplot_idx]
        dpct = cfg["density_pct"]
        x = np.arange(n_patterns)
        cda_algo = "bsr_spmm_multicore_snfin0_cdain1"
        cda_idx = ALGORITHMS.index(cda_algo)
        offset = (cda_idx - (n_algos - 1) / 2) * bar_width
        ymin, ymax = ax.get_ylim()

        for pi, pat in enumerate(PATTERNS):
            cda_val = data.get(dpct, {}).get(pat, {}).get(cda_algo, 0.0)
            other_vals = [
                data.get(dpct, {}).get(pat, {}).get(a, 0.0)
                for a in ALGORITHMS if a != cda_algo
            ]
            if cda_val > 0 and any(ov > cda_val for ov in other_vals):
                marker_y = cda_val + 0.03 * (ymax - ymin)
                ax.annotate(
                    "",
                    xy=(x[pi] + offset, cda_val),
                    xytext=(x[pi] + offset, marker_y),
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
                )

    fig.suptitle("SpMM Throughput by Sparsity Pattern and Density", fontsize=14, y=1.01)
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
        description="Plot SpMM throughput across sparsity patterns and densities")
    parser.add_argument(
        "--data-dir",
        default="/home/user/tt-metal/profiles_sc26/csvs",
        help="Root directory containing registry/host_code/test CSV files")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(script_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    print("Loading throughput data...")
    data = load_throughput_data(args.data_dir)

    # Check if we have any data
    has_data = any(
        data.get(cfg["density_pct"], {}).get(pat, {}).get(algo, 0) > 0
        for cfg in DENSITY_REGISTRIES
        for pat in PATTERNS
        for algo in ALGORITHMS
    )
    if not has_data:
        print("WARNING: No valid throughput data found. "
              "Skipping plot.", file=sys.stderr)
        return

    output_path = os.path.join(figures_dir, "throughput.png")
    plot_throughput(data, output_path)


if __name__ == "__main__":
    main()
