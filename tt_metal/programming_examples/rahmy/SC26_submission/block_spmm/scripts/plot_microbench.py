#!/usr/bin/env python3
"""Plot microbenchmark ablation study results as stacked bar charts.

Reads device CSV data from MicrobenchD25 (registry 0) and MicrobenchD5 (registry 1).
For each algorithm and block size, computes ablation savings by comparing the full
runtime against the runtime with each component skipped.

Produces two figures (one per density level) saved to scripts/figures/.

Usage:
    python scripts/plot_microbench.py
    python scripts/plot_microbench.py --data-dir /path/to/csvs
"""

import argparse
import csv
import os
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
    "bsr_spmm_multicore_snf": "SnF in0\nnaive in1",
    "bsr_spmm_multicore_snfin0_cdain1": "SnF in0\nCDA in1",
}

# Host-code name suffixes for each ablation variant
ABLATION_SUFFIXES = {
    "full": "",
    "no_a_read": "_no_a_read",
    "no_b_read": "_no_b_read",
    "no_compute": "_no_compute",
    "no_write": "_no_write",
}

BLOCK_SIZES = [32, 64, 128, 256]

# Fixed problem parameters for microbenchmark registries
M, N, K = 8192, 8192, 8192

COMPONENT_COLORS = {
    "A read": "#4472C4",    # blue
    "B read": "#ED7D31",    # orange
    "Compute": "#70AD47",   # green
    "Write": "#FF4B4B",     # red
}

COMPONENT_ORDER = ["A read", "B read", "Compute", "Write"]

DENSITY_CONFIGS = {
    25: {"registry": "MicrobenchD25", "density_pct": 25},
    5:  {"registry": "MicrobenchD5",  "density_pct": 5},
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def get_test_name(R, C, density_pct):
    """Build the test name string matching profiling_suite output."""
    return f"parametric_M{M}_N{N}_K{K}_R{R}_C{C}_d{density_pct}"


def get_device_csv_path(data_dir, registry, host_code_name, test_name):
    """Return the path to the .device.csv file."""
    return Path(data_dir) / registry / host_code_name / f"{test_name}.device.csv"


def read_max_kernel_time_ns(csv_path):
    """Read device CSV and return the maximum kernel execution time in ns.

    Uses the BRISC-KERNEL zone as a proxy for total per-core runtime and
    takes the maximum across all entries (cores x invocations). If no
    BRISC-KERNEL zone is found, falls back to the maximum GPU execution
    time across all zones.
    """
    if not csv_path.exists():
        return None

    kernel_times = []
    all_times = []

    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    t = float(row["GPU execution time"])
                except (KeyError, ValueError):
                    continue
                all_times.append(t)
                name = row.get("name", "")
                if "BRISC-KERNEL" in name or "NCRISC-KERNEL" in name:
                    kernel_times.append(t)
    except Exception as e:
        print(f"  WARNING: Could not read {csv_path}: {e}", file=sys.stderr)
        return None

    if kernel_times:
        return max(kernel_times)
    elif all_times:
        return max(all_times)
    return None


def load_ablation_data(data_dir, registry, density_pct):
    """Load timing data for all algorithms, block sizes, and ablation variants.

    Returns:
        dict: {algo: {block_size: {variant: time_ns}}}
    """
    data = {}
    for algo in ALGORITHMS:
        data[algo] = {}
        for bs in BLOCK_SIZES:
            data[algo][bs] = {}
            test_name = get_test_name(bs, bs, density_pct)
            for variant, suffix in ABLATION_SUFFIXES.items():
                hc_name = algo + suffix
                csv_path = get_device_csv_path(data_dir, registry, hc_name, test_name)
                t = read_max_kernel_time_ns(csv_path)
                if t is not None:
                    data[algo][bs][variant] = t
                else:
                    print(f"  WARNING: Missing data for {hc_name} / {test_name} "
                          f"(looked at {csv_path})", file=sys.stderr)
    return data


def compute_ablation_savings(data):
    """Compute component savings from ablation data.

    saving_X = full_time - no_X_time  (clamped to >= 0)

    Returns:
        dict: {algo: {block_size: {component_name: saving_ns}}}
    """
    savings = {}
    component_map = {
        "A read": "no_a_read",
        "B read": "no_b_read",
        "Compute": "no_compute",
        "Write": "no_write",
    }
    for algo in ALGORITHMS:
        savings[algo] = {}
        for bs in BLOCK_SIZES:
            savings[algo][bs] = {}
            variants = data[algo][bs]
            full = variants.get("full")
            if full is None:
                continue
            for comp_name, variant_key in component_map.items():
                ablated = variants.get(variant_key)
                if ablated is not None:
                    savings[algo][bs][comp_name] = max(0, full - ablated)
                else:
                    savings[algo][bs][comp_name] = 0.0
    return savings


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def plot_microbench(savings, density_pct, output_path):
    """Create a grouped + stacked bar chart for one density level."""
    fig, ax = plt.subplots(figsize=(10, 6))

    n_groups = len(BLOCK_SIZES)
    n_algos = len(ALGORITHMS)
    bar_width = 0.22
    group_width = n_algos * bar_width + 0.15

    x_centers = np.arange(n_groups) * group_width

    for algo_idx, algo in enumerate(ALGORITHMS):
        bottoms = np.zeros(n_groups)
        for comp in COMPONENT_ORDER:
            values = []
            for bs in BLOCK_SIZES:
                values.append(savings[algo].get(bs, {}).get(comp, 0.0))
            values = np.array(values)
            offset = (algo_idx - (n_algos - 1) / 2) * bar_width
            bars = ax.bar(
                x_centers + offset,
                values,
                bar_width,
                bottom=bottoms,
                color=COMPONENT_COLORS[comp],
                edgecolor="white",
                linewidth=0.5,
                label=comp if algo_idx == 0 else "",
            )
            bottoms += values

    ax.set_xticks(x_centers)
    block_labels = []
    for bs in BLOCK_SIZES:
        sub_labels = [ALGO_PAPER_NAMES[a] for a in ALGORITHMS]
        block_labels.append(f"{bs}x{bs}")
    ax.set_xticklabels(block_labels, fontsize=10)

    # Add algorithm labels below
    for gi, bs in enumerate(BLOCK_SIZES):
        for ai, algo in enumerate(ALGORITHMS):
            offset = (ai - (n_algos - 1) / 2) * bar_width
            x = x_centers[gi] + offset
            ax.text(x, -0.06 * ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else -10,
                    ALGO_PAPER_NAMES[algo],
                    ha="center", va="top", fontsize=6, rotation=0)

    ax.set_xlabel("Block Size (R=C)", fontsize=12)
    ax.set_ylabel("Runtime (ns)", fontsize=12)
    ax.set_title(f"Microbenchmark Ablation (density={density_pct}%)", fontsize=14)

    # Legend (deduplicated)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left", framealpha=0.9)

    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
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
        description="Plot microbenchmark ablation stacked bar charts")
    parser.add_argument(
        "--data-dir",
        default="/home/user/tt-metal/profiles_sc26/csvs",
        help="Root directory containing registry/host_code/test CSV files")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(script_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    for density_pct, cfg in DENSITY_CONFIGS.items():
        print(f"\n--- Density = {density_pct}% (registry: {cfg['registry']}) ---")
        data = load_ablation_data(args.data_dir, cfg["registry"], density_pct)
        savings = compute_ablation_savings(data)

        # Check if we have any data
        has_data = any(
            savings[algo].get(bs, {}).get(comp, 0) > 0
            for algo in ALGORITHMS
            for bs in BLOCK_SIZES
            for comp in COMPONENT_ORDER
        )
        if not has_data:
            print(f"  WARNING: No valid ablation data found for density={density_pct}%. "
                  f"Skipping plot.", file=sys.stderr)
            continue

        output_path = os.path.join(figures_dir, f"microbench_d{density_pct}.png")
        plot_microbench(savings, density_pct, output_path)


if __name__ == "__main__":
    main()
