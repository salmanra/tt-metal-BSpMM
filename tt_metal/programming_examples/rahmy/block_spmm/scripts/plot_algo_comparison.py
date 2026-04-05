#!/usr/bin/env python3
"""3-algorithm comparison plots from profiles_april3 data."""

import os
import re
import matplotlib.pyplot as plt
import numpy as np

DDA_ROOT = "/home/user/tt-metal/profiles_april3/csvs"
OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")

ALGORITHMS = [
    ("bsr_spmm_multicore_naive", "Naive", "#1f77b4"),
    ("bsr_spmm_multicore_snf_in0_naive_in1", "SnF", "#ff7f0e"),
    ("bsr_spmm_multicore_snf_in0_dda_in1", "DDA", "#2ca02c"),
]

PATTERNS = ["row", "col", "multi_diag", "random"]
PATTERN_LABELS = ["Row", "Col", "Multi-diag", "Random"]


def classify_pattern(filename):
    if "_row_" in filename:
        return "row"
    elif "_col_" in filename:
        return "col"
    elif "_multi_diag_" in filename:
        return "multi_diag"
    else:
        return "random"


def parse_tflops_from_dir(registry_name, algo_name):
    """Return {pattern: tflops} or {density_ppm: tflops} from sparse logs."""
    d = os.path.join(DDA_ROOT, registry_name, algo_name)
    result = {}
    if not os.path.isdir(d):
        return result
    for f in os.listdir(d):
        if not f.endswith("_sparse.log"):
            continue
        path = os.path.join(d, f)
        tflops = None
        with open(path) as fh:
            for line in fh:
                m = re.search(r"Device TFLOP/s:\s+([\d.]+)", line)
                if m:
                    tflops = float(m.group(1))
        if tflops is not None:
            result[f] = tflops
    return result


def get_pattern_tflops(registry_name, algo_name):
    """Return {pattern: tflops} for a pattern registry."""
    raw = parse_tflops_from_dir(registry_name, algo_name)
    result = {}
    for fname, tflops in raw.items():
        pat = classify_pattern(fname)
        result[pat] = tflops
    return result


def get_density_tflops(registry_name, algo_name):
    """Return [(ppm, tflops)] sorted by density for an ultra-sparse registry."""
    raw = parse_tflops_from_dir(registry_name, algo_name)
    results = []
    for fname, tflops in raw.items():
        m = re.search(r"dppm(\d+)", fname)
        if m:
            results.append((int(m.group(1)), tflops))
    results.sort()
    return results


def plot_pattern_comparison():
    registries = [
        ("PatternD5", "d=5%"),
        ("PatternD10", "d=10%"),
        ("PatternD25", "d=25%"),
        ("PatternD50", "d=50%"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
    bar_width = 0.25

    for ax_idx, (reg_name, density_label) in enumerate(registries):
        ax = axes[ax_idx]
        x = np.arange(len(PATTERNS))

        for algo_idx, (algo_name, algo_label, color) in enumerate(ALGORITHMS):
            data = get_pattern_tflops(reg_name, algo_name)
            vals = [data.get(p, 0) for p in PATTERNS]
            offsets = x + algo_idx * bar_width
            bars = ax.bar(offsets, vals, bar_width, label=algo_label, color=color,
                         edgecolor="white", linewidth=0.5)
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, v + 0.3,
                           f"{v:.1f}", ha='center', va='bottom', fontsize=7)

        ax.set_xlabel("Sparsity Pattern")
        ax.set_title(density_label, fontsize=12)
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels(PATTERN_LABELS, rotation=30, ha='right')
        ax.grid(axis='y', alpha=0.3)

    axes[0].set_ylabel("Device TFLOP/s")
    axes[0].legend(loc='upper left', fontsize=9)
    fig.suptitle("3-Algorithm Comparison by Sparsity Pattern (R=C=256)", fontsize=14)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, "algo_comparison_pattern.png")
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.close(fig)


def plot_ultrasparse_comparison():
    registries = [
        ("UltraLowDensity32", "R=C=32"),
        ("UltraLowDensity64", "R=C=64"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bar_width = 0.25

    for ax_idx, (reg_name, block_label) in enumerate(registries):
        ax = axes[ax_idx]

        # Get density points from DDA (all algos should have the same)
        dda_data = get_density_tflops(reg_name, ALGORITHMS[2][0])
        density_ppms = [d[0] for d in dda_data]
        density_labels = [str(d) for d in density_ppms]
        x = np.arange(len(density_ppms))

        for algo_idx, (algo_name, algo_label, color) in enumerate(ALGORITHMS):
            algo_data = get_density_tflops(reg_name, algo_name)
            ppm_to_tflops = {d[0]: d[1] for d in algo_data}
            vals = [ppm_to_tflops.get(ppm, 0) for ppm in density_ppms]
            offsets = x + algo_idx * bar_width
            bars = ax.bar(offsets, vals, bar_width, label=algo_label, color=color,
                         edgecolor="white", linewidth=0.5)
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, v + 0.1,
                           f"{v:.1f}", ha='center', va='bottom', fontsize=7)

        ax.set_xlabel("Density (PPM)")
        ax.set_ylabel("Device TFLOP/s")
        ax.set_title(block_label, fontsize=12)
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels(density_labels)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(loc='upper left', fontsize=9)

    fig.suptitle("3-Algorithm Comparison: Ultra-Low Density", fontsize=14)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, "algo_comparison_ultrasparse.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    plot_pattern_comparison()
    plot_ultrasparse_comparison()
