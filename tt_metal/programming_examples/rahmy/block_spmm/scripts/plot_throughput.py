#!/usr/bin/env python3
"""Throughput plots: Device TFLOP/s across sparsity patterns and densities."""

import os
import re
import matplotlib.pyplot as plt
import numpy as np

CSV_ROOT = "/home/user/tt-metal/profiles_sc26/csvs"
OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")

ALGORITHMS = [
    ("bsr_spmm_multicore_naive", "Naive"),
    ("bsr_spmm_multicore_snf_in0_naive_in1", "SnF"),
    ("bsr_spmm_multicore_snf_in0_dda_in1", "DDA"),
]

PATTERN_REGISTRIES = [
    ("PatternD5",  "d=5%"),
    ("PatternD10", "d=10%"),
    ("PatternD25", "d=25%"),
    ("PatternD50", "d=50%"),
]

PATTERNS = ["random", "row", "col", "multi_diag"]
PATTERN_LABELS = ["Random", "Row", "Col", "Multi-diag"]


def parse_tflops(sparse_log_path):
    """Extract Device TFLOP/s from sparse log."""
    with open(sparse_log_path) as f:
        for line in f:
            m = re.search(r"Device TFLOP/s:\s+([\d.]+)", line)
            if m:
                return float(m.group(1))
    return None


def find_sparse_log(registry, host_code, pattern_tag):
    """Find the sparse log matching a pattern tag in a registry."""
    d = os.path.join(CSV_ROOT, registry, host_code)
    if not os.path.isdir(d):
        return None
    for f in sorted(os.listdir(d)):
        if not f.endswith("_sparse.log"):
            continue
        if pattern_tag == "random":
            # The random case has no pattern prefix (just "parametric_M...")
            if not any(p in f for p in ["_row_", "_col_", "_multi_diag_"]):
                return os.path.join(d, f)
        elif f"_{pattern_tag}_" in f:
            return os.path.join(d, f)
    return None


def plot_throughput():
    fig, axes = plt.subplots(1, len(PATTERN_REGISTRIES), figsize=(16, 5), sharey=True)

    algo_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    bar_width = 0.25

    for reg_idx, (registry, density_label) in enumerate(PATTERN_REGISTRIES):
        ax = axes[reg_idx]
        x = np.arange(len(PATTERNS))

        for algo_idx, (algo_name, algo_label) in enumerate(ALGORITHMS):
            tflops = []
            for pat in PATTERNS:
                log = find_sparse_log(registry, algo_name, pat)
                val = parse_tflops(log) if log else 0
                tflops.append(val or 0)

            offsets = x + algo_idx * bar_width
            bars = ax.bar(offsets, tflops, bar_width, label=algo_label,
                         color=algo_colors[algo_idx], edgecolor="white", linewidth=0.5)
            for bar, v in zip(bars, tflops):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, v + 0.3, f"{v:.1f}",
                           ha='center', va='bottom', fontsize=7)

        ax.set_xlabel("Sparsity Pattern")
        ax.set_title(density_label)
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels(PATTERN_LABELS, rotation=30, ha='right')
        ax.grid(axis='y', alpha=0.3)

    axes[0].set_ylabel("Device TFLOP/s")
    axes[0].legend(loc='upper left')
    fig.suptitle("Throughput: 3 Algorithms x 4 Patterns x 4 Densities (R=C=256)", fontsize=13)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, "throughput.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_throughput()
