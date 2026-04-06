#!/usr/bin/env python3
"""DDA vs GPU throughput by sparsity pattern, density, and block size."""

import os
import re
import csv
import matplotlib.pyplot as plt
import numpy as np

DDA_ROOT = "/home/user/tt-metal/profiles_sc26_april5/csvs"
GPU_DIR = "/home/user/tt-metal/tt_metal/programming_examples/rahmy/gpu-normalized"
OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
DDA_HC = "bsr_spmm_multicore_snfin0_cdain1"

PATTERNS = ["row", "random", "multi_diag", "col"]
PATTERN_LABELS = ["Row", "Random", "Multi-diag", "Col"]
PATTERN_COLORS = ["#d62728", "#1f77b4", "#9467bd", "#2ca02c"]

# Block size → (GPU csv filename, list of (density_label, registry_name))
AXES = [
    ("32×32", "sweep_pattern_32.csv", [
        ("0.003%", "PatternUltra32_30"),
        ("0.01%",  "PatternUltra32_100"),
        ("0.03%",  "PatternUltra32_300"),
        ("0.1%",   "PatternUltra32_1000"),
        ("0.3%",   "PatternUltra32_3000"),
    ]),
    ("64×64", "sweep_pattern_64.csv", [
        ("0.006%", "PatternUltra64_60"),
        ("0.02%",  "PatternUltra64_200"),
        ("0.06%",  "PatternUltra64_600"),
        ("0.2%",   "PatternUltra64_2000"),
        ("0.6%",   "PatternUltra64_6000"),
        ("1%",     "PatternUltra64_10000"),
    ]),
    ("128×128", "sweep_pattern_128.csv", [
        ("5%",  "PatternD5_128"),
        ("10%", "PatternD10_128"),
        ("25%", "PatternD25_128"),
        ("50%", "PatternD50_128"),
    ]),
    ("256×256", "sweep_pattern_256.csv", [
        ("5%",  "PatternD5"),
        ("10%", "PatternD10"),
        ("25%", "PatternD25"),
        ("50%", "PatternD50"),
    ]),
]


def classify_pattern(case_name):
    if "_row_" in case_name:
        return "row"
    elif "_col_" in case_name:
        return "col"
    elif "_multi_diag_" in case_name:
        return "multi_diag"
    else:
        return "random"


def parse_dda_tflops(registry_name):
    """Return {pattern: tflops} from DDA sparse logs."""
    d = os.path.join(DDA_ROOT, registry_name, DDA_HC)
    result = {}
    if not os.path.isdir(d):
        return result
    for f in os.listdir(d):
        if not f.endswith("_sparse.log"):
            continue
        pat = classify_pattern(f)
        path = os.path.join(d, f)
        with open(path) as fh:
            for line in fh:
                m = re.search(r"Device TFLOP/s:\s+([\d.]+)", line)
                if m:
                    result[pat] = float(m.group(1))
    return result


def parse_gpu_csv(gpu_csv_path):
    """Return {(registry_idx, pattern): tflops} from GPU CSV."""
    result = {}
    with open(gpu_csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            reg = int(row["Registry"])
            pat = classify_pattern(row["Case"])
            result[(reg, pat)] = float(row["Avg_TFLOPs"])
    return result


def registry_to_gpu_index(registry_name):
    """Map registry name to the index used in the GPU CSV Registry column."""
    # The GPU CSVs use the profiling_suite.hpp registry indices
    name_to_idx = {
        "PatternD5": 2, "PatternD10": 3, "PatternD25": 4, "PatternD50": 5,
        "PatternD5_128": 13, "PatternD10_128": 14, "PatternD25_128": 15, "PatternD50_128": 16,
        "PatternUltra32_30": 17, "PatternUltra32_100": 18, "PatternUltra32_300": 19,
        "PatternUltra32_1000": 20, "PatternUltra32_3000": 21, "PatternUltra32_10000": 22,
        "PatternUltra64_60": 23, "PatternUltra64_200": 24, "PatternUltra64_600": 25,
        "PatternUltra64_2000": 26, "PatternUltra64_6000": 27, "PatternUltra64_10000": 28,
    }
    return name_to_idx.get(registry_name)


def plot_figure(axes_specs, title, out_name):
    """Group by pattern: x-axis is pattern, density pairs are bars within each group."""
    n_axes = len(axes_specs)
    fig, axes = plt.subplots(1, n_axes, figsize=(12 * n_axes // 2, 6))
    if n_axes == 1:
        axes = [axes]

    # Colors per density (cycle through a palette)
    density_palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#e377c2", "#17becf", "#ff7f0e", "#8c564b"]

    for ax_idx, (block_label, gpu_csv_name, densities) in enumerate(axes_specs):
        ax = axes[ax_idx]
        gpu_data = parse_gpu_csv(os.path.join(GPU_DIR, gpu_csv_name))

        n_densities = len(densities)
        n_patterns = len(PATTERNS)
        bar_width = 0.35
        # Each pattern group: n_densities pairs of bars (DDA+GPU)
        pair_width = bar_width * 2 + 0.05
        density_spacing = pair_width + 0.1
        pattern_spacing = n_densities * density_spacing + 0.8

        colors = [density_palette[i % len(density_palette)] for i in range(n_densities)]

        for p_idx, pat in enumerate(PATTERNS):
            pattern_center = p_idx * pattern_spacing

            for d_idx, (density_label, registry_name) in enumerate(densities):
                dda_data = parse_dda_tflops(registry_name)
                gpu_reg_idx = registry_to_gpu_index(registry_name)

                x_center = pattern_center + d_idx * density_spacing

                dda_val = dda_data.get(pat, 0)
                gpu_val = gpu_data.get((gpu_reg_idx, pat), 0) if gpu_reg_idx is not None else 0

                # DDA bar (solid)
                ax.bar(x_center - bar_width/2, dda_val, bar_width,
                       color=colors[d_idx], edgecolor="white", linewidth=0.5,
                       label=f"d={density_label} (DDA)" if p_idx == 0 else "")
                # GPU bar (hatched)
                ax.bar(x_center + bar_width/2, gpu_val, bar_width,
                       color=colors[d_idx], alpha=0.4, hatch="//",
                       edgecolor=colors[d_idx], linewidth=0.5,
                       label=f"d={density_label} (GPU)" if p_idx == 0 else "")

        # X-axis: pattern labels
        pattern_centers = [p_idx * pattern_spacing + (n_densities - 1) * density_spacing / 2
                          for p_idx in range(n_patterns)]
        ax.set_xticks(pattern_centers)
        ax.set_xticklabels(PATTERN_LABELS, fontsize=9)
        ax.set_xlabel("Sparsity Pattern")

        ax.set_title(f"R=C={block_label}", fontsize=12)
        ax.grid(axis='y', alpha=0.3)

        if ax_idx == 0:
            ax.set_ylabel("TFLOP/s")
        # Each axis gets its own legend (densities may differ between axes)
        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        unique_handles = []
        unique_labels = []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen[l] = True
                unique_handles.append(h)
                unique_labels.append(l)
        ax.legend(unique_handles, unique_labels, loc='upper left', fontsize=7, ncol=2)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, out_name)
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    plot_figure(
        AXES[:2],
        "DDA (N150) vs GPU: Ultra-Sparse (R=C=32, R=C=64)",
        "sweep_pattern_dda_vs_gpu_ultrasparse.png",
    )
    plot_figure(
        AXES[2:],
        "DDA (N150) vs GPU: Standard Density (R=C=128, R=C=256)",
        "sweep_pattern_dda_vs_gpu_standard.png",
    )
