#!/usr/bin/env python3
"""Scaling sweep plots: DDA performance vs N, K, block size, and density."""

import os
import re
import matplotlib.pyplot as plt
import numpy as np

CSV_ROOT = "/home/user/tt-metal/profiles_sc26/csvs"
OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")

DDA = "bsr_spmm_multicore_snf_in0_dda_in1"


def parse_tflops(sparse_log_path):
    """Extract Device TFLOP/s from sparse log."""
    with open(sparse_log_path) as f:
        for line in f:
            m = re.search(r"Device TFLOP/s:\s+([\d.]+)", line)
            if m:
                return float(m.group(1))
    return None


def parse_param_from_filename(filename, param):
    """Extract a numeric parameter from a filename like parametric_M8192_N512_..."""
    m = re.search(rf"{param}(\d+)", filename)
    return int(m.group(1)) if m else None


def parse_density_from_filename(filename):
    """Extract density from filename: d10 -> 10, dppm1000 -> 0.1 (as percent)."""
    m = re.search(r"_dppm(\d+)", filename)
    if m:
        return int(m.group(1)) / 10000 * 100  # ppm to percent
    m = re.search(r"_d(\d+)", filename)
    if m:
        return int(m.group(1))
    return None


def collect_sweep(registry, host_code, param_name, param_extractor=None):
    """Collect (param_value, tflops) pairs from a registry."""
    d = os.path.join(CSV_ROOT, registry, host_code)
    if not os.path.isdir(d):
        return [], []
    results = []
    for f in sorted(os.listdir(d)):
        if not f.endswith("_sparse.log"):
            continue
        if param_extractor:
            val = param_extractor(f)
        else:
            val = parse_param_from_filename(f, param_name)
        tflops = parse_tflops(os.path.join(d, f))
        if val is not None and tflops is not None:
            results.append((val, tflops))
    results.sort()
    return [r[0] for r in results], [r[1] for r in results]


def collect_sweep_multi_algo(registry, param_name, param_extractor=None):
    """Collect sweep data for all 3 algorithms (if available)."""
    algos = [
        ("bsr_spmm_multicore_naive", "Naive"),
        ("bsr_spmm_multicore_snf_in0_naive_in1", "SnF"),
        ("bsr_spmm_multicore_snf_in0_dda_in1", "DDA"),
    ]
    data = {}
    for algo_name, algo_label in algos:
        xs, ys = collect_sweep(registry, algo_name, param_name, param_extractor)
        if xs:
            data[algo_label] = (xs, ys)
    return data


def plot_scaling():
    sweeps = [
        ("SweepN", "N", "Output Width N", None),
        ("SweepK", "K", "Reduction Dimension K", None),
        ("SweepBlockSize", "R", "Block Size (R=C)", None),
        ("SweepDensity", None, "Density (%)", parse_density_from_filename),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    algo_colors = {"Naive": "#1f77b4", "SnF": "#ff7f0e", "DDA": "#2ca02c"}

    for idx, (registry, param, xlabel, extractor) in enumerate(sweeps):
        ax = axes[idx]
        data = collect_sweep_multi_algo(registry, param, extractor)

        for algo_label, (xs, ys) in data.items():
            ax.plot(xs, ys, 'o-', label=algo_label, color=algo_colors[algo_label],
                    markersize=6, linewidth=2)
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                            xytext=(0, 7), ha='center', fontsize=7)

        ax.set_xlabel(xlabel)
        ax.set_title(f"Sweep: {registry}")
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.set_ylabel("Device TFLOP/s")
        ax.legend(loc='best', fontsize=8)

    fig.suptitle("Scaling Sweeps (R=C=256, d=10% unless swept)", fontsize=13)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, "scaling.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_scaling()
