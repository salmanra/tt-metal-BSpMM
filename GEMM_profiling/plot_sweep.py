#!/usr/bin/env python3
"""Plot TFLOP/s and GB/s from sweep.sh CSV output.

Usage:
    python GEMM_profiling/plot_sweep.py                              # uses default CSV path
    python GEMM_profiling/plot_sweep.py GEMM_profiling/sweep_results.csv
    python GEMM_profiling/plot_sweep.py results.csv -o my_plot.png
"""

import argparse
import csv
import sys

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot GEMM sweep results")
    parser.add_argument("csv_file", nargs="?", default="GEMM_profiling/sweep_results.csv",
                        help="Path to sweep_results.csv")
    parser.add_argument("-o", "--output-dir", default="GEMM_profiling",
                        help="Output directory for plots (default: GEMM_profiling)")
    args = parser.parse_args()

    # Read CSV
    sizes, avg_tflops, max_tflops, avg_gbs, max_gbs = [], [], [], [], []
    with open(args.csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sizes.append(int(row["N"]))
            avg_tflops.append(float(row["Avg_TFLOPs"]))
            max_tflops.append(float(row["Max_TFLOPs"]))
            avg_gbs.append(float(row["Avg_GBs"]))
            max_gbs.append(float(row["Max_GBs"]))

    if not sizes:
        print("No data found in CSV.", file=sys.stderr)
        sys.exit(1)

    size_labels = [str(n) for n in sizes]

    # TFLOP/s plot
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(size_labels, avg_tflops, "o-", label="Avg TFLOP/s", color="tab:blue", linewidth=2)
    ax1.plot(size_labels, max_tflops, "s--", label="Max TFLOP/s", color="tab:cyan", linewidth=2)
    ax1.set_xlabel("Matrix Size (M=K=N)")
    ax1.set_ylabel("TFLOP/s")
    ax1.set_title("GEMM Throughput (TFLOP/s)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    for i, (a, m) in enumerate(zip(avg_tflops, max_tflops)):
        ax1.annotate(f"{a:.1f}", (size_labels[i], a), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=8)
        ax1.annotate(f"{m:.1f}", (size_labels[i], m), textcoords="offset points",
                     xytext=(0, -15), ha="center", fontsize=8)
    fig1.tight_layout()
    tflops_path = f"{args.output_dir}/sweep_tflops.png"
    fig1.savefig(tflops_path, dpi=150)
    print(f"TFLOP/s plot saved to {tflops_path}")

    # GB/s plot
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(size_labels, avg_gbs, "o-", label="Avg GB/s", color="tab:orange", linewidth=2)
    ax2.plot(size_labels, max_gbs, "s--", label="Max GB/s", color="tab:red", linewidth=2)
    ax2.set_xlabel("Matrix Size (M=K=N)")
    ax2.set_ylabel("GB/s")
    ax2.set_title("GEMM Bandwidth (GB/s)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    for i, (a, m) in enumerate(zip(avg_gbs, max_gbs)):
        ax2.annotate(f"{a:.0f}", (size_labels[i], a), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=8)
        ax2.annotate(f"{m:.0f}", (size_labels[i], m), textcoords="offset points",
                     xytext=(0, -15), ha="center", fontsize=8)
    fig2.tight_layout()
    gbs_path = f"{args.output_dir}/sweep_gbs.png"
    fig2.savefig(gbs_path, dpi=150)
    print(f"GB/s plot saved to {gbs_path}")


if __name__ == "__main__":
    main()
