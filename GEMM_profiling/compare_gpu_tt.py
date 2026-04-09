#!/usr/bin/env python3
"""
compare_gpu_tt.py

Join the GPU baseline (``gemm_gpu.csv``) against the TT sweep results
(``gemm_tt.csv``) by ``Case`` and emit a side-by-side comparison table
covering ms, TFLOPs/s, GB/s, and the TT/GPU throughput ratio.

Usage:
    # From the repo root (/home/user/tt-metal):
    python GEMM_profiling/compare_gpu_tt.py
    python GEMM_profiling/compare_gpu_tt.py --gpu-csv ... --tt-csv ... --out ...
"""

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_GPU_CSV = Path(
    "tt_metal/programming_examples/rahmy/gpu-normalized/gemm_gpu.csv"
)
DEFAULT_TT_CSV = Path("GEMM_profiling/gemm_tt.csv")
DEFAULT_OUT = Path("GEMM_profiling/gemm_comparison.csv")


def build_comparison(gpu_csv: Path, tt_csv: Path) -> pd.DataFrame:
    """Outer-join GPU and TT sweep results on ``Case`` and compute ratios.

    Output columns:
        Registry, Case, M, K, N,
        GPU_ms, TT_ms,
        GPU_TFLOPs, TT_TFLOPs, TFLOPs_ratio,
        GPU_GBs, TT_GBs, GBs_ratio

    ``TFLOPs_ratio`` and ``GBs_ratio`` are TT/GPU — values < 1 mean GPU is
    faster. Cells stay blank where one side is missing or failed.
    """
    gpu = pd.read_csv(gpu_csv)
    tt = pd.read_csv(tt_csv)

    keep = ["Registry", "Case", "M", "K", "N", "Avg_ms", "Avg_TFLOPs", "Avg_GBs"]
    gpu = gpu[keep].rename(columns={
        "Avg_ms": "GPU_ms",
        "Avg_TFLOPs": "GPU_TFLOPs",
        "Avg_GBs": "GPU_GBs",
    })
    tt = tt[keep].rename(columns={
        "Avg_ms": "TT_ms",
        "Avg_TFLOPs": "TT_TFLOPs",
        "Avg_GBs": "TT_GBs",
    })

    # Join on (Registry, Case): the case name "gemm_M8192_N8192_K8192" is the
    # anchor of every non-Reg-4 sweep, so it shows up once per registry — a
    # plain Case-only join would cross-product those rows.
    merged = gpu.merge(
        tt[["Registry", "Case", "TT_ms", "TT_TFLOPs", "TT_GBs"]],
        on=["Registry", "Case"],
        how="outer",
        sort=False,
    )

    merged["TFLOPs_ratio"] = merged["TT_TFLOPs"] / merged["GPU_TFLOPs"]
    merged["GBs_ratio"] = merged["TT_GBs"] / merged["GPU_GBs"]

    col_order = [
        "Registry", "Case", "M", "K", "N",
        "GPU_ms", "TT_ms",
        "GPU_TFLOPs", "TT_TFLOPs", "TFLOPs_ratio",
        "GPU_GBs", "TT_GBs", "GBs_ratio",
    ]
    return merged[col_order]


def print_summary(df: pd.DataFrame) -> None:
    """Print mean/median TT/GPU ratios overall and per registry."""
    valid = df.dropna(subset=["TFLOPs_ratio"])
    if valid.empty:
        print("\nNo valid GPU+TT pairs to summarize.")
        return

    print(f"\nSummary across {len(valid)} valid pair(s):")
    print(f"  Mean   TT/GPU TFLOPs: {valid['TFLOPs_ratio'].mean():.3f}")
    print(f"  Median TT/GPU TFLOPs: {valid['TFLOPs_ratio'].median():.3f}")
    print(f"  Mean   TT/GPU GB/s:   {valid['GBs_ratio'].mean():.3f}")
    print(f"  Median TT/GPU GB/s:   {valid['GBs_ratio'].median():.3f}")

    print("\nPer registry (mean TT/GPU TFLOPs):")
    per_reg = valid.groupby("Registry")["TFLOPs_ratio"].agg(["count", "mean"])
    for reg, row in per_reg.iterrows():
        print(f"  Reg {int(reg)}: {row['mean']:.3f}  ({int(row['count'])} cases)")


def main():
    parser = argparse.ArgumentParser(
        description="Compare TT vs GPU GEMM sweep results side-by-side.")
    parser.add_argument("--gpu-csv", type=Path, default=DEFAULT_GPU_CSV,
                        help="Path to gemm_gpu.csv (baseline)")
    parser.add_argument("--tt-csv", type=Path, default=DEFAULT_TT_CSV,
                        help="Path to gemm_tt.csv (TT sweep results)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="Path to comparison CSV")
    args = parser.parse_args()

    if not args.gpu_csv.exists():
        raise SystemExit(f"GPU CSV not found: {args.gpu_csv}")
    if not args.tt_csv.exists():
        raise SystemExit(
            f"TT CSV not found: {args.tt_csv}\n"
            f"  Run `python GEMM_profiling/sweep_from_gpu_csv.py` first."
        )

    df = build_comparison(args.gpu_csv, args.tt_csv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, float_format="%.3f")
    print(f"Saved {args.out}\n")

    # Pretty-print the table to the console.
    with pd.option_context("display.max_columns", None,
                           "display.width", 200):
        print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print_summary(df)


if __name__ == "__main__":
    main()
