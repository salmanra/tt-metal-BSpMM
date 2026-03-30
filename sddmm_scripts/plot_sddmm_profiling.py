#!/usr/bin/env python3
"""
plot_sddmm_profiling.py

Plot SDDMM profiling results — throughput bar charts for density, N, K,
and block-size sweeps.

Usage:
    python sddmm_scripts/plot_sddmm_profiling.py
    python sddmm_scripts/plot_sddmm_profiling.py --data-dir /path/to/csvs --out-dir plots/
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Algorithm metadata ─────────────────────────────────────────────────────────

SDDMM_ALGOS = [
    "bsr_sddmm_multicore_naive",
]

SDDMM_ALGO_LABEL = {
    "bsr_sddmm_multicore_naive": "Naive",
}

SDDMM_ALGO_COLOR = {
    "bsr_sddmm_multicore_naive": "#1565C0",
}

SDDMM_DATA_DIR = Path("sddmm_profiles/naive/csvs")



# ── Helpers (from spmm_scripts/plot_profiling_plan.py) ─────────────────────────

def parse_log_metadata(filepath):
    """Parse matrix metadata (H, W, R, C, nblocks) from a pretty_print log file."""
    result = {}
    try:
        with open(filepath, "r") as f:
            for line in f:
                if "(H x W)" in line:
                    parts = line.split(":")[1].strip().split(" x ")
                    result["H"], result["W"] = int(parts[0]), int(parts[1])
                elif "(R x C)" in line:
                    parts = line.split(":")[1].strip().split(" x ")
                    result["R"], result["C"] = int(parts[0]), int(parts[1])
                elif "Number of blocks" in line:
                    result["nblocks"] = int(line.split(":")[1].strip())
    except FileNotFoundError:
        pass
    return result


def _parse_parametric(stem: str) -> dict | None:
    m = re.match(
        r"parametric_M(\d+)_N(\d+)_K(\d+)_R(\d+)_C(\d+)_d(\d+)", stem
    )
    if not m:
        return None
    return dict(zip(["M", "N", "K", "R", "C", "density"],
                    [int(x) for x in m.groups()]))


# ── SDDMM-specific ────────────────────────────────────────────────────────────

def extract_device_tflops(log_path: Path) -> float | None:
    """Extract Device TFLOP/s from a mask log file (appended by read_sddmm_profiler.py)."""
    if not log_path.exists():
        return None
    try:
        with open(log_path) as f:
            content = f.read()
    except Exception:
        return None
    for pat in [
        r"Device\s+TFLOP/s:\s*([\d.]+)",
        r"TFLOP/?s:\s*([\d.]+)",
    ]:
        match = re.search(pat, content, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def load_sddmm_sweep(data_dir: Path, registry: str, sweep_param: str) -> pd.DataFrame:
    """
    Load timing data for all SDDMM algorithms in a parametric sweep.
    Returns a DataFrame with columns: algo, <sweep_param>, ..., tflops
    """
    rows = []
    reg_dir = data_dir / registry
    for algo in SDDMM_ALGOS:
        algo_dir = reg_dir / algo
        if not algo_dir.exists():
            continue
        for csv in sorted(algo_dir.glob("*.csv")):
            if csv.suffix != ".csv" or csv.stem.endswith(".device"):
                continue
            params = _parse_parametric(csv.stem)
            if params is None:
                continue
            log = csv.parent / f"{csv.stem}_mask.log"
            meta = parse_log_metadata(log)
            nblocks = meta.get("nblocks")
            tflops = extract_device_tflops(log)
            if tflops is not None and nblocks is not None:
                rows.append({
                    "algo": algo, **params,
                    "nblocks": nblocks,
                    "tflops": tflops,
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["algo", sweep_param]).reset_index(drop=True)
    return df


# ── Shared bar-chart helper ────────────────────────────────────────────────────

def _bar_chart(df: pd.DataFrame, sweep_col: str, fmt_label, title_param: str,
               out_dir: Path, out_name: str, clean: bool = False) -> None:
    """Reusable grouped bar chart of TFLOPs/s vs. a sweep parameter."""
    values = sorted(df[sweep_col].unique())
    tick_labels = [fmt_label(v) for v in values]

    n_algos = len(SDDMM_ALGOS)
    x = np.arange(len(values))
    if n_algos == 1:
        bar_w = 0.5
    else:
        bar_w = 0.75 / n_algos

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, algo in enumerate(SDDMM_ALGOS):
        sub = df[df["algo"] == algo]
        ys = []
        for v in values:
            row = sub[sub[sweep_col] == v]
            ys.append(row["tflops"].iloc[0] if not row.empty else 0)
        offset = 0 if n_algos == 1 else (i - (n_algos - 1) / 2) * bar_w
        bars = ax.bar(x + offset, ys, bar_w,
                      label=SDDMM_ALGO_LABEL[algo], color=SDDMM_ALGO_COLOR[algo],
                      edgecolor="white", linewidth=0.5, zorder=3)
        if not clean:
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                            f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=10)
    ax.set_ylabel("Throughput (TFLOPs/s)")
    ax.set_ylim(bottom=0)
    if not clean:
        row0 = df.iloc[0]
        fixed = []
        if sweep_col != "M":
            fixed.append(f"M={row0['M']}")
        if sweep_col != "N":
            fixed.append(f"N={row0['N']}")
        if sweep_col != "K":
            fixed.append(f"K={row0['K']}")
        if sweep_col != "R":
            fixed.append(f"R=C={row0['R']}")
        if sweep_col != "density":
            fixed.append(f"d={row0['density']}%")
        ax.set_title(
            f"SDDMM Throughput vs. {title_param}\n"
            f"({', '.join(fixed)})",
            pad=10, fontweight="bold",
        )
        if n_algos > 1:
            ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    else:
        ax.tick_params(axis="both", length=0)

    fig.tight_layout()
    suffix = "_clean" if clean else ""
    out = out_dir / f"{out_name}{suffix}.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ── Per-sweep figure functions ─────────────────────────────────────────────────

def make_sddmm_density_figure(data_dir: Path, out_dir: Path, clean: bool = False) -> None:
    """Bar chart — SDDMM throughput vs. sparsity density."""
    df = load_sddmm_sweep(data_dir, "SDDMMSweepDensity", "density")
    if df.empty:
        print("WARNING: No SDDMM density sweep data found. Skipping.")
        return
    _bar_chart(df, "density", lambda v: f"{v}%", "Sparsity Density",
               out_dir, "sddmm_density_throughput", clean)


def make_sddmm_n_figure(data_dir: Path, out_dir: Path, clean: bool = False) -> None:
    """Bar chart — SDDMM throughput vs. N (dense output width)."""
    df = load_sddmm_sweep(data_dir, "SDDMMSweepN", "N")
    if df.empty:
        print("WARNING: No SDDMM N-sweep data found. Skipping.")
        return
    _bar_chart(df, "N", str, "N (Dense Output Width)",
               out_dir, "sddmm_n_throughput", clean)


def make_sddmm_k_figure(data_dir: Path, out_dir: Path, clean: bool = False) -> None:
    """Bar chart — SDDMM throughput vs. K (reduction dimension)."""
    df = load_sddmm_sweep(data_dir, "SDDMMSweepK", "K")
    if df.empty:
        print("WARNING: No SDDMM K-sweep data found. Skipping.")
        return
    _bar_chart(df, "K", str, "K (Reduction Dimension)",
               out_dir, "sddmm_k_throughput", clean)


def make_sddmm_blocksize_figure(data_dir: Path, out_dir: Path, clean: bool = False) -> None:
    """Bar chart — SDDMM throughput vs. block size (R=C)."""
    df = load_sddmm_sweep(data_dir, "SDDMMSweepBlockSize", "R")
    if df.empty:
        print("WARNING: No SDDMM block-size sweep data found. Skipping.")
        return
    _bar_chart(df, "R", lambda v: f"{v}\u00d7{v}", "Block Size (R=C)",
               out_dir, "sddmm_blocksize_throughput", clean)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot SDDMM profiling results.")
    parser.add_argument("--data-dir", type=Path, default=SDDMM_DATA_DIR,
                        help="Root directory with CSV data")
    parser.add_argument("--out-dir", type=Path, default=Path("sddmm_plots"),
                        help="Output directory for figures")
    parser.add_argument("--clean", action="store_true",
                        help="Generate clean figures (no titles/annotations)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    make_sddmm_density_figure(args.data_dir, args.out_dir, clean=args.clean)
    make_sddmm_n_figure(args.data_dir, args.out_dir, clean=args.clean)
    make_sddmm_k_figure(args.data_dir, args.out_dir, clean=args.clean)
    make_sddmm_blocksize_figure(args.data_dir, args.out_dir, clean=args.clean)


if __name__ == "__main__":
    main()
