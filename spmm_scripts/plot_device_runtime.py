#!/usr/bin/env python3
"""
plot_device_runtime.py

Plot throughput from profiles_device_runtime/csvs, reading TFLOPs directly
from the sparse.log files (Device TFLOP/s line) instead of computing them.

Produces two figures per algorithm variant:
  1. Throughput (TFLOPs/s) grouped by density, colored by sparsity pattern
  2. Same data as % of N150 theoretical peak (74 TFLOPs/s)

Usage:
    python spmm_scripts/plot_device_runtime.py
    python spmm_scripts/plot_device_runtime.py --out-dir my_plots/
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_DIR = Path("/home/user/tt-metal/profiles_device_runtime/csvs")

N150_PEAK_TFLOPS = 74.0


# ── Style ─────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "legend.fontsize":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})

_PATTERN_COLOR = {
    "random":     "#1565C0",
    "row":        "#E53935",
    "col":        "#43A047",
    "multi_diag": "#7B1FA2",
}

_SPARSITY_PATTERN_ORDER = ["row", "col", "multi_diag", "random"]
_SPARSITY_PATTERN_LABELS = {
    "random":     "Random",
    "col":        "Column",
    "multi_diag": "Multi-Diag",
    "row":        "Row",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_tflops_from_log(log_path: Path) -> float | None:
    """Read the 'Device TFLOP/s:' value from a sparse.log file."""
    try:
        with open(log_path, "r") as f:
            for line in f:
                if "Device TFLOP/s:" in line:
                    return float(line.split(":")[1].strip())
    except (FileNotFoundError, ValueError):
        pass
    return None


def _parse_sparsity_pattern_stem(stem: str) -> tuple[str | None, dict | None]:
    # Strip _Disable__... suffix produced by --no-zones profiling runs
    stem_clean = re.sub(r"_Disable__.*$", "", stem)
    m = re.match(
        r"parametric_(?:(multi_diag|col|row)_)?M(\d+)_N(\d+)_K(\d+)_R(\d+)_C(\d+)_d(\d+)",
        stem_clean,
    )
    if not m:
        return None, None
    pattern = m.group(1) or "random"
    params = dict(zip(["M", "N", "K", "R", "C", "density"],
                      [int(x) for x in m.groups()[1:]]))
    return pattern, params


# ── Registry / density config ────────────────────────────────────────────────

_DENSITY_PANELS = [
    ("ProfileSweepSparsityPatternD5",  "5%",  5),
    ("ProfileSweepSparsityPatternD10", "10%", 10),
    ("ProfileSweepSparsityPattern",    "25%", 25),
    ("ProfileSweepSparsityPatternD50", "50%", 50),
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_sparsity_pattern_throughput(data_dir: Path, registry: str,
                                      algo_dir_name: str) -> pd.DataFrame:
    """Load throughput for an algo in a sparsity pattern sweep registry."""
    rows = []
    algo_dir = data_dir / registry / algo_dir_name
    if not algo_dir.exists():
        return pd.DataFrame()
    for log in sorted(algo_dir.glob("*_sparse.log")):
        case_stem = log.name.removesuffix("_sparse.log")
        pattern, params = _parse_sparsity_pattern_stem(case_stem)
        if pattern is None:
            continue
        tflops = parse_tflops_from_log(log)
        if tflops is None:
            continue
        rows.append({
            "pattern": pattern,
            **params,
            "tflops": tflops,
        })
    return pd.DataFrame(rows)


# ── Plotting ──────────────────────────────────────────────────────────────────

def make_sparsity_by_density(data_dir: Path, out_dir: Path,
                              algo_dir_name: str) -> None:
    """
    Single-axis grouped bar chart.
    Groups = density levels (5%, 10%, 25%, 50%).
    Bars within each group = sparsity patterns (Row, Column, Multi-Diag, Random).
    """
    all_rows = []
    for registry, density_label, density_val in _DENSITY_PANELS:
        df = load_sparsity_pattern_throughput(data_dir, registry, algo_dir_name)
        if df.empty:
            continue
        df["density_label"] = density_label
        df["density_val"] = density_val
        all_rows.append(df)

    if not all_rows:
        print(f"WARNING: No data for {algo_dir_name}. Skipping sparsity-by-density plot.")
        return

    combined = pd.concat(all_rows, ignore_index=True)

    patterns = [p for p in _SPARSITY_PATTERN_ORDER if p in combined["pattern"].values]
    density_labels = [d for _, d, _ in _DENSITY_PANELS if d in combined["density_label"].values]

    n_patterns = len(patterns)
    n_densities = len(density_labels)
    x = np.arange(n_densities)
    total_bar_width = 0.75
    bar_w = total_bar_width / max(n_patterns, 1)

    fig, ax = plt.subplots(figsize=(12, 6))

    for j, pat in enumerate(patterns):
        ys = []
        for dlabel in density_labels:
            row = combined[(combined["density_label"] == dlabel) & (combined["pattern"] == pat)]
            ys.append(row["tflops"].iloc[0] if not row.empty else 0)
        offset = (j - (n_patterns - 1) / 2) * bar_w
        bars = ax.bar(
            x + offset, ys, bar_w,
            label=_SPARSITY_PATTERN_LABELS.get(pat, pat),
            color=_PATTERN_COLOR.get(pat, "#888888"),
            edgecolor="white", linewidth=0.5, zorder=3,
        )
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.2,
                        f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Density = {d}" for d in density_labels], fontsize=11)
    ax.set_ylabel("Throughput (TFLOPs/s)")
    ax.set_ylim(bottom=0, top=N150_PEAK_TFLOPS * 1.1)
    ax.axhline(N150_PEAK_TFLOPS, color="black", linewidth=1.0, linestyle="--",
               alpha=0.3, label=f"N150 peak ({N150_PEAK_TFLOPS:.0f} TFLOPs/s)")
    ax.set_title(
        f"Tenstorrent N150 — CDA {algo_dir_name}: Throughput vs. Sparsity Pattern by Density\n"
        "8192x8192x8192, R=C=256  (device-runtime measurement)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = out_dir / f"sparsity_by_density_{algo_dir_name}.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def make_sparsity_by_density_pct(data_dir: Path, out_dir: Path,
                                  algo_dir_name: str) -> None:
    """
    Same as make_sparsity_by_density but y-axis is % of N150 theoretical peak.
    """
    all_rows = []
    for registry, density_label, density_val in _DENSITY_PANELS:
        df = load_sparsity_pattern_throughput(data_dir, registry, algo_dir_name)
        if df.empty:
            continue
        df["density_label"] = density_label
        df["density_val"] = density_val
        all_rows.append(df)

    if not all_rows:
        print(f"WARNING: No data for {algo_dir_name}. Skipping pct-peak plot.")
        return

    combined = pd.concat(all_rows, ignore_index=True)

    patterns = [p for p in _SPARSITY_PATTERN_ORDER if p in combined["pattern"].values]
    density_labels = [d for _, d, _ in _DENSITY_PANELS if d in combined["density_label"].values]

    n_patterns = len(patterns)
    n_densities = len(density_labels)
    x = np.arange(n_densities)
    total_bar_width = 0.75
    bar_w = total_bar_width / max(n_patterns, 1)

    fig, ax = plt.subplots(figsize=(12, 6))

    for j, pat in enumerate(patterns):
        ys = []
        for dlabel in density_labels:
            row = combined[(combined["density_label"] == dlabel) & (combined["pattern"] == pat)]
            tflops = row["tflops"].iloc[0] if not row.empty else 0
            ys.append(tflops / N150_PEAK_TFLOPS * 100)
        offset = (j - (n_patterns - 1) / 2) * bar_w
        bars = ax.bar(
            x + offset, ys, bar_w,
            label=_SPARSITY_PATTERN_LABELS.get(pat, pat),
            color=_PATTERN_COLOR.get(pat, "#888888"),
            edgecolor="white", linewidth=0.5, zorder=3,
        )
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                        f"{h:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Density = {d}" for d in density_labels], fontsize=11)
    ax.set_ylabel(f"% of N150 Theoretical Peak ({N150_PEAK_TFLOPS:.0f} TFLOPs/s)")
    ax.set_ylim(bottom=0, top=100)
    ax.axhline(100, color="black", linewidth=1.0, linestyle="--", alpha=0.3,
               label="100% peak")
    ax.set_title(
        f"Tenstorrent N150 — CDA {algo_dir_name}: Throughput as % of Peak ({N150_PEAK_TFLOPS:.0f} TFLOPs/s)\n"
        "8192x8192x8192, R=C=256  (device-runtime measurement)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = out_dir / f"sparsity_by_density_{algo_dir_name}_pct_peak.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot throughput from device-runtime profiling (TFLOPs read from sparse.log)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="CSV root directory")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("spmm_plots/device_runtime"),
                        help="Output directory for PNG figures")
    parser.add_argument("--fig", choices=["sparsity", "sparsity_pct", "all"],
                        default="all", help="Which figure to generate")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Discover all algo variants across all registries
    all_algo_dirs = set()
    for registry, _, _ in _DENSITY_PANELS:
        reg_path = args.data_dir / registry
        if reg_path.exists():
            for d in reg_path.iterdir():
                if d.is_dir():
                    all_algo_dirs.add(d.name)
    all_algo_dirs = sorted(all_algo_dirs)

    print(f"Algorithm variants found: {all_algo_dirs}")

    for algo in all_algo_dirs:
        if args.fig in ("sparsity", "all"):
            make_sparsity_by_density(args.data_dir, args.out_dir, algo)
        if args.fig in ("sparsity_pct", "all"):
            make_sparsity_by_density_pct(args.data_dir, args.out_dir, algo)

    # ── Summary table ──
    print(f"\n{'Registry':<45} {'Algo':<30} {'Pattern':<15} {'TFLOPs/s':>10} {'% Peak':>8}")
    print("-" * 112)
    for registry, density_label, _ in _DENSITY_PANELS:
        for algo in all_algo_dirs:
            df = load_sparsity_pattern_throughput(args.data_dir, registry, algo)
            if df.empty:
                continue
            for pat in _SPARSITY_PATTERN_ORDER:
                row = df[df["pattern"] == pat]
                if not row.empty:
                    t = row["tflops"].iloc[0]
                    pct = t / N150_PEAK_TFLOPS * 100
                    print(f"{registry:<45} {algo:<30} {pat:<15} {t:>10.2f} {pct:>7.1f}%")


if __name__ == "__main__":
    main()
