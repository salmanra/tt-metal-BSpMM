#!/usr/bin/env python3
"""
plot_sddmm_profiling.py

Comprehensive SDDMM profiling visualization suite.

Generates 9 figures across 4 categories:
  Category A (Figs 1-4): Throughput comparison — Naive vs CDA
  Category B (Fig 5):    CDA speedup analysis per sweep
  Category C (Figs 6-8): Ablation analysis — cost breakdown
  Category D (Fig 9):    Device-level load imbalance & variability

Usage:
    python sddmm_scripts/plot_sddmm_profiling.py
    python sddmm_scripts/plot_sddmm_profiling.py --fig 5
    python sddmm_scripts/plot_sddmm_profiling.py --data-dir /path/to/opt/csvs --out-dir plots/
    python sddmm_scripts/plot_sddmm_profiling.py --clean
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


# ── Algorithm metadata ────────────────────────────────────────────────────────

SDDMM_ALGOS = [
    "bsr_sddmm_multicore_naive",
    "bsr_sddmm_multicore_CDA",
]

SDDMM_ALGO_LABEL = {
    "bsr_sddmm_multicore_naive": "Naive",
    "bsr_sddmm_multicore_CDA":  "CDA",
}

SDDMM_ALGO_COLOR = {
    "bsr_sddmm_multicore_naive": "#1565C0",   # blue
    "bsr_sddmm_multicore_CDA":  "#E53935",    # red
}

ABLATION_VARIANTS = [
    "",              # full run
    "_no_b_read",    # skip sparse B mask reads
    "_no_c_read",    # skip dense C reads (+ CDA sharing)
    "_no_d_read",    # skip dense D reads (+ CDA sharing)
    "_no_compute",   # skip matmul + Hadamard
    "_no_write",     # skip output DRAM writes
]

ABLATION_LABELS = {
    "no_b_read":  "B-reads\n(sparse mask)",
    "no_c_read":  "C-reads\n(dense left)",
    "no_d_read":  "D-reads\n(dense right)",
    "no_compute": "Compute\n(matmul+Had.)",
    "no_write":   "Writes\n(output)",
}

ABLATION_COLORS = {
    "no_b_read":  "#AB47BC",   # purple
    "no_c_read":  "#1565C0",   # blue
    "no_d_read":  "#E53935",   # red
    "no_compute": "#FB8C00",   # orange
    "no_write":   "#43A047",   # green
}

OPT_DATA_DIR = Path("sddmm_profiles/opt/csvs")
NAIVE_DATA_DIR = Path("sddmm_profiles/naive/csvs")

# The "Device program Loop" zone wraps 10 EnqueueProgram calls (1 warmup outside).
NUM_ITERS = 10


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


# ── Data loading helpers ──────────────────────────────────────────────────────

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


def get_metric(csv_path: Path, zone: str = "Device program Loop") -> float | None:
    """Extract total_ns for a named zone from a host profiler CSV."""
    try:
        df = pd.read_csv(csv_path, usecols=["name", "total_ns"])
        row = df[df["name"] == zone]
        return float(row["total_ns"].iloc[0]) if not row.empty else None
    except Exception:
        return None


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


def extract_device_kernel_stats(log_path: Path) -> dict | None:
    """
    Parse TRISC1 kernel duration stats from a mask log file.
    Returns dict with keys: avg_cycles, min_cycles, max_cycles, std_cycles, count
    """
    if not log_path.exists():
        return None
    try:
        with open(log_path) as f:
            content = f.read()
    except Exception:
        return None
    # Check for the TRISC1 section
    if "TRISC1 kernel duration" not in content:
        return None
    result = {}
    patterns = [
        ("count",      r"Count:\s+(\d+)"),
        ("avg_cycles", r"Avg:\s+(\d+)\s+cycles"),
        ("min_cycles", r"Min:\s+(\d+)\s+cycles"),
        ("max_cycles", r"Max:\s+(\d+)\s+cycles"),
        ("std_cycles", r"Std:\s+(\d+)\s+cycles"),
    ]
    for key, pat in patterns:
        m = re.search(pat, content)
        if m:
            result[key] = int(m.group(1))
    if "avg_cycles" not in result:
        return None
    return result


def _host_tflops(nblocks: int, R: int, C: int, K: int, total_ns: float) -> float:
    """
    Compute host-side TFLOPs/s from total nanoseconds.
    SDDMM FLOPs = 2*nblocks*R*C*K (matmul) + nblocks*R*C (Hadamard)
    """
    flops = 2 * nblocks * R * C * K + nblocks * R * C
    return flops / 1e12 / (total_ns / 1e9)


# ── Sweep data loading ────────────────────────────────────────────────────────

def load_sddmm_sweep(data_dir: Path, registry: str, sweep_param: str,
                      algos: list | None = None) -> pd.DataFrame:
    """
    Load timing data for SDDMM algorithms in a parametric sweep.
    Uses host-side "Device program Loop" timing to compute TFLOPs/s
    (captures CDA's data movement benefits, unlike device TFLOP/s).
    """
    if algos is None:
        algos = SDDMM_ALGOS
    rows = []
    reg_dir = data_dir / registry
    for algo in algos:
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
            ns = get_metric(csv)
            if ns is not None and nblocks is not None:
                ns_per_iter = ns / NUM_ITERS
                ms = ns_per_iter / 1e6
                rows.append({
                    "algo": algo, **params,
                    "nblocks": nblocks,
                    "ms": ms,
                    "tflops": _host_tflops(nblocks, params["R"],
                                           params["C"], params["K"],
                                           ns_per_iter),
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["algo", sweep_param]).reset_index(drop=True)
    return df


# ── Ablation data loading ─────────────────────────────────────────────────────

def load_sddmm_ablation(data_dir: Path, registry: str) -> pd.DataFrame:
    """
    Load timing for all (algo, variant, test_case) triples in a registry.
    Returns DataFrame with columns: algo, variant, M, N, K, R, C, density, ms
    """
    rows = []
    reg_dir = data_dir / registry
    for algo in SDDMM_ALGOS:
        for suffix in ABLATION_VARIANTS:
            algo_dir = reg_dir / f"{algo}{suffix}"
            if not algo_dir.exists():
                continue
            variant = suffix.lstrip("_") or "full"
            for csv in sorted(algo_dir.glob("*.csv")):
                if csv.suffix != ".csv" or csv.stem.endswith(".device"):
                    continue
                params = _parse_parametric(csv.stem)
                if params is None:
                    continue
                ns = get_metric(csv)
                if ns is not None:
                    rows.append({
                        "algo": algo,
                        "variant": variant,
                        **params,
                        "ms": ns / NUM_ITERS / 1e6,
                    })
    return pd.DataFrame(rows)


# ── Device stats loading ──────────────────────────────────────────────────────

def load_sddmm_device_stats(data_dir: Path, registry: str,
                             sweep_param: str) -> pd.DataFrame:
    """
    Load TRISC1 kernel duration stats for full-run (non-ablation) directories.
    Returns DataFrame with: algo, <params>, avg_cycles, max_avg_ratio, cv_pct
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
            stats = extract_device_kernel_stats(log)
            if stats is None:
                continue
            avg = stats["avg_cycles"]
            rows.append({
                "algo": algo, **params,
                "avg_cycles": avg,
                "max_avg_ratio": stats["max_cycles"] / avg if avg > 0 else 0,
                "cv_pct": stats.get("std_cycles", 0) / avg * 100 if avg > 0 else 0,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["algo", sweep_param]).reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Category A: Throughput Comparison (Figs 1-4)
# ══════════════════════════════════════════════════════════════════════════════

def _bar_chart(df: pd.DataFrame, sweep_col: str, fmt_label, title_param: str,
               out_dir: Path, out_name: str, algos: list | None = None,
               clean: bool = False) -> None:
    """Reusable grouped bar chart of TFLOPs/s vs. a sweep parameter."""
    if algos is None:
        algos = SDDMM_ALGOS
    values = sorted(df[sweep_col].unique())
    tick_labels = [fmt_label(v) for v in values]

    n_algos = len(algos)
    x = np.arange(len(values))
    bar_w = 0.5 if n_algos == 1 else 0.75 / n_algos

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, algo in enumerate(algos):
        sub = df[df["algo"] == algo]
        ys = []
        for v in values:
            row = sub[sub[sweep_col] == v]
            ys.append(row["tflops"].iloc[0] if not row.empty else 0)
        offset = 0 if n_algos == 1 else (i - (n_algos - 1) / 2) * bar_w
        bars = ax.bar(x + offset, ys, bar_w,
                      label=SDDMM_ALGO_LABEL.get(algo, algo),
                      color=SDDMM_ALGO_COLOR.get(algo, "#888888"),
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
            ax.legend(fontsize=9, loc="upper right")
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


def make_figure1(data_dir: Path, naive_data_dir: Path, out_dir: Path,
                 clean: bool = False) -> None:
    """Fig 1: Throughput vs. Sparsity Density (Naive vs CDA)."""
    df = load_sddmm_sweep(data_dir, "SDDMMSweepDensity", "density")
    if df.empty:
        print("WARNING: No density sweep data found. Skipping Figure 1.")
        return
    _bar_chart(df, "density", lambda v: f"{v}%", "Sparsity Density",
               out_dir, "sddmm_fig1_density_throughput", clean=clean)


def make_figure2(data_dir: Path, naive_data_dir: Path, out_dir: Path,
                 clean: bool = False) -> None:
    """Fig 2: Throughput vs. N (Dense Output Width)."""
    df = load_sddmm_sweep(data_dir, "SDDMMSweepN", "N")
    if df.empty:
        print("WARNING: No N-sweep data found. Skipping Figure 2.")
        return
    _bar_chart(df, "N", str, "N (Dense Output Width)",
               out_dir, "sddmm_fig2_n_throughput", clean=clean)


def make_figure3(data_dir: Path, naive_data_dir: Path, out_dir: Path,
                 clean: bool = False) -> None:
    """Fig 3: Throughput vs. K (Reduction Dimension)."""
    df = load_sddmm_sweep(data_dir, "SDDMMSweepK", "K")
    if df.empty:
        print("WARNING: No K-sweep data found. Skipping Figure 3.")
        return
    _bar_chart(df, "K", str, "K (Reduction Dimension)",
               out_dir, "sddmm_fig3_k_throughput", clean=clean)


def make_figure4(data_dir: Path, naive_data_dir: Path, out_dir: Path,
                 clean: bool = False) -> None:
    """Fig 4: Throughput vs. Block Size (Naive only — CDA data not available)."""
    df = load_sddmm_sweep(naive_data_dir, "SDDMMSweepBlockSize", "R",
                           algos=["bsr_sddmm_multicore_naive"])
    if df.empty:
        print("WARNING: No block-size sweep data found. Skipping Figure 4.")
        return
    _bar_chart(df, "R", lambda v: f"{v}\u00d7{v}", "Block Size (R=C)",
               out_dir, "sddmm_fig4_blocksize_throughput",
               algos=["bsr_sddmm_multicore_naive"], clean=clean)


# ══════════════════════════════════════════════════════════════════════════════
# Category B: CDA Speedup Analysis (Fig 5)
# ══════════════════════════════════════════════════════════════════════════════

_SWEEP_CONFIGS = [
    ("SDDMMSweepDensity", "density", lambda v: f"{v}%",   "Sparsity Density"),
    ("SDDMMSweepN",       "N",       str,                 "N (Dense Width)"),
    ("SDDMMSweepK",       "K",       str,                 "K (Reduction Dim.)"),
]


def make_figure5(data_dir: Path, naive_data_dir: Path, out_dir: Path,
                 clean: bool = False) -> None:
    """
    Fig 5: CDA Speedup over Naive — 3-column layout.
    Top row: absolute runtime bars (Naive vs CDA).
    Bottom row: speedup ratio bars.
    """
    fig = plt.figure(figsize=(16, 7), constrained_layout=False)
    outer = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    for col_idx, (registry, sweep_col, fmt_label, title) in enumerate(_SWEEP_CONFIGS):
        df = load_sddmm_sweep(data_dir, registry, sweep_col)
        if df.empty:
            continue

        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[col_idx],
            height_ratios=[3, 1], hspace=0.35,
        )
        ax_top = fig.add_subplot(inner[0])
        ax_bot = fig.add_subplot(inner[1])

        values = sorted(df[sweep_col].unique())
        tick_labels = [fmt_label(v) for v in values]
        x = np.arange(len(values))
        bar_w = 0.35

        # Collect ms per algo
        algo_ms = {}
        for algo in SDDMM_ALGOS:
            sub = df[df["algo"] == algo]
            algo_ms[algo] = [
                sub.loc[sub[sweep_col] == v, "ms"].iloc[0]
                if not sub[sub[sweep_col] == v].empty else 0
                for v in values
            ]

        # Top panel: absolute runtime bars
        for i, algo in enumerate(SDDMM_ALGOS):
            offset = (i - 0.5) * bar_w
            bars = ax_top.bar(
                x + offset, algo_ms[algo], bar_w,
                label=SDDMM_ALGO_LABEL[algo], color=SDDMM_ALGO_COLOR[algo],
                edgecolor="white", linewidth=0.5, zorder=3,
            )
            if not clean:
                for bar in bars:
                    h = bar.get_height()
                    if h > 0:
                        ax_top.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                                    f"{h:.1f}", ha="center", va="bottom",
                                    fontsize=7)

        ax_top.set_xticks(x)
        ax_top.set_xticklabels(tick_labels, fontsize=8)
        ax_top.set_ylabel("Runtime (ms)")
        ax_top.set_ylim(bottom=0)
        if not clean:
            ax_top.set_title(title, fontweight="bold")
            ax_top.legend(fontsize=7, loc="upper left")
            ax_top.grid(axis="y", alpha=0.25)
            ax_top.set_axisbelow(True)
        else:
            ax_top.set_title(title)
            ax_top.tick_params(axis="both", length=0)

        # Bottom panel: speedup ratio
        naive_key = "bsr_sddmm_multicore_naive"
        cda_key = "bsr_sddmm_multicore_CDA"
        naive_ms = algo_ms.get(naive_key, [])
        cda_ms = algo_ms.get(cda_key, [])
        if naive_ms and cda_ms:
            speedups = [
                (n / c if c > 0 and n > 0 else 1.0)
                for n, c in zip(naive_ms, cda_ms)
            ]
            colors = ["#43A047" if s > 1.0 else "#E53935" for s in speedups]
            bars = ax_bot.bar(x, speedups, 0.5, color=colors,
                              edgecolor="white", linewidth=0.5, zorder=3)
            if not clean:
                for bar, s in zip(bars, speedups):
                    ax_bot.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + 0.005,
                                f"{s:.3f}x", ha="center", va="bottom",
                                fontsize=7.5, fontweight="bold")
            ax_bot.axhline(1.0, color="black", linewidth=0.8,
                           linestyle="--", alpha=0.5)
            ax_bot.set_xticks(x)
            ax_bot.set_xticklabels(tick_labels, fontsize=8)
            ax_bot.set_ylabel("Speedup\n(CDA vs Naive)")
            pad = 0.05
            ax_bot.set_ylim(min(speedups) - pad, max(speedups) + pad)
            if not clean:
                ax_bot.grid(axis="y", alpha=0.25)
                ax_bot.set_axisbelow(True)
            else:
                ax_bot.tick_params(axis="both", length=0)

    if not clean:
        fig.suptitle(
            "SDDMM: CDA Speedup over Naive\n"
            "(M=8192, N=8192, K=8192, R=C=256 unless swept)",
            fontsize=13, fontweight="bold", y=1.02,
        )

    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.08)
    suffix = "_clean" if clean else ""
    out = out_dir / f"sddmm_fig5_cda_speedup{suffix}.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Category C: Ablation Analysis (Figs 6-8)
# ══════════════════════════════════════════════════════════════════════════════

_SKIP_VARIANTS = ["no_b_read", "no_c_read", "no_d_read", "no_compute", "no_write"]


def _compute_savings(abl_df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """
    From an ablation DataFrame, compute % savings for each skip variant.
    savings = (full_ms - skip_ms) / full_ms * 100
    Returns DataFrame with columns: algo, <group_cols>, variant, savings_pct
    """
    full = abl_df[abl_df["variant"] == "full"]
    rows = []
    for _, full_row in full.iterrows():
        key = {col: full_row[col] for col in ["algo"] + group_cols}
        for v in _SKIP_VARIANTS:
            match = abl_df[
                (abl_df["variant"] == v) &
                (abl_df["algo"] == full_row["algo"])
            ]
            for col in group_cols:
                match = match[match[col] == full_row[col]]
            if match.empty:
                continue
            skip_ms = match.iloc[0]["ms"]
            savings = (full_row["ms"] - skip_ms) / full_row["ms"] * 100
            rows.append({**key, "variant": v, "savings_pct": savings})
    return pd.DataFrame(rows)


def make_figure6(data_dir: Path, naive_data_dir: Path, out_dir: Path,
                 clean: bool = False) -> None:
    """
    Fig 6: Ablation savings at reference point (d=25%, M=N=K=8192, R=C=256).
    Grouped bar chart: 5 component groups x 2 algorithms.
    """
    abl = load_sddmm_ablation(data_dir, "SDDMMSweepDensity")
    if abl.empty:
        print("WARNING: No ablation data found. Skipping Figure 6.")
        return

    # Filter to d=25% reference case
    abl = abl[abl["density"] == 25]
    if abl.empty:
        print("WARNING: No d=25% ablation data. Skipping Figure 6.")
        return

    savings = _compute_savings(abl, ["density"])
    if savings.empty:
        print("WARNING: Could not compute savings. Skipping Figure 6.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    n_algos = len(SDDMM_ALGOS)
    x = np.arange(len(_SKIP_VARIANTS))
    bar_w = 0.75 / n_algos

    for i, algo in enumerate(SDDMM_ALGOS):
        sub = savings[savings["algo"] == algo]
        ys = []
        for v in _SKIP_VARIANTS:
            row = sub[sub["variant"] == v]
            ys.append(row["savings_pct"].iloc[0] if not row.empty else 0)
        offset = (i - (n_algos - 1) / 2) * bar_w
        bars = ax.bar(x + offset, ys, bar_w,
                      label=SDDMM_ALGO_LABEL[algo],
                      color=SDDMM_ALGO_COLOR[algo],
                      edgecolor="white", linewidth=0.5, zorder=3)
        if not clean:
            for bar in bars:
                h = bar.get_height()
                va = "bottom" if h >= 0 else "top"
                y_pos = h + 0.3 if h >= 0 else h - 0.3
                ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                        f"{h:.1f}%", ha="center", va=va, fontsize=7.5,
                        fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)

    variant_labels = [ABLATION_LABELS[v] for v in _SKIP_VARIANTS]
    ax.set_xticks(x)
    ax.set_xticklabels(variant_labels, fontsize=9)
    ax.set_ylabel("Runtime Savings (%)")
    if not clean:
        ax.set_title(
            "SDDMM Ablation Savings — Which Component Costs the Most?\n"
            "(M=N=K=8192, R=C=256, density=25%)",
            pad=10, fontweight="bold",
        )
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    else:
        ax.tick_params(axis="both", length=0)

    fig.tight_layout()
    suffix = "_clean" if clean else ""
    out = out_dir / f"sddmm_fig6_ablation_savings{suffix}.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def make_figure7(data_dir: Path, naive_data_dir: Path, out_dir: Path,
                 clean: bool = False) -> None:
    """
    Fig 7: Ablation savings across density sweep — 2x3 line plot grid.
    Each subplot: one skip type. X=density, Y=% savings, two lines.
    """
    abl = load_sddmm_ablation(data_dir, "SDDMMSweepDensity")
    if abl.empty:
        print("WARNING: No ablation data found. Skipping Figure 7.")
        return

    savings = _compute_savings(abl, ["density"])
    if savings.empty:
        print("WARNING: Could not compute savings. Skipping Figure 7.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    if not clean:
        fig.suptitle(
            "SDDMM Ablation Savings Across Density\n"
            "% runtime reduction when each component is skipped "
            "(M=N=K=8192, R=C=256)",
            fontsize=13, fontweight="bold",
        )

    densities = sorted(savings["density"].unique())

    for idx, variant in enumerate(_SKIP_VARIANTS):
        row, col = divmod(idx, 3)
        ax = axes[row, col]
        accent = ABLATION_COLORS[variant]

        for algo in SDDMM_ALGOS:
            sub = savings[(savings["algo"] == algo) & (savings["variant"] == variant)]
            sub = sub.sort_values("density")
            marker = "" if clean else "o"
            ax.plot(densities,
                    [sub.loc[sub["density"] == d, "savings_pct"].iloc[0]
                     if not sub[sub["density"] == d].empty else np.nan
                     for d in densities],
                    f"{marker}-", label=SDDMM_ALGO_LABEL[algo],
                    color=SDDMM_ALGO_COLOR[algo], linewidth=1.8,
                    markersize=6, zorder=3)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
        ax.set_xticks(densities)
        ax.set_xticklabels([f"{d}%" for d in densities], fontsize=8)
        ax.set_xlabel("Density")
        ax.set_ylabel("Savings (%)")
        ax.set_title(ABLATION_LABELS[variant].replace("\n", " "),
                     pad=8, color=accent, fontweight="bold")
        if not clean:
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)
        else:
            ax.tick_params(axis="both", length=0)

    # Hide the 6th (empty) subplot
    axes[1, 2].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    suffix = "_clean" if clean else ""
    out = out_dir / f"sddmm_fig7_ablation_by_density{suffix}.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def make_figure8(data_dir: Path, naive_data_dir: Path, out_dir: Path,
                 clean: bool = False) -> None:
    """
    Fig 8: Cost breakdown — horizontal stacked bars (one per algorithm).
    Components: B-reads, C-reads, D-reads, Compute, Writes, Overhead.
    Negative component savings clamped to 0; absorbed into Overhead.
    """
    abl = load_sddmm_ablation(data_dir, "SDDMMSweepDensity")
    if abl.empty:
        print("WARNING: No ablation data found. Skipping Figure 8.")
        return

    abl = abl[abl["density"] == 25]
    if abl.empty:
        print("WARNING: No d=25% ablation data. Skipping Figure 8.")
        return

    components = _SKIP_VARIANTS  # same order
    comp_labels = [ABLATION_LABELS[v].replace("\n", " ") for v in components]
    comp_colors = [ABLATION_COLORS[v] for v in components]

    fig, ax = plt.subplots(figsize=(12, 4))

    y_positions = np.arange(len(SDDMM_ALGOS))
    bar_height = 0.5

    for algo_idx, algo in enumerate(SDDMM_ALGOS):
        full_row = abl[(abl["algo"] == algo) & (abl["variant"] == "full")]
        if full_row.empty:
            continue
        full_ms = full_row.iloc[0]["ms"]

        fractions = []
        for v in components:
            skip_row = abl[(abl["algo"] == algo) & (abl["variant"] == v)]
            if skip_row.empty:
                fractions.append(0)
            else:
                saving = (full_ms - skip_row.iloc[0]["ms"]) / full_ms * 100
                fractions.append(max(saving, 0))  # clamp negative to 0

        overhead = max(100 - sum(fractions), 0)
        fractions.append(overhead)

        # Draw stacked horizontal bars
        all_fractions = fractions
        all_colors = comp_colors + ["#BDBDBD"]
        left = 0
        for frac, color in zip(all_fractions, all_colors):
            ax.barh(algo_idx, frac, bar_height, left=left, color=color,
                    edgecolor="white", linewidth=0.5, zorder=3)
            if not clean and frac > 3:
                ax.text(left + frac / 2, algo_idx, f"{frac:.1f}%",
                        ha="center", va="center", fontsize=8,
                        fontweight="bold", color="white")
            left += frac

    ax.set_yticks(y_positions)
    ax.set_yticklabels([SDDMM_ALGO_LABEL[a] for a in SDDMM_ALGOS], fontsize=10)
    ax.set_xlabel("Runtime Fraction (%)")
    ax.set_xlim(0, 105)

    if not clean:
        ax.set_title(
            "SDDMM Cost Breakdown (M=N=K=8192, R=C=256, density=25%)\n"
            "Fractions are independent upper bounds — ops partially overlap",
            pad=10, fontweight="bold",
        )
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=c, label=l)
            for c, l in zip(comp_colors + ["#BDBDBD"],
                            comp_labels + ["Overhead"])
        ]
        ax.legend(handles=legend_elements, fontsize=8, loc="upper right",
                  ncol=3, bbox_to_anchor=(1.0, -0.15))
    else:
        ax.tick_params(axis="both", length=0)

    fig.tight_layout()
    suffix = "_clean" if clean else ""
    out = out_dir / f"sddmm_fig8_cost_breakdown{suffix}.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Category D: Device-Level Analysis (Fig 9)
# ══════════════════════════════════════════════════════════════════════════════

def make_figure9(data_dir: Path, naive_data_dir: Path, out_dir: Path,
                 clean: bool = False) -> None:
    """
    Fig 9: Load Imbalance & Kernel Variability — 2x3 grid.
    Top row: Max/Avg cycle ratio across Density, N, K sweeps.
    Bottom row: Coefficient of variation (Std/Avg %) across same sweeps.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    if not clean:
        fig.suptitle(
            "SDDMM Device-Level Analysis: Load Imbalance & Kernel Variability\n"
            "(TRISC1 compute kernel duration across cores)",
            fontsize=13, fontweight="bold",
        )

    metrics = [
        ("max_avg_ratio", "Max / Avg Cycles", "Load Imbalance"),
        ("cv_pct",        "CV (Std/Avg %)",   "Kernel Variability"),
    ]

    for row_idx, (metric_col, ylabel, row_title) in enumerate(metrics):
        for col_idx, (registry, sweep_col, fmt_label, title) in enumerate(_SWEEP_CONFIGS):
            ax = axes[row_idx, col_idx]
            df = load_sddmm_device_stats(data_dir, registry, sweep_col)
            if df.empty:
                ax.set_title(f"{title}\n(no data)", fontsize=9)
                continue

            values = sorted(df[sweep_col].unique())

            for algo in SDDMM_ALGOS:
                sub = df[df["algo"] == algo].sort_values(sweep_col)
                ys = [
                    sub.loc[sub[sweep_col] == v, metric_col].iloc[0]
                    if not sub[sub[sweep_col] == v].empty else np.nan
                    for v in values
                ]
                marker = "" if clean else "o"
                ax.plot(values, ys, f"{marker}-",
                        label=SDDMM_ALGO_LABEL[algo],
                        color=SDDMM_ALGO_COLOR[algo],
                        linewidth=1.8, markersize=6, zorder=3)

            if metric_col == "max_avg_ratio":
                ax.axhline(1.0, color="black", linewidth=0.8,
                           linestyle="--", alpha=0.4)

            tick_labels = [fmt_label(v) for v in values]
            ax.set_xticks(values)
            ax.set_xticklabels(tick_labels, fontsize=8)
            ax.set_ylabel(ylabel)
            if not clean:
                ax.set_title(title, fontweight="bold", fontsize=10)
                ax.legend(fontsize=8)
                ax.grid(axis="y", alpha=0.25)
                ax.set_axisbelow(True)
            else:
                ax.set_title(title, fontsize=10)
                ax.tick_params(axis="both", length=0)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    suffix = "_clean" if clean else ""
    out = out_dir / f"sddmm_fig9_device_analysis{suffix}.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ── Entry point ───────────────────────────────────────────────────────────────

ALL_FIGURES = {
    "1": make_figure1,
    "2": make_figure2,
    "3": make_figure3,
    "4": make_figure4,
    "5": make_figure5,
    "6": make_figure6,
    "7": make_figure7,
    "8": make_figure8,
    "9": make_figure9,
}


def main():
    parser = argparse.ArgumentParser(
        description="Plot SDDMM profiling results — comprehensive suite.")
    parser.add_argument("--data-dir", type=Path, default=OPT_DATA_DIR,
                        help="Primary data directory (opt, with CDA + ablation)")
    parser.add_argument("--naive-data-dir", type=Path, default=NAIVE_DATA_DIR,
                        help="Fallback data directory for naive-only sweeps")
    parser.add_argument("--out-dir", type=Path, default=Path("sddmm_plots"),
                        help="Output directory for figures")
    parser.add_argument("--fig", choices=list(ALL_FIGURES.keys()) + ["all"],
                        default="all",
                        help="Which figure to generate (default: all)")
    parser.add_argument("--clean", action="store_true",
                        help="Generate clean figures (no titles/annotations)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    figs_to_run = ALL_FIGURES if args.fig == "all" else {args.fig: ALL_FIGURES[args.fig]}

    for num, fn in figs_to_run.items():
        fn(args.data_dir, args.naive_data_dir, args.out_dir, clean=args.clean)

    # Generate clean variants alongside normal when running all
    if args.fig == "all" and not args.clean:
        clean_dir = args.out_dir / "clean"
        clean_dir.mkdir(parents=True, exist_ok=True)
        for num, fn in ALL_FIGURES.items():
            fn(args.data_dir, args.naive_data_dir, clean_dir, clean=True)


if __name__ == "__main__":
    main()
