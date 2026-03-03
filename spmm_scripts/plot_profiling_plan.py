#!/usr/bin/env python3
"""
plot_profiling_plan.py

Plot the results of the full BSR SpMM profiling plan in three figures.

  Figure 1 — Cost analysis:  B-matrix DRAM reads dominate runtime
  Figure 2 — Scaling sweeps: how performance scales with N, K, density, block size
  Figure 3 — Ablation detail: % savings per test case for each skipped component

Usage:
    python spmm_scripts/plot_profiling_plan.py
    python spmm_scripts/plot_profiling_plan.py --data-dir /path/to/csvs --out-dir plots/
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd


# ── Algorithm metadata ─────────────────────────────────────────────────────────

# Ordered from fastest to slowest (for consistent legend / bar ordering)
ALGOS = [
    "bsr_spmm_multicore_snf",
    "bsr_spmm_multicore_load_balanced_new_DM",
    "bsr_spmm_multicore_naive_new_DM",
    "bsr_spmm_multicore_load_balanced",
    "bsr_spmm_multicore_reuse_iteration",
]

ALGO_LABEL = {
    "bsr_spmm_multicore_snf":                 "SNF",
    "bsr_spmm_multicore_load_balanced_new_DM": "LB (new DM)",
    "bsr_spmm_multicore_naive_new_DM":         "Naive (new DM)",
    "bsr_spmm_multicore_load_balanced":        "Load Balanced",
    "bsr_spmm_multicore_reuse_iteration":      "Reuse Iter.",
}

ALGO_COLOR = {
    "bsr_spmm_multicore_snf":                 "#1565C0",
    "bsr_spmm_multicore_load_balanced_new_DM": "#E53935",
    "bsr_spmm_multicore_naive_new_DM":         "#43A047",
    "bsr_spmm_multicore_load_balanced":        "#F57C00",
    "bsr_spmm_multicore_reuse_iteration":      "#7B1FA2",
}

ABLATION_VARIANTS = ["", "_no_a_read", "_no_b_read", "_no_compute", "_no_write"]

# Reference case used for ablation analysis
ABLATION_REGISTRY = "ProfileSuiteLargeSparseVersioning"
ABLATION_LARGE_BLOCKS_REGISTRY = "ProfileSuiteLargeSparseLargeBlocksVersioning"
ABLATION_CASE     = "profile_case_sparse_fill_random_large_R64_C64"

NUM_ITERS = 10


# ── Style ──────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "legend.fontsize":  9,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "figure.dpi":       150,
})


# ── Data loading ───────────────────────────────────────────────────────────────

# ── Log / metadata parsing ──────────────────────────────────────────

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

def get_metric(csv_path: Path, zone: str = "Device program Loop") -> float | None:
    """Read total_ns for one profiler zone from a host-code CSV."""
    try:
        df = pd.read_csv(csv_path, usecols=["name", "total_ns"])
        row = df[df["name"] == zone]
        return float(row["total_ns"].iloc[0]) if not row.empty else None
    except Exception:
        return None


def load_ablation(data_dir: Path, ablation_reg, ablation_case) -> pd.DataFrame:
    """
    Load ablation timing data for every algorithm × variant pair.
    Returns a DataFrame with columns: algo, variant, ms
    """
    rows = []
    reg_dir = data_dir / ablation_reg
    for algo in ALGOS:
        for suffix in ABLATION_VARIANTS:
            path = reg_dir / f"{algo}{suffix}" / f"{ablation_case}.csv"
            ns = get_metric(path)
            if ns is not None:
                ns = ns / NUM_ITERS
                rows.append({
                    "algo":    algo,
                    "variant": suffix.lstrip("_") or "full",
                    "ms":      ns / 1e6,
                })
    return pd.DataFrame(rows)


def _parse_parametric(stem: str) -> dict | None:
    m = re.match(
        r"parametric_M(\d+)_N(\d+)_K(\d+)_R(\d+)_C(\d+)_d(\d+)", stem
    )
    if not m:
        return None
    return dict(zip(["M", "N", "K", "R", "C", "density"],
                    [int(x) for x in m.groups()]))


def _tflops_per_sec(nblocks: int, R: int, C: int, N: int, ms: float) -> float:
    """
    Compute SpMM throughput in TFLOPs/s.
    FLOPs = 2 × nblocks × R × C × N
    nblocks is the actual nnz block count read from the _sparse.log file.
    """
    flops = 2 * nblocks * R * C * N
    return flops / 1e12 / (ms / 1e3)


def load_sweep(data_dir: Path, registry: str, sweep_param: str) -> pd.DataFrame:
    """
    Load timing data for all algorithms in a parametric sweep.
    Returns a DataFrame with columns: algo, <sweep_param>, ..., ms, tflops
    nblocks (and thus TFLOPs) is read from the sibling _sparse.log file.
    """
    rows = []
    reg_dir = data_dir / registry
    for algo in ALGOS:
        algo_dir = reg_dir / algo
        if not algo_dir.exists():
            continue
        for csv in sorted(algo_dir.glob("*.csv")):
            if csv.suffix != ".csv" or csv.stem.endswith(".device"):
                continue
            params = _parse_parametric(csv.stem)
            if params is None:
                continue
            log = csv.parent / f"{csv.stem}_sparse.log"
            meta = parse_log_metadata(log)
            nblocks = meta.get("nblocks")
            ns = get_metric(csv)
            if ns is not None and nblocks is not None:
                ns = ns / NUM_ITERS
                ms = ns / 1e6
                rows.append({
                    "algo": algo, **params, "ms": ms,
                    "nblocks": nblocks,
                    "tflops": _tflops_per_sec(nblocks, params["R"],
                                              params["C"], params["N"], ms),
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["algo", sweep_param]).reset_index(drop=True)
    return df


# ── Figure 1: Cost analysis ────────────────────────────────────────────────────

def _ablation_panel(ax: plt.Axes, abl: pd.DataFrame, clean: bool = False) -> None:
    """
    Panel A: side-by-side bars comparing Full vs No-B-read for each algorithm.
    Immediately shows that removing B-reads cuts ~70% of runtime.
    """
    full_ms  = [abl.loc[(abl.algo == a) & (abl.variant == "full"),  "ms"].iloc[0]
                for a in ALGOS if not abl.loc[(abl.algo == a) & (abl.variant == "full"),  "ms"].empty]
    nob_ms   = [abl.loc[(abl.algo == a) & (abl.variant == "no_b_read"), "ms"].iloc[0]
                for a in ALGOS if not abl.loc[(abl.algo == a) & (abl.variant == "no_b_read"), "ms"].empty]

    x      = np.arange(len(ALGOS))
    w      = 0.38
    labels = [ALGO_LABEL[a] for a in ALGOS]

    bars_full = ax.bar(x - w/2, full_ms, w, label="Full",
                       color="#455A64", edgecolor="white", linewidth=0.5)
    bars_nob  = ax.bar(x + w/2, nob_ms,  w, label="No B-reads",
                       color="#64B5F6", edgecolor="white", linewidth=0.5)

    # Annotate savings percentage above each Full bar
    if not clean:
        for bar_f, bar_n in zip(bars_full, bars_nob):
            saving = (bar_f.get_height() - bar_n.get_height()) / bar_f.get_height()
            ax.text(bar_f.get_x() + bar_f.get_width() / 2,
                    bar_f.get_height() + 4,
                    f"−{saving:.0%}", ha="center", va="bottom",
                    fontsize=8, color="#B71C1C", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    if not clean:
        ax.set_ylabel("Runtime (ms)")
        ax.set_title("Full vs. No-B-read Runtime\n"
                     "(M=K=8192, R=C=64, density=25%, random sparsity)",
                     pad=10)
        ax.legend()
    if not clean:
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    else:
        ax.tick_params(axis="both", length=0)
    ax.set_ylim(0, max(full_ms) * 1.18)


def _breakdown_panel(ax: plt.Axes, abl: pd.DataFrame, clean: bool = False) -> None:
    """
    Panel B: stacked percentage bar showing estimated cost breakdown.
    B-reads = full − no_b_read.  Compute = full − no_compute.
    Writes  = full − no_write.   A-reads = full − no_a_read (≈ 0).
    Remainder is pipeline / dispatch overhead.
    """
    component_color = {
        "B-reads":  "#E53935",
        "Compute":  "#FB8C00",
        "Writes":   "#43A047",
        "A-reads":  "#AB47BC",
        "Overhead": "#B0BEC5",
    }

    def variant_ms(algo, v):
        row = abl.loc[(abl.algo == algo) & (abl.variant == v), "ms"]
        return row.iloc[0] if not row.empty else np.nan

    fracs = {c: [] for c in component_color}
    for algo in ALGOS:
        full = variant_ms(algo, "full")
        no_b = variant_ms(algo, "no_b_read")
        no_c = variant_ms(algo, "no_compute")
        no_w = variant_ms(algo, "no_write")
        no_a = variant_ms(algo, "no_a_read")

        b  = max(0, full - no_b)
        c  = max(0, full - no_c)
        w  = max(0, full - no_w)
        a  = max(0, full - no_a)
        oh = max(0, full - b - c - w - a)

        fracs["B-reads"].append(b  / full * 100)
        fracs["Compute"].append(c  / full * 100)
        fracs["Writes"].append(w  / full * 100)
        fracs["A-reads"].append(a  / full * 100)
        fracs["Overhead"].append(oh / full * 100)

    x      = np.arange(len(ALGOS))
    bottom = np.zeros(len(ALGOS))
    for label, vals in fracs.items():
        vals = np.array(vals)
        ax.bar(x, vals, bottom=bottom, label=label,
               color=component_color[label], edgecolor="white", linewidth=0.5)
        # Annotate B-reads fraction in the middle of its segment
        if label == "B-reads" and not clean:
            for i, (v, b) in enumerate(zip(vals, bottom)):
                if v > 5:
                    ax.text(i, b + v / 2, f"{v:.0f}%",
                            ha="center", va="center",
                            fontsize=8.5, color="white", fontweight="bold")
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([ALGO_LABEL[a] for a in ALGOS], rotation=20, ha="right")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    if not clean:
        ax.set_ylabel("% of total runtime")
        ax.set_title("Estimated Cost Breakdown\n"
                     "(fractions are independent upper bounds — ops partially overlap)",
                     pad=10)
        ax.legend(loc="upper right", ncol=2)
    if not clean:
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    else:
        ax.tick_params(axis="both", length=0)


def make_figure1(data_dir: Path, out_dir: Path, clean: bool = False) -> None:
    abl = load_ablation(data_dir, ABLATION_REGISTRY, ABLATION_CASE)
    if abl.empty:
        print("WARNING: No ablation data found. Skipping Figure 1.")
        return

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5.5))
    if not clean:
        fig.suptitle(
            "BSR SpMM Cost Analysis: Dense B-Matrix DRAM Reads Dominate (~70% of Runtime)",
            fontsize=13, fontweight="bold",
        )
    _ablation_panel(ax_a, abl, clean=clean)
    _breakdown_panel(ax_b, abl, clean=clean)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    suffix = "_clean" if clean else ""
    out = out_dir / f"fig1_cost_analysis{suffix}.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 2: Scaling sweeps ───────────────────────────────────────────────────

def _sweep_panel(ax: plt.Axes, df: pd.DataFrame, param: str,
                 xlabel: str, title: str,
                 xscale: str = "linear",
                 algos: list = None,
                 clean: bool = False) -> None:
    """Generic line-plot panel for one sweep axis (Y-axis: TFLOPs/s)."""
    if algos is None:
        algos = ALGOS
    has_data = False
    marker = "" if clean else "o"
    for algo in algos:
        sub = df[df["algo"] == algo].sort_values(param)
        if sub.empty:
            continue
        ax.plot(sub[param], sub["tflops"], f"{marker}-",
                label=ALGO_LABEL[algo], color=ALGO_COLOR[algo],
                linewidth=1.8, markersize=5, zorder=3)
        has_data = True

    if not has_data:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        return

    if not clean:
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Throughput (TFLOPs/s)")
        ax.set_title(title, pad=8)
    ax.set_xscale(xscale)
    ax.set_ylim(bottom=0)
    if not clean:
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
    else:
        ax.tick_params(axis="both", length=0)
    if not clean:
        ax.legend(fontsize=8)


def make_figure2(data_dir: Path, out_dir: Path, clean: bool = False) -> None:
    n_df  = load_sweep(data_dir, "ProfileSweepN",         "N")
    k_df  = load_sweep(data_dir, "ProfileSweepK",         "K")
    d_df  = load_sweep(data_dir, "ProfileSweepDensity",   "density")
    bs_df = load_sweep(data_dir, "ProfileSweepBlockSize", "R")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    if not clean:
        fig.suptitle("BSR SpMM Throughput Sweeps",
                     fontsize=13, fontweight="bold")

    # ── Top-left: N sweep (show all algorithms — good for comparison) ──
    _sweep_panel(
        axes[0, 0], n_df, "N", "Output width N (columns of B / C)",
        "Efficiency vs. output width N\n(M=K=8192, R=C=64, density=25%)",
        xscale="log", clean=clean,
    )
    n_ticks = sorted(n_df["N"].unique()) if not n_df.empty else []
    axes[0, 0].set_xticks(n_ticks)
    axes[0, 0].set_xticklabels([str(v) for v in n_ticks])
    axes[0, 0].tick_params(axis="x", which="minor", bottom=False)

    # ── Top-right: K sweep (reduction dim — also compare all algos) ────
    _sweep_panel(
        axes[0, 1], k_df, "K", "Reduction dimension K",
        "Efficiency vs. reduction dimension K\n(M=N=8192, R=C=64, density=25%)\n"
        r"$\it{Note: K=1024\ anomaly\ likely\ dispatch\ overhead}$",
        xscale="log", clean=clean,
    )
    k_ticks = sorted(k_df["K"].unique()) if not k_df.empty else []
    axes[0, 1].set_xticks(k_ticks)
    axes[0, 1].set_xticklabels([str(v) for v in k_ticks])
    axes[0, 1].tick_params(axis="x", which="minor", bottom=False)

    # ── Bottom-left: density sweep ──────────────────────────────────────
    _sweep_panel(
        axes[1, 0], d_df, "density", "Sparsity density (%)",
        "Efficiency vs. sparsity density\n(M=N=K=8192, R=C=64)",
        xscale="linear", clean=clean,
    )

    # ── Bottom-right: block size sweep ──────────────────────────────────
    _sweep_panel(
        axes[1, 1], bs_df, "R", "BSR block size R=C",
        "Effect of BSR block size\n(M=N=K=8192, density=25%)",
        xscale="log", clean=clean,
    )
    # No linear guide here — block size has a non-linear sweet spot
    block_ticks = sorted(bs_df["R"].unique()) if not bs_df.empty else []
    axes[1, 1].set_xticks(block_ticks)
    axes[1, 1].set_xticklabels([f"{r}×{r}" for r in block_ticks])
    axes[1, 1].tick_params(axis="x", which="minor", bottom=False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    suffix = "_clean" if clean else ""
    out = out_dir / f"fig2_scaling_sweeps{suffix}.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 3: Ablation detail across all test cases ────────────────────────────

# Short display name for each test-case stem
def _case_label(stem: str) -> str:
    stem = stem.replace("profile_case_sparse_", "").replace("_large", "")
    # extract pattern and block size
    for pat, short in [("fill_lower_triangular", "tri"),
                       ("fill_column", "col"),
                       ("fill_random", "rand"),
                       ("fill_row", "row"),
                       ("diagonal",  "diag")]:
        if pat in stem:
            # extract block size suffix: R32_C32 / R64_C64 / R128_C128
            # or the mis-named triangular form: large32_C32
            m = re.search(r"[_]?(\d+)[_]C\d+", stem)
            size = m.group(1) if m else "?"
            return f"{short}\n{size}×{size}"
    return stem


def _block_size(s: str) -> int:
    """Extract BSR block size from a case stem (for sort key)."""
    m = re.search(r"(\d+)[_]C\d+$", s)
    return int(m.group(1)) if m else 0


# Explicit pattern ordering: row → diag → col → tri → rand
_PATTERN_RANK = {
    "fill_row":              0,
    "diagonal":              1,
    "fill_column":           2,
    "fill_lower_triangular": 3,
    "fill_random":           4,
}

def _pattern_rank(s: str) -> int:
    for pat, rank in _PATTERN_RANK.items():
        if pat in s:
            return rank
    return 99


def _sorted_cases(pivot: "pd.DataFrame") -> list:
    """Return case names sorted by explicit pattern rank then block size."""
    return sorted(pivot["case"].unique(), key=lambda s: (
        _pattern_rank(s),
        _block_size(s),
    ))


def load_ablation_all_cases(data_dir: Path, ablation_reg) -> pd.DataFrame:
    """
    Load full × ablation timing for every (algo, variant, test_case) triple.
    Returns a DataFrame with columns: algo, variant, case, ms
    """
    rows = []
    reg_dir = data_dir / ablation_reg
    # Collect all available test-case stems from the base (full) algorithm
    base_dir = reg_dir / ALGOS[0]
    cases = sorted(
        p.stem for p in base_dir.glob("*.csv")
        if not p.stem.endswith(".device")
    )
    for algo in ALGOS:
        for suffix in ABLATION_VARIANTS:
            algo_dir = reg_dir / f"{algo}{suffix}"
            for case in cases:
                path = algo_dir / f"{case}.csv"
                ns = get_metric(path)
                if ns is not None:
                    ns = ns / NUM_ITERS
                    rows.append({
                        "algo":    algo,
                        "variant": suffix.lstrip("_") or "full",
                        "case":    case,
                        "ms":      ns / 1e6,
                    })
    return pd.DataFrame(rows)


def make_ablation_plot(data_dir: Path, out_dir: Path, ablation_reg, fig_num, clean: bool = False) -> None:
    # shared logic for figs 3 and 4 (they're the same)
    df = load_ablation_all_cases(data_dir, ablation_reg)
    if df.empty:
        print(f"WARNING: No ablation data found. Skipping Figure {fig_num}.")
        return

    # Pivot so each row is (algo, case) with columns for each variant's ms
    pivot = df.pivot_table(index=["algo", "case"], columns="variant",
                        values="ms").reset_index()

    # Compute % savings for each skipped component
    for v in ["no_a_read", "no_b_read", "no_compute", "no_write"]:
        if v in pivot.columns:
            pivot[f"saving_{v}"] = (pivot["full"] - pivot[v]) / pivot["full"] * 100

    # Build ordered case list: group by pattern, ordered by block size within each.
    all_cases = _sorted_cases(pivot)
    case_labels = [_case_label(c) for c in all_cases]
    x = np.arange(len(all_cases))

    variants_cfg = [
        ("no_a_read",  "No-A-read savings\n(skip sparse A DRAM reads)",     "#7B1FA2"),
        ("no_b_read",  "No-B-read savings\n(skip dense B DRAM reads)",       "#C62828"),
        ("no_compute", "No-Compute savings\n(skip tile multiply)",            "#E65100"),
        ("no_write",   "No-Write savings\n(skip output DRAM writes)",         "#1B5E20"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    if not clean:
        fig.suptitle(
            "Ablation Savings Across All Test Cases\n"
            "% runtime reduction when each component is skipped",
            fontsize=13, fontweight="bold",
        )

    for ax, (variant, title, accent) in zip(axes.flat, variants_cfg):
        col = f"saving_{variant}"
        if col not in pivot.columns:
            ax.set_visible(False)
            continue

        marker = "" if clean else "o"
        for algo in ALGOS:
            sub = pivot[pivot["algo"] == algo].set_index("case")
            ys = [sub.loc[c, col] if c in sub.index else np.nan
                for c in all_cases]
            ax.plot(x, ys, f"{marker}-", label=ALGO_LABEL[algo],
                    color=ALGO_COLOR[algo], linewidth=1.6, markersize=5, zorder=3)

        # Zero-savings reference line
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)

        ax.set_xticks(x)
        ax.set_xticklabels(case_labels, fontsize=7.5)
        ax.set_ylabel("Runtime savings (%)")
        ax.set_title(title, pad=8, color=accent, fontweight="bold")
        if not clean:
            ax.legend(fontsize=8)
        if not clean:
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)
        else:
            ax.tick_params(axis="both", length=0)

        # Shade pattern groups (every 3 cases = one block-size triplet per pattern)
        for i in range(0, len(all_cases), 6):
            ax.axvspan(i - 0.5, i + 2.5, alpha=0.04, color="black", zorder=0)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    suffix = "_clean" if clean else ""
    out = out_dir / f'fig{fig_num}_ablation_detail{suffix}.png'
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def make_figure3(data_dir: Path, out_dir: Path, clean: bool = False) -> None:
    make_ablation_plot(data_dir, out_dir, ABLATION_REGISTRY, 3, clean=clean)

def make_figure4(data_dir: Path, out_dir: Path, clean: bool = False) -> None:
    '''
    Large Blocks Ablation fig
    '''
    make_ablation_plot(data_dir, out_dir, ABLATION_LARGE_BLOCKS_REGISTRY, 4, clean=clean)


# ── Figure 7: No-B-read savings with sparsity-regime highlights ────────────────

def make_figure7(data_dir: Path, out_dir: Path, clean: bool = False) -> None:
    """
    Figure 7: No-B-read savings panel from fig 3 with highlighted bar regions:
      - 'hyper-sparse'  for diagonal + column test cases  (deep purple)
      - 'semi-sparse'   for triangular + random test cases (deep blue)
    """
    df = load_ablation_all_cases(data_dir, ABLATION_REGISTRY)
    if df.empty:
        print("WARNING: No ablation data found. Skipping Figure 7.")
        return

    pivot = df.pivot_table(index=["algo", "case"], columns="variant",
                           values="ms").reset_index()

    if "no_b_read" not in pivot.columns or "full" not in pivot.columns:
        print("WARNING: no_b_read or full data missing. Skipping Figure 7.")
        return

    col = "saving_no_b_read"
    pivot[col] = (pivot["full"] - pivot["no_b_read"]) / pivot["full"] * 100

    all_cases   = _sorted_cases(pivot)
    case_labels = [_case_label(c) for c in all_cases]
    x = np.arange(len(all_cases))

    fig, ax = plt.subplots(figsize=(11, 5))

    # ── Algorithm lines ──────────────────────────────────────────────────
    marker = "" if clean else "o"
    for algo in ALGOS:
        sub = pivot[pivot["algo"] == algo].set_index("case")
        ys = [sub.loc[c, col] if c in sub.index else np.nan for c in all_cases]
        ax.plot(x, ys, f"{marker}-", label=ALGO_LABEL[algo],
                color=ALGO_COLOR[algo], linewidth=1.8, markersize=5, zorder=3)

    if not clean:
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)

    # ── Sparsity-regime highlighted regions ──────────────────────────────
    HYPER_PATTERNS = ["fill_row", "diagonal", "fill_column"]
    SEMI_PATTERNS  = ["fill_lower_triangular", "fill_random"]
    HYPER_COLOR    = "#4A148C"   # deep purple
    SEMI_COLOR     = "#0D47A1"   # deep blue
    REGION_ALPHA   = 0.20        # bolder fill (both modes)

    hyper_xs = [i for i, c in enumerate(all_cases)
                if any(p in c for p in HYPER_PATTERNS)]
    semi_xs  = [i for i, c in enumerate(all_cases)
                if any(p in c for p in SEMI_PATTERNS)]

    if hyper_xs:
        ax.axvspan(min(hyper_xs) - 0.5, max(hyper_xs) + 0.5,
                   alpha=REGION_ALPHA, color=HYPER_COLOR, zorder=0)
    if semi_xs:
        ax.axvspan(min(semi_xs) - 0.5, max(semi_xs) + 0.5,
                   alpha=REGION_ALPHA, color=SEMI_COLOR, zorder=0)

    # Blended transform: data-x, axes-y — keeps label pinned near top
    blend = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

    if hyper_xs:
        mid_x = (min(hyper_xs) + max(hyper_xs)) / 2
        ax.text(mid_x, 0.97, "hyper-sparse",
                transform=blend, ha="center", va="top",
                fontsize=11, color=HYPER_COLOR, fontweight="bold")
    if semi_xs:
        mid_x = (min(semi_xs) + max(semi_xs)) / 2
        ax.text(mid_x, 0.97, "semi-sparse",
                transform=blend, ha="center", va="top",
                fontsize=11, color=SEMI_COLOR, fontweight="bold")

    # Subtle alternating group shading (every 3 cases = one block-size group)
    if not clean:
        for i in range(0, len(all_cases), 6):
            ax.axvspan(i - 0.5, i + 2.5, alpha=0.04, color="black", zorder=0)

    # ── Axes decoration ──────────────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels(case_labels, fontsize=7.5)
    ax.set_ylabel("Runtime savings (%)")
    ax.set_title("No-B-read savings  (skip dense B DRAM reads)",
                 pad=8, color="#C62828", fontweight="bold")
    if not clean:
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    else:
        ax.tick_params(axis="both", length=0)

    if not clean:
        fig.suptitle(
            "BSR SpMM No-B-read Savings Across All Test Cases",
            fontsize=13, fontweight="bold",
        )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    suffix = "_clean" if clean else ""
    out = out_dir / f"fig7_nob_savings{suffix}.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot BSR SpMM profiling plan results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path("/home/user/tt-metal/profiles_opt_noc_flip_writer/csvs"),
        help="Root directory containing the registry subdirectories (*.csv files)",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("spmm_plots"),
        help="Output directory for PNG figures",
    )
    args = parser.parse_args()

    if not args.data_dir.exists():
        raise SystemExit(f"Data directory not found: {args.data_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Reading from: {args.data_dir}")
    print(f"Writing to:   {args.out_dir}\n")

    make_figure1(args.data_dir, args.out_dir)
    make_figure2(args.data_dir, args.out_dir)
    make_figure3(args.data_dir, args.out_dir)
    make_figure4(args.data_dir, args.out_dir)
    make_figure7(args.data_dir, args.out_dir)
    make_figure7(args.data_dir, args.out_dir, clean=True)
    make_figure1(args.data_dir, args.out_dir, clean=True)
    make_figure2(args.data_dir, args.out_dir, clean=True)
    make_figure3(args.data_dir, args.out_dir, clean=True)
    make_figure4(args.data_dir, args.out_dir, clean=True)


if __name__ == "__main__":
    main()
