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
    "bsr_spmm_multicore_snfin0_cdain1",
    "bsr_spmm_multicore_snf",
    "bsr_spmm_multicore_load_balanced_new_DM",
    "bsr_spmm_multicore_naive_new_DM",
    "bsr_spmm_multicore_load_balanced",
    "bsr_spmm_multicore_reuse_iteration",
]

ALGO_LABEL = {
    "bsr_spmm_multicore_snfin0_cdain1":       "SNF in0 CDA in1",
    "bsr_spmm_multicore_snf":                 "SNF",
    "bsr_spmm_multicore_load_balanced_new_DM": "LB (new DM)",
    "bsr_spmm_multicore_naive_new_DM":         "Naive (new DM)",
    "bsr_spmm_multicore_load_balanced":        "Load Balanced",
    "bsr_spmm_multicore_reuse_iteration":      "Naive",
}

ALGO_COLOR = {
    "bsr_spmm_multicore_snfin0_cdain1":       "#BB6500",
    "bsr_spmm_multicore_snf":                 "#1565C0",
    "bsr_spmm_multicore_load_balanced_new_DM": "#E53935",
    "bsr_spmm_multicore_naive_new_DM":         "#43A047",
    "bsr_spmm_multicore_load_balanced":        "#F57C00",
    "bsr_spmm_multicore_reuse_iteration":      "#7B1FA2",
}

ABLATION_VARIANTS = ["", "_no_a_read", "_no_b_read", "_no_compute", "_no_write"]

# Suffix appended to directory names when plotting flip-noc data.
# Set by --flip-noc CLI flag; used by all loading functions.
DIR_SUFFIX = ""

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
    """Parse matrix metadata (H, W, R, C, nblocks, in1_block_w) from a pretty_print log file."""
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
                elif "in1_block_w" in line:
                    # "Dense block width (in1_block_w): 4 tiles (128 columns)"
                    tiles_str = line.split(":")[1].strip().split()[0]
                    result["in1_block_w"] = int(tiles_str)
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
            path = reg_dir / f"{algo}{suffix}{DIR_SUFFIX}" / f"{ablation_case}.csv"
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
        algo_dir = reg_dir / f"{algo}{DIR_SUFFIX}"
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
    # Load dense block widths for block size sweep to annotate labels
    sweep_bw = _load_dense_block_widths(data_dir, ["ProfileSweepBlockSize"])
    bs_labels = []
    for r in block_ticks:
        lbl = f"A:{r}×{r}"
        # Find any sweep case with this R to get its dense block width
        for case, dw in sweep_bw.items():
            if _parse_parametric(case) and _parse_parametric(case)["R"] == r:
                lbl += f"\nB:{r}×{dw}"
                break
        bs_labels.append(lbl)
    axes[1, 1].set_xticklabels(bs_labels, fontsize=7)
    axes[1, 1].tick_params(axis="x", which="minor", bottom=False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    suffix = "_clean" if clean else ""
    out = out_dir / f"fig2_scaling_sweeps{suffix}.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 3: Ablation detail across all test cases ────────────────────────────

# Short display name for each test-case stem
def _load_dense_block_widths(data_dir: Path, registries) -> dict:
    """
    Scan dense log files to build a mapping: case_stem → dense block width
    in columns (in1_block_w * TILE_WIDTH).  Uses the first algo's directory
    in each registry (all algos share the same matrix geometry).
    """
    result = {}
    for reg in (registries if isinstance(registries, list) else [registries]):
        first_algo_dir = data_dir / reg / f"{ALGOS[0]}{DIR_SUFFIX}"
        if not first_algo_dir.exists():
            continue
        for log in first_algo_dir.glob("*_dense.log"):
            case = log.stem.removesuffix("_dense")
            if case in result:
                continue
            meta = parse_log_metadata(log)
            if "in1_block_w" in meta:
                # in1_block_w is in tiles; convert to columns (tile_width=32)
                result[case] = meta["in1_block_w"] * 32
    return result


def _case_label(stem: str, dense_bw: dict | None = None) -> str:
    stem_clean = stem.replace("profile_case_sparse_", "").replace("_large", "")
    for pat, short in [("fill_lower_triangular", "tri"),
                       ("fill_column", "col"),
                       ("fill_random", "rand"),
                       ("fill_row", "row"),
                       ("diagonal",  "diag")]:
        if pat in stem_clean:
            m = re.search(r"[_]?(\d+)[_]C\d+", stem_clean)
            size = m.group(1) if m else "?"
            label = f"{short}\nA:{size}×{size}"
            if dense_bw and stem in dense_bw:
                dw = dense_bw[stem]
                label += f"\nB:{size}×{dw}"
            return label
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
    base_dir = reg_dir / f"{ALGOS[0]}{DIR_SUFFIX}"
    cases = sorted(
        p.stem for p in base_dir.glob("*.csv")
        if not p.stem.endswith(".device")
    )
    for algo in ALGOS:
        for suffix in ABLATION_VARIANTS:
            algo_dir = reg_dir / f"{algo}{suffix}{DIR_SUFFIX}"
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


def _load_ablation_merged(data_dir: Path, registries, block_filter=None) -> pd.DataFrame:
    """
    Load and concatenate ablation data from multiple registries.
    If block_filter is given, only keep cases whose block size is in the set.
    """
    frames = []
    for reg in registries:
        part = load_ablation_all_cases(data_dir, reg)
        if not part.empty and block_filter is not None:
            part = part[part["case"].apply(_block_size).isin(block_filter)]
        frames.append(part)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return combined.drop_duplicates(subset=["algo", "variant", "case"])


def make_ablation_plot(data_dir: Path, out_dir: Path, ablation_reg, fig_num, clean: bool = False) -> None:
    # shared logic for figs 3 and 4 (they're the same)
    if isinstance(ablation_reg, list):
        regs_list = ablation_reg[0]
        df = _load_ablation_merged(data_dir, *ablation_reg)
    else:
        regs_list = [ablation_reg]
        df = load_ablation_all_cases(data_dir, ablation_reg)
    if df.empty:
        print(f"WARNING: No ablation data found. Skipping Figure {fig_num}.")
        return

    dense_bw = _load_dense_block_widths(data_dir, regs_list)

    # Pivot so each row is (algo, case) with columns for each variant's ms
    pivot = df.pivot_table(index=["algo", "case"], columns="variant",
                        values="ms").reset_index()

    # Compute % savings for each skipped component
    for v in ["no_a_read", "no_b_read", "no_compute", "no_write"]:
        if v in pivot.columns:
            pivot[f"saving_{v}"] = (pivot["full"] - pivot[v]) / pivot["full"] * 100

    # Build ordered case list: group by pattern, ordered by block size within each.
    all_cases = _sorted_cases(pivot)
    case_labels = [_case_label(c, dense_bw) for c in all_cases]
    x = np.arange(len(all_cases))

    variants_cfg = [
        ("no_a_read",  "No-A-read savings\n(skip sparse A DRAM reads)",     "#7B1FA2"),
        ("no_b_read",  "No-B-read savings\n(skip dense B DRAM reads)",       "#C62828"),
        ("no_compute", "No-Compute savings\n(skip tile multiply)",            "#E65100"),
        ("no_write",   "No-Write savings\n(skip output DRAM writes)",         "#1B5E20"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
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
        ax.set_xticklabels(case_labels, fontsize=6.5)
        ax.set_ylabel("Runtime savings (%)")
        ax.set_title(title, pad=8, color=accent, fontweight="bold")
        if not clean:
            ax.legend(fontsize=8)
        if not clean:
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)
        else:
            ax.tick_params(axis="both", length=0)

        # Shade alternating pattern groups
        # Detect group size from the data (number of block sizes per pattern)
        n_sizes = len(set(_block_size(c) for c in all_cases))
        grp = n_sizes  # cases per pattern group
        for i in range(0, len(all_cases), 2 * grp):
            ax.axvspan(i - 0.5, i + grp - 0.5, alpha=0.04, color="black", zorder=0)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    suffix = "_clean" if clean else ""
    out = out_dir / f'fig{fig_num}_ablation_detail{suffix}.png'
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def make_figure3(data_dir: Path, out_dir: Path, clean: bool = False) -> None:
    # Merge standard registry (32/64/128) with 256×256 from large blocks registry
    regs = [[ABLATION_REGISTRY, ABLATION_LARGE_BLOCKS_REGISTRY], {32, 64, 128, 256}]
    make_ablation_plot(data_dir, out_dir, regs, 3, clean=clean)

def make_figure4(data_dir: Path, out_dir: Path, clean: bool = False) -> None:
    '''
    Large Blocks Ablation fig
    '''
    make_ablation_plot(data_dir, out_dir, ABLATION_LARGE_BLOCKS_REGISTRY, 4, clean=clean)

def _pattern_name(stem: str) -> str:
    """Extract the sparsity pattern name from a case stem."""
    for pat in _PATTERN_RANK:
        if pat in stem:
            return pat
    return stem


def _pattern_label(pat: str) -> str:
    """Short display label for a pattern name."""
    _MAP = {
        "fill_row": "row",
        "diagonal": "diag",
        "fill_column": "col",
        "fill_lower_triangular": "tri",
        "fill_random": "rand",
    }
    return _MAP.get(pat, pat)


def make_figure9(data_dir: Path, out_dir: Path, clean: bool = False) -> None:
    """
    Fig 9: ablation detail for 256×256 block size, with 32×32 overlaid as
    dashed lines on the same x-axis (matched by sparsity pattern).
    """
    # Load 256 data (primary) and 32 data (reference)
    df_256 = _load_ablation_merged(
        data_dir, [ABLATION_LARGE_BLOCKS_REGISTRY], block_filter={256})
    df_32 = _load_ablation_merged(
        data_dir, [ABLATION_REGISTRY], block_filter={32})

    if df_256.empty:
        print("WARNING: No 256×256 ablation data found. Skipping Figure 9.")
        return

    def _pivot_savings(df):
        pivot = df.pivot_table(index=["algo", "case"], columns="variant",
                               values="ms").reset_index()
        for v in ["no_a_read", "no_b_read", "no_compute", "no_write"]:
            if v in pivot.columns:
                pivot[f"saving_{v}"] = (pivot["full"] - pivot[v]) / pivot["full"] * 100
        return pivot

    pivot_256 = _pivot_savings(df_256)
    pivot_32  = _pivot_savings(df_32) if not df_32.empty else None

    # X-axis: one tick per sparsity pattern (from 256 cases, sorted by pattern rank)
    cases_256 = _sorted_cases(pivot_256)
    dense_bw = _load_dense_block_widths(data_dir, [ABLATION_LARGE_BLOCKS_REGISTRY, ABLATION_REGISTRY])
    dense_bw_32 = _load_dense_block_widths(data_dir, [ABLATION_REGISTRY])
    pat_labels = []
    for c in cases_256:
        pat = _pattern_label(_pattern_name(c))
        dw256 = dense_bw.get(c)
        # Find matching 32 case for this pattern
        c32_stem = None
        for k in dense_bw_32:
            if _pattern_name(k) == _pattern_name(c) and _block_size(k) == 32:
                c32_stem = k
                break
        dw32 = dense_bw_32.get(c32_stem) if c32_stem else None
        line2 = f"A:256×256"
        if dw256:
            line2 += f" B:256×{dw256}"
        line3 = f"A:32×32"
        if dw32:
            line3 += f" B:32×{dw32}"
        pat_labels.append(f"{pat}\n{line2}\n{line3}")
    x = np.arange(len(cases_256))

    # Build pattern→case mapping for 32×32
    case_32_by_pat = {}
    if pivot_32 is not None:
        for c in pivot_32["case"].unique():
            case_32_by_pat[_pattern_name(c)] = c

    variants_cfg = [
        ("no_a_read",  "No-A-read savings\n(skip sparse A DRAM reads)",     "#7B1FA2"),
        ("no_b_read",  "No-B-read savings\n(skip dense B DRAM reads)",       "#C62828"),
        ("no_compute", "No-Compute savings\n(skip tile multiply)",            "#E65100"),
        ("no_write",   "No-Write savings\n(skip output DRAM writes)",         "#1B5E20"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    if not clean:
        fig.suptitle(
            "Ablation Savings — 256×256 blocks (solid) vs 32×32 blocks (dashed)\n"
            "% runtime reduction when each component is skipped",
            fontsize=13, fontweight="bold",
        )

    for ax, (variant, title, accent) in zip(axes.flat, variants_cfg):
        col = f"saving_{variant}"
        if col not in pivot_256.columns:
            ax.set_visible(False)
            continue

        marker = "" if clean else "o"
        for algo in ALGOS:
            # 256 lines (solid)
            sub = pivot_256[pivot_256["algo"] == algo].set_index("case")
            ys = [sub.loc[c, col] if c in sub.index else np.nan
                  for c in cases_256]
            ax.plot(x, ys, f"{marker}-", label=f"{ALGO_LABEL[algo]} (256)",
                    color=ALGO_COLOR[algo], linewidth=1.6, markersize=5, zorder=3)

            # 32 lines (dashed)
            if pivot_32 is not None and col in pivot_32.columns:
                sub32 = pivot_32[pivot_32["algo"] == algo].set_index("case")
                ys32 = []
                for c256 in cases_256:
                    pat = _pattern_name(c256)
                    c32 = case_32_by_pat.get(pat)
                    if c32 is not None and c32 in sub32.index:
                        ys32.append(sub32.loc[c32, col])
                    else:
                        ys32.append(np.nan)
                ax.plot(x, ys32, f"{marker}--", label=f"{ALGO_LABEL[algo]} (32)",
                        color=ALGO_COLOR[algo], linewidth=1.2, markersize=4,
                        alpha=0.55, zorder=2)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)

        ax.set_xticks(x)
        ax.set_xticklabels(pat_labels, fontsize=6)
        ax.set_ylabel("Runtime savings (%)")
        ax.set_title(title, pad=8, color=accent, fontweight="bold")
        if not clean:
            ax.legend(fontsize=6.5, ncol=2)
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)
        else:
            ax.tick_params(axis="both", length=0)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    suffix = "_clean" if clean else ""
    out = out_dir / f"fig9_ablation_detail{suffix}.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 10: Throughput across ablation test cases ──────────────────────────

def _load_throughput_all_cases(data_dir: Path, registries, block_filter=None) -> pd.DataFrame:
    """
    Load full-run timing + metadata for every (algo, test_case) pair and
    compute TFLOPs/s.  Returns DataFrame with columns: algo, case, ms, tflops.
    """
    rows = []
    for reg in registries:
        reg_dir = data_dir / reg
        base_dir = reg_dir / f"{ALGOS[0]}{DIR_SUFFIX}"
        if not base_dir.exists():
            continue
        cases = sorted(
            p.stem for p in base_dir.glob("*.csv")
            if not p.stem.endswith(".device")
        )
        for algo in ALGOS:
            algo_dir = reg_dir / f"{algo}{DIR_SUFFIX}"
            for case in cases:
                if block_filter is not None and _block_size(case) not in block_filter:
                    continue
                csv_path = algo_dir / f"{case}.csv"
                ns = get_metric(csv_path)
                if ns is None:
                    continue
                # Read nblocks and block size from sparse log
                log_path = algo_dir / f"{case}_sparse.log"
                meta = parse_log_metadata(log_path)
                nblocks = meta.get("nblocks")
                R = meta.get("R")
                C = meta.get("C")
                # Read N from dense log
                dense_log = algo_dir / f"{case}_dense.log"
                dense_meta = parse_log_metadata(dense_log)
                N = dense_meta.get("W")
                if nblocks is None or R is None or C is None or N is None:
                    continue
                ms = ns / NUM_ITERS / 1e6
                rows.append({
                    "algo": algo,
                    "case": case,
                    "ms": ms,
                    "tflops": _tflops_per_sec(nblocks, R, C, N, ms),
                })
    df = pd.DataFrame(rows)
    return df.drop_duplicates(subset=["algo", "case"])


def make_figure10(data_dir: Path, out_dir: Path, clean: bool = False) -> None:
    """
    Figure 10: Throughput (TFLOPs/s) of each algorithm across the same test
    cases used in fig 3 (block sizes 32, 64, 128, 256).
    """
    df = _load_throughput_all_cases(
        data_dir,
        [ABLATION_REGISTRY, ABLATION_LARGE_BLOCKS_REGISTRY],
        block_filter={32, 64, 128, 256},
    )
    if df.empty:
        print("WARNING: No throughput data found. Skipping Figure 10.")
        return

    dense_bw = _load_dense_block_widths(data_dir, [ABLATION_REGISTRY, ABLATION_LARGE_BLOCKS_REGISTRY])
    all_cases   = _sorted_cases(df)
    case_labels = [_case_label(c, dense_bw) for c in all_cases]
    x = np.arange(len(all_cases))

    fig, ax = plt.subplots(figsize=(15, 6))

    marker = "" if clean else "o"
    for algo in ALGOS:
        sub = df[df["algo"] == algo].set_index("case")
        ys = [sub.loc[c, "tflops"] if c in sub.index else np.nan
              for c in all_cases]
        ax.plot(x, ys, f"{marker}-", label=ALGO_LABEL[algo],
                color=ALGO_COLOR[algo], linewidth=1.8, markersize=5, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(case_labels, fontsize=6.5)
    ax.set_ylabel("Throughput (TFLOPs/s)")
    ax.set_ylim(bottom=0)
    if not clean:
        ax.set_title(
            "SpMM Throughput Across Sparsity Patterns and Block Sizes\n"
            "(M=K=8192, N=8192, density varies by pattern)",
            pad=10, fontweight="bold",
        )
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    else:
        ax.tick_params(axis="both", length=0)

    # Alternating group shading
    if not clean:
        n_sizes = len(set(_block_size(c) for c in all_cases))
        grp = n_sizes
        for i in range(0, len(all_cases), 2 * grp):
            ax.axvspan(i - 0.5, i + grp - 0.5, alpha=0.04, color="black", zorder=0)

    fig.tight_layout()
    suffix = "_clean" if clean else ""
    out = out_dir / f"fig10_throughput_cases{suffix}.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


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

    dense_bw = _load_dense_block_widths(data_dir, [ABLATION_REGISTRY])
    all_cases   = _sorted_cases(pivot)
    case_labels = [_case_label(c, dense_bw) for c in all_cases]
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
    ax.set_xticklabels(case_labels, fontsize=6.5)
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


# ── Figure 8: All four ablation panels on one axis ────────────────────────────

def make_figure8(data_dir: Path, out_dir: Path, clean: bool = False) -> None:
    """
    Figure 8: overlay all four ablation savings (no_a_read, no_b_read,
    no_compute, no_write) on a single axis, averaged across algorithms.
    One line per ablation variant.
    """
    df = _load_ablation_merged(
        data_dir,
        [ABLATION_REGISTRY, ABLATION_LARGE_BLOCKS_REGISTRY],
        block_filter={32, 64, 128, 256},
    )
    if df.empty:
        print("WARNING: No ablation data found. Skipping Figure 8.")
        return

    pivot = df.pivot_table(index=["algo", "case"], columns="variant",
                           values="ms").reset_index()

    variants_cfg = [
        ("no_a_read",  "A-reads skipped",  "#7B1FA2"),
        ("no_b_read",  "B-reads skipped",  "#C62828"),
        ("no_compute", "Compute skipped",  "#E65100"),
        ("no_write",   "Writes skipped",   "#1B5E20"),
    ]

    for v, _, _ in variants_cfg:
        if v in pivot.columns:
            pivot[f"saving_{v}"] = (pivot["full"] - pivot[v]) / pivot["full"] * 100

    dense_bw = _load_dense_block_widths(data_dir, [ABLATION_REGISTRY, ABLATION_LARGE_BLOCKS_REGISTRY])
    all_cases   = _sorted_cases(pivot)
    case_labels = [_case_label(c, dense_bw) for c in all_cases]
    x = np.arange(len(all_cases))

    fig, ax = plt.subplots(figsize=(13, 6))

    marker = "" if clean else "o"
    for variant, label, color in variants_cfg:
        col = f"saving_{variant}"
        if col not in pivot.columns:
            continue
        # Average savings across all algorithms for each case
        means = []
        for c in all_cases:
            vals = pivot.loc[pivot["case"] == c, col].dropna()
            means.append(vals.mean() if not vals.empty else np.nan)
        ax.plot(x, means, f"{marker}-", label=label, color=color,
                linewidth=2.0, markersize=6, zorder=3)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)

    # Alternating group shading
    if not clean:
        n_sizes = len(set(_block_size(c) for c in all_cases))
        grp = n_sizes
        for i in range(0, len(all_cases), 2 * grp):
            ax.axvspan(i - 0.5, i + grp - 0.5, alpha=0.04, color="black", zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(case_labels, fontsize=6.5)
    ax.set_ylabel("Runtime savings (%)")
    if not clean:
        ax.set_title(
            "Ablation Savings Overview  (all four components, averaged across algorithms)",
            pad=10, fontweight="bold",
        )
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    else:
        ax.tick_params(axis="both", length=0)

    fig.tight_layout()
    suffix = "_clean" if clean else ""
    out = out_dir / f"fig8_ablation_combined{suffix}.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 11: Throughput across sparsity patterns (SweepSparsityPattern) ─────

SPARSITY_PATTERN_REGISTRY = "ProfileSweepSparsityPattern"

# Mapping from filename pattern prefix → display label
_SPARSITY_PATTERN_LABELS = {
    "random":     "Random",
    "col":        "Column",
    "diag":       "Diagonal",
    "multi_diag": "Multi-Diag",
    "row":        "Row",
}

# Desired display order
_SPARSITY_PATTERN_ORDER = ["random", "row", "col", "diag", "multi_diag"]


def _parse_sparsity_pattern_stem(stem: str) -> tuple[str | None, dict | None]:
    """
    Parse a SweepSparsityPattern filename stem.
    Handles:
      parametric_M8192_N8192_K8192_R256_C256_d25         → ("random", {...})
      parametric_col_M8192_N8192_K8192_R256_C256_d25     → ("col",    {...})
      parametric_multi_diag_M8192_N8192_K8192_R256_C256_d25 → ("multi_diag", {...})
    """
    m = re.match(
        r"parametric_(?:(multi_diag|col|row)_)?M(\d+)_N(\d+)_K(\d+)_R(\d+)_C(\d+)_d(\d+)",
        stem,
    )
    if not m:
        return None, None
    pattern = m.group(1) or "random"
    params = dict(zip(["M", "N", "K", "R", "C", "density"],
                      [int(x) for x in m.groups()[1:]]))
    return pattern, params


def load_sweep_sparsity_pattern(data_dir: Path, registry: str = SPARSITY_PATTERN_REGISTRY) -> pd.DataFrame:
    """
    Load timing + metadata for every (algo, sparsity_pattern) pair in
    a sparsity-pattern sweep registry and compute TFLOPs/s.
    """
    rows = []
    reg_dir = data_dir / registry
    for algo in ALGOS:
        algo_dir = reg_dir / f"{algo}{DIR_SUFFIX}"
        if not algo_dir.exists():
            continue
        for csv in sorted(algo_dir.glob("*.csv")):
            if csv.stem.endswith(".device"):
                continue
            pattern, params = _parse_sparsity_pattern_stem(csv.stem)
            if pattern is None or params is None:
                continue
            ns = get_metric(csv)
            if ns is None:
                continue
            # Read nblocks from sparse log
            log = csv.parent / f"{csv.stem}_sparse.log"
            meta = parse_log_metadata(log)
            nblocks = meta.get("nblocks")
            if nblocks is None:
                continue
            ms = ns / NUM_ITERS / 1e6
            rows.append({
                "algo": algo,
                "pattern": pattern,
                **params,
                "ms": ms,
                "nblocks": nblocks,
                "tflops": _tflops_per_sec(nblocks, params["R"],
                                          params["C"], params["N"], ms),
            })
    return pd.DataFrame(rows)


def _make_sparsity_pattern_figure(
    data_dir: Path, out_dir: Path, registry: str, fig_num: int,
    clean: bool = False,
) -> None:
    """Grouped bar chart of throughput across sparsity patterns for one registry."""
    df = load_sweep_sparsity_pattern(data_dir, registry)
    if df.empty:
        print(f"WARNING: No sparsity pattern data found for {registry}. Skipping Figure {fig_num}.")
        return

    patterns = [p for p in _SPARSITY_PATTERN_ORDER if p in df["pattern"].values]
    pat_labels = [_SPARSITY_PATTERN_LABELS.get(p, p) for p in patterns]

    n_algos = len(ALGOS)
    x = np.arange(len(patterns))
    total_bar_width = 0.75
    bar_w = total_bar_width / n_algos

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, algo in enumerate(ALGOS):
        sub = df[df["algo"] == algo]
        ys = []
        for pat in patterns:
            row = sub[sub["pattern"] == pat]
            ys.append(row["tflops"].iloc[0] if not row.empty else 0)
        offset = (i - (n_algos - 1) / 2) * bar_w
        bars = ax.bar(x + offset, ys, bar_w,
                      label=ALGO_LABEL[algo], color=ALGO_COLOR[algo],
                      edgecolor="white", linewidth=0.5, zorder=3)
        if not clean:
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                            f"{h:.3f}", ha="left", va="bottom",
                            fontsize=6.5, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(pat_labels, fontsize=10)
    ax.set_ylabel("Throughput (TFLOPs/s)")
    ax.set_ylim(bottom=0)
    if not clean:
        row0 = df.iloc[0]
        ax.set_title(
            f"SpMM Throughput vs. Sparsity Pattern\n"
            f"(M={row0['M']}, N={row0['N']}, K={row0['K']}, "
            f"R=C={row0['R']}, density={row0['density']}%)",
            pad=10, fontweight="bold",
        )
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    else:
        ax.tick_params(axis="both", length=0)

    fig.tight_layout()
    suffix = "_clean" if clean else ""
    out = out_dir / f"fig{fig_num}_sparsity_pattern_throughput{suffix}.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def make_figure11(data_dir: Path, out_dir: Path, clean: bool = False) -> None:
    _make_sparsity_pattern_figure(data_dir, out_dir, "ProfileSweepSparsityPattern", 11, clean)


def make_figure13(data_dir: Path, out_dir: Path, clean: bool = False) -> None:
    _make_sparsity_pattern_figure(data_dir, out_dir, "ProfileSweepSparsityPatternD10", 13, clean)


def make_figure14(data_dir: Path, out_dir: Path, clean: bool = False) -> None:
    _make_sparsity_pattern_figure(data_dir, out_dir, "ProfileSweepSparsityPatternD5", 14, clean)


def make_figure15(data_dir: Path, out_dir: Path, clean: bool = False) -> None:
    _make_sparsity_pattern_figure(data_dir, out_dir, "ProfileSweepSparsityPatternD50", 15, clean)


# ── Figure 16: Combined sparsity-pattern throughput, grouped by algorithm ─────

_PATTERN_COLOR = {
    "random":     "#1565C0",
    "row":        "#E53935",
    "col":        "#43A047",
    "diag":       "#F57C00",
    "multi_diag": "#7B1FA2",
}

# (registry, density_label) tuples ordered by density
_FIG16_PANELS = [
    ("ProfileSweepSparsityPatternD5",  "5%"),
    ("ProfileSweepSparsityPatternD10", "10%"),
    ("ProfileSweepSparsityPattern",    "25%"),
    ("ProfileSweepSparsityPatternD50", "50%"),
]


def make_figure16(data_dir: Path, out_dir: Path, clean: bool = False) -> None:
    """
    Figure 16: 2×2 grid combining figs 11, 13, 14, 15.
    Each subplot is one density level, bars grouped by algorithm,
    colored by sparsity pattern.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True)

    for ax, (registry, density_label) in zip(axes.flat, _FIG16_PANELS):
        df = load_sweep_sparsity_pattern(data_dir, registry)
        if df.empty:
            ax.set_title(f"Density = {density_label} (no data)")
            continue

        patterns = [p for p in _SPARSITY_PATTERN_ORDER if p in df["pattern"].values]
        algos_present = [a for a in ALGOS if a in df["algo"].values]
        n_patterns = len(patterns)
        x = np.arange(len(algos_present))
        total_bar_width = 0.75
        bar_w = total_bar_width / max(n_patterns, 1)

        for j, pat in enumerate(patterns):
            ys = []
            for algo in algos_present:
                row = df[(df["algo"] == algo) & (df["pattern"] == pat)]
                ys.append(row["tflops"].iloc[0] if not row.empty else 0)
            offset = (j - (n_patterns - 1) / 2) * bar_w
            bars = ax.bar(x + offset, ys, bar_w,
                          label=_SPARSITY_PATTERN_LABELS.get(pat, pat),
                          color=_PATTERN_COLOR[pat],
                          edgecolor="white", linewidth=0.5, zorder=3)
            if not clean:
                for bar in bars:
                    h = bar.get_height()
                    if h > 0:
                        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                                f"{h:.3f}", ha="left", va="bottom",
                                fontsize=5.5, rotation=45)

        ax.set_xticks(x)
        ax.set_xticklabels([ALGO_LABEL[a] for a in algos_present],
                           fontsize=8, rotation=15, ha="right")
        if not clean:
            ax.set_title(f"Density = {density_label}", fontweight="bold")
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)
        else:
            ax.set_title(f"Density = {density_label}")
            ax.tick_params(axis="both", length=0)

    # Shared y-axis: start at 0, with headroom for value labels
    global_max = max(ax.get_ylim()[1] for ax in axes.flat)
    axes.flat[0].set_ylim(0, global_max * 1.12)

    # Common y-label on the left subplots
    for ax in axes[:, 0]:
        ax.set_ylabel("Throughput (TFLOPs/s)")

    # Single shared legend at the top
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if not clean:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels),
                   fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.0))
        fig.suptitle(
            "SpMM Throughput vs. Sparsity Pattern (grouped by algorithm).\n8192x8192x8192, R=C=256",
            fontsize=13, fontweight="bold", y=1.04,
        )

    fig.tight_layout()
    suffix = "_clean" if clean else ""
    out = out_dir / f"fig16_sparsity_pattern_by_algo{suffix}.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 17: Fig 16 + cuSPARSE (4090) as 7th algorithm ─────────────────────

CUSPARSE_CSV = Path("spmm_scripts/cusparse4090.csv")

# Registry ID → density string (matches _FIG16_PANELS ordering)
_CUSPARSE_REG_DENSITY = {8: "25%", 9: "10%", 10: "5%", 11: "50%"}


def _parse_cusparse_pattern(case_name: str) -> str | None:
    """Extract sparsity pattern from a cuSPARSE case name, skipping bare 'diag'."""
    m = re.match(r"parametric_(?:(multi_diag|col|row)_)?M", case_name)
    if not m:
        # Check if it's bare diag — skip it
        if re.match(r"parametric_diag_", case_name):
            return None
        return None
    return m.group(1) or "random"


def load_cusparse(csv_path: Path) -> pd.DataFrame:
    """
    Load cuSPARSE benchmark CSV.
    Columns: registry_id, case_name, M, N, K, R, C, ms, tflops_avg, tflops_max, gbs_avg, gbs_max
    Returns DataFrame with columns: pattern, density, tflops (avg TFLOPs/s).
    Bare 'diag' cases are excluded.
    """
    rows = []
    try:
        with open(csv_path) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 12:
                    continue
                reg_id = int(parts[0])
                case_name = parts[1]
                density = _CUSPARSE_REG_DENSITY.get(reg_id)
                if density is None:
                    continue
                pattern = _parse_cusparse_pattern(case_name)
                if pattern is None:
                    continue
                rows.append({
                    "pattern": pattern,
                    "density": density,
                    "tflops": float(parts[8]),
                })
    except FileNotFoundError:
        pass
    return pd.DataFrame(rows)


CUSPARSE_ALGO = "cusparse_4090"
CUSPARSE_LABEL = "cuSPARSE (4090)"
CUSPARSE_COLOR = "#212121"


def make_figure17(data_dir: Path, out_dir: Path, cusparse_csv: Path = CUSPARSE_CSV,
                  clean: bool = False) -> None:
    """
    Figure 17: Same as fig 16 but with cuSPARSE (4090) added as a 7th algorithm.
    """
    cusparse_df = load_cusparse(cusparse_csv)
    if cusparse_df.empty:
        print("WARNING: No cuSPARSE data found. Skipping Figure 17.")
        return

    all_algos = ALGOS + [CUSPARSE_ALGO]
    all_labels = {**ALGO_LABEL, CUSPARSE_ALGO: CUSPARSE_LABEL}
    all_colors = {**ALGO_COLOR, CUSPARSE_ALGO: CUSPARSE_COLOR}

    fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharey=True)

    for ax, (registry, density_label) in zip(axes.flat, _FIG16_PANELS):
        df = load_sweep_sparsity_pattern(data_dir, registry)
        # Merge cuSPARSE data for this density
        cuda_sub = cusparse_df[cusparse_df["density"] == density_label].copy()
        if not cuda_sub.empty:
            cuda_sub["algo"] = CUSPARSE_ALGO
            if not df.empty:
                df = pd.concat([df, cuda_sub[["algo", "pattern", "tflops"]]],
                               ignore_index=True)
            else:
                df = cuda_sub[["algo", "pattern", "tflops"]].copy()

        if df.empty:
            ax.set_title(f"Density = {density_label} (no data)")
            continue

        patterns = [p for p in _SPARSITY_PATTERN_ORDER if p in df["pattern"].values]
        algos_present = [a for a in all_algos if a in df["algo"].values]
        n_patterns = len(patterns)
        x = np.arange(len(algos_present))
        total_bar_width = 0.75
        bar_w = total_bar_width / max(n_patterns, 1)

        for j, pat in enumerate(patterns):
            ys = []
            for algo in algos_present:
                row = df[(df["algo"] == algo) & (df["pattern"] == pat)]
                ys.append(row["tflops"].iloc[0] if not row.empty else 0)
            offset = (j - (n_patterns - 1) / 2) * bar_w
            bars = ax.bar(x + offset, ys, bar_w,
                          label=_SPARSITY_PATTERN_LABELS.get(pat, pat),
                          color=_PATTERN_COLOR[pat],
                          edgecolor="white", linewidth=0.5, zorder=3)
            if not clean:
                for bar in bars:
                    h = bar.get_height()
                    if h > 0:
                        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                                f"{h:.3f}", ha="left", va="bottom",
                                fontsize=5.5, rotation=45)

        ax.set_xticks(x)
        ax.set_xticklabels([all_labels[a] for a in algos_present],
                           fontsize=7, rotation=20, ha="right")
        if not clean:
            ax.set_title(f"Density = {density_label}", fontweight="bold")
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)
        else:
            ax.set_title(f"Density = {density_label}")
            ax.tick_params(axis="both", length=0)

    # Shared y-axis: start at 0, with headroom for value labels
    global_max = max(ax.get_ylim()[1] for ax in axes.flat)
    axes.flat[0].set_ylim(0, global_max * 1.12)

    # Common y-label on the left subplots
    for ax in axes[:, 0]:
        ax.set_ylabel("Throughput (TFLOPs/s)")

    # Single shared legend at the top
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if not clean:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels),
                   fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.0))
        fig.suptitle(
            "SpMM Throughput vs. Sparsity Pattern — TT vs cuSPARSE (4090)\n"
            "8192x8192x8192, R=C=256",
            fontsize=13, fontweight="bold", y=1.04,
        )

    fig.tight_layout()
    suffix = "_clean" if clean else ""
    out = out_dir / f"fig17_sparsity_pattern_vs_cuda{suffix}.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 12: Throughput across densities (bar chart like fig 11) ────────────

DENSITY_SWEEP_DATA_DIR = Path("/home/user/tt-metal/profiles_opt_noc_full_profiling_suite/csvs")
DENSITY_SWEEP_REGISTRY = "ProfileSweepDensity"


def make_figure12(data_dir: Path, out_dir: Path, clean: bool = False) -> None:
    """
    Figure 12: Grouped bar chart — throughput (TFLOPs/s) for each algorithm
    across density values in ProfileSweepDensity (from a separate data dir).
    """
    df = load_sweep(DENSITY_SWEEP_DATA_DIR, DENSITY_SWEEP_REGISTRY, "density")
    if df.empty:
        print("WARNING: No density sweep data found. Skipping Figure 12.")
        return

    densities = sorted(df["density"].unique())
    density_labels = [f"{d}%" for d in densities]

    n_algos = len(ALGOS)
    x = np.arange(len(densities))
    total_bar_width = 0.75
    bar_w = total_bar_width / n_algos

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, algo in enumerate(ALGOS):
        sub = df[df["algo"] == algo]
        ys = []
        for d in densities:
            row = sub[sub["density"] == d]
            ys.append(row["tflops"].iloc[0] if not row.empty else 0)
        offset = (i - (n_algos - 1) / 2) * bar_w
        bars = ax.bar(x + offset, ys, bar_w,
                      label=ALGO_LABEL[algo], color=ALGO_COLOR[algo],
                      edgecolor="white", linewidth=0.5, zorder=3)
        if not clean:
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                            f"{h:.3f}", ha="left", va="bottom",
                            fontsize=6.5, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(density_labels, fontsize=10)
    ax.set_ylabel("Throughput (TFLOPs/s)")
    ax.set_ylim(bottom=0)
    if not clean:
        row0 = df.iloc[0]
        ax.set_title(
            f"SpMM Throughput vs. Sparsity Density\n"
            f"(M={row0['M']}, N={row0['N']}, K={row0['K']}, "
            f"R=C={row0['R']})",
            pad=10, fontweight="bold",
        )
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    else:
        ax.tick_params(axis="both", length=0)

    fig.tight_layout()
    suffix = "_clean" if clean else ""
    out = out_dir / f"fig12_density_throughput{suffix}.png"
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
        default=Path("/home/user/tt-metal/profiles_opt_noc_full_profiling_suite/csvs"),
        help="Root directory containing the registry subdirectories (*.csv files)",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("spmm_plots"),
        help="Output directory for PNG figures",
    )
    parser.add_argument(
        "--flip-noc", action="store_true",
        help="Plot flip-noc data (directories with _flip_noc suffix)",
    )
    args = parser.parse_args()

    global DIR_SUFFIX
    if args.flip_noc:
        DIR_SUFFIX = "_flip_noc"

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
    make_figure8(args.data_dir, args.out_dir)
    make_figure9(args.data_dir, args.out_dir)
    make_figure10(args.data_dir, args.out_dir)
    make_figure11(args.data_dir, args.out_dir)
    make_figure12(args.data_dir, args.out_dir)
    make_figure13(args.data_dir, args.out_dir)
    make_figure14(args.data_dir, args.out_dir)
    make_figure15(args.data_dir, args.out_dir)
    make_figure16(args.data_dir, args.out_dir)
    make_figure17(args.data_dir, args.out_dir)
    clean_dir = args.out_dir / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    make_figure1(args.data_dir, clean_dir, clean=True)
    make_figure2(args.data_dir, clean_dir, clean=True)
    make_figure3(args.data_dir, clean_dir, clean=True)
    make_figure4(args.data_dir, clean_dir, clean=True)
    make_figure7(args.data_dir, clean_dir, clean=True)
    make_figure8(args.data_dir, clean_dir, clean=True)
    make_figure9(args.data_dir, clean_dir, clean=True)
    make_figure10(args.data_dir, clean_dir, clean=True)
    make_figure11(args.data_dir, clean_dir, clean=True)
    make_figure12(args.data_dir, clean_dir, clean=True)
    make_figure13(args.data_dir, clean_dir, clean=True)
    make_figure14(args.data_dir, clean_dir, clean=True)
    make_figure15(args.data_dir, clean_dir, clean=True)
    make_figure16(args.data_dir, clean_dir, clean=True)
    make_figure17(args.data_dir, clean_dir, clean=True)


if __name__ == "__main__":
    main()
