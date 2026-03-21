#!/usr/bin/env python3
"""
plot_cdav2_direction_sweep.py

Recreate Figure 4 (ablation detail for large blocks) and Figure 16 (sparsity
pattern throughput grouped by algorithm) using only CDA direction-sweep data
from profiles_opt_noc_CDAV2/csvs/.

The "algorithms" axis becomes the direction-sweep variants:
  - bsr_spmm_multicore_snfin0_cdain1  (L2R+B2T, default)
  - snfin0_cdain1_R2L_T2B
  - snfin0_cdain1_L2R_B2T_flip_noc
  (+ any other direction variants present in the data)

Usage:
    python spmm_scripts/plot_cdav2_direction_sweep.py
    python spmm_scripts/plot_cdav2_direction_sweep.py --data-dir /path/to/csvs --out-dir plots/
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_DATA_DIR = Path("/home/user/tt-metal/profiles_opt_noc_CDAV2/csvs")

DEFAULT_OUT_DIR = Path("/home/user/tt-metal/spmm_plots/cdav2")

ABLATION_REGISTRY = "ProfileSuiteLargeSparseLargeBlocksVersioning"

NUM_ITERS = 10


# ── Direction variant metadata ────────────────────────────────────────────────

# Discover direction variants from the data (see _discover_variants below).
# We map directory names to short display labels and colors.

# 4 direction combos × 2 NoC configs = 8 variants
# Colors: one per direction combo (shared between optimal and flip noc)
# Line styles: solid = optimal NoC, dashed = flip NoC
_DIR_COLORS = {
    "L2R_B2T": "#BB6500",
    "L2R_T2B": "#1565C0",
    "R2L_B2T": "#E53935",
    "R2L_T2B": "#43A047",
}

_VARIANT_LABEL = {
    "bsr_spmm_multicore_snfin0_cdain1": "L2R+B2T (old)",
    "snfin0_cdain1_L2R_B2T": "L2R+B2T",
    "snfin0_cdain1_L2R_T2B": "L2R+T2B",
    "snfin0_cdain1_R2L_B2T": "R2L+B2T",
    "snfin0_cdain1_R2L_T2B": "R2L+T2B",
    "snfin0_cdain1_L2R_B2T_flip_noc": "L2R+B2T flip",
    "snfin0_cdain1_L2R_T2B_flip_noc": "L2R+T2B flip",
    "snfin0_cdain1_R2L_B2T_flip_noc": "R2L+B2T flip",
    "snfin0_cdain1_R2L_T2B_flip_noc": "R2L+T2B flip",
}

_VARIANT_COLOR = {
    "bsr_spmm_multicore_snfin0_cdain1": _DIR_COLORS["L2R_B2T"],
    "snfin0_cdain1_L2R_B2T": _DIR_COLORS["L2R_B2T"],
    "snfin0_cdain1_L2R_T2B": _DIR_COLORS["L2R_T2B"],
    "snfin0_cdain1_R2L_B2T": _DIR_COLORS["R2L_B2T"],
    "snfin0_cdain1_R2L_T2B": _DIR_COLORS["R2L_T2B"],
    "snfin0_cdain1_L2R_B2T_flip_noc": _DIR_COLORS["L2R_B2T"],
    "snfin0_cdain1_L2R_T2B_flip_noc": _DIR_COLORS["L2R_T2B"],
    "snfin0_cdain1_R2L_B2T_flip_noc": _DIR_COLORS["R2L_B2T"],
    "snfin0_cdain1_R2L_T2B_flip_noc": _DIR_COLORS["R2L_T2B"],
}

_VARIANT_LINESTYLE = {
    "bsr_spmm_multicore_snfin0_cdain1": ":",
    "snfin0_cdain1_L2R_B2T": "-",
    "snfin0_cdain1_L2R_T2B": "-",
    "snfin0_cdain1_R2L_B2T": "-",
    "snfin0_cdain1_R2L_T2B": "-",
    "snfin0_cdain1_L2R_B2T_flip_noc": "--",
    "snfin0_cdain1_L2R_T2B_flip_noc": "--",
    "snfin0_cdain1_R2L_B2T_flip_noc": "--",
    "snfin0_cdain1_R2L_T2B_flip_noc": "--",
}

# Desired display order: optimal noc first, then flip-noc
_VARIANT_ORDER = [
    "snfin0_cdain1_L2R_B2T",
    "snfin0_cdain1_L2R_T2B",
    "snfin0_cdain1_R2L_B2T",
    "snfin0_cdain1_R2L_T2B",
    "snfin0_cdain1_L2R_B2T_flip_noc",
    "snfin0_cdain1_L2R_T2B_flip_noc",
    "snfin0_cdain1_R2L_B2T_flip_noc",
    "snfin0_cdain1_R2L_T2B_flip_noc",
    "bsr_spmm_multicore_snfin0_cdain1",  # fallback if explicit L2R_B2T missing
]

# Ablation suffixes
ABLATION_SUFFIXES = ["", "_no_a_read", "_no_b_read", "_no_compute", "_no_write"]


# ── Style ─────────────────────────────────────────────────────────────────────

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    }
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def get_metric(csv_path: Path, zone: str = "Device program Loop") -> float | None:
    """Read total_ns for one profiler zone from a host-code CSV."""
    try:
        df = pd.read_csv(csv_path, usecols=["name", "total_ns"])
        row = df[df["name"] == zone]
        return float(row["total_ns"].iloc[0]) if not row.empty else None
    except Exception:
        return None


def parse_log_metadata(filepath: Path) -> dict:
    """Parse matrix metadata from a pretty_print log file."""
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
                    tiles_str = line.split(":")[1].strip().split()[0]
                    result["in1_block_w"] = int(tiles_str)
    except FileNotFoundError:
        pass
    return result


def _tflops_per_sec(nblocks: int, R: int, C: int, N: int, ms: float) -> float:
    flops = 2 * nblocks * R * C * N
    return flops / 1e12 / (ms / 1e3)


_ABLATION_TAGS = {"_no_a_read", "_no_b_read", "_no_compute", "_no_write"}


def _is_ablation_dir(name: str) -> bool:
    """Check if a directory name contains an ablation tag anywhere."""
    return any(tag in name for tag in _ABLATION_TAGS)


def _discover_base_variants(data_dir: Path, registry: str) -> list[str]:
    """Discover which base (non-ablation) direction variants have data."""
    reg_dir = data_dir / registry
    if not reg_dir.exists():
        return []
    dirs = sorted(d.name for d in reg_dir.iterdir() if d.is_dir())
    # Base variants: no ablation tag anywhere in the name
    base = [d for d in dirs if not _is_ablation_dir(d)]
    # Drop the old default if the explicit L2R_B2T variant exists
    if "snfin0_cdain1_L2R_B2T" in base and "bsr_spmm_multicore_snfin0_cdain1" in base:
        base.remove("bsr_spmm_multicore_snfin0_cdain1")
    # Sort by preferred order
    order_map = {v: i for i, v in enumerate(_VARIANT_ORDER)}
    base.sort(key=lambda d: order_map.get(d, 999))
    return base


def _ablation_dir_name(base_variant: str, ablation_suffix: str) -> str:
    """
    Build the directory name for an ablation variant.
    e.g. base="snfin0_cdain1_R2L_T2B", suffix="_no_a_read"
         → "snfin0_cdain1_no_a_read_R2L_T2B"
    For the default algo "bsr_spmm_multicore_snfin0_cdain1", ablation dirs
    are "bsr_spmm_multicore_snfin0_cdain1_no_a_read" etc.
    """
    if not ablation_suffix:
        return base_variant

    # Default algo keeps its full name + suffix
    if base_variant == "bsr_spmm_multicore_snfin0_cdain1":
        return base_variant + ablation_suffix

    # Direction variants: insert ablation before direction tags
    # e.g. "snfin0_cdain1_R2L_T2B" → "snfin0_cdain1_no_a_read_R2L_T2B"
    # The base prefix is "snfin0_cdain1", direction tags follow
    prefix = "snfin0_cdain1"
    if base_variant.startswith(prefix):
        direction_part = base_variant[len(prefix) :]  # e.g. "_R2L_T2B" or "_L2R_B2T_flip_noc"
        return prefix + ablation_suffix + direction_part

    # Fallback: just append
    return base_variant + ablation_suffix


def _variant_label(name: str) -> str:
    return _VARIANT_LABEL.get(name, name.replace("snfin0_cdain1_", ""))


def _variant_color(name: str) -> str:
    return _VARIANT_COLOR.get(name, "#888888")


# ── Figure 4: Ablation detail for large blocks ───────────────────────────────


def _case_label(stem: str, dense_bw: dict | None = None) -> str:
    stem_clean = stem.replace("profile_case_sparse_", "").replace("_large", "")
    for pat, short in [
        ("fill_lower_triangular", "tri"),
        ("fill_column", "col"),
        ("fill_random", "rand"),
        ("fill_row", "row"),
        ("diagonal", "diag"),
    ]:
        if pat in stem_clean:
            m = re.search(r"[_]?(\d+)[_]C\d+", stem_clean)
            size = m.group(1) if m else "?"
            label = f"{short}\nA:{size}x{size}"
            if dense_bw and stem in dense_bw:
                dw = dense_bw[stem]
                label += f"\nB:{size}x{dw}"
            return label
    return stem


def _block_size(s: str) -> int:
    m = re.search(r"(\d+)[_]C\d+$", s)
    return int(m.group(1)) if m else 0


_PATTERN_RANK = {
    "fill_row": 0,
    "diagonal": 1,
    "fill_column": 2,
    "fill_lower_triangular": 3,
    "fill_random": 4,
}


def _pattern_rank(s: str) -> int:
    for pat, rank in _PATTERN_RANK.items():
        if pat in s:
            return rank
    return 99


def _sorted_cases(pivot: pd.DataFrame) -> list:
    return sorted(pivot["case"].unique(), key=lambda s: (_pattern_rank(s), _block_size(s)))


def _load_dense_block_widths(data_dir: Path, registry: str, variants: list[str]) -> dict:
    result = {}
    for v in variants:
        d = data_dir / registry / v
        if not d.exists():
            continue
        for log in d.glob("*_dense.log"):
            case = log.stem.removesuffix("_dense")
            if case in result:
                continue
            meta = parse_log_metadata(log)
            if "in1_block_w" in meta:
                result[case] = meta["in1_block_w"] * 32
        if result:
            break  # all variants share the same geometry
    return result


def load_ablation_all_cases(data_dir: Path, registry: str, variants: list[str]) -> pd.DataFrame:
    """Load full x ablation timing for every (variant, ablation, case) triple."""
    rows = []
    reg_dir = data_dir / registry

    # Collect case stems from the first available variant
    cases = []
    for v in variants:
        d = reg_dir / v
        if d.exists():
            cases = sorted(p.stem for p in d.glob("*.csv") if not p.stem.endswith(".device"))
            break

    for base_var in variants:
        for abl_suffix in ABLATION_SUFFIXES:
            abl_dir_name = _ablation_dir_name(base_var, abl_suffix)
            abl_dir = reg_dir / abl_dir_name
            for case in cases:
                path = abl_dir / f"{case}.csv"
                ns = get_metric(path)
                if ns is not None:
                    ns = ns / NUM_ITERS
                    rows.append(
                        {
                            "variant": base_var,
                            "ablation": abl_suffix.lstrip("_") or "full",
                            "case": case,
                            "ms": ns / 1e6,
                        }
                    )
    return pd.DataFrame(rows)


def make_figure4(data_dir: Path, out_dir: Path) -> None:
    """
    Figure 4: Ablation savings (%) across test cases for each skipped component.
    2x2 grid: no_a_read, no_b_read, no_compute, no_write.
    Lines are direction-sweep variants instead of different algorithms.
    """
    variants = _discover_base_variants(data_dir, ABLATION_REGISTRY)
    if not variants:
        print(f"WARNING: No direction variants found in {ABLATION_REGISTRY}. Skipping Figure 4.")
        return

    df = load_ablation_all_cases(data_dir, ABLATION_REGISTRY, variants)
    if df.empty:
        print("WARNING: No ablation data loaded. Skipping Figure 4.")
        return

    dense_bw = _load_dense_block_widths(data_dir, ABLATION_REGISTRY, variants)

    # Pivot: each row is (variant, case) with columns for each ablation's ms
    pivot = df.pivot_table(index=["variant", "case"], columns="ablation", values="ms").reset_index()

    # Compute % savings
    for v in ["no_a_read", "no_b_read", "no_compute", "no_write"]:
        if v in pivot.columns:
            pivot[f"saving_{v}"] = (pivot["full"] - pivot[v]) / pivot["full"] * 100

    all_cases = _sorted_cases(pivot)
    case_labels = [_case_label(c, dense_bw) for c in all_cases]
    x = np.arange(len(all_cases))

    variants_cfg = [
        ("no_a_read", "No-A-read savings\n(skip sparse A DRAM reads)", "#7B1FA2"),
        ("no_b_read", "No-B-read savings\n(skip dense B DRAM reads)", "#C62828"),
        ("no_compute", "No-Compute savings\n(skip tile multiply)", "#E65100"),
        ("no_write", "No-Write savings\n(skip output DRAM writes)", "#1B5E20"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        "Ablation Savings: CDA Direction Sweep (Large Blocks 256x256)\n"
        "% runtime reduction when each component is skipped",
        fontsize=13,
        fontweight="bold",
    )

    for ax, (abl_name, title, accent) in zip(axes.flat, variants_cfg):
        col = f"saving_{abl_name}"
        if col not in pivot.columns:
            ax.set_visible(False)
            continue

        for var in variants:
            sub = pivot[pivot["variant"] == var].set_index("case")
            ys = [sub.loc[c, col] if c in sub.index else np.nan for c in all_cases]
            ls = _VARIANT_LINESTYLE.get(var, "-")
            marker = "o" if ls == "-" else "s" if ls == "--" else "^"
            ax.plot(
                x,
                ys,
                label=_variant_label(var),
                color=_variant_color(var),
                linestyle=ls,
                marker=marker,
                linewidth=1.6,
                markersize=4,
                zorder=3,
            )

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(case_labels, fontsize=6.5)
        ax.set_ylabel("Runtime savings (%)")
        ax.set_title(title, pad=8, color=accent, fontweight="bold")
        ax.legend(fontsize=6.5, loc="best", ncol=2)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

        # Shade alternating pattern groups
        n_sizes = len(set(_block_size(c) for c in all_cases))
        grp = max(n_sizes, 1)
        for i in range(0, len(all_cases), 2 * grp):
            ax.axvspan(i - 0.5, i + grp - 0.5, alpha=0.04, color="black", zorder=0)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = out_dir / "fig4_ablation_direction_sweep.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 16: Sparsity pattern throughput by direction variant ───────────────

_SPARSITY_PATTERN_LABELS = {
    "random": "Random",
    "col": "Column",
    "diag": "Diagonal",
    "multi_diag": "Multi-Diag",
    "row": "Row",
}

_SPARSITY_PATTERN_ORDER = ["random", "row", "col", "diag", "multi_diag"]

_PATTERN_COLOR = {
    "random": "#1565C0",
    "row": "#E53935",
    "col": "#43A047",
    "diag": "#F57C00",
    "multi_diag": "#7B1FA2",
}

_FIG16_PANELS = [
    ("ProfileSweepSparsityPatternD5", "5%"),
    ("ProfileSweepSparsityPatternD10", "10%"),
    ("ProfileSweepSparsityPattern", "25%"),
    ("ProfileSweepSparsityPatternD50", "50%"),
]


def _parse_sparsity_pattern_stem(stem: str) -> tuple[str | None, dict | None]:
    m = re.match(
        r"parametric_(?:(multi_diag|col|row)_)?M(\d+)_N(\d+)_K(\d+)_R(\d+)_C(\d+)_d(\d+)",
        stem,
    )
    if not m:
        return None, None
    pattern = m.group(1) or "random"
    params = dict(zip(["M", "N", "K", "R", "C", "density"], [int(x) for x in m.groups()[1:]]))
    return pattern, params


def load_sweep_sparsity_pattern(data_dir: Path, registry: str) -> pd.DataFrame:
    """Load timing + metadata for every (direction_variant, sparsity_pattern) pair."""
    variants = _discover_base_variants(data_dir, registry)
    rows = []
    reg_dir = data_dir / registry
    for var in variants:
        var_dir = reg_dir / var
        if not var_dir.exists():
            continue
        for csv in sorted(var_dir.glob("*.csv")):
            if csv.stem.endswith(".device"):
                continue
            pattern, params = _parse_sparsity_pattern_stem(csv.stem)
            if pattern is None or params is None:
                continue
            ns = get_metric(csv)
            if ns is None:
                continue
            log = csv.parent / f"{csv.stem}_sparse.log"
            meta = parse_log_metadata(log)
            nblocks = meta.get("nblocks")
            if nblocks is None:
                continue
            ms = ns / NUM_ITERS / 1e6
            rows.append(
                {
                    "variant": var,
                    "pattern": pattern,
                    **params,
                    "ms": ms,
                    "nblocks": nblocks,
                    "tflops": _tflops_per_sec(nblocks, params["R"], params["C"], params["N"], ms),
                }
            )
    return pd.DataFrame(rows)


def make_figure16(data_dir: Path, out_dir: Path) -> None:
    """
    Figure 16: 2x2 grid, one subplot per density level.
    Bars grouped by direction variant, colored by sparsity pattern.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True)

    for ax, (registry, density_label) in zip(axes.flat, _FIG16_PANELS):
        df = load_sweep_sparsity_pattern(data_dir, registry)
        if df.empty:
            ax.set_title(f"Density = {density_label} (no data)")
            continue

        patterns = [p for p in _SPARSITY_PATTERN_ORDER if p in df["pattern"].values]
        # Order variants by our preferred order
        present = df["variant"].unique().tolist()
        order_map = {v: i for i, v in enumerate(_VARIANT_ORDER)}
        variants_present = sorted(present, key=lambda v: order_map.get(v, 999))

        n_patterns = len(patterns)
        x = np.arange(len(variants_present))
        total_bar_width = 0.75
        bar_w = total_bar_width / max(n_patterns, 1)

        for j, pat in enumerate(patterns):
            ys = []
            for var in variants_present:
                row = df[(df["variant"] == var) & (df["pattern"] == pat)]
                ys.append(row["tflops"].iloc[0] if not row.empty else 0)
            offset = (j - (n_patterns - 1) / 2) * bar_w
            bars = ax.bar(
                x + offset,
                ys,
                bar_w,
                label=_SPARSITY_PATTERN_LABELS.get(pat, pat),
                color=_PATTERN_COLOR.get(pat, "#888888"),
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        h + 0.002,
                        f"{h:.3f}",
                        ha="left",
                        va="bottom",
                        fontsize=5.5,
                        rotation=45,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels([_variant_label(v) for v in variants_present], fontsize=7, rotation=15, ha="right")
        ax.set_title(f"Density = {density_label}", fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    # Shared y-axis
    global_max = max(ax.get_ylim()[1] for ax in axes.flat)
    axes.flat[0].set_ylim(0, global_max * 1.12)

    for ax in axes[:, 0]:
        ax.set_ylabel("Throughput (TFLOPs/s)")

    # Shared legend
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=len(labels), fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.0)
    )
    fig.suptitle(
        "SpMM Throughput vs. Sparsity Pattern (CDA Direction Sweep)\n" "8192x8192x8192, R=C=256",
        fontsize=13,
        fontweight="bold",
        y=1.04,
    )

    fig.tight_layout()
    out = out_dir / "fig16_sparsity_pattern_direction_sweep.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ── Single-variant sparsity sweep: grouped by density level ───────────────────

_DENSITY_PANELS = [
    ("ProfileSweepSparsityPatternD5", "5%", 5),
    ("ProfileSweepSparsityPatternD10", "10%", 10),
    ("ProfileSweepSparsityPattern", "25%", 25),
    ("ProfileSweepSparsityPatternD50", "50%", 50),
]

FAVORITE_VARIANT = "snfin0_cdain1_R2L_T2B"

# Bar order for single-variant sparsity plots
_SINGLE_VARIANT_PATTERN_ORDER = ["row", "col", "multi_diag", "random"]


def _load_single_variant_sparsity(data_dir: Path, registry: str, variant: str) -> pd.DataFrame:
    """Load sparsity pattern data for one specific variant."""
    rows = []
    var_dir = data_dir / registry / variant
    if not var_dir.exists():
        return pd.DataFrame()
    for csv in sorted(var_dir.glob("*.csv")):
        if csv.stem.endswith(".device"):
            continue
        pattern, params = _parse_sparsity_pattern_stem(csv.stem)
        if pattern is None:
            continue
        ns = get_metric(csv)
        if ns is None:
            continue
        log = csv.parent / f"{csv.stem}_sparse.log"
        meta = parse_log_metadata(log)
        nblocks = meta.get("nblocks")
        if nblocks is None:
            continue
        ms = ns / NUM_ITERS / 1e6
        rows.append(
            {
                "pattern": pattern,
                **params,
                "ms": ms,
                "nblocks": nblocks,
                "tflops": _tflops_per_sec(nblocks, params["R"], params["C"], params["N"], ms),
            }
        )
    return pd.DataFrame(rows)


def make_sparsity_by_density(data_dir: Path, out_dir: Path) -> None:
    """
    Single-axis grouped bar chart.
    Groups = density levels (5%, 10%, 25%, 50%).
    Bars within each group = sparsity patterns (Random, Row, Column, Multi-Diag).
    Data from R2L+T2B only.
    """
    # Collect data across all density registries
    all_rows = []
    for registry, density_label, density_val in _DENSITY_PANELS:
        df = _load_single_variant_sparsity(data_dir, registry, FAVORITE_VARIANT)
        if df.empty:
            continue
        df["density_label"] = density_label
        df["density_val"] = density_val
        all_rows.append(df)

    if not all_rows:
        print("WARNING: No data for R2L+T2B. Skipping sparsity-by-density plot.")
        return

    combined = pd.concat(all_rows, ignore_index=True)

    # Determine patterns present
    patterns = [p for p in _SINGLE_VARIANT_PATTERN_ORDER if p in combined["pattern"].values]
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
            x + offset,
            ys,
            bar_w,
            label=_SPARSITY_PATTERN_LABELS.get(pat, pat),
            color=_PATTERN_COLOR.get(pat, "#888888"),
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.2,
                    f"{h:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=0,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Density = {d}" for d in density_labels], fontsize=11)
    ax.set_ylabel("Throughput (TFLOPs/s)")
    ax.set_ylim(bottom=0)
    ax.set_title(
        "CDA R2L+T2B: Throughput vs. Sparsity Pattern by Density\n" "8192x8192x8192, R=C=256",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = out_dir / "sparsity_by_density_R2L_T2B.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


N150_PEAK_TFLOPS = 74.0


def make_sparsity_by_density_pct(data_dir: Path, out_dir: Path) -> None:
    """
    Same as make_sparsity_by_density but y-axis is % of N150 theoretical peak (74 TFLOPs/s).
    """
    all_rows = []
    for registry, density_label, density_val in _DENSITY_PANELS:
        df = _load_single_variant_sparsity(data_dir, registry, FAVORITE_VARIANT)
        if df.empty:
            continue
        df["density_label"] = density_label
        df["density_val"] = density_val
        all_rows.append(df)

    if not all_rows:
        print("WARNING: No data for R2L+T2B. Skipping sparsity-by-density % plot.")
        return

    combined = pd.concat(all_rows, ignore_index=True)

    patterns = [p for p in _SINGLE_VARIANT_PATTERN_ORDER if p in combined["pattern"].values]
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
            x + offset,
            ys,
            bar_w,
            label=_SPARSITY_PATTERN_LABELS.get(pat, pat),
            color=_PATTERN_COLOR.get(pat, "#888888"),
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.3,
                    f"{h:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=0,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Density = {d}" for d in density_labels], fontsize=11)
    ax.set_ylabel("% of N150 Theoretical Peak (74 TFLOPs/s)")
    ax.set_ylim(bottom=0, top=100)
    ax.axhline(100, color="black", linewidth=1.0, linestyle="--", alpha=0.3, label="100% peak")
    ax.set_title(
        "CDA R2L+T2B: Throughput as % of N150 Peak (74 TFLOPs/s)\n" "8192x8192x8192, R=C=256",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = out_dir / "sparsity_by_density_R2L_T2B_pct_peak.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Plot CDA direction sweep figures 4 and 16")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Path to csvs/ directory")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory for plots")
    parser.add_argument(
        "--fig",
        choices=["4", "16", "sparsity", "sparsity_pct", "all"],
        default="all",
        help="Which figure to generate (default: all)",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.fig in ("4", "all"):
        make_figure4(args.data_dir, args.out_dir)

    if args.fig in ("16", "all"):
        make_figure16(args.data_dir, args.out_dir)

    if args.fig in ("sparsity", "all"):
        make_sparsity_by_density(args.data_dir, args.out_dir)

    if args.fig in ("sparsity_pct", "all"):
        make_sparsity_by_density_pct(args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()
