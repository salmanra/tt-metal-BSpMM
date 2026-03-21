#!/usr/bin/env python3
"""
plot_mathapprox_comparison.py

Compare CDA R2L+T2B performance with vs without math_approx_mode.

  Figure 4 analog — Runtime comparison per test case (Large Blocks 256x256)
  Figure 16 analog — Sparsity pattern throughput (2x2 density grid)

Data sources:
  profiles_opt_noc_CDA_mathapprox/csvs/       (math_approx_mode = true)
  profiles_opt_noc_CDA_no_mathapprox/csvs/     (math_approx_mode = false)

Usage:
    python spmm_scripts/plot_mathapprox_comparison.py
    python spmm_scripts/plot_mathapprox_comparison.py --fig 4
    python spmm_scripts/plot_mathapprox_comparison.py --fig 16
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Defaults ──────────────────────────────────────────────────────────────────

BASE_DIR = Path("/home/user/tt-metal")
DATA_APPROX = BASE_DIR / "profiles_opt_noc_CDA_mathapprox" / "csvs"
DATA_NO_APPROX = BASE_DIR / "profiles_opt_noc_CDA_no_mathapprox" / "csvs"
DEFAULT_OUT_DIR = BASE_DIR / "spmm_plots" / "mathapprox"

ABLATION_REGISTRY = "ProfileSuiteLargeSparseLargeBlocksVersioning"
NUM_ITERS = 10

# The two configs being compared
CONFIGS = [
    {"name": "math_approx ON", "dir": DATA_APPROX, "color": "#1565C0"},
    {"name": "math_approx OFF", "dir": DATA_NO_APPROX, "color": "#E53935"},
]


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
    try:
        df = pd.read_csv(csv_path, usecols=["name", "total_ns"])
        row = df[df["name"] == zone]
        return float(row["total_ns"].iloc[0]) if not row.empty else None
    except Exception:
        return None


def parse_log_metadata(filepath: Path) -> dict:
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


def _discover_variant(data_dir: Path, registry: str) -> str | None:
    """Find the single host-code variant directory in a registry."""
    reg_dir = data_dir / registry
    if not reg_dir.exists():
        return None
    dirs = [d.name for d in reg_dir.iterdir() if d.is_dir()]
    return dirs[0] if dirs else None


# ── Case label helpers (from plot_cdav2_direction_sweep.py) ───────────────────


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


def _block_size(s: str) -> int:
    m = re.search(r"(\d+)[_]C\d+$", s)
    return int(m.group(1)) if m else 0


# ── Figure 4: Runtime comparison per test case ───────────────────────────────


def make_figure4(out_dir: Path) -> None:
    """
    Side-by-side bar chart comparing runtime (ms) per test case
    for math_approx ON vs OFF.
    """
    # Collect cases from first config
    variant = _discover_variant(CONFIGS[0]["dir"], ABLATION_REGISTRY)
    if variant is None:
        print("WARNING: No variant found. Skipping Figure 4.")
        return

    case_dir = CONFIGS[0]["dir"] / ABLATION_REGISTRY / variant
    cases = sorted(
        (p.stem for p in case_dir.glob("*.csv") if not p.stem.endswith(".device")),
        key=lambda s: (_pattern_rank(s), _block_size(s)),
    )

    # Load dense block widths for labels
    dense_bw = {}
    for log in case_dir.glob("*_dense.log"):
        c = log.stem.removesuffix("_dense")
        meta = parse_log_metadata(log)
        if "in1_block_w" in meta:
            dense_bw[c] = meta["in1_block_w"] * 32

    # Load timing for each config
    config_ms = {}
    for cfg in CONFIGS:
        v = _discover_variant(cfg["dir"], ABLATION_REGISTRY)
        if v is None:
            continue
        ms_list = []
        for case in cases:
            p = cfg["dir"] / ABLATION_REGISTRY / v / f"{case}.csv"
            ns = get_metric(p)
            ms_list.append(ns / NUM_ITERS / 1e6 if ns else 0)
        config_ms[cfg["name"]] = ms_list

    x = np.arange(len(cases))
    n_configs = len(config_ms)
    total_w = 0.7
    bar_w = total_w / n_configs
    case_labels = [_case_label(c, dense_bw) for c in cases]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

    # Top panel: absolute runtime
    ax = axes[0]
    for i, cfg in enumerate(CONFIGS):
        name = cfg["name"]
        if name not in config_ms:
            continue
        offset = (i - (n_configs - 1) / 2) * bar_w
        bars = ax.bar(
            x + offset,
            config_ms[name],
            bar_w,
            label=name,
            color=cfg["color"],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.05,
                    f"{h:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=0,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(case_labels, fontsize=8)
    ax.set_ylabel("Runtime (ms)")
    ax.set_title(
        f"CDA R2L+T2B Runtime: math_approx ON vs OFF\n" f"({ABLATION_REGISTRY}, Large Blocks 256x256)",
        fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    # Bottom panel: speedup (approx_ON / approx_OFF)
    ax2 = axes[1]
    approx_ms = config_ms.get("math_approx ON", [])
    no_approx_ms = config_ms.get("math_approx OFF", [])
    if approx_ms and no_approx_ms:
        speedups = [(na / a if a > 0 and na > 0 else 1.0) for a, na in zip(approx_ms, no_approx_ms)]
        colors = ["#43A047" if s > 1.0 else "#E53935" for s in speedups]
        bars = ax2.bar(x, speedups, 0.5, color=colors, edgecolor="white", linewidth=0.5, zorder=3)
        for bar, s in zip(bars, speedups):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{s:.3f}x",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold",
            )
        ax2.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax2.set_xticks(x)
        ax2.set_xticklabels(case_labels, fontsize=8)
        ax2.set_ylabel("Speedup\n(approx ON vs OFF)")
        ax2.set_ylim(min(speedups) - 0.05, max(speedups) + 0.05)
        ax2.grid(axis="y", alpha=0.25)
        ax2.set_axisbelow(True)

    fig.tight_layout()
    out = out_dir / "fig4_mathapprox_comparison.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 16: Sparsity pattern throughput comparison ─────────────────────────

_SPARSITY_PATTERN_LABELS = {
    "random": "Random",
    "col": "Column",
    "diag": "Diagonal",
    "multi_diag": "Multi-Diag",
    "row": "Row",
}
_SPARSITY_PATTERN_ORDER = ["random", "row", "col", "diag", "multi_diag"]

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


def _load_sparsity_data(data_dir: Path, registry: str) -> pd.DataFrame:
    """Load sparsity pattern timing from one data dir."""
    variant = _discover_variant(data_dir, registry)
    if variant is None:
        return pd.DataFrame()
    rows = []
    var_dir = data_dir / registry / variant
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


def make_figure16(out_dir: Path) -> None:
    """
    Figure 16: 2x2 density grid. Each subplot has sparsity patterns on x-axis
    with side-by-side bars for math_approx ON vs OFF, plus speedup annotations.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True)

    for ax, (registry, density_label) in zip(axes.flat, _FIG16_PANELS):
        # Load both configs
        dfs = {}
        for cfg in CONFIGS:
            dfs[cfg["name"]] = _load_sparsity_data(cfg["dir"], registry)

        # Find common patterns
        all_patterns = set()
        for df in dfs.values():
            if not df.empty:
                all_patterns.update(df["pattern"].values)
        patterns = [p for p in _SPARSITY_PATTERN_ORDER if p in all_patterns]

        if not patterns:
            ax.set_title(f"Density = {density_label} (no data)")
            continue

        x = np.arange(len(patterns))
        n_configs = len(CONFIGS)
        total_w = 0.6
        bar_w = total_w / n_configs

        for i, cfg in enumerate(CONFIGS):
            df = dfs[cfg["name"]]
            ys = []
            for pat in patterns:
                row = df[df["pattern"] == pat] if not df.empty else pd.DataFrame()
                ys.append(row["tflops"].iloc[0] if not row.empty else 0)
            offset = (i - (n_configs - 1) / 2) * bar_w
            bars = ax.bar(
                x + offset, ys, bar_w, label=cfg["name"], color=cfg["color"], edgecolor="white", linewidth=0.5, zorder=3
            )
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        h + 0.15,
                        f"{h:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=6.5,
                        rotation=45,
                    )

        # Annotate speedup between each pair of bars
        df_on = dfs.get("math_approx ON", pd.DataFrame())
        df_off = dfs.get("math_approx OFF", pd.DataFrame())
        if not df_on.empty and not df_off.empty:
            for j, pat in enumerate(patterns):
                t_on = df_on.loc[df_on["pattern"] == pat, "tflops"]
                t_off = df_off.loc[df_off["pattern"] == pat, "tflops"]
                if not t_on.empty and not t_off.empty:
                    spd = t_on.iloc[0] / t_off.iloc[0]
                    top = max(t_on.iloc[0], t_off.iloc[0])
                    color = "#43A047" if spd > 1.0 else "#E53935"
                    ax.text(
                        x[j],
                        top + 1.5,
                        f"{spd:.2f}x",
                        ha="center",
                        va="bottom",
                        fontsize=7.5,
                        fontweight="bold",
                        color=color,
                    )

        pat_labels = [_SPARSITY_PATTERN_LABELS.get(p, p) for p in patterns]
        ax.set_xticks(x)
        ax.set_xticklabels(pat_labels, fontsize=9)
        ax.set_title(f"Density = {density_label}", fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    # Shared y-axis
    global_max = max(ax.get_ylim()[1] for ax in axes.flat)
    axes.flat[0].set_ylim(0, global_max * 1.15)

    for ax in axes[:, 0]:
        ax.set_ylabel("Throughput (TFLOPs/s)")

    # Shared legend
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=len(labels), fontsize=10, frameon=False, bbox_to_anchor=(0.5, 1.0)
    )
    fig.suptitle(
        "SpMM Throughput: math_approx ON vs OFF (CDA R2L+T2B)\n" "8192x8192x8192, R=C=256",
        fontsize=13,
        fontweight="bold",
        y=1.04,
    )

    fig.tight_layout()
    out = out_dir / "fig16_mathapprox_comparison.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Compare CDA performance with vs without math_approx_mode")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fig", choices=["4", "16", "all"], default="all")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.fig in ("4", "all"):
        make_figure4(args.out_dir)
    if args.fig in ("16", "all"):
        make_figure16(args.out_dir)


if __name__ == "__main__":
    main()
