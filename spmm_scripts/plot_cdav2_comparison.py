#!/usr/bin/env python3
"""
plot_cdav2_comparison.py

Compare throughput of CDA V2 (snfin0_cdain1) against the original CDA V1
from the full profiling suite. Only registries present in the CDAV2 data
are plotted.

Usage:
    python spmm_scripts/plot_cdav2_comparison.py
    python spmm_scripts/plot_cdav2_comparison.py --out-dir my_plots/
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Paths ─────────────────────────────────────────────────────────────────────

CDAV2_DIR = Path("/home/user/tt-metal/profiles_opt_noc_CDAV2/csvs")
V1_DIR = Path("/home/user/tt-metal/profiles_opt_noc_full_profiling_suite/csvs")

ALGO = "bsr_spmm_multicore_snfin0_cdain1"
NUM_ITERS = 10

# ── Style ─────────────────────────────────────────────────────────────────────

V1_COLOR = "#1565C0"
V2_COLOR = "#E53935"
V1_LABEL = "CDA V1 (bottom→top)"
V2_LABEL = "CDA V2 (top→bottom)"

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


# ── Sparsity pattern parsing ─────────────────────────────────────────────────

_SPARSITY_PATTERN_ORDER = ["random", "row", "col", "multi_diag"]
_SPARSITY_PATTERN_LABELS = {
    "random":     "Random",
    "col":        "Column",
    "multi_diag": "Multi-Diag",
    "row":        "Row",
}


def _parse_sparsity_pattern_stem(stem: str) -> tuple[str | None, dict | None]:
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


# ── Ablation case parsing ────────────────────────────────────────────────────

_PATTERN_RANK = {
    "fill_row": 0, "diagonal": 1, "fill_column": 2,
    "fill_lower_triangular": 3, "fill_random": 4,
}

def _pattern_rank(s: str) -> int:
    for pat, rank in _PATTERN_RANK.items():
        if pat in s:
            return rank
    return 99

def _block_size(s: str) -> int:
    m = re.search(r"(\d+)[_]C\d+$", s)
    return int(m.group(1)) if m else 0

def _case_label(stem: str) -> str:
    stem_clean = stem.replace("profile_case_sparse_", "").replace("_large", "")
    for pat, short in [("fill_lower_triangular", "tri"),
                       ("fill_column", "col"),
                       ("fill_random", "rand"),
                       ("fill_row", "row"),
                       ("diagonal", "diag")]:
        if pat in stem_clean:
            m = re.search(r"[_]?(\d+)[_]C\d+", stem_clean)
            size = m.group(1) if m else "?"
            return f"{short}\n{size}x{size}"
    return stem


# ── Data loading ──────────────────────────────────────────────────────────────

def load_sparsity_pattern_throughput(data_dir: Path, registry: str) -> pd.DataFrame:
    """Load throughput for cdain1 algo in a sparsity pattern sweep registry."""
    rows = []
    algo_dir = data_dir / registry / ALGO
    if not algo_dir.exists():
        return pd.DataFrame()
    for csv in sorted(algo_dir.glob("*.csv")):
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
        rows.append({
            "pattern": pattern,
            **params,
            "ms": ms,
            "nblocks": nblocks,
            "tflops": _tflops_per_sec(nblocks, params["R"], params["C"],
                                      params["N"], ms),
        })
    return pd.DataFrame(rows)


def load_ablation_throughput(data_dir: Path, registry: str) -> pd.DataFrame:
    """Load throughput for cdain1 in an ablation-style registry (non-parametric cases)."""
    rows = []
    algo_dir = data_dir / registry / ALGO
    if not algo_dir.exists():
        return pd.DataFrame()
    for csv in sorted(algo_dir.glob("*.csv")):
        if csv.stem.endswith(".device"):
            continue
        ns = get_metric(csv)
        if ns is None:
            continue
        sparse_log = csv.parent / f"{csv.stem}_sparse.log"
        dense_log = csv.parent / f"{csv.stem}_dense.log"
        meta = parse_log_metadata(sparse_log)
        dense_meta = parse_log_metadata(dense_log)
        nblocks = meta.get("nblocks")
        R = meta.get("R")
        C = meta.get("C")
        N = dense_meta.get("W")
        if nblocks is None or R is None or C is None or N is None:
            continue
        ms = ns / NUM_ITERS / 1e6
        rows.append({
            "case": csv.stem,
            "ms": ms,
            "tflops": _tflops_per_sec(nblocks, R, C, N, ms),
        })
    return pd.DataFrame(rows)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_sparsity_pattern_comparison(v1_df: pd.DataFrame, v2_df: pd.DataFrame,
                                     registry: str, density_label: str,
                                     ax: plt.Axes) -> None:
    """Grouped bar chart comparing V1 vs V2 throughput across sparsity patterns."""
    patterns = [p for p in _SPARSITY_PATTERN_ORDER
                if p in v2_df["pattern"].values]
    pat_labels = [_SPARSITY_PATTERN_LABELS.get(p, p) for p in patterns]

    x = np.arange(len(patterns))
    w = 0.35

    v1_vals = []
    v2_vals = []
    for pat in patterns:
        r1 = v1_df[v1_df["pattern"] == pat]
        r2 = v2_df[v2_df["pattern"] == pat]
        v1_vals.append(r1["tflops"].iloc[0] if not r1.empty else 0)
        v2_vals.append(r2["tflops"].iloc[0] if not r2.empty else 0)

    bars1 = ax.bar(x - w/2, v1_vals, w, label=V1_LABEL, color=V1_COLOR,
                   edgecolor="white", linewidth=0.5, zorder=3)
    bars2 = ax.bar(x + w/2, v2_vals, w, label=V2_LABEL, color=V2_COLOR,
                   edgecolor="white", linewidth=0.5, zorder=3)

    # Annotate speedup above each pair
    for i, (v1, v2) in enumerate(zip(v1_vals, v2_vals)):
        if v1 > 0 and v2 > 0:
            speedup = v2 / v1
            color = "#1B5E20" if speedup >= 1.0 else "#B71C1C"
            label = f"{speedup:.2f}x"
            ypos = max(v1, v2) * 1.02
            ax.text(x[i], ypos, label, ha="center", va="bottom",
                    fontsize=8, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(pat_labels, fontsize=10)
    ax.set_ylabel("Throughput (TFLOPs/s)")
    ax.set_ylim(bottom=0)
    ax.set_title(f"{registry}\n(density={density_label})", pad=8, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)


def plot_ablation_case_comparison(v1_df: pd.DataFrame, v2_df: pd.DataFrame,
                                  registry: str, ax: plt.Axes) -> None:
    """Grouped bar chart comparing V1 vs V2 throughput across ablation test cases."""
    # Use V2 cases as the reference set
    cases = sorted(v2_df["case"].unique(), key=lambda s: (_pattern_rank(s), _block_size(s)))
    case_labels = [_case_label(c) for c in cases]

    x = np.arange(len(cases))
    w = 0.35

    v1_vals = []
    v2_vals = []
    for c in cases:
        r1 = v1_df[v1_df["case"] == c]
        r2 = v2_df[v2_df["case"] == c]
        v1_vals.append(r1["tflops"].iloc[0] if not r1.empty else 0)
        v2_vals.append(r2["tflops"].iloc[0] if not r2.empty else 0)

    bars1 = ax.bar(x - w/2, v1_vals, w, label=V1_LABEL, color=V1_COLOR,
                   edgecolor="white", linewidth=0.5, zorder=3)
    bars2 = ax.bar(x + w/2, v2_vals, w, label=V2_LABEL, color=V2_COLOR,
                   edgecolor="white", linewidth=0.5, zorder=3)

    for i, (v1, v2) in enumerate(zip(v1_vals, v2_vals)):
        if v1 > 0 and v2 > 0:
            speedup = v2 / v1
            color = "#1B5E20" if speedup >= 1.0 else "#B71C1C"
            label = f"{speedup:.2f}x"
            ypos = max(v1, v2) * 1.02
            ax.text(x[i], ypos, label, ha="center", va="bottom",
                    fontsize=8, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(case_labels, fontsize=7)
    ax.set_ylabel("Throughput (TFLOPs/s)")
    ax.set_ylim(bottom=0)
    ax.set_title(f"{registry}", pad=8, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)


# ── Registry detection ────────────────────────────────────────────────────────

# Sparsity pattern sweep registries and their density labels
_SWEEP_REGISTRIES = {
    "ProfileSweepSparsityPattern":     "25%",
    "ProfileSweepSparsityPatternD5":   "5%",
    "ProfileSweepSparsityPatternD10":  "10%",
    "ProfileSweepSparsityPatternD50":  "50%",
}

# Ablation-style registries (non-parametric case names)
_ABLATION_REGISTRIES = [
    "ProfileSuiteLargeSparseLargeBlocksVersioning",
]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot CDA V2 vs V1 throughput comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cdav2-dir", type=Path, default=CDAV2_DIR,
                        help="CDAV2 CSV root directory")
    parser.add_argument("--v1-dir", type=Path, default=V1_DIR,
                        help="V1 (full profiling suite) CSV root directory")
    parser.add_argument("--out-dir", type=Path, default=Path("spmm_plots/cdav2"),
                        help="Output directory for PNG figures")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Discover which registries are present in CDAV2
    cdav2_registries = [d.name for d in args.cdav2_dir.iterdir() if d.is_dir()]
    print(f"CDAV2 registries found: {cdav2_registries}")

    # ── Sparsity pattern sweep registries ──
    sweep_regs = [r for r in cdav2_registries if r in _SWEEP_REGISTRIES]
    # Sort by density
    sweep_regs.sort(key=lambda r: float(_SWEEP_REGISTRIES[r].rstrip("%")))

    if sweep_regs:
        n = len(sweep_regs)
        ncols = min(n, 2)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * nrows),
                                 squeeze=False)
        fig.suptitle(
            "CDA V2 vs V1 Throughput — Sparsity Pattern Sweeps\n"
            f"(algorithm: {ALGO})",
            fontsize=13, fontweight="bold",
        )

        for idx, reg in enumerate(sweep_regs):
            ax = axes[idx // ncols, idx % ncols]
            v2_df = load_sparsity_pattern_throughput(args.cdav2_dir, reg)
            v1_df = load_sparsity_pattern_throughput(args.v1_dir, reg)
            if v2_df.empty:
                ax.text(0.5, 0.5, f"No V2 data for {reg}",
                        ha="center", va="center", transform=ax.transAxes)
                continue
            plot_sparsity_pattern_comparison(
                v1_df, v2_df, reg, _SWEEP_REGISTRIES[reg], ax)

        # Hide unused axes
        for idx in range(len(sweep_regs), nrows * ncols):
            axes[idx // ncols, idx % ncols].set_visible(False)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        out = args.out_dir / "cdav2_vs_v1_sparsity_sweeps.png"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved {out}")

    # ── Ablation-style registries ──
    ablation_regs = [r for r in cdav2_registries if r in _ABLATION_REGISTRIES]

    for reg in ablation_regs:
        v2_df = load_ablation_throughput(args.cdav2_dir, reg)
        v1_df = load_ablation_throughput(args.v1_dir, reg)
        if v2_df.empty:
            print(f"No V2 data for {reg}, skipping.")
            continue

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle(
            f"CDA V2 vs V1 Throughput — {reg}\n"
            f"(algorithm: {ALGO})",
            fontsize=13, fontweight="bold",
        )
        plot_ablation_case_comparison(v1_df, v2_df, reg, ax)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        out = args.out_dir / f"cdav2_vs_v1_{reg}.png"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved {out}")

    # ── Summary table ──
    print("\n=== Summary ===")
    print(f"{'Registry':<50} {'Pattern/Case':<25} {'V1 TFLOPs/s':>12} {'V2 TFLOPs/s':>12} {'Speedup':>8}")
    print("-" * 110)
    for reg in sweep_regs:
        v1_df = load_sparsity_pattern_throughput(args.v1_dir, reg)
        v2_df = load_sparsity_pattern_throughput(args.cdav2_dir, reg)
        for pat in _SPARSITY_PATTERN_ORDER:
            r1 = v1_df[v1_df["pattern"] == pat] if not v1_df.empty else pd.DataFrame()
            r2 = v2_df[v2_df["pattern"] == pat] if not v2_df.empty else pd.DataFrame()
            v1 = r1["tflops"].iloc[0] if not r1.empty else 0
            v2 = r2["tflops"].iloc[0] if not r2.empty else 0
            if v1 > 0 and v2 > 0:
                sp = v2 / v1
                print(f"{reg:<50} {pat:<25} {v1:>12.4f} {v2:>12.4f} {sp:>7.2f}x")
    for reg in ablation_regs:
        v1_df = load_ablation_throughput(args.v1_dir, reg)
        v2_df = load_ablation_throughput(args.cdav2_dir, reg)
        if v2_df.empty:
            continue
        for _, row2 in v2_df.iterrows():
            c = row2["case"]
            r1 = v1_df[v1_df["case"] == c] if not v1_df.empty else pd.DataFrame()
            v1 = r1["tflops"].iloc[0] if not r1.empty else 0
            v2 = row2["tflops"]
            if v1 > 0 and v2 > 0:
                sp = v2 / v1
                short = c.replace("profile_case_sparse_", "")
                print(f"{reg:<50} {short:<25} {v1:>12.4f} {v2:>12.4f} {sp:>7.2f}x")


if __name__ == "__main__":
    main()
