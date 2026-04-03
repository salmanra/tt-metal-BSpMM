#!/usr/bin/env python3
"""
plot_lowdensity.py

Plot throughput for ultra-low density profiling experiments.
Reads Device TFLOP/s from sparse.log files in UltraLowDensity32 and
UltraLowDensity64 registry directories.

Produces one grouped bar chart per block size (R=C=32 and R=C=64),
with density on the x-axis and bars colored by algorithm variant.

Usage:
    python spmm_scripts/plot_lowdensity.py
    python spmm_scripts/plot_lowdensity.py --data-dir /path/to/csvs --out-dir my_plots/
"""

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────

DATA_DIR = Path("/home/user/tt-metal/profiles_sc26/csvs")
GPU_CSV = Path("/home/user/tt-metal/tt_metal/programming_examples/rahmy/gpu-normalized/ultra_sparse_gpu_ELL.csv")

N150_PEAK_TFLOPS = 74.0

# ── Style ────────────────────────────────────────────────────────────────────

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

_ALGO_COLOR = {
    "bsr_spmm_multicore_load_balanced_new_DM": "#E53935",
    "bsr_spmm_multicore_snf":                  "#43A047",
    "bsr_spmm_multicore_snfin0_cdain1":        "#1565C0",
}

_ALGO_LABEL = {
    "bsr_spmm_multicore_load_balanced_new_DM": "Naive",
    "bsr_spmm_multicore_snf":                  "SnF",
    "bsr_spmm_multicore_snfin0_cdain1":        "CDA",
}

_ALGO_ORDER = [
    "bsr_spmm_multicore_load_balanced_new_DM",
    "bsr_spmm_multicore_snf",
    "bsr_spmm_multicore_snfin0_cdain1",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

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


def ppm_to_label(ppm: int) -> str:
    """Convert PPM value to a human-readable density label."""
    pct = ppm / 10000.0
    if pct >= 1.0:
        return f"{pct:.0f}%"
    elif pct >= 0.01:
        return f"{pct:.2f}%"
    else:
        return f"{pct:.3f}%"


def load_registry_data(data_dir: Path, registry: str) -> list[dict]:
    """Load all TFLOPs data from a registry directory."""
    rows = []
    reg_dir = data_dir / registry
    if not reg_dir.exists():
        return rows
    for algo_dir in sorted(reg_dir.iterdir()):
        if not algo_dir.is_dir():
            continue
        algo_name = algo_dir.name
        for log in sorted(algo_dir.glob("*_sparse.log")):
            stem = log.name.removesuffix("_sparse.log")
            # Match the new dppm format
            m = re.match(
                r"parametric_M(\d+)_N(\d+)_K(\d+)_R(\d+)_C(\d+)_dppm(\d+)",
                stem,
            )
            if not m:
                continue
            ppm = int(m.group(6))
            tflops = parse_tflops_from_log(log)
            if tflops is None:
                continue
            rows.append({
                "algo": algo_name,
                "ppm": ppm,
                "R": int(m.group(4)),
                "C": int(m.group(5)),
                "tflops": tflops,
            })
    return rows


def load_gpu_data(csv_path: Path, block_size: int) -> list[dict]:
    """Load GPU TFLOPs from the ultra-sparse ELL CSV, filtered by block size."""
    rows = []
    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row["Block"]) != block_size:
                    continue
                m = re.search(r"_dppm(\d+)", row["Case"])
                if not m:
                    continue
                rows.append({
                    "ppm": int(m.group(1)),
                    "tflops": float(row["Avg_TFLOPs"]),
                })
    except FileNotFoundError:
        print(f"WARNING: GPU CSV not found at {csv_path}")
    return rows


# ── Plotting ─────────────────────────────────────────────────────────────────

def make_lowdensity_chart(data_dir: Path, out_dir: Path,
                          registry: str, block_size: int) -> None:
    """
    Grouped bar chart: x-axis = density (PPM), bars = algorithm variants.
    Matches the visual style of plot_device_runtime.py charts.
    """
    rows = load_registry_data(data_dir, registry)
    if not rows:
        print(f"WARNING: No data for {registry}. Skipping.")
        return

    # Collect available algos and density points
    algos = [a for a in _ALGO_ORDER if any(r["algo"] == a for r in rows)]
    ppm_vals = sorted(set(r["ppm"] for r in rows))

    n_algos = len(algos)
    n_densities = len(ppm_vals)
    x = np.arange(n_densities)
    total_bar_width = 0.75
    bar_w = total_bar_width / max(n_algos, 1)

    fig, ax = plt.subplots(figsize=(12, 6))

    for j, algo in enumerate(algos):
        ys = []
        for ppm in ppm_vals:
            match = [r for r in rows if r["algo"] == algo and r["ppm"] == ppm]
            ys.append(match[0]["tflops"] if match else 0)
        offset = (j - (n_algos - 1) / 2) * bar_w
        bars = ax.bar(
            x + offset, ys, bar_w,
            label=_ALGO_LABEL.get(algo, algo),
            color=_ALGO_COLOR.get(algo, "#888888"),
            edgecolor="white", linewidth=0.5, zorder=3,
        )
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.2,
                        f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{ppm_to_label(p)}\n({p} PPM)" for p in ppm_vals],
                       fontsize=10)
    ax.set_ylabel("Throughput (TFLOPs/s)")
    ax.set_ylim(bottom=0, top=N150_PEAK_TFLOPS * 1.1)
    ax.axhline(N150_PEAK_TFLOPS, color="black", linewidth=1.0, linestyle="--",
               alpha=0.3, label=f"N150 peak ({N150_PEAK_TFLOPS:.0f} TFLOPs/s)")
    ax.set_title(
        f"Tenstorrent N150 — Ultra-Low Density Throughput\n"
        f"8192x8192x8192, R=C={block_size}  (device-runtime measurement)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = out_dir / f"lowdensity_throughput_R{block_size}.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def make_lowdensity_chart_with_gpu(data_dir: Path, out_dir: Path,
                                    registry: str, block_size: int,
                                    gpu_data: list[dict]) -> None:
    """
    Same as make_lowdensity_chart but with an extra bar for RTX 4090 cuSPARSE.
    """
    rows = load_registry_data(data_dir, registry)
    if not rows:
        print(f"WARNING: No data for {registry}. Skipping GPU comparison plot.")
        return

    algos = [a for a in _ALGO_ORDER if any(r["algo"] == a for r in rows)]
    ppm_vals = sorted(set(r["ppm"] for r in rows))

    # Series: TT algos + GPU
    series = [(algo, _ALGO_LABEL.get(algo, algo), _ALGO_COLOR.get(algo, "#888")) for algo in algos]
    series.append(("gpu", "RTX 4090 (cuSPARSE), 51 SMs", "#FF9800"))

    n_series = len(series)
    n_densities = len(ppm_vals)
    x = np.arange(n_densities)
    total_bar_width = 0.80
    bar_w = total_bar_width / max(n_series, 1)

    fig, ax = plt.subplots(figsize=(12, 6))

    gpu_by_ppm = {r["ppm"]: r["tflops"] for r in gpu_data}

    for j, (key, label, color) in enumerate(series):
        ys = []
        for ppm in ppm_vals:
            if key == "gpu":
                ys.append(gpu_by_ppm.get(ppm, 0))
            else:
                match = [r for r in rows if r["algo"] == key and r["ppm"] == ppm]
                ys.append(match[0]["tflops"] if match else 0)
        offset = (j - (n_series - 1) / 2) * bar_w
        bars = ax.bar(
            x + offset, ys, bar_w,
            label=label, color=color,
            edgecolor="white", linewidth=0.5, zorder=3,
        )
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.2,
                        f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{ppm_to_label(p)}\n({p} PPM)" for p in ppm_vals],
                       fontsize=10)
    ax.set_ylabel("Throughput (TFLOPs/s)")
    y_max = max(
        max((r["tflops"] for r in rows), default=0),
        max((r["tflops"] for r in gpu_data), default=0),
    )
    ax.set_ylim(bottom=0, top=max(y_max * 1.25, 1))
    ax.set_title(
        f"Tenstorrent N150 vs. RTX 4090 — Ultra-Low Density Throughput\n"
        f"8192x8192x8192, R=C={block_size}  (device-runtime measurement)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = out_dir / f"lowdensity_throughput_R{block_size}_vs_gpu.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot ultra-low density throughput (TFLOPs from sparse.log)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="CSV root directory")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("spmm_plots/lowdensity"),
                        help="Output directory for PNG figures")
    parser.add_argument("--gpu-csv", type=Path, default=GPU_CSV,
                        help="Path to GPU ELL benchmark CSV")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    make_lowdensity_chart(args.data_dir, args.out_dir,
                          "UltraLowDensity32", 32)
    make_lowdensity_chart(args.data_dir, args.out_dir,
                          "UltraLowDensity64", 64)

    # GPU comparison versions
    gpu32 = load_gpu_data(args.gpu_csv, 32)
    gpu64 = load_gpu_data(args.gpu_csv, 64)
    if gpu32:
        make_lowdensity_chart_with_gpu(args.data_dir, args.out_dir,
                                       "UltraLowDensity32", 32, gpu32)
    if gpu64:
        make_lowdensity_chart_with_gpu(args.data_dir, args.out_dir,
                                       "UltraLowDensity64", 64, gpu64)

    # ── Summary table ──
    print(f"\n{'Registry':<25} {'Algo':<45} {'PPM':>8} {'Density':>10} {'TFLOPs/s':>10} {'% Peak':>8}")
    print("-" * 110)
    for registry, bs in [("UltraLowDensity32", 32), ("UltraLowDensity64", 64)]:
        rows = load_registry_data(args.data_dir, registry)
        for algo in _ALGO_ORDER:
            for r in sorted([r for r in rows if r["algo"] == algo], key=lambda r: r["ppm"]):
                pct = r["tflops"] / N150_PEAK_TFLOPS * 100
                print(f"{registry:<25} {algo:<45} {r['ppm']:>8} {ppm_to_label(r['ppm']):>10} {r['tflops']:>10.2f} {pct:>7.1f}%")


if __name__ == "__main__":
    main()
