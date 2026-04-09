#!/usr/bin/env python3
"""
plot_sddmm_profiling.py

Build a single CSV comparison table from SDDMM profiling sweeps —
one row per parametric test case across the density / N / K sweeps,
with throughput columns for both the naive and CDA algorithms plus
their speedup ratio.

Usage:
    python sddmm_scripts/plot_sddmm_profiling.py
    python sddmm_scripts/plot_sddmm_profiling.py --data-dir /path/to/csvs --out-dir tables/
"""

import argparse
import re
from pathlib import Path

import pandas as pd


# ── Algorithm metadata ─────────────────────────────────────────────────────────

SDDMM_ALGOS = [
    "bsr_sddmm_multicore_naive",
    "bsr_sddmm_multicore_CDA",
]

SDDMM_DATA_DIR = Path("sddmm_profiles/opt_noc/csvs")

# (display label, registry directory, varying parameter key)
SDDMM_SWEEPS = [
    ("Density", "SDDMMSweepDensity", "density"),
    ("N",       "SDDMMSweepN",       "N"),
    ("K",       "SDDMMSweepK",       "K"),
]


# ── Helpers (from spmm_scripts/plot_profiling_plan.py) ─────────────────────────

def parse_log_metadata(filepath):
    """Parse matrix metadata (H, W, R, C, nblocks) and reported Device TFLOP/s
    from a pretty_print log file."""
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
                elif "Device TFLOP/s" in line:
                    result["tflops"] = float(line.split(":")[1].strip())
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

def _algo_throughput(algo_dir: Path) -> dict[tuple, dict]:
    """
    Walk a single algorithm's CSVs in a sweep directory and return a mapping
    keyed by parameter tuple → {nblocks, tflops}.

    TFLOP/s is read directly from the `Device TFLOP/s` line in the matching
    `<stem>_mask.log` file rather than being recomputed from host-side timing.
    """
    out: dict[tuple, dict] = {}
    if not algo_dir.exists():
        return out
    for csv in sorted(algo_dir.glob("*.csv")):
        if csv.stem.endswith(".device"):
            continue
        params = _parse_parametric(csv.stem)
        if params is None:
            continue
        log = csv.parent / f"{csv.stem}_mask.log"
        meta = parse_log_metadata(log)
        nblocks = meta.get("nblocks")
        tflops = meta.get("tflops")
        if nblocks is None or tflops is None:
            continue
        key = (params["M"], params["N"], params["K"],
               params["R"], params["C"], params["density"])
        out[key] = {
            "nblocks": nblocks,
            "tflops": tflops,
        }
    return out


def _format_tflops(t: float | None) -> str:
    if t is None or pd.isna(t):
        return "---"
    return f"{t:.2f}"


def _format_speedup(s: float | None) -> str:
    if s is None or pd.isna(s):
        return "---"
    return f"{s:.2f}$\\times$"


def build_latex_table(df: pd.DataFrame) -> str:
    """
    Render the SDDMM comparison DataFrame as a booktabs LaTeX table.

    - Drops parameter columns whose value is constant across the whole table.
    - Formats the speedup as "1.72$\\times$".
    - Groups rows by sweep with \\midrule separators (sweep label via \\multirow).

    Requires: \\usepackage{booktabs}, \\usepackage{multirow}.
    """
    # Identify constant parameter columns and drop them.
    candidate_cols = ["M", "N", "K", "R", "C", "Density", "nblocks"]
    kept = [c for c in candidate_cols if df[c].nunique() > 1]

    label_map = {
        "M": "$M$", "N": "$N$", "K": "$K$", "R": "$R$", "C": "$C$",
        "Density": "Density",
        "nblocks": "\\# blocks",
    }
    headers = ["Sweep", *(label_map[c] for c in kept),
               "Naive (TFLOPs/s)", "CDA (TFLOPs/s)", "Speedup"]
    col_spec = "l" + "r" * (len(headers) - 1)

    # Build a short note about the dropped (constant) columns.
    dropped = [c for c in candidate_cols if c not in kept and c != "nblocks"]
    fixed_parts = []
    for c in dropped:
        v = int(df[c].iloc[0])
        if c == "Density":
            fixed_parts.append(f"density={v}\\%")
        else:
            fixed_parts.append(f"${c}={v}$")
    fixed_note = (
        f" Fixed across all rows: {', '.join(fixed_parts)}." if fixed_parts else ""
    )

    lines = [
        "% Requires \\usepackage{booktabs} and \\usepackage{multirow}.",
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{SDDMM throughput: naive vs.\\ CDA across the density, "
        "$N$, and $K$ sweeps." + fixed_note + "}",
        "\\label{tab:sddmm_naive_vs_cda}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]

    sweep_groups = list(df.groupby("Sweep", sort=False))
    for i, (sweep_name, group) in enumerate(sweep_groups):
        if i > 0:
            lines.append("\\midrule")
        n = len(group)
        for j, (_, row) in enumerate(group.iterrows()):
            cells = []
            if j == 0:
                cells.append(f"\\multirow{{{n}}}{{*}}{{{sweep_name}}}")
            else:
                cells.append("")
            for c in kept:
                v = row[c]
                if c == "Density":
                    cells.append(f"{int(v)}\\%")
                else:
                    cells.append(f"{int(v)}")
            n_tf = row["naive_tflops"]
            c_tf = row["cda_tflops"]
            n_str = _format_tflops(n_tf)
            c_str = _format_tflops(c_tf)
            # Bold the larger of the two TFLOP/s values on this row.
            if pd.notna(n_tf) and pd.notna(c_tf):
                if n_tf > c_tf:
                    n_str = f"\\textbf{{{n_str}}}"
                elif c_tf > n_tf:
                    c_str = f"\\textbf{{{c_str}}}"
            cells.append(n_str)
            cells.append(c_str)
            cells.append(_format_speedup(row["speedup"]))
            lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def build_sddmm_table(data_dir: Path) -> pd.DataFrame:
    """
    Build a single naive-vs-CDA comparison table across the density/N/K sweeps.
    Columns: Sweep, M, N, K, R, C, Density, nblocks, naive_tflops, cda_tflops, speedup
    """
    rows = []
    for sweep_label, registry, _varying in SDDMM_SWEEPS:
        reg_dir = data_dir / registry
        naive = _algo_throughput(reg_dir / "bsr_sddmm_multicore_naive")
        cda   = _algo_throughput(reg_dir / "bsr_sddmm_multicore_CDA")
        # Union of test cases so a missing run still shows up as a blank cell.
        keys = sorted(set(naive) | set(cda))
        for key in keys:
            M, N, K, R, C, density = key
            n_entry = naive.get(key, {})
            c_entry = cda.get(key, {})
            n_tf = n_entry.get("tflops")
            c_tf = c_entry.get("tflops")
            nblocks = n_entry.get("nblocks") or c_entry.get("nblocks")
            speedup = (c_tf / n_tf) if (n_tf and c_tf) else None
            rows.append({
                "Sweep": sweep_label,
                "M": M, "N": N, "K": K, "R": R, "C": C,
                "Density": density,
                "nblocks": nblocks,
                "naive_tflops": n_tf,
                "cda_tflops": c_tf,
                "speedup": speedup,
            })
    return pd.DataFrame(rows)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build the SDDMM naive-vs-CDA comparison table.")
    parser.add_argument("--data-dir", type=Path, default=SDDMM_DATA_DIR,
                        help="Root directory with CSV data")
    parser.add_argument("--out-dir", type=Path, default=Path("sddmm_plots"),
                        help="Output directory for the table")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = build_sddmm_table(args.data_dir)
    if df.empty:
        print("WARNING: No SDDMM sweep data found. Nothing to write.")
        return

    out_csv = args.out_dir / "sddmm_naive_vs_cda.csv"
    df.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"Saved {out_csv}")

    out_tex = args.out_dir / "sddmm_naive_vs_cda.tex"
    out_tex.write_text(build_latex_table(df))
    print(f"Saved {out_tex}")

    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
