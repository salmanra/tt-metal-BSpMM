import os
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Wormhole hardware specs ──────────────────────────────────────────
PEAK_TFLOPS = 74       # HiFi4, 80 Tensix cores
DRAM_BW_GB_S = 256     # GB/s (from roofline_utils.py WH_DRAM_THROUGHPUT)

NUM_ITERS = 10

ALGORITHM_COLORS = ["red", "orange", "steelblue", "mediumblue", "midnightblue"]


# ── Configuration ────────────────────────────────────────────────────

def build_config(profiles_dir):
    """Return directory paths and algorithm data directories."""
    csv_dir = os.path.join(profiles_dir, "csvs")
    suite = os.path.join(csv_dir, "ProfileSuiteLargeSparseVersioning")

    algorithm_dirs = [
        os.path.join(suite, "bsr_spmm_multicore_snf"),
        os.path.join(suite, "bsr_spmm_multicore_load_balanced"),
        os.path.join(suite, "bsr_spmm_multicore_reuse_iteration"),
    ]
    algorithm_labels = [os.path.basename(d) for d in algorithm_dirs]

    json_dir = os.path.join(profiles_dir, "jsons")
    png_dir = os.path.join(profiles_dir, "pngs")
    for d in [csv_dir, json_dir, png_dir]:
        os.makedirs(d, exist_ok=True)

    return algorithm_dirs, algorithm_labels, json_dir, png_dir


# ── File discovery ───────────────────────────────────────────────────

def discover_files(algorithm_dirs):
    """Find the union of CSV / log files across all algorithm directories."""
    def union_by_suffix(suffix):
        return sorted(set.union(*(
            {f for f in os.listdir(d) if f.endswith(suffix)}
            for d in algorithm_dirs
        )))

    csv_files = [f for f in union_by_suffix(".csv") if not f.endswith(".device.csv")]
    sparse_logs = union_by_suffix("sparse.log")
    dense_logs = union_by_suffix("dense.log")

    short_names = [
        name.replace("profile_case_sparse_", "").replace(".csv", "").replace("fill_", "")
        for name in csv_files
    ]
    return csv_files, sparse_logs, dense_logs, short_names


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


# ── Metrics computation ─────────────────────────────────────────────

def compute_operational_intensity(M, K, N, R, C, nblocks, total_ops):
    """Compute ideal and pessimistic operational intensity."""
    sparse_values_bytes = nblocks * R * C * 2                   # bfloat16
    sparse_indices_bytes = nblocks * 4 + (M // R + 1) * 4      # col_indices + indptr
    output_bytes = M * N * 2                                    # bfloat16

    # Ideal: dense matrix read once
    dense_bytes_ideal = K * N * 2
    total_bytes_ideal = sparse_values_bytes + sparse_indices_bytes + dense_bytes_ideal + output_bytes
    oi_ideal = total_ops / total_bytes_ideal

    # Pessimistic: dense column-strip re-read per sparse block
    dense_bytes_pessimistic = nblocks * C * N * 2
    total_bytes_pessimistic = sparse_values_bytes + sparse_indices_bytes + dense_bytes_pessimistic + output_bytes
    oi_pessimistic = total_ops / total_bytes_pessimistic

    return oi_ideal, oi_pessimistic


def collect_metrics(algorithm_dirs, csv_files, sparse_logs, dense_logs, short_names):
    """Parse profiles for every algorithm and return a list of per-algorithm dicts."""
    data_dicts = [{} for _ in algorithm_dirs]

    for alg_idx, alg_dir in enumerate(algorithm_dirs):
        for case_idx, csv_name in enumerate(csv_files):
            csv_path = os.path.join(alg_dir, csv_name)
            sparse_log_path = os.path.join(alg_dir, sparse_logs[case_idx])
            dense_log_path = os.path.join(alg_dir, dense_logs[case_idx])

            df = pd.read_csv(csv_path)
            sparse_meta = parse_log_metadata(sparse_log_path)
            dense_meta = parse_log_metadata(dense_log_path)

            M, K = sparse_meta["H"], sparse_meta["W"]
            N = dense_meta["W"]
            R, C = sparse_meta["R"], sparse_meta["C"]
            nblocks = sparse_meta["nblocks"]

            total_ops = nblocks * R * C * N * 2   # 1 mul + 1 add per nnz element per dense column
            tflop_count = total_ops / 1e12

            oi_ideal, oi_pessimistic = compute_operational_intensity(M, K, N, R, C, nblocks, total_ops)

            # Extract device program loop time
            loop_rows = df[df["name"] == "Device program Loop"]
            if loop_rows.empty:
                loop_seconds = np.nan
            else:
                loop_seconds = int(loop_rows["total_ns"].values[0]) / 1e9

            tflops = tflop_count / (loop_seconds / NUM_ITERS) if not np.isnan(loop_seconds) else np.nan

            data_dicts[alg_idx][short_names[case_idx]] = {
                "FLOP count": tflop_count,
                "Program Loop total seconds": loop_seconds,
                "TFLOP/s": tflops,
                "oi_ideal": oi_ideal,
                "oi_pessimistic": oi_pessimistic,
            }

    return data_dicts


# ── Plotting ─────────────────────────────────────────────────────────

def plot_tflops_bar_chart(data_dicts, algorithm_labels, group_labels, output_path):
    """Grouped bar chart comparing TFLOP/s across algorithms and test cases."""
    n_groups = len(group_labels)
    n_algs = len(data_dicts)

    bar_values = np.array([
        [d[k]["TFLOP/s"] for k in group_labels]
        for d in data_dicts
    ])
    max_val = np.nanmax(bar_values) * 1.05

    fig, ax = plt.subplots(figsize=(10, 10))
    bar_width = 0.18
    x = np.arange(n_groups)

    for i in range(n_algs):
        valid = ~np.isnan(bar_values[i])
        ax.bar(x[valid] + i * bar_width, bar_values[i, valid],
               width=bar_width, color=ALGORITHM_COLORS[i], label=algorithm_labels[i])

        # Mark missing data with a red X
        missing = np.isnan(bar_values[i])
        for idx in np.where(missing)[0]:
            ax.plot(x[idx] + i * bar_width + bar_width / 2, max_val / 5,
                    marker="x", color="red", markersize=12, markeredgewidth=3)

    ax.set_xlabel("Test Case")
    ax.set_ylabel(f"TFLOP/s (Peak = {PEAK_TFLOPS})")
    ax.set_title("Sparse Algorithms Runtime Comparison")
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(group_labels, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


def plot_roofline(data_dicts, algorithm_labels, group_labels, oi_key, title, output_path):
    """Log-log roofline plot with measured data points."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Roofline envelope
    oi_range = np.logspace(-1, 4, 500)
    bw_ceiling = DRAM_BW_GB_S * oi_range / 1e3   # TFLOP/s = GB/s * FLOPs/byte / 1e3
    roofline = np.minimum(bw_ceiling, PEAK_TFLOPS)
    ax.plot(oi_range, roofline, "k-", linewidth=2, label="Roofline")

    # Ridge point
    ridge_oi = PEAK_TFLOPS / (DRAM_BW_GB_S / 1e3)
    ax.axvline(ridge_oi, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.annotate(f"Ridge: {ridge_oi:.0f} F/B", xy=(ridge_oi, PEAK_TFLOPS),
                xytext=(ridge_oi * 1.5, PEAK_TFLOPS * 0.5),
                arrowprops=dict(arrowstyle="->", color="gray"), fontsize=8, color="gray")

    # Scatter measured data points
    for i, d in enumerate(data_dicts):
        ois = [d[k][oi_key] for k in group_labels]
        perfs = [d[k]["TFLOP/s"] for k in group_labels]
        ax.scatter(ois, perfs, color=ALGORITHM_COLORS[i], label=algorithm_labels[i], s=80, zorder=5)

    # Annotate test case names
    for k in group_labels:
        oi_val = data_dicts[0][k][oi_key]
        perfs = [d[k]["TFLOP/s"] for d in data_dicts if not np.isnan(d[k]["TFLOP/s"])]
        if perfs:
            ax.annotate(k, xy=(oi_val, min(perfs)),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=7)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Operational Intensity (FLOPs/byte)")
    ax.set_ylabel("Performance (TFLOP/s)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


# ── Main ─────────────────────────────────────────────────────────────

def main():
    profiles_dir = "/home/user/tt-metal/profiles_opt_noc/"
    algorithm_dirs, algorithm_labels, json_dir, png_dir = build_config(profiles_dir)

    csv_files, sparse_logs, dense_logs, short_names = discover_files(algorithm_dirs)
    data_dicts = collect_metrics(algorithm_dirs, csv_files, sparse_logs, dense_logs, short_names)

    # Dump raw metrics to JSON
    with open(os.path.join(json_dir, "data_dicts.json"), "w") as f:
        json.dump(data_dicts, f, indent=4)

    group_labels = list(data_dicts[0].keys())

    # Bar chart
    plot_tflops_bar_chart(data_dicts, algorithm_labels, group_labels,
                          os.path.join(png_dir, "fig2_tflops_opt_nocsv2.png"))

    # Roofline plots
    plot_roofline(data_dicts, algorithm_labels, group_labels, "oi_ideal",
                  "Roofline (Ideal: Dense Matrix Read Once)",
                  os.path.join(png_dir, "roofline_idealv2.png"))

    plot_roofline(data_dicts, algorithm_labels, group_labels, "oi_pessimistic",
                  "Roofline (Pessimistic: Dense Re-read Per Block)",
                  os.path.join(png_dir, "roofline_pessimisticv2.png"))


if __name__ == "__main__":
    main()
