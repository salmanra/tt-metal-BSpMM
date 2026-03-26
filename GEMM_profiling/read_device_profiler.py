# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Read device profiler log and print TRISC1 kernel duration / device utilization.

Run this after run_minimal_matmul.py completes (the profiler log is flushed at process exit).

Usage:
    python GEMM_profiling/read_device_profiler.py
    python GEMM_profiling/read_device_profiler.py --M 4096 --K 4096 --N 4096 --fidelity HiFi4
"""

import argparse

import numpy as np
from models.utility_functions import is_wormhole_b0, is_blackhole
from tt_metal.tools.profiler.process_device_log import import_log_run_stats
import tt_metal.tools.profiler.device_post_proc_config as device_post_proc_config
from tt_metal.tools.profiler.common import PROFILER_LOGS_DIR, PROFILER_DEVICE_SIDE_LOG

profiler_log_path = PROFILER_LOGS_DIR / PROFILER_DEVICE_SIDE_LOG


def main():
    parser = argparse.ArgumentParser(description="Read device profiler log from a prior run_minimal_matmul.py run")
    parser.add_argument("--M", type=int, default=4096)
    parser.add_argument("--K", type=int, default=4096)
    parser.add_argument("--N", type=int, default=4096)
    parser.add_argument("--grid-x", type=int, default=8)
    parser.add_argument("--grid-y", type=int, default=8)
    parser.add_argument("--fidelity", type=str, default="HiFi4", choices=["LoFi", "HiFi2", "HiFi3", "HiFi4"])
    args = parser.parse_args()

    if not profiler_log_path.exists():
        print(f"Profiler log not found: {profiler_log_path}")
        print("Run run_minimal_matmul.py with TT_METAL_DEVICE_PROFILER=1 first, then wait for process to fully exit.")
        return

    print(f"Reading: {profiler_log_path}")
    print(f"  File size: {profiler_log_path.stat().st_size:,} bytes")

    setup = device_post_proc_config.default_setup()
    setup.deviceInputLog = profiler_log_path
    deviceData = import_log_run_stats(setup)

    device_freq = deviceData["deviceInfo"]["freq"]
    trisc1 = deviceData["devices"][0]["cores"]["DEVICE"]["analysis"]["device_trisc1_kernel_duration"]["stats"]["Average"]

    print(f"  Device freq:      {device_freq} MHz")
    print(f"  TRISC1 avg:       {np.mean(trisc1):.0f} cycles")
    print(f"  TRISC1 avg time:  {np.mean(trisc1) / device_freq / 1e3:.3f} ms")

    # Utilization calculation
    M, K, N = args.M, args.K, args.N
    LoFi_cycle = 16
    cycle_per_tile = {
        "LoFi": LoFi_cycle,
        "HiFi2": LoFi_cycle * 2,
        "HiFi3": LoFi_cycle * 3,
        "HiFi4": LoFi_cycle * 4,
    }[args.fidelity]
    num_cores = args.grid_x * args.grid_y
    ideal_cycles = M * K * N / 32 / 32 / 32 * cycle_per_tile / num_cores
    total_flops = 2 * M * K * N

    device_utilization = ideal_cycles / np.mean(trisc1)
    device_time_s = np.mean(trisc1) / device_freq / 1e6
    device_tflops = total_flops / device_time_s / 1e12

    print(f"\n  For {M}x{K}x{N} {args.fidelity} on {args.grid_x}x{args.grid_y} grid:")
    print(f"    Ideal cycles:   {ideal_cycles:.0f}")
    print(f"    Device util:    {device_utilization * 100:.2f}%")
    print(f"    Device TFLOP/s: {device_tflops:.2f}")


if __name__ == "__main__":
    main()
