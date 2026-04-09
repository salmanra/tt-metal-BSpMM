#!/usr/bin/env python3
"""
sweep_from_gpu_csv.py

Run TT GEMM (minimal_matmul) on the same (M, K, N) test cases as the GPU
baseline at tt_metal/programming_examples/rahmy/gpu-normalized/gemm_gpu.csv
and write a results CSV with the same schema, so the two files can be
joined row-for-row by ``Case``.

Each test case is timed via the **device profiler** (TRISC1 kernel duration),
not host wall-clock — same two-step pattern as ``sweep_square.sh``:

    TT_METAL_DEVICE_PROFILER=1 python run_minimal_matmul.py --M ... --K ... --N ... --trace
    python read_device_profiler.py --M ... --K ... --N ...

Registry 4 (M=N=32768) is skipped by default to avoid OOM on the 32768²
output tensor; pass --include-reg4 to attempt those rows anyway.

Usage:
    # From the repo root (/home/user/tt-metal):
    python GEMM_profiling/sweep_from_gpu_csv.py
    python GEMM_profiling/sweep_from_gpu_csv.py --out GEMM_profiling/gemm_tt.csv
    python GEMM_profiling/sweep_from_gpu_csv.py --no-trace
    python GEMM_profiling/sweep_from_gpu_csv.py --include-reg4
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path


# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_GPU_CSV = Path(
    "tt_metal/programming_examples/rahmy/gpu-normalized/gemm_gpu.csv"
)
DEFAULT_OUT = Path("GEMM_profiling/gemm_tt.csv")
RUNNER = Path("GEMM_profiling/run_minimal_matmul.py")
PROFILER_READER = Path("GEMM_profiling/read_device_profiler.py")

# Registries to skip by default. Reg 4 is the M=N=32768 set.
SKIP_REGISTRIES = {4}

# Bytes per element for the GB/s calculation. minimal_matmul defaults to bf16.
BYTES_PER_ELEM = 2


# ── stdout parsing ────────────────────────────────────────────────────────────

# read_device_profiler.py prints lines like:
#     TRISC1 avg time:  0.832 ms
#     Device TFLOP/s: 65.12
RE_DEV_MS = re.compile(r"TRISC1 avg time:\s+([\d.]+)\s*ms")
RE_DEV_TFLOPS = re.compile(r"Device TFLOP/s:\s+([\d.]+)")


def _run_subprocess(cmd: list[str], *, env: dict | None = None) -> subprocess.CompletedProcess | None:
    """Run a subprocess, print the command, and return the CompletedProcess.

    Returns None on FileNotFoundError. Caller is responsible for inspecting
    the return code.
    """
    print(f"  $ {' '.join(cmd)}")
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    except FileNotFoundError as e:
        print(f"    ERROR: {e}", file=sys.stderr)
        return None


def _print_stderr_tail(proc: subprocess.CompletedProcess, n: int = 8) -> None:
    for line in proc.stderr.strip().splitlines()[-n:]:
        print(f"    | {line}", file=sys.stderr)


def run_one(M: int, K: int, N: int, *, trace: bool) -> dict | None:
    """Time one (M, K, N) case via the device profiler.

    Two-step pattern (matches sweep_square.sh):
        1. Run run_minimal_matmul.py with TT_METAL_DEVICE_PROFILER=1.
        2. Run read_device_profiler.py and parse TRISC1 avg time / Device TFLOP/s.

    Returns {Avg_ms, Avg_TFLOPs, Avg_GBs} from the device measurement,
    or None on any failure.
    """
    # Step 1: run the matmul with the device profiler enabled.
    runner_cmd = [sys.executable, str(RUNNER),
                  "--M", str(M), "--K", str(K), "--N", str(N)]
    if trace:
        runner_cmd.append("--trace")
    env = {**os.environ, "TT_METAL_DEVICE_PROFILER": "1"}
    proc = _run_subprocess(runner_cmd, env=env)
    if proc is None:
        return None
    if proc.returncode != 0:
        print(f"    runner exited {proc.returncode}", file=sys.stderr)
        _print_stderr_tail(proc)
        return None

    # Step 2: read the device profiler log.
    reader_cmd = [sys.executable, str(PROFILER_READER),
                  "--M", str(M), "--K", str(K), "--N", str(N)]
    rproc = _run_subprocess(reader_cmd)
    if rproc is None:
        return None
    if rproc.returncode != 0:
        print(f"    profiler reader exited {rproc.returncode}", file=sys.stderr)
        _print_stderr_tail(rproc)
        return None

    m_ms = RE_DEV_MS.search(rproc.stdout)
    m_tf = RE_DEV_TFLOPS.search(rproc.stdout)
    if not (m_ms and m_tf):
        print("    couldn't parse device profiler output", file=sys.stderr)
        return None

    avg_ms = float(m_ms.group(1))
    avg_tflops = float(m_tf.group(1))

    # Compute GB/s from device time and matrix shapes (read_device_profiler.py
    # doesn't print it). bf16 = 2 bytes/elem, matching the runner default.
    total_bytes = (M * K + K * N + M * N) * BYTES_PER_ELEM
    device_time_s = avg_ms / 1e3
    avg_gbs = total_bytes / device_time_s / 1e9

    return {
        "Avg_ms": avg_ms,
        "Avg_TFLOPs": avg_tflops,
        "Avg_GBs": avg_gbs,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def load_test_cases(gpu_csv: Path, include_reg4: bool) -> list[dict]:
    cases = []
    with open(gpu_csv, newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("Registry"):
                continue
            reg = int(row["Registry"])
            if reg in SKIP_REGISTRIES and not include_reg4:
                continue
            cases.append({
                "Registry": reg,
                "Case": row["Case"],
                "M": int(row["M"]),
                "K": int(row["K"]),
                "N": int(row["N"]),
            })
    return cases


def main():
    parser = argparse.ArgumentParser(
        description="Run TT GEMM sweep on the GPU baseline test cases.")
    parser.add_argument("--gpu-csv", type=Path, default=DEFAULT_GPU_CSV,
                        help="Path to gemm_gpu.csv (input test cases)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="Path to results CSV")
    parser.add_argument("--include-reg4", action="store_true",
                        help="Also run registry 4 (M=N=32768) cases (may OOM)")
    parser.add_argument("--no-trace", action="store_true",
                        help="Disable trace mode (trace is on by default)")
    args = parser.parse_args()

    if not args.gpu_csv.exists():
        sys.exit(f"GPU CSV not found: {args.gpu_csv}")
    if not RUNNER.exists():
        sys.exit(f"Runner not found: {RUNNER} (run from repo root)")
    if not PROFILER_READER.exists():
        sys.exit(f"Profiler reader not found: {PROFILER_READER} (run from repo root)")

    cases = load_test_cases(args.gpu_csv, include_reg4=args.include_reg4)
    if not cases:
        sys.exit("No test cases to run.")
    print(f"Loaded {len(cases)} test cases from {args.gpu_csv}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["Registry", "Case", "M", "K", "N",
              "Avg_ms", "Avg_TFLOPs", "Max_TFLOPs", "Avg_GBs", "Max_GBs"]
    trace = not args.no_trace

    failed = []
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()

        for i, tc in enumerate(cases, 1):
            print(f"\n[{i}/{len(cases)}] {tc['Case']}  "
                  f"(M={tc['M']}, K={tc['K']}, N={tc['N']})")
            metrics = run_one(tc["M"], tc["K"], tc["N"], trace=trace)
            row = {
                "Registry": tc["Registry"],
                "Case": tc["Case"],
                "M": tc["M"], "K": tc["K"], "N": tc["N"],
                "Avg_ms": "", "Avg_TFLOPs": "",
                "Max_TFLOPs": "", "Avg_GBs": "", "Max_GBs": "",
            }
            if metrics is None:
                failed.append(tc["Case"])
            else:
                row["Avg_ms"] = f"{metrics['Avg_ms']:.3f}"
                row["Avg_TFLOPs"] = f"{metrics['Avg_TFLOPs']:.3f}"
                row["Avg_GBs"] = f"{metrics['Avg_GBs']:.3f}"
            writer.writerow(row)
            fh.flush()  # progress is recoverable if the sweep is interrupted

    print(f"\nResults saved to {args.out}")
    if failed:
        print(f"WARNING: {len(failed)} case(s) failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
