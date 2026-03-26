# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Run minimal_matmul experimental operation and verify correctness.

Usage:
    # Default: 4096x4096x4096 with minimal_matmul
    python GEMM_profiling/run_minimal_matmul.py

    # Custom matrix dimensions
    python GEMM_profiling/run_minimal_matmul.py --M 2048 --K 2048 --N 2048

    # Block sizes (in tiles): uniform via --block, or per-dimension overrides
    python GEMM_profiling/run_minimal_matmul.py --block 4
    python GEMM_profiling/run_minimal_matmul.py --M-block 8 --K-block 4 --N-block 8

    # Subblock sizes: uniform via --subblock, or per-dimension overrides
    python GEMM_profiling/run_minimal_matmul.py --subblock 4
    python GEMM_profiling/run_minimal_matmul.py --subblock-h 2 --subblock-w 4

    # Math fidelity and data type
    python GEMM_profiling/run_minimal_matmul.py --fidelity LoFi --dtype bfloat8_b
    python GEMM_profiling/run_minimal_matmul.py --fidelity HiFi2 --dtype bfloat16 --fp16-acc

    # Bias and fused activation
    python GEMM_profiling/run_minimal_matmul.py --bias --activation gelu

    # Trace mode for device-only timing
    python GEMM_profiling/run_minimal_matmul.py --trace --iterations 100
    python GEMM_profiling/run_minimal_matmul.py --trace --trace-region-size 8000000

    # Use ttnn.matmul with auto-configured block sizes (from benchmark lookup table)
    python GEMM_profiling/run_minimal_matmul.py --ttnn-matmul
    python GEMM_profiling/run_minimal_matmul.py --ttnn-matmul --M 8192 --K 8192 --N 8192
    python GEMM_profiling/run_minimal_matmul.py --ttnn-matmul --fidelity HiFi2 --dtype bfloat8_b

    # ttnn.matmul with explicit block sizes (overrides auto-config)
    python GEMM_profiling/run_minimal_matmul.py --ttnn-matmul --M-block 2 --K-block 4 --N-block 2

    # Device selection and iteration count
    python GEMM_profiling/run_minimal_matmul.py --device-id 1 --iterations 50
"""

import argparse
import os
import sys
import time

import torch
import ttnn
from models.utility_functions import profiler, is_wormhole_b0, is_blackhole
from tt_metal.tools.profiler.common import PROFILER_LOGS_DIR, PROFILER_DEVICE_SIDE_LOG, rm

profiler_log_path = PROFILER_LOGS_DIR / PROFILER_DEVICE_SIDE_LOG


def get_device_frequency():
    if is_wormhole_b0():
        return 1000
    elif is_blackhole():
        return 1350
    else:
        return None


def get_profiler_build_enabled():
    return os.getenv("TT_METAL_DEVICE_PROFILER") is not None


SUBBLOCK_HW_CHOICES = [
    (4, 2),
    (2, 4),
    (8, 1),
    (1, 8),  # subblock_hw = 8
    (7, 1),
    (1, 7),  # subblock_hw = 7
    (3, 2),
    (2, 3),
    (6, 1),
    (1, 6),  # subblock_hw = 6
    (5, 1),
    (1, 5),  # subblock_hw = 5
    (2, 2),
    (4, 1),
    (1, 4),  # subblock_hw = 4
    (3, 1),
    (1, 3),  # subblock_hw = 3
    (2, 1),
    (1, 2),  # subblock_hw = 2
    (1, 1),  # subblock_hw = 1
]


def get_subblock_sizes(m_tiles_per_core, n_tiles_per_core, fp32_dest_acc_en=False):
    for out_subblock_h, out_subblock_w in SUBBLOCK_HW_CHOICES:
        if fp32_dest_acc_en:
            if (out_subblock_h * out_subblock_w) > 4:
                continue
        if m_tiles_per_core % out_subblock_h == 0 and n_tiles_per_core % out_subblock_w == 0:
            return (out_subblock_h, out_subblock_w)
    return (1, 1)


# Optimal block division factors from test_benchmark.py for square matmuls (bfloat16).
# Keyed by per-core dim in elements. Values: (in0_block_w_div, num_out_blocks_h, num_out_blocks_w)
BENCHMARK_SQUARE_CONFIGS_BF16 = {
    64: (1, 1, 1),
    128: (1, 1, 1),
    256: (1, 1, 1),
    384: (4, 1, 1),
    512: (1, 2, 2),
    1024: (2, 4, 4),
    2048: (4, 8, 8),
}


def get_ttnn_matmul_block_sizes(M, K, N, grid_size):
    """Auto-compute (in0_block_w, out_block_h, out_block_w) for MatmulMultiCoreReuseMultiCastProgramConfig.

    For square per-core shapes with a known benchmark config, uses optimal division factors.
    Otherwise falls back to 1-tile blocks (always fits in L1).
    """
    per_core_m = M // grid_size[1]
    per_core_k = K // grid_size[0]
    per_core_n = N // grid_size[0]
    per_core_m_tiles = per_core_m // 32
    per_core_k_tiles = per_core_k // 32
    per_core_n_tiles = per_core_n // 32

    if per_core_m == per_core_k == per_core_n and per_core_m in BENCHMARK_SQUARE_CONFIGS_BF16:
        in0_block_w_div, num_out_blocks_h, num_out_blocks_w = BENCHMARK_SQUARE_CONFIGS_BF16[per_core_m]
        in0_block_w = per_core_k_tiles // in0_block_w_div
        out_block_h = per_core_m_tiles // num_out_blocks_h
        out_block_w = per_core_n_tiles // num_out_blocks_w
        print(f"  Auto-config: per_core={per_core_m}, using benchmark config "
              f"(in0_block_w_div={in0_block_w_div}, num_out_blocks_h={num_out_blocks_h}, num_out_blocks_w={num_out_blocks_w})")
    else:
        in0_block_w = 1
        out_block_h = 1
        out_block_w = 1
        print(f"  Auto-config: per_core=({per_core_m},{per_core_k},{per_core_n}), no benchmark match — using fallback (1,1,1)")

    return in0_block_w, out_block_h, out_block_w


def run_minimal_matmul(
    device,
    M,
    K,
    N,
    M_block_size=8,
    K_block_size=8,
    N_block_size=8,
    subblock_h=2,
    subblock_w=2,
    use_bias=False,
    activation=None,
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_acc=False,
    dtype=ttnn.bfloat16,
    core_grid=None,
    num_iterations=10,
    trace=False,
    use_ttnn_matmul=False,
    auto_block_sizes=False,
):
    print(f"\n{'='*60}")
    print(f"  Minimal MatMul: ({M}, {K}) x ({K}, {N})")
    if not (use_ttnn_matmul and auto_block_sizes):
        print(f"  Block: {M_block_size}x{K_block_size}x{N_block_size}, Subblock: {subblock_h}x{subblock_w}")
    print(f"  Fidelity: {math_fidelity}, FP32 acc: {fp32_acc}, dtype: {dtype}")
    print(f"  Bias: {use_bias}, Activation: {activation}")
    print(f"  Iterations: {num_iterations}")
    print(f"  Trace mode: {trace}")
    print(f"  Op: {'ttnn.matmul' if use_ttnn_matmul else 'minimal_matmul'}")
    print(f"{'='*60}")

    tile_h = 32
    tile_w = 32

    # Create random input tensors in float32 for golden reference
    torch_input = torch.randn((1, 1, M, K), dtype=torch.float32)
    torch_weight = torch.randn((1, 1, K, N), dtype=torch.float32)
    torch_bias = torch.randn((1, N), dtype=torch.float32) if use_bias else None

    # Compute golden reference
    with torch.no_grad():
        torch_output = torch_input @ torch_weight
        if use_bias:
            torch_output = torch_output + torch_bias
        if activation == "gelu":
            torch_output = torch.nn.functional.gelu(torch_output)

    # Resolve core grid
    core_grid = core_grid or device.compute_with_storage_grid_size()
    grid_size = (core_grid.x, core_grid.y)

    # Convert to TT tensors on device
    tt_input = ttnn.from_torch(
        torch_input,
        tile=ttnn.Tile((tile_h, 32)),
        dtype=dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_weight = ttnn.from_torch(
        torch_weight,
        tile=ttnn.Tile((32, tile_w)),
        dtype=dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_bias = None
    if use_bias:
        tt_bias = ttnn.from_torch(torch_bias, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)

    # Fused activation
    activation_fn = None
    if activation == "gelu":
        activation_fn = (ttnn.UnaryOpType.GELU, False)

    # Compute kernel config (match benchmark: math_approx_mode=True, throttle=NO_THROTTLE)
    compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=math_fidelity,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
        throttle_level=ttnn.ThrottleLevel.NO_THROTTLE,
    )

    # Matmul config (for minimal_matmul path)
    matmul_config = ttnn.MinimalMatmulConfig(
        M_block_size=M_block_size,
        K_block_size=K_block_size,
        N_block_size=N_block_size,
        subblock_h=subblock_h,
        subblock_w=subblock_w,
        compute_with_storage_grid_size=core_grid,
    )

    calc_device_utilization = get_profiler_build_enabled()

    print(f"  Core grid: {core_grid}")
    print(f"  Running...")
    profiler.clear()

    if use_ttnn_matmul:
        # Build MatmulMultiCoreReuseMultiCastProgramConfig matching the benchmark
        per_core_M = M // grid_size[1] // tile_h
        per_core_N = N // grid_size[0] // tile_w
        if auto_block_sizes:
            in0_block_w, out_block_h, out_block_w = get_ttnn_matmul_block_sizes(M, K, N, grid_size)
        else:
            in0_block_w = K_block_size
            out_block_h = M_block_size
            out_block_w = N_block_size
        out_subblock_h, out_subblock_w = get_subblock_sizes(out_block_h, out_block_w)

        program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=in0_block_w,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            out_block_h=out_block_h,
            out_block_w=out_block_w,
            per_core_M=per_core_M,
            per_core_N=per_core_N,
            transpose_mcast=False,
            fused_activation=None,
        )

        output_tile = ttnn.Tile([tile_h, tile_w])

        print(f"  in0_block_w: {in0_block_w}, per_core_M: {per_core_M}, per_core_N: {per_core_N}")
        print(f"  out_block_h: {out_block_h}, out_block_w: {out_block_w}")
        print(f"  out_subblock_h: {out_subblock_h}, out_subblock_w: {out_subblock_w}")

        def _run_op():
            return ttnn.matmul(
                tt_input,
                tt_weight,
                program_config=program_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=dtype,
                compute_kernel_config=compute_config,
                output_tile=output_tile,
            )

    else:

        def _run_op():
            return ttnn.experimental.minimal_matmul(
                tt_input,
                tt_weight,
                bias_tensor=tt_bias,
                fused_activation=activation_fn,
                compute_kernel_config=compute_config,
                config=matmul_config,
            )

    if trace:
        # Warmup: first call compiles, rest warm up caches (match benchmark's 1+5)
        for i in range(6):
            tt_output = _run_op()

        if calc_device_utilization:
            ttnn.ReadDeviceProfiler(device)
            rm(profiler_log_path)

        ttnn.synchronize_device(device)

        # Capture trace with all iterations inside
        trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        for _ in range(num_iterations):
            tt_output = _run_op()
        ttnn.end_trace_capture(device, trace_id, cq_id=0)

        # Timed replay — single replay of all iterations
        profiler.start(f"run")
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        profiler.end(f"run")
        ttnn.release_trace(device, trace_id)
        total_duration = profiler.get(f"run")
    else:
        # Warmup: first call compiles, rest warm up caches
        for i in range(5):
            tt_output = _run_op()

        if calc_device_utilization:
            ttnn.ReadDeviceProfiler(device)
            rm(profiler_log_path)

        ttnn.synchronize_device(device)

        # Timed run — batch all iterations under one timer
        profiler.start(f"run")
        for i in range(num_iterations):
            tt_output = _run_op()
        ttnn.synchronize_device(device)
        profiler.end(f"run")
        total_duration = profiler.get(f"run")

    device_freq = get_device_frequency()
    avg_dur = total_duration / num_iterations

    # Convert output back to torch for validation
    tt_output_torch = ttnn.to_torch(tt_output).to(torch.float32)

    # Compute PCC (Pearson Correlation Coefficient)
    torch_flat = torch_output.flatten()
    tt_flat = tt_output_torch.flatten()
    pcc = torch.corrcoef(torch.stack([torch_flat, tt_flat]))[0, 1].item()

    # Compute relative RMSE
    rmse = torch.nn.functional.mse_loss(torch_output, tt_output_torch).sqrt().item()
    rel_rmse = rmse / torch_output.std().item()

    # Report results
    total_flops = 2 * M * N * K
    dtype_bytes = {ttnn.bfloat16: 2, ttnn.bfloat8_b: 1, ttnn.bfloat4_b: 0.5}
    elem_size = dtype_bytes.get(dtype, 2)
    total_bytes = (M * K + K * N + M * N) * elem_size

    # Utilization calculation (matching benchmark)
    LoFi_cycle = 16
    cycle_per_tile = {
        ttnn.MathFidelity.LoFi: LoFi_cycle,
        ttnn.MathFidelity.HiFi2: LoFi_cycle * 2,
        ttnn.MathFidelity.HiFi3: LoFi_cycle * 3,
        ttnn.MathFidelity.HiFi4: LoFi_cycle * 4,
    }[math_fidelity]
    num_cores = grid_size[0] * grid_size[1]
    ideal_cycles = M * K * N / tile_h / tile_w / 32 * cycle_per_tile / num_cores
    if device_freq is not None:
        inference_cycles = avg_dur * device_freq * 1e6
        host_utilization = ideal_cycles / inference_cycles
    else:
        host_utilization = None

    print(f"\n  Results:")
    print(f"    PCC:            {pcc:.7f}")
    print(f"    Relative RMSE:  {rel_rmse:.6f}")
    mode_label = "Trace mode" if trace else "Iterations"
    warmup_note = "" if trace else " (after 5 warmup)"
    print(f"    {mode_label}:     {num_iterations}{warmup_note}")
    print(f"    Avg duration:   {avg_dur*1000:.2f} ms")
    print(f"    Avg TFLOP/s:    {total_flops / avg_dur / 1e12:.2f}")
    print(f"    Avg GB/s:       {total_bytes / avg_dur / 1e9:.2f}")
    if host_utilization is not None:
        print(f"    Host util:      {host_utilization * 100:.2f}% (vs {grid_size[0]}x{grid_size[1]} grid, {device_freq} MHz)")

    return {"pcc": pcc, "relative_rmse": rel_rmse, "avg_duration": avg_dur}


def main():
    parser = argparse.ArgumentParser(description="Run minimal_matmul on Tenstorrent device")
    parser.add_argument("--M", type=int, default=4096, help="M dimension (rows of A)")
    parser.add_argument("--K", type=int, default=4096, help="K dimension (inner)")
    parser.add_argument("--N", type=int, default=4096, help="N dimension (cols of B)")
    parser.add_argument("--block", type=int, default=8, help="Block size for M, K, N (tiles)")
    parser.add_argument("--M-block", type=int, default=None, help="M block size (overrides --block)")
    parser.add_argument("--K-block", type=int, default=None, help="K block size (overrides --block)")
    parser.add_argument("--N-block", type=int, default=None, help="N block size (overrides --block)")
    parser.add_argument("--subblock", type=int, default=2, help="Subblock size for h and w")
    parser.add_argument("--subblock-h", type=int, default=None, help="Subblock height (overrides --subblock)")
    parser.add_argument("--subblock-w", type=int, default=None, help="Subblock width (overrides --subblock)")
    parser.add_argument("--bias", action="store_true", help="Add bias")
    parser.add_argument("--activation", type=str, default=None, choices=["gelu"], help="Fused activation")
    parser.add_argument(
        "--fidelity",
        type=str,
        default="HiFi4",
        choices=["LoFi", "HiFi2", "HiFi4"],
        help="Math fidelity",
    )
    parser.add_argument("--fp16-acc", action="store_true", help="Use fp16 accumulation (default is fp32)")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "bfloat8_b", "bfloat4_b"],
        help="Data type",
    )
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations to run")
    parser.add_argument("--ttnn-matmul", action="store_true", help="Use ttnn.matmul instead of minimal_matmul")
    parser.add_argument("--trace", action="store_true", help="Use trace capture/replay for device-only timing")
    parser.add_argument(
        "--trace-region-size", type=int, default=3855488, help="Trace region size in bytes (only with --trace)"
    )
    parser.add_argument("--device-id", type=int, default=0, help="Device ID")
    args = parser.parse_args()

    # Detect if user explicitly set any block size
    user_set_block = args.M_block is not None or args.K_block is not None or args.N_block is not None
    # --block has a default of 8, so check if it was explicitly passed
    if "--block" in sys.argv:
        user_set_block = True
    auto_block_sizes = not user_set_block

    # Resolve block/subblock sizes
    M_block = args.M_block if args.M_block is not None else args.block
    K_block = args.K_block if args.K_block is not None else args.block
    N_block = args.N_block if args.N_block is not None else args.block
    subblock_h = args.subblock_h if args.subblock_h is not None else args.subblock
    subblock_w = args.subblock_w if args.subblock_w is not None else args.subblock

    # Resolve dtype
    dtype_map = {
        "bfloat16": ttnn.bfloat16,
        "bfloat8_b": ttnn.bfloat8_b,
        "bfloat4_b": ttnn.bfloat4_b,
    }
    dtype = dtype_map[args.dtype]

    # Resolve fidelity
    fidelity_map = {
        "LoFi": ttnn.MathFidelity.LoFi,
        "HiFi2": ttnn.MathFidelity.HiFi2,
        "HiFi4": ttnn.MathFidelity.HiFi4,
    }
    math_fidelity = fidelity_map[args.fidelity]

    # Open device (match benchmark: l1_small_size=24576, large trace region)
    device = ttnn.open_device(
        device_id=args.device_id,
        l1_small_size=24576,
        trace_region_size=args.trace_region_size if args.trace else 0,
    )

    try:
        result = run_minimal_matmul(
            device=device,
            M=args.M,
            K=args.K,
            N=args.N,
            M_block_size=M_block,
            K_block_size=K_block,
            N_block_size=N_block,
            subblock_h=subblock_h,
            subblock_w=subblock_w,
            use_bias=args.bias,
            activation=args.activation,
            math_fidelity=math_fidelity,
            fp32_acc=not args.fp16_acc,
            dtype=dtype,
            num_iterations=args.iterations,
            trace=args.trace,
            use_ttnn_matmul=args.ttnn_matmul,
            auto_block_sizes=auto_block_sizes,
        )

        print(f"\n{'='*60}")
        if result["pcc"] > 0.9995:
            print("  PASS - PCC > 0.9995")
        else:
            print(f"  WARN - PCC {result['pcc']:.7f} is below 0.9995")
        print(f"{'='*60}\n")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
