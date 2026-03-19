# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Run minimal_matmul experimental operation and verify correctness.

Usage:
    python GEMM_profiling/run_minimal_matmul.py
    python GEMM_profiling/run_minimal_matmul.py --M 2048 --K 2048 --N 2048
    python GEMM_profiling/run_minimal_matmul.py --M 4096 --K 4096 --N 4096 --block 8 --subblock 2 --fidelity HiFi4
    python GEMM_profiling/run_minimal_matmul.py --bias --activation gelu
    python GEMM_profiling/run_minimal_matmul.py --trace --iterations 10  # device-only timing via trace replay
"""

import argparse
import time

import torch
import ttnn

# TODO: do we want to test with math fidelity less than perfect?
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
    fp32_acc=True,
    dtype=ttnn.bfloat16,
    core_grid=None,
    num_iterations=10,
    trace=False,
):
    print(f"\n{'='*60}")
    print(f"  Minimal MatMul: ({M}, {K}) x ({K}, {N})")
    print(f"  Block: {M_block_size}x{K_block_size}x{N_block_size}, Subblock: {subblock_h}x{subblock_w}")
    print(f"  Fidelity: {math_fidelity}, FP32 acc: {fp32_acc}, dtype: {dtype}")
    print(f"  Bias: {use_bias}, Activation: {activation}")
    print(f"  Iterations: {num_iterations}")
    print(f"  Trace mode: {trace}")
    print(f"{'='*60}")

    # Create random input tensors in float32 for golden reference
    torch_input = torch.randn((M, K), dtype=torch.float32)
    torch_weight = torch.randn((K, N), dtype=torch.float32)
    torch_bias = torch.randn((1, N), dtype=torch.float32) if use_bias else None

    # Compute golden reference
    with torch.no_grad():
        torch_output = torch_input @ torch_weight
        if use_bias:
            torch_output = torch_output + torch_bias
        if activation == "gelu":
            torch_output = torch.nn.functional.gelu(torch_output)

    # Convert to TT tensors on device
    tt_input = ttnn.from_torch(torch_input, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
    tt_weight = ttnn.from_torch(torch_weight, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
    tt_bias = None
    if use_bias:
        tt_bias = ttnn.from_torch(torch_bias, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)

    # Fused activation
    activation_fn = None
    if activation == "gelu":
        activation_fn = (ttnn.UnaryOpType.GELU, False)

    # Compute kernel config
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=True,
    )

    # Matmul config
    core_grid = core_grid or device.compute_with_storage_grid_size()
    matmul_config = ttnn.MinimalMatmulConfig(
        M_block_size=M_block_size,
        K_block_size=K_block_size,
        N_block_size=N_block_size,
        subblock_h=subblock_h,
        subblock_w=subblock_w,
        compute_with_storage_grid_size=core_grid,
    )

    print(f"  Core grid: {core_grid}")
    print(f"  Running...")

    def _run_op():
        return ttnn.experimental.minimal_matmul(
            tt_input,
            tt_weight,
            bias_tensor=tt_bias,
            fused_activation=activation_fn,
            compute_kernel_config=compute_config,
            config=matmul_config,
        )

    durations = []

    if trace:
        # Warmup (compile the program binary)
        tt_output = _run_op()
        ttnn.synchronize_device(device)

        # Capture trace
        trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        tt_output = _run_op()
        ttnn.end_trace_capture(device, trace_id, cq_id=0)

        # Timed replay — measures pure device execution
        for i in range(num_iterations):
            start = time.perf_counter()
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            durations.append(time.perf_counter() - start)

        ttnn.release_trace(device, trace_id)
    else:
        # First call compiles, subsequent calls use cache
        start = time.perf_counter()
        tt_output = _run_op()
        ttnn.synchronize_device(device)
        durations.append(time.perf_counter() - start)

        for i in range(num_iterations):
            start = time.perf_counter()
            tt_output = _run_op()
            ttnn.synchronize_device(device)
            durations.append(time.perf_counter() - start)

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
    # Data movement: read A (M*K) + read B (K*N) + write C (M*N), bfloat16 = 2 bytes
    dtype_bytes = {ttnn.bfloat16: 2, ttnn.bfloat8_b: 1, ttnn.bfloat4_b: 0.5}
    elem_size = dtype_bytes.get(dtype, 2)
    total_bytes = (M * K + K * N + M * N) * elem_size
    print(f"\n  Results:")
    print(f"    PCC:            {pcc:.7f}")
    print(f"    Relative RMSE:  {rel_rmse:.6f}")
    if trace:
        avg_dur = sum(durations) / len(durations)
        min_dur = min(durations)
        print(f"    Trace mode:     {num_iterations} iterations")
        print(f"    Avg device:     {avg_dur*1000:.2f} ms")
        print(f"    Min device:     {min_dur*1000:.2f} ms")
        print(f"    Avg TFLOP/s:    {total_flops / avg_dur / 1e12:.2f}")
        print(f"    Max TFLOP/s:   {total_flops / min_dur / 1e12:.2f}")
        print(f"    Avg GB/s:       {total_bytes / avg_dur / 1e9:.2f}")
        print(f"    Max GB/s:      {total_bytes / min_dur / 1e9:.2f}")
    elif num_iterations == 1:
        print(f"    Host duration:  {durations[0]*1000:.2f} ms")
        print(f"    TFLOP/s:        {total_flops / durations[0] / 1e12:.2f}")
        print(f"    GB/s:           {total_bytes / durations[0] / 1e9:.2f}")
    else:
        print(f"    First iter:     {durations[0]*1000:.2f} ms (includes compile)")
        avg_cached = sum(durations[1:]) / (num_iterations - 1)
        print(f"    Avg cached:     {avg_cached*1000:.2f} ms ({num_iterations} iters)")
        print(f"    Avg TFLOP/s:    {total_flops / avg_cached / 1e12:.2f}")
        print(f"    Avg GB/s:       {total_bytes / avg_cached / 1e9:.2f}")

    return {"pcc": pcc, "relative_rmse": rel_rmse, "durations": durations}


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
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations to run")
    parser.add_argument("--trace", action="store_true", help="Use trace capture/replay for device-only timing")
    parser.add_argument("--trace-region-size", type=int, default=400000, help="Trace region size in bytes (only with --trace)")
    parser.add_argument("--device-id", type=int, default=0, help="Device ID")
    args = parser.parse_args()

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

    # Open device
    device = ttnn.open_device(
        device_id=args.device_id,
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
