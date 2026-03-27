#!/bin/bash

TT_METAL_DEVICE_PROFILER=1 python GEMM_profiling/run_minimal_matmul.py --ttnn-matmul --M 4096 --K 4096 --N 4096 --fp16-acc --trace
python GEMM_profiling/read_device_profiler.py --M 4096 --K 4096 --N 4096


TT_METAL_DEVICE_PROFILER=1 python GEMM_profiling/run_minimal_matmul.py --ttnn-matmul --M 2048 --K 2048 --N 2048 --fp16-acc --trace
python GEMM_profiling/read_device_profiler.py --M 2048 --K 2048 --N 2048

TT_METAL_DEVICE_PROFILER=1 python GEMM_profiling/run_minimal_matmul.py --ttnn-matmul --M 8192 --K 8192 --N 8192 --fp16-acc --trace
python GEMM_profiling/read_device_profiler.py --M 8192 --K 8192 --N 8192