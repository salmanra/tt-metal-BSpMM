#!/bin/bash

python GEMM_profiling/run_minimal_matmul.py --trace --M 1024 --K 1024 --N 1024 | grep TFLOP/s
python GEMM_profiling/run_minimal_matmul.py --trace --M 2048 --K 2048 --N 2048 | grep TFLOP/s
python GEMM_profiling/run_minimal_matmul.py --trace --M 4096 --K 4096 --N 4096 | grep TFLOP/s
python GEMM_profiling/run_minimal_matmul.py --trace --M 8192 --K 8192 --N 8192 | grep TFLOP/s
python GEMM_profiling/run_minimal_matmul.py --trace --M 16384 --K 16384 --N 16384 | grep TFLOP/s