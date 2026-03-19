#!/bin/bash

OUTPUT_CSV="${1:-GEMM_profiling/sweep_results.csv}"
echo "N,Avg_TFLOPs,Max_TFLOPs,Avg_GBs,Max_GBs" > "$OUTPUT_CSV"

for N in 1024 2048 4096 8192 16384; do
    echo "Running M=K=N=$N ..."
    OUTPUT=$(python GEMM_profiling/run_minimal_matmul.py --trace --M $N --K $N --N $N)
    echo "$OUTPUT" | grep -E "TFLOP/s|GB/s"

    AVG_TFLOPS=$(echo "$OUTPUT" | grep "Avg TFLOP/s" | awk '{print $NF}')
    MAX_TFLOPS=$(echo "$OUTPUT" | grep "Max TFLOP/s" | awk '{print $NF}')
    AVG_GBS=$(echo "$OUTPUT" | grep "Avg GB/s" | awk '{print $NF}')
    MAX_GBS=$(echo "$OUTPUT" | grep "Max GB/s" | awk '{print $NF}')

    echo "$N,$AVG_TFLOPS,$MAX_TFLOPS,$AVG_GBS,$MAX_GBS" >> "$OUTPUT_CSV"
done

echo ""
echo "Results saved to $OUTPUT_CSV"
