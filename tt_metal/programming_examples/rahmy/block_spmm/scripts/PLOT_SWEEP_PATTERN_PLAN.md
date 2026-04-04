# Plot Plan: DDA vs GPU Throughput by Sparsity Pattern

## Figure layout

4 axes (subplots), one per block size: **32x32**, **64x64**, **128x128**, **256x256**.

Within each axis:
- **X-axis**: density points (varies per block size)
- At each density tick: **4 grouped bars** (Row, Col, Multi-diag, Random)
- Each grouped bar is a **pair of 2**: DDA TFLOP/s (from `profiles_april4`) and GPU TFLOP/s (from `gpu-normalized/`)
- **Y-axis**: TFLOP/s

## Data sources

### DDA (Tenstorrent N150)
- Location: `/home/user/tt-metal/profiles_april4/csvs/{RegistryName}/bsr_spmm_multicore_snf_in0_dda_in1/`
- Metric: `Device TFLOP/s` parsed from `*_sparse.log` files

### GPU baseline
- Location: `/home/user/tt-metal/tt_metal/programming_examples/rahmy/gpu-normalized/sweep_pattern_{block}.csv`
- Metric: `Avg_TFLOPs` column
- Pattern identification: filename contains `_row_`, `_col_`, `_multi_diag_`, or none (= random)

## Axis 1: R=C=32 (Ultra-sparse)

| Density (PPM) | Registry (DDA) | Registry (GPU) |
|---------------|----------------|----------------|
| 30            | 17 (PatternUltra32_30) | 17 |
| 100           | 18 (PatternUltra32_100) | 18 |
| 300           | 19 (PatternUltra32_300) | 19 |
| 1000          | 20 (PatternUltra32_1000) | 20 |
| 3000          | 21 (PatternUltra32_3000) | 21 |
| 10000         | 22 (PatternUltra32_10000) | 22 |

6 density ticks × 4 patterns × 2 bars = 48 bars

## Axis 2: R=C=64 (Ultra-sparse)

| Density (PPM) | Registry (DDA) | Registry (GPU) |
|---------------|----------------|----------------|
| 60            | 23 (PatternUltra64_60) | 23 |
| 200           | 24 (PatternUltra64_200) | 24 |
| 600           | 25 (PatternUltra64_600) | 25 |
| 2000          | 26 (PatternUltra64_2000) | 26 |
| 6000          | 27 (PatternUltra64_6000) | 27 |
| 10000         | 28 (PatternUltra64_10000) | 28 |

6 density ticks × 4 patterns × 2 bars = 48 bars

## Axis 3: R=C=128

| Density (%) | Registry (DDA) | Registry (GPU) |
|-------------|----------------|----------------|
| 5           | 13 (PatternD5_128) | 13 |
| 10          | 14 (PatternD10_128) | 14 |
| 25          | 15 (PatternD25_128) | 15 |
| 50          | 16 (PatternD50_128) | 16 |

4 density ticks × 4 patterns × 2 bars = 32 bars

## Axis 4: R=C=256

| Density (%) | Registry (DDA) | Registry (GPU) |
|-------------|----------------|----------------|
| 5           | 2 (PatternD5) | 2 |
| 10          | 3 (PatternD10) | 3 |
| 25          | 4 (PatternD25) | 4 |
| 50          | 5 (PatternD50) | 5 |

4 density ticks × 4 patterns × 2 bars = 32 bars

## Pattern identification

In both DDA sparse logs and GPU CSVs, the pattern is encoded in the filename:
- `_row_` → Row
- `_col_` → Col
- `_multi_diag_` → Multi-diag
- No pattern prefix (just `parametric_M...`) → Random

## Bar styling
- DDA bars: solid color
- GPU bars: hatched or lighter shade of same color
- 4 pattern colors: one per pattern (Row, Col, Multi-diag, Random)
- Annotate bar values for readability

## Output
- File: `scripts/figures/sweep_pattern_dda_vs_gpu.png`
- Script: `scripts/plot_sweep_pattern.py`
