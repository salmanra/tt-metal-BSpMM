# Final Block SpMM Profiling Plan

This document describes the final set of profiling experiments for the block SpMM paper.
For infrastructure details (ablation flags, host code architecture, bash script usage),
see `PROFILING_PLAN.md`.

---

## Algorithms

We use 3 of the 6 implemented algorithms. The other 3 (`load_balanced`, `reuse_iteration`,
`naive_new_DM`) are retired from the final profiling.

| Code name (HostCodeRegistryProfiling index) | Paper name              | Description |
|--------------------------------------------|-------------------------|-------------|
| `bsr_spmm_multicore_load_balanced_new_DM` [4] | **Naive**               | Both in0 and in1 read directly from DRAM; load-balanced row assignment |
| `bsr_spmm_multicore_snf` [0]                  | **SnF in0 naive in1**   | Store-and-forward for sparse A across core columns; naive DRAM reads for dense B |
| `bsr_spmm_multicore_snfin0_cdain1` [5]         | **SnF in0 CDA in1**    | Store-and-forward for sparse A; chain-of-direct-addressing for dense B across core rows |

---

## Experiment 1: Microbenchmark (Ablation)

**Goal**: For each algorithm, isolate the cost of each operation (A read, B read,
compute, write) by measuring runtime savings when that operation is skipped.
Compare across block sizes to show how operation costs scale with tile count per block.

### Parameters

| Parameter | Values |
|-----------|--------|
| Algorithms | Naive [4], SnF [0], CDA [5] |
| M, N, K | 8192 each |
| Sparsity pattern | random |
| Block size R=C | 32, 64, 128, 256 |
| Density | 25%, 5% |
| Variants | full, no_a_read, no_b_read, no_compute, no_write |

### Run matrix

3 algos x 4 block sizes x 2 densities x 5 variants = **120 runs**

### Host code indices

For each algorithm index `i` in {0, 4, 5}:

| Variant | Registry index |
|---------|---------------|
| full | `i` |
| no_a_read | `i + 6` |
| no_b_read | `i + 12` |
| no_compute | `i + 18` |
| no_write | `i + 24` |

### Profile cases needed

8 parametric random cases (4 block sizes x 2 densities):

```
profile_case_parametric_random<8192, 8192, 8192,  32,  32, 25>
profile_case_parametric_random<8192, 8192, 8192,  64,  64, 25>
profile_case_parametric_random<8192, 8192, 8192, 128, 128, 25>
profile_case_parametric_random<8192, 8192, 8192, 256, 256, 25>   # already in registry 7
profile_case_parametric_random<8192, 8192, 8192,  32,  32,  5>
profile_case_parametric_random<8192, 8192, 8192,  64,  64,  5>
profile_case_parametric_random<8192, 8192, 8192, 128, 128,  5>
profile_case_parametric_random<8192, 8192, 8192, 256, 256,  5>
```

**Infrastructure needed**: Registry 7 (`ProfileSweepBlockSizeRegistry`) covers the d=25%
cases. Need a new registry for d=5% block size sweep.

### Expected output

For each (algorithm, block_size, density): a stacked/grouped bar chart showing the
runtime of the full algorithm decomposed into A-read, B-read, compute, and write
components (each measured as `full_time - no_X_time`).

---

## Experiment 2: Pure Throughput (Algorithm x Pattern x Density)

**Goal**: Demonstrate that CDA wins across most sparsity patterns and densities.
Identify and explain the cases where it doesn't (low-sparsity random, multi_diag).

### Parameters

| Parameter | Values |
|-----------|--------|
| Algorithms | Naive [4], SnF [0], CDA [5] |
| M, N, K | 8192 each |
| Block size R=C | 256 |
| Sparsity pattern | row, col, multi_diag, random |
| Density | 5%, 10%, 25%, 50% |

### Run matrix

3 algos x 4 patterns x 4 densities = **48 runs**

### Profile case registries (already exist)

| Registry | Density | Patterns |
|----------|---------|----------|
| 8 (`ProfileSweepSparsityPatternRegistry`) | 25% | row, col, multi_diag, random |
| 9 (`ProfileSweepSparsityPatternRegistryD10`) | 10% | row, col, multi_diag, random |
| 10 (`ProfileSweepSparsityPatternRegistryD5`) | 5% | row, col, multi_diag, random |
| 11 (`ProfileSweepSparsityPatternRegistryD50`) | 50% | row, col, multi_diag, random |

All registries use M=N=K=8192, R=C=256 -- perfect match.

### Expected output

Heatmap or grouped bar chart:
- Rows = sparsity patterns (row, col, multi_diag, random)
- Columns = densities (5%, 10%, 25%, 50%)
- Within each cell: 3 bars (Naive, SnF, CDA) showing throughput

**Key narrative**: CDA wins in most cells. It loses in:
- **Low-sparsity random**: CDA's chain synchronization overhead dominates when few
  blocks need to be amortized across the chain.
- **Multi-diagonal**: Scattered column access pattern defeats CDA's column-chain
  locality assumption (the chain forwards B columns sequentially, but multi_diag
  touches columns non-contiguously).

---

## Experiment 3: Scaling Sweeps (CDA only)

**Goal**: Characterize how the CDA algorithm's throughput scales with each parameter
independently.

**Fixed**: Algorithm = SnF in0 CDA in1 (index 5), sparsity pattern = random

### 3a: Sweep N (output width)

| Parameter | Values |
|-----------|--------|
| M, K | 8192 |
| R=C | 256 |
| Density | 10% |
| N | 512, 1024, 2048, 4096, 8192 |

**5 runs**. Needs new registry (existing registry 4 uses d=25%).

### 3b: Sweep K (reduction dimension)

| Parameter | Values |
|-----------|--------|
| M, N | 8192 |
| R=C | 256 |
| Density | 10% |
| K | 512, 1024, 2048, 4096, 8192 |

**5 runs**. Needs new registry (existing registry 6 uses d=25%).

### 3c: Sweep Block Size

| Parameter | Values |
|-----------|--------|
| M, N, K | 8192 |
| Density | 10% |
| R=C | 32, 64, 128, 256 |

**4 runs**. Needs new registry (existing registry 7 uses d=25%).

### 3d: Sweep Density

| Parameter | Values |
|-----------|--------|
| M, N, K | 8192 |
| R=C | 256 |
| Density | 5%, 10%, 25%, 50%, 75% |

**5 runs**. Registry 5 (`ProfileSweepDensityRegistry`) already matches.

### Total scaling runs: 5 + 5 + 4 + 5 = **19 runs**

### Expected output

4 line plots (one per sweep axis):
- X-axis = swept parameter
- Y-axis = throughput (TFLOP/s) or runtime (us)
- Single line (CDA algorithm)

---

## Summary

| Experiment | Description | Runs |
|-----------|-------------|------|
| 1. Microbenchmark | 3 algos x 4 block sizes x 2 densities x 5 variants | 120 |
| 2. Pure Throughput | 3 algos x 4 patterns x 4 densities | 48 |
| 3. Scaling Sweeps | CDA only, 4 sweep axes | 19 |
| **Total** | | **187** |

---

## Infrastructure Changes Required

### New profile case registries

| New registry | Based on | Change |
|-------------|----------|--------|
| Microbench block size sweep d=5% | Registry 7 | Change density from 25% to 5% |
| Sweep N d=10% | Registry 4 | Change density from 25% to 10% |
| Sweep K d=10% | Registry 6 | Change density from 25% to 10% |
| Sweep block size d=10% | Registry 7 | Change density from 25% to 10% |

### Host code filtering

Either create a new 3-algorithm host code registry (indices 0, 4, 5 + their ablation
variants) or use `--host-code` flags in the bash script to select only these three.

### New bash script phases

Add experiment-specific phases to `run_profiling_plan.sh`:
- `microbench` — runs experiment 1
- `throughput` — runs experiment 2
- `scaling` — runs experiment 3

### Plotting

Update plotting scripts to use paper names:
- `load_balanced_new_DM` -> "Naive"
- `snf` -> "SnF in0 naive in1"
- `snfin0_cdain1` -> "SnF in0 CDA in1"
