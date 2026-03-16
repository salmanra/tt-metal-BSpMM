# Block SpMM Profiling Plan

## Overview

This document describes the profiling infrastructure added to the `block_spmm` project
to support two goals:

1. **Ablation profiling** — isolate individual cost components by skipping them
   (sparse A reads, dense B reads, tile multiplies, DRAM writes) to get upper-bound
   and microbenchmark measurements for the paper.
2. **Parametric cost-class sweeps** — systematically vary matrix dimensions and
   sparsity to measure how performance scales with each cost metric.

---

## Profiling Goals

### Goal 1: Ablation Runs

Each algorithm is run four additional times, each skipping one cost component:

| Variant name  | Skipped component | Kernel flag            | Measures                           |
|---------------|-------------------|------------------------|------------------------------------|
| `no_a_read`   | Sparse A DRAM reads | `SKIP_IN0_DRAM_READ=1` | Upper bound without sparse traffic |
| `no_b_read`   | Dense B DRAM reads  | `SKIP_IN1_DRAM_READ=1` | Upper bound without dense traffic  |
| `no_compute`  | Tile multiply       | `SKIP_COMPUTE=1`       | Pure data movement cost            |
| `no_write`    | Output DRAM writes  | `SKIP_DRAM_WRITE=1`    | Upper bound without write traffic  |

The CB handshake protocol is always preserved: `cb_push_back`/`cb_pop_front` still
execute even when the underlying operation is skipped, so no downstream kernel deadlocks.

### Goal 1b: NoC Assignment Comparison (flip_noc)

Each algorithm has a hardcoded "optimal" NoC assignment (`transpose_NoCs`). To measure
the impact, the `use_optimal_noc` template parameter lets us instantiate each algorithm
with its non-optimal NoC and compare against the optimal baseline.

The flip_noc variants mirror the full ablation layout but with the NoC flipped:

| Registry indices | Group            | Description                                |
|------------------|------------------|--------------------------------------------|
| `[25-29]`        | `flip_noc full`  | Full algorithms with non-optimal NoC       |
| `[30-34]`        | `flip_noc no_a`  | Non-optimal NoC + skip sparse A reads      |
| `[35-39]`        | `flip_noc no_b`  | Non-optimal NoC + skip dense B reads       |
| `[40-44]`        | `flip_noc no_c`  | Non-optimal NoC + skip compute             |
| `[45-49]`        | `flip_noc no_w`  | Non-optimal NoC + skip DRAM writes         |

Optimal NoC assignments per algorithm:
- `snf`, `naive_new_DM`, `load_balanced_new_DM`: `transpose_NoCs = true`
- `reuse_iteration`, `load_balanced`: `transpose_NoCs = false`

### Goal 2: Parametric Sweeps

Four sweep registries explore one axis at a time:

| Registry index | Name                  | Swept axis | Fixed parameters             |
|----------------|-----------------------|------------|------------------------------|
| 4              | `ProfileSweepN`       | N (output width) | M=K=8192, R=C=64, density=25% |
| 5              | `ProfileSweepDensity` | Density %  | M=N=K=8192, R=C=64           |
| 6              | `ProfileSweepK`       | K (reduction) | M=N=8192, R=C=64, density=25% |
| 7              | `ProfileSweepBlockSize` | R=C block size | M=N=K=8192, density=25%   |
| 8              | `ProfileSweepSparsityPattern` | Sparsity pattern | M=N=K=8192, R=C=256, density=25% |
| 9              | `ProfileSweepSparsityPatternD10` | Sparsity pattern | M=N=K=8192, R=C=256, density=10% |
| 10             | `ProfileSweepSparsityPatternD5` | Sparsity pattern | M=N=K=8192, R=C=256, density=5% |
| 11             | `ProfileSweepSparsityPatternD50` | Sparsity pattern | M=N=K=8192, R=C=256, density=50% |

Each case is named `parametric_M{M}_N{N}_K{K}_R{R}_C{C}_d{density%}` for
unambiguous identification in Tracy output.

**Cost metric estimates** (useful for understanding which axis drives each cost):

```
sparse_reads  ∝ nnz_blocks × Rt × Ct
dense_reads   ∝ nnz_blocks × Ct × Nt
tile_mults    ∝ nnz_blocks × Rt × Ct × Nt
dram_writes   ∝ nnz_rows   × Rt × Nt
```

where `Rt = R/32`, `Ct = C/32`, `Nt = N/32` are tile counts.

---

## Codebase Changes

### Device Kernel SKIP Guards

Each device kernel file now has preprocessor ablation flags with defensive `#ifndef`
guards (default = 0, so normal runs are unaffected):

| Kernel file                 | Flags added                              |
|-----------------------------|------------------------------------------|
| `reader_snf_in0_reader.cpp` | `SKIP_IN0_DRAM_READ`, `SKIP_DRAM_WRITE`  |
| `reader_snf_in1_reader.cpp` | `SKIP_IN1_DRAM_READ`, `SKIP_DRAM_WRITE`  |
| `reader_block_iter.cpp`     | `SKIP_IN0_DRAM_READ`, `SKIP_IN1_DRAM_READ` |
| `writer_block_load_balanced.cpp` | `SKIP_DRAM_WRITE`                   |
| `writer_block_iter.cpp`     | `SKIP_DRAM_WRITE`                        |
| `reader_in0_naive.cpp`      | `SKIP_IN0_DRAM_READ`, `SKIP_DRAM_WRITE`  |
| `reader_in1_naive.cpp`      | `SKIP_IN1_DRAM_READ`, `SKIP_DRAM_WRITE`  |
| `bmm_iter.cpp`              | `SKIP_COMPUTE`                           |
| `bmm_iter_old_profiling.cpp`| `SKIP_COMPUTE`                           |

The `SKIP_COMPUTE` replacement loop preserves the CB protocol:
- Drains `cb_in0` and `cb_in1` on every block iteration.
- Pushes a dummy output tile to `cb_out` only on the last block of each output block,
  matching what the real compute kernel does.

### Host Code Architecture

Each of the 5 host code `.cpp` files was refactored as follows:

1. The original function body moved to `bsr_spmm_multicore_X_impl(...)` with an
   added `const std::map<std::string, std::string>& extra_defines = {}` parameter.
2. After `auto zone_defines = spmm_zone_config::get_zone_defines();`, the line
   `zone_defines.insert(extra_defines.begin(), extra_defines.end());` merges any
   skip flags into the defines map used by all `CreateKernel` calls.
3. The public `bsr_spmm_multicore_X(...)` function (matching `HostCodeFunctionPtr`)
   becomes a thin wrapper that calls `_impl` with `{}`.
4. Four skip wrapper functions (`_no_a_read`, `_no_b_read`, `_no_compute`,
   `_no_write`) call `_impl` with the appropriate single-entry defines map.
5. All functions (impl, public, ablation wrappers) have a third template parameter
   `bool use_optimal_noc = true`. When `false`, the `transpose_NoCs` value is
   flipped from its optimal setting, allowing profiling of the non-optimal NoC
   assignment. Explicit instantiations for `<false, true, false>` are provided.

Files modified:
- `inc/host_code/bsr_spmm_multicore_snf.cpp`
- `inc/host_code/bsr_spmm_multicore_load_balanced.cpp`
- `inc/host_code/bsr_spmm_multicore_reuse_iteration.cpp`
- `inc/host_code/bsr_spmm_multicore_naive_new_DM.cpp`
- `inc/host_code/bsr_spmm_multicore_load_balanced_new_DM.cpp`

### `host_code.hpp`

- Added `DECLARE_ABLATION_WRAPPERS(func_name)` macro to declare the 4 skip wrappers
  for each of the 5 algorithms (20 new function declarations total).
- `HostCodeRegistryProfiling` extended from 5 entries to 50:
  - `[0-4]`   Full algorithms (unchanged)
  - `[5-9]`   `no_a_read` variants
  - `[10-14]` `no_b_read` variants
  - `[15-19]` `no_compute` variants
  - `[20-24]` `no_write` variants
  - `[25-29]` `flip_noc` full algorithms (non-optimal NoC)
  - `[30-34]` `flip_noc no_a_read` variants
  - `[35-39]` `flip_noc no_b_read` variants
  - `[40-44]` `flip_noc no_compute` variants
  - `[45-49]` `flip_noc no_write` variants

### `profiling_suite.hpp`

- Added `profile_case_parametric_random<M, N, K, R, C, DensityPercent>` template
  that creates a random BSR matrix and dense matrix with the given dimensions.
  Test name encoded as `parametric_M{M}_N{N}_K{K}_R{R}_C{C}_d{density%}`.
- Added 4 new static registry arrays: `ProfileSweepNRegistry`,
  `ProfileSweepDensityRegistry`, `ProfileSweepKRegistry`,
  `ProfileSweepBlockSizeRegistry`.

### `src/profile_block.cpp`

- Added `case 4..7` to the registry switch statement.

---

## Running the Profiling Plan

### Prerequisites

```bash
cd /home/user/tt-metal
# Ensure Tracy capture-release binary is on PATH or in the working directory
```

### List all registries and host codes

```bash
./spmm_scripts/run_profiling_plan.sh --list
```

### Full profiling run (both phases)

```bash
./spmm_scripts/run_profiling_plan.sh
```

### Ablation phase only

Runs all 4 skip variants × 5 algorithms against the reference registry (registry 2
by default):

```bash
./spmm_scripts/run_profiling_plan.sh --phase ablation
```

Use a different reference registry:

```bash
./spmm_scripts/run_profiling_plan.sh --phase ablation --ablation-registry 3
```

### Sweep phase only

Runs base algorithms (host codes 0-4) against all 4 sweep registries (4-7):

```bash
./spmm_scripts/run_profiling_plan.sh --phase sweep
```

Run a single sweep registry:

```bash
./spmm_scripts/run_profiling_plan.sh --phase sweep --registry 5
```

### Restrict to one algorithm

The `--host-code` flag accepts an algorithm index 0-4:
- For the **sweep phase**: directly selects host codes 0-4 in `HostCodeRegistryProfiling`.
- For the **ablation phase**: selects the algorithm offset within each ablation group
  (e.g. `--host-code 0` runs `snf_no_a_read`, `snf_no_b_read`, etc.).

```bash
# Only snf ablations:
./spmm_scripts/run_profiling_plan.sh --phase ablation --host-code 0

# Only load_balanced sweeps:
./spmm_scripts/run_profiling_plan.sh --phase sweep --host-code 1
```

### Flip-NoC phase

Runs all 5 algorithms with non-optimal NoC assignment, plus their 4 ablation variants
(25 runs total per profile case), against the reference registry:

```bash
./spmm_scripts/run_profiling_plan.sh --phase flip_noc
```

Use a different reference registry:

```bash
./spmm_scripts/run_profiling_plan.sh --phase flip_noc --ablation-registry 3
```

Run only the full (non-ablation) flip_noc variants — useful if you only want to
compare optimal vs non-optimal NoC without the ablation breakdown. Unfortunately
the script runs all 5 groups together, so use `--host-code` to restrict to one
algorithm:

```bash
# Only snf with non-optimal NoC (full + all 4 ablations):
./spmm_scripts/run_profiling_plan.sh --phase flip_noc --host-code 0

# Only load_balanced_new_DM with non-optimal NoC:
./spmm_scripts/run_profiling_plan.sh --phase flip_noc --host-code 4
```

### Skip rebuild

```bash
./spmm_scripts/run_profiling_plan.sh --no-build --phase sweep
```

### Dry run (preview commands without executing)

```bash
./spmm_scripts/run_profiling_plan.sh --dry-run
```

---

## Output

Traces and CSV files are written by `profile_block` and `export_to_csv` to:

```
/home/user/tt-metal/profiles_opt_noc_flip_writer/bsr/<registry_name>/<host_code_name>/
```

Each file is named after the test case (e.g. `parametric_M8192_N8192_K8192_R64_C64_d25`).

---

## Key Invariants

- **CB handshake protocol is never broken.** Every `cb_push_back` and `cb_pop_front`
  that would happen in a normal run still happens in ablation runs. Only the actual
  DRAM reads/writes or compute instructions are skipped.
- **`zone_defines` merge is non-destructive.** Zone profiling flags from
  `spmm_zone_config::get_zone_defines()` are applied first, then skip flags are
  inserted. Skip flags and zone flags are orthogonal.
- **Existing registries (0-3) and host codes are unchanged.** The refactoring
  preserves all original function signatures and behavior.
