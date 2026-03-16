# SpMM in BSR format
In this readme:

1. Directory Structure
2. Specifying which version of the SpMM algorithm you want to run
3. How to add a test case
4. How to build and run the test suite
5. How to run the profiling plan

### Directory structure
```
block_spmm/
-- inc/
-- -- host_code.hpp          # HostCodeRegistry + HostCodeRegistryProfiling
-- -- host_code/             # one .cpp per algorithm implementation
-- -- -- bsr_spmm_multicore_snf.cpp
-- -- -- bsr_spmm_multicore_load_balanced.cpp
-- -- -- bsr_spmm_multicore_reuse_iteration.cpp
-- -- -- bsr_spmm_multicore_naive_new_DM.cpp
-- -- -- bsr_spmm_multicore_load_balanced_new_DM.cpp
-- -- -- ...
-- -- -- spmm_zone_config.hpp
-- -- test_suite.hpp
-- -- profiling_suite.hpp    # parametric test cases + sweep registries
-- -- bsr_matrix.hpp
-- -- bmm_op.hpp
-- kernels/
-- -- compute/               # compute kernels (bmm_iter.cpp, ...)
-- -- dataflow/              # reader/writer kernels
-- -- common/                # shared headers (spmm_profiling.hpp, spmm_reader_common.hpp, ...)
-- src/
-- -- test_block.cpp
-- -- profile_block.cpp
-- -- export_to_csv.cpp
-- -- run_block.cpp
-- -- ...
-- analysis_tools/           # Python scripts for output analysis
-- PROFILING_PLAN.md         # detailed profiling infrastructure docs
```

**src/** - contains the **.cpp** files which run the program. ***test_block.cpp*** runs the selected version of the program chosen from the registry defined in *host_code.hpp* along the entire test suite defined in *test_suite.hpp* and checks that all tests pass (Pearson's Correlation Coefficient between sequential result and multicore result is greater than 0.99). ***profile_block.cpp*** runs the selected host code on the selected profiling case with Tracy ZoneScoped macros for capturing profiling data. ***export_to_csv.cpp*** extracts Tracy trace data to CSV files for analysis.

## Specifying a program to run
Since we are iterating over increasingly optimized SpMM impls, we want an easy way to go back and forth between versions.
***host_code.hpp*** introduces two registries:

- **HostCodeRegistry** — the base algorithms, used by `test_block`.
- **HostCodeRegistryProfiling** — an extended registry (50 entries) used by `profile_block`, organized into groups:

| Indices   | Group              | Description                                 |
|-----------|--------------------|---------------------------------------------|
| `[0-4]`   | Full               | Base algorithms (same as HostCodeRegistry)  |
| `[5-9]`   | `no_a_read`        | Skip sparse A DRAM reads                    |
| `[10-14]` | `no_b_read`        | Skip dense B DRAM reads                     |
| `[15-19]` | `no_compute`       | Skip tile multiply                          |
| `[20-24]` | `no_write`         | Skip output DRAM writes                     |
| `[25-29]` | `flip_noc full`    | Full algorithms with non-optimal NoC        |
| `[30-34]` | `flip_noc no_a`    | Non-optimal NoC + skip sparse A reads       |
| `[35-39]` | `flip_noc no_b`    | Non-optimal NoC + skip dense B reads        |
| `[40-44]` | `flip_noc no_c`    | Non-optimal NoC + skip compute              |
| `[45-49]` | `flip_noc no_w`    | Non-optimal NoC + skip DRAM writes          |

Within each group, the 5 algorithms are ordered: `snf`, `load_balanced`, `reuse_iteration`, `naive_new_DM`, `load_balanced_new_DM`.

Both the **profile_block** and **test_block** executables take an index into their respective registry. If none is provided, they will run the first program in the registry.

When you want to develop a new version of the program, add the new function declaration to the top of the namespace in ***host_code.hpp***, then add that function pointer to the HostCodeRegistry, then define it towards the bottom of the namespace. This will allow any host program (profiling, testing, visualizing, debugging) to quickly access all versions of the code.

## How to add a test case
In ***test_suite.hpp***:
1. Add a function declaration at the top of the namespace for your test case. It should take no args and return a tuple.
2. Add the name of your test case to the TestRegistry.
3. Add the function definition at the bottom of the namespace (anywhere following the TestRegistry).

### How to write a test case

***test_suite.hpp*** defines a test registry which any host code can call upon to obtain any number of test cases by index into the test registry. Each test case is a function with no arguments which returns a BSR matrix, a dense matrix, and the name of the test case. The calling program is then free to use the returned data structures as it pleases (testing, profiling, visualizing, debugging...).

In general, each test case is structured as below:
```C++
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_case_name() {
        // matmul params setup
        uint32_t M = 1024;
        uint32_t N = 1024;
        uint32_t K = 1024;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 32;
        uint32_t block_matrix_height = M / R;

        // create matrices. Constructors provide options for block placement
        bsr_matrix<float> bsr(M, K, R, C, nblocks, RAND);
        dense_matrix<float> dense(K, N, RAND);

        // cast to bfloat16 for testing on Tenstorrent
        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_case_name");
    }
```

For tests which require more control over the shape, values, and placement of nonzero blocks in the sparse matrix, the programmer can create a BSR matrix by explicitly passing in its data, indices, and indptr vectors.

```C++
std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_checkerboard() {
    uint32_t M = 256;
    uint32_t N = 256;
    uint32_t K = 256;
    uint32_t R = 64;
    uint32_t C = 64;
    uint32_t nblocks = 8;
    uint32_t block_matrix_height = M / R;

    // custom data array
    std::vector<float> data(R*C*nblocks);
    for (int k = 0; k < nblocks; k++){
        for (int i = 0; i < R*C; i++){
            data[k*nblocks + i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
    }

    // checkerboard pattern of nonzero blocks
    std::vector<int> indptr = {0, 2, 4, 6, 8};
    std::vector<int> indices = {0, 2, 1, 3, 0, 2, 1, 3};

    // constructor from data vectors
    bsr_matrix<float> bsr(data, indptr, indices, M, K, R, C, nblocks);
    dense_matrix<float> dense(K, N, RAND);

    bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
    dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
    return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_checkerboard");
}
```

## How to build and run the test suite
Say I have added a test to *test_suite.hpp* and want to run it on my latest SpMM implementation. I will have to:
1. Rebuild **tt_metal** with programming_examples enabled.
2. Run the test suite.

```shell
./build_metal.sh --build-programming-examples # should be fast, will only rebuild modified examples
./build/programming_examples/rahmy/test_block # default args -> runs the entire test suite using the first function in the HostCodeRegistry
```
## How to run the profiling plan

The automated profiling script `spmm_scripts/run_profiling_plan.sh` handles building with profiling enabled, launching Tracy capture, running profile_block, and exporting CSVs. It supports three phases: **ablation** (skip one cost component at a time), **sweep** (vary one matrix parameter at a time), and **flip_noc** (compare optimal vs non-optimal NoC assignment).

```shell
# List all registries and host codes
./spmm_scripts/run_profiling_plan.sh --list

# Run everything (ablation + sweep + flip_noc)
./spmm_scripts/run_profiling_plan.sh

# Ablation only (all 5 algorithms × 4 skip variants)
./spmm_scripts/run_profiling_plan.sh --phase ablation

# Sweep only (vary N, density, K, or block size)
./spmm_scripts/run_profiling_plan.sh --phase sweep

# Flip-NoC only (non-optimal NoC + ablation variants)
./spmm_scripts/run_profiling_plan.sh --phase flip_noc

# Restrict to one algorithm (0=snf, 1=load_balanced, ...)
./spmm_scripts/run_profiling_plan.sh --phase ablation --host-code 0

# Skip rebuild if already built with profiling
./spmm_scripts/run_profiling_plan.sh --no-build --phase sweep

# Preview commands without executing
./spmm_scripts/run_profiling_plan.sh --dry-run
```

See [PROFILING_PLAN.md](PROFILING_PLAN.md) for the full description of the profiling infrastructure, including kernel skip flags, the `_impl` host code pattern, sweep registries, and output locations.

### Manual single-trace capture

For one-off traces outside the profiling plan, you can still run the manual process:

```shell
./build_metal.sh --enable-profiler --build-programming-examples
./capture-release -f -o {path-to-stored-traces}/{new-test-name}.tracy & # Note the ampersand!!!
./build/programming_examples/rahmy/profile_block {program_id} {test_id}
```

**Tracy** is the open-source profiler that Tenstorrent includes in the source builds. Since TT cards are typically set up in non-interactive workstations, we launch the Tracy GUI on a separate machine from the one used to run code and capture traces. The **capture-release** tool is built into the **tt-metal** project. As the test case is running, the console should display live profiling information. If it isn't, check that ***./capture-release*** was running in the background before you ran the test case, and check that your last rebuild of **tt-metal** enabled profiling.
