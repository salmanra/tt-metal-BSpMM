# SpMM in BSR format
In this readme:

1. Directory Structure
2. Specifying which version of the SpMM algorithm you want to run
3. How to add a test case
4. How to build and run the test suite

### Directory structure
```
block_spmm/
-- inc/
-- -- host_code.hpp
-- -- test_suite.hpp
-- kernels/
-- -- compute/
-- -- dataflow/
-- src/
-- -- block.cpp
-- -- profile_block.cpp
-- -- test_block.cpp
-- test/
-- -- *.txt
```

**src/** - contains the **.cpp** files which run the program. ***test_block.cpp*** runs the selected version of the program chosen from the registry defined in *host_code.hpp* along the entire test suite defined in *test_suite.hpp* and checks that all tests pass (Pearson's Correlation Coefficient between sequential result and multicore result is greater than 0.99). ***profile_block.cpp*** runs the selected host code on the selected test case 10 times with Tracy ZoneScoped macros for capturing profiling data.

## Specifying a program to run
Since we are iterating over increasingly optimized SpMM impls, we want an easy way to go back and forth between versions.
***host_code.hpp*** introduces a HostCodeRegistry which includes the function pointers to all the different versions in development.

Both the **profile_block** and **test_block** executable take an index into the HostCodeRegistry. If none is provided, they will run the first program in the registry. Try to keep the newest version at the top so you can run these executables without thinking to hard about the command line args.

When you want to develop a new version of the program, add the new function declaration to the top of the namespace in ***host_code.hpp***, then add that function pointer to the HostCodeRegistry, then define it towards the bottom of the namespace. This will allow any host program (profiling, testing, visualizing, debugging) to quickly access all version of the code.

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
## How to capture a profiling trace
Say my latest test case is big and interesting and I want to capture a trace of my program running the test case. I will have to:
1. Rebuild **tt-metal** with progrmaming_examples *and* profiling enabled.
2. Start a background process of the Tracy profiler listening on the capture port.
3. Run the profile build, specifying a program implementation and the test case number (index into the TestRegistry).
4. Move the trace file from the remote machine to any machine with Tracy GUI.
5. Open the saved trace in the GUI.

**Tracy** is the open-source profiler that Tenstorrent includes in the source builds. Since TT cards are typically set up in non-interactive workstations, we have to launch the Tracy GUI on a separate machine from the one we use to run our code and capture a trace. On the remote Wormhole card we have access to, we have the **capture** tool from Tracy which is built into the **tt-metal** project. After creating symbolic link to the ***./capture-release*** command, we can achieve steps 1. 2. and 3. with the following commands:

```shell
./build_metal.sh --enable-profiler --build-programming-examples
./capture-release -f -o {path-to-stored-traces}/{new-test-name}.tracy & # Note the ampersand!!!
./build/programming_examples/rahmy/profile_block {program_id} {test_id} # default behavior -> use first function in HostCodeRegistry, use test case 30
```
As the test case is running, the console should display a wealth of live profiling information. If it isn't, check that ***./capture-release*** was running in the background before you ran the test case, and check that your last rebuild of **tt-metal** enabled profiling.
