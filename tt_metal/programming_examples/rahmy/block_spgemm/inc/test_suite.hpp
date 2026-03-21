#pragma once

#include <cstdint>
#include <random>
#include "include_me.hpp"
#include "sparse_common/bsr_matrix.hpp"
#include "tt-metalium/bfloat16.hpp"

using namespace tt;

namespace bsr_spgemm_test_suite {

    // SpGEMM test cases return two BSR matrices (A and B) and a test name.
    // Verification: compare device output against A.spgemm(B) on CPU.

    // Diagonal x Diagonal: simplest case, output is diagonal, structure matches inputs
    inline std::tuple<bsr_matrix<bfloat16>, bsr_matrix<bfloat16>, std::string> test_diagonal_times_diagonal() {
        // 4x4 block grid, 32x32 blocks, diagonal with 4 blocks
        bsr_matrix<bfloat16> a(128, 128, 32, 32, 4, FILL_DIAG, RAND);
        bsr_matrix<bfloat16> b(128, 128, 32, 32, 4, FILL_DIAG, RAND);
        return {a, b, "diagonal_times_diagonal"};
    }

    // Single block: 1x1 block grid, trivial case for debugging
    inline std::tuple<bsr_matrix<bfloat16>, bsr_matrix<bfloat16>, std::string> test_single_block() {
        bsr_matrix<bfloat16> a(32, 32, 32, 32, 1, FILL_ROW, RAND);
        bsr_matrix<bfloat16> b(32, 32, 32, 32, 1, FILL_ROW, RAND);
        return {a, b, "single_block"};
    }

    // Row x Column: A has blocks in first row, B has blocks in first column
    // Tests fill-in: output should have blocks everywhere in the product of those patterns
    inline std::tuple<bsr_matrix<bfloat16>, bsr_matrix<bfloat16>, std::string> test_row_times_col() {
        // A: 4x4 block grid, first row filled (4 blocks)
        bsr_matrix<bfloat16> a(128, 128, 32, 32, 4, FILL_ROW, RAND);
        // B: 4x4 block grid, first column filled (4 blocks)
        bsr_matrix<bfloat16> b(128, 128, 32, 32, 4, FILL_COL, RAND);
        return {a, b, "row_times_col"};
    }

    // Random sparse: general test case with random sparsity pattern
    inline std::tuple<bsr_matrix<bfloat16>, bsr_matrix<bfloat16>, std::string> test_random_sparse() {
        // 8x8 block grid, ~25% density (16 blocks out of 64)
        bsr_matrix<bfloat16> a(256, 256, 32, 32, 16, RAND);
        bsr_matrix<bfloat16> b(256, 256, 32, 32, 16, RAND);
        return {a, b, "random_sparse"};
    }

    // Identity x Sparse: output should equal B (good correctness check)
    inline std::tuple<bsr_matrix<bfloat16>, bsr_matrix<bfloat16>, std::string> test_identity_times_sparse() {
        // A: diagonal identity blocks
        bsr_matrix<bfloat16> a(128, 128, 32, 32, 4, FILL_DIAG, ID);
        // B: random sparse
        bsr_matrix<bfloat16> b(128, 128, 32, 32, 6, FILL_ROW, RAND);
        return {a, b, "identity_times_sparse"};
    }

    // Triangular x Triangular: tests fill-in (output has more nonzeros than either input)
    inline std::tuple<bsr_matrix<bfloat16>, bsr_matrix<bfloat16>, std::string> test_triangular_times_triangular() {
        bsr_matrix<bfloat16> a(128, 128, 32, 32, 0, FILL_TRIL, RAND);
        bsr_matrix<bfloat16> b(128, 128, 32, 32, 0, FILL_TRIL, RAND);
        return {a, b, "triangular_times_triangular"};
    }

    // Nonsquare: A is tall, B is wide
    inline std::tuple<bsr_matrix<bfloat16>, bsr_matrix<bfloat16>, std::string> test_nonsquare() {
        // A: 8x4 block grid (256x128), B: 4x8 block grid (128x256)
        bsr_matrix<bfloat16> a(256, 128, 32, 32, 8, FILL_ROW, RAND);
        bsr_matrix<bfloat16> b(128, 256, 32, 32, 8, FILL_ROW, RAND);
        return {a, b, "nonsquare"};
    }

    // Large random: bigger test for stress testing
    inline std::tuple<bsr_matrix<bfloat16>, bsr_matrix<bfloat16>, std::string> test_large_random() {
        // 16x16 block grid, ~10% density
        bsr_matrix<bfloat16> a(512, 512, 32, 32, 26, RAND);
        bsr_matrix<bfloat16> b(512, 512, 32, 32, 26, RAND);
        return {a, b, "large_random"};
    }

    using TestFunctionPtr = std::tuple<bsr_matrix<bfloat16>, bsr_matrix<bfloat16>, std::string> (*)();

    static TestFunctionPtr TestRegistry[] = {
        test_diagonal_times_diagonal,     // 0
        test_single_block,                // 1
        test_row_times_col,               // 2
        test_random_sparse,               // 3
        test_identity_times_sparse,       // 4
        test_triangular_times_triangular, // 5
        test_nonsquare,                   // 6
        test_large_random,                // 7
    };

} // namespace bsr_spgemm_test_suite
