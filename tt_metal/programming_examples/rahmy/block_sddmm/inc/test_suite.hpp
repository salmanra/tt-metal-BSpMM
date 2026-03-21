#pragma once

#include <cstdint>
#include <random>
#include "include_me.hpp"
#include "sparse_common/bsr_matrix.hpp"
#include "tt-metalium/bfloat16.hpp"

using namespace tt;

namespace bsr_sddmm_test_suite {

    // SDDMM test cases return: sampling mask (BSR), dense C, dense D, test name.
    // SDDMM: A = B ⊙ (C × D)
    // Verification: compare device output against sampling_mask.sddmm(c, d) on CPU.

    using SDDMMTestReturnType = std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, dense_matrix<bfloat16>, std::string>;

    // Single block: 1×1 block grid, trivial case
    inline SDDMMTestReturnType test_single_block() {
        bsr_matrix<bfloat16> mask(32, 32, 32, 32, 1, FILL_ROW, RAND);
        dense_matrix<bfloat16> c(32, 64, RAND);
        dense_matrix<bfloat16> d(64, 32, RAND);
        return {mask, c, d, "single_block"};
    }

    // Diagonal mask: only diagonal blocks are sampled
    inline SDDMMTestReturnType test_diagonal_mask() {
        bsr_matrix<bfloat16> mask(128, 128, 32, 32, 4, FILL_DIAG, RAND);
        dense_matrix<bfloat16> c(128, 64, RAND);
        dense_matrix<bfloat16> d(64, 128, RAND);
        return {mask, c, d, "diagonal_mask"};
    }

    // Full mask: all blocks nonzero (degenerates to full GEMM then hadamard)
    inline SDDMMTestReturnType test_full_mask() {
        // 2×2 block grid, all 4 blocks present
        bsr_matrix<bfloat16> mask(64, 64, 32, 32, 4, FILL_ROW, UNIFORM);
        dense_matrix<bfloat16> c(64, 32, RAND);
        dense_matrix<bfloat16> d(32, 64, RAND);
        return {mask, c, d, "full_mask"};
    }

    // Identity mask: mask blocks are identity matrices
    // Output should equal corresponding blocks of C×D
    inline SDDMMTestReturnType test_identity_mask() {
        bsr_matrix<bfloat16> mask(128, 128, 32, 32, 4, FILL_DIAG, ID);
        dense_matrix<bfloat16> c(128, 64, RAND);
        dense_matrix<bfloat16> d(64, 128, RAND);
        return {mask, c, d, "identity_mask"};
    }

    // Random sparse mask: general case with random sparsity pattern
    inline SDDMMTestReturnType test_random_sparse_mask() {
        // 8×8 block grid, ~25% density (16 blocks out of 64)
        bsr_matrix<bfloat16> mask(256, 256, 32, 32, 16, RAND);
        dense_matrix<bfloat16> c(256, 128, RAND);
        dense_matrix<bfloat16> d(128, 256, RAND);
        return {mask, c, d, "random_sparse_mask"};
    }

    // Nonsquare: tall mask × wide dense product
    inline SDDMMTestReturnType test_nonsquare() {
        // mask: 8×4 block grid (256×128), C: 256×64, D: 64×128
        bsr_matrix<bfloat16> mask(256, 128, 32, 32, 8, FILL_ROW, RAND);
        dense_matrix<bfloat16> c(256, 64, RAND);
        dense_matrix<bfloat16> d(64, 128, RAND);
        return {mask, c, d, "nonsquare"};
    }

    // Large random: bigger test for stress testing
    inline SDDMMTestReturnType test_large_random() {
        // 16×16 block grid, ~10% density
        bsr_matrix<bfloat16> mask(512, 512, 32, 32, 26, RAND);
        dense_matrix<bfloat16> c(512, 256, RAND);
        dense_matrix<bfloat16> d(256, 512, RAND);
        return {mask, c, d, "large_random"};
    }

    using TestFunctionPtr = SDDMMTestReturnType (*)();

    static TestFunctionPtr TestRegistry[] = {
        test_single_block,        // 0
        test_diagonal_mask,       // 1
        test_full_mask,           // 2
        test_identity_mask,       // 3
        test_random_sparse_mask,  // 4
        test_nonsquare,           // 5
        test_large_random,        // 6
    };

} // namespace bsr_sddmm_test_suite
