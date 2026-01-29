#pragma once

#include <cstdint>
#include <cmath>
#include <random>
#include "include_me.hpp"
#include "block_spmm/inc/bsr_matrix.hpp"
#include "tt-metalium/bfloat16.hpp"
#include "tt-metalium/bfloat4.hpp"

using namespace tt;

using CoreSpec = std::variant<CoreCoord, CoreRange, CoreRangeSet>;

namespace profiling_suite {
    // ehh... what are the interesting test cases to profile?

    // invariably, big test cases are interesting.
    // Dense
        // Square
        // Tall (we're screwed)
        // Wide
    // Sparse (p < 0.1)
        // single input block
        // diagonal
            // perm
        // column fill
        // row fill
    // Semi-sparse (0.1 < p < 0.5)
        // Random placement
        // Row fill
        // Column fill
        // Checkerboard
        // Diagonal

    // What were the largest dimensions the dense cases supported?
        //         // NOTE: Maximum number of tiles in output is 120 * 16^2 = 30,720 (eg. [1, 1, 5120, 6144])
        //                                                          < 2^15 = 32,768
        // --> M=N=2^7=4096 is the largest s quare output which is power of 2
        // I tried M=8192 N=4096 K=512 and, after a bunch of time, got an error (floating point exception) which is apparently a divide by 0 error
        //      where does the matmul kernel divide by 0? curious but not important.
        //
        //


    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    ///////// Profile Case Declarations ////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    using ProfileCaseReturnType = std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string>;
    template <uint32_t>
    ProfileCaseReturnType profile_case_dense_square();

    template <uint32_t>
    ProfileCaseReturnType profile_case_dense_tall();
    template <uint32_t>
    ProfileCaseReturnType profile_case_dense_wide();

    // template on block sizes R and C.
    // we could template on fill_type, but too late :P
    template <uint32_t, uint32_t>
    ProfileCaseReturnType profile_case_sparse_single_input_block();
    template <uint32_t, uint32_t>
    ProfileCaseReturnType profile_case_sparse_diagonal();
    template <uint32_t, uint32_t>
    ProfileCaseReturnType profile_case_sparse_fill_row();
    template <uint32_t, uint32_t>
    ProfileCaseReturnType profile_case_sparse_fill_column();
    template <uint32_t, uint32_t>
    ProfileCaseReturnType profile_case_sparse_fill_random();

    // Large sparse cases (32768x32768 matrices)
    // DensityPercent: density as percentage (e.g., 25 = 0.25, 10 = 0.10)
    template <uint32_t, uint32_t, uint32_t>
    ProfileCaseReturnType profile_case_sparse_diagonal_large();
    template <uint32_t, uint32_t, uint32_t>
    ProfileCaseReturnType profile_case_sparse_fill_column_large();
    template <uint32_t, uint32_t, uint32_t>
    ProfileCaseReturnType profile_case_sparse_fill_row_large();
    template <uint32_t, uint32_t, uint32_t>
    ProfileCaseReturnType profile_case_sparse_fill_random_large();

    ProfileCaseReturnType profile_case_sanity_check();

    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    ///////// Profile Case Registry ////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    using ProfileCaseFunctionPtr = ProfileCaseReturnType (*)();
    static ProfileCaseFunctionPtr ProfileCaseRegistry[] = {
        profile_case_sparse_single_input_block<32, 32>, // 3
        profile_case_sparse_single_input_block<64, 64>, // 4
        profile_case_sparse_single_input_block<128, 128>, // 5
        profile_case_sparse_diagonal<32, 32>, // 6
        profile_case_sparse_diagonal<64, 64>, // 7
        profile_case_sparse_diagonal<128, 128>, // 8
        profile_case_sparse_fill_column<32, 32>, // 9
        profile_case_sparse_fill_column<64, 64>, // 10
        profile_case_sparse_fill_column<128, 128>, // 11
        profile_case_sparse_fill_row<32, 32>, // 12
        profile_case_sparse_fill_row<64, 64>, // 13
        profile_case_sparse_fill_row<128, 128>, // 14
        profile_case_sparse_fill_random<32, 32>, // 15
        profile_case_sparse_fill_random<64, 64>, // 16
        profile_case_sparse_fill_random<128, 128>, // 17
    };

    static ProfileCaseFunctionPtr ProfileDenseAblationRegistry[] = {
        profile_case_dense_square<512>, // 0
        profile_case_dense_square<1024>, // 1
        profile_case_dense_square<2048>, // 2
        profile_case_dense_square<4096>, // 3
    };

    static ProfileCaseFunctionPtr ProfileLargeSparseRegistry[] = {
        profile_case_sparse_diagonal_large<32, 32, 25>, // 0 - 25% density
        profile_case_sparse_diagonal_large<64, 64, 25>, // 1
        profile_case_sparse_diagonal_large<128, 128, 25>, // 2
        profile_case_sparse_fill_column_large<32, 32, 25>, // 3
        profile_case_sparse_fill_column_large<64, 64, 25>, // 4
        profile_case_sparse_fill_column_large<128, 128, 25>, // 5
        profile_case_sparse_fill_row_large<32, 32, 25>, // 6
        profile_case_sparse_fill_row_large<64, 64, 25>, // 7
        profile_case_sparse_fill_row_large<128, 128, 25>, // 8
        profile_case_sparse_fill_random_large<32, 32, 25>, // 9
        profile_case_sparse_fill_random_large<64, 64, 25>, // 10
        profile_case_sparse_fill_random_large<128, 128, 25>, // 11
    };

    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    ///////// Profile Case Definitions /////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////
    ///////// Sanity Check /////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    inline ProfileCaseReturnType profile_case_sanity_check() {
        uint32_t M = 32;
        uint32_t N = 32;
        uint32_t K = 32;

        uint32_t R = 32;
        uint32_t C = 32;
        uint32_t nblocks = 1;

        bsr_matrix<float> src0(M, K, R, C, nblocks, RAND);
        dense_matrix<float> src1(K, N, RAND);

        bsr_matrix<bfloat16> src0_bfoat16 = src0.bfloat16_cast();
        dense_matrix<bfloat16> src1_bfloat16 = src1.bfloat16_cast();
        return std::make_tuple(src0_bfoat16, src1_bfloat16, "profile_case_sanity_check");
    }
    ////////////////////////////////////////////////////////////////////////////
    ///////// Large Cases //////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    template <uint32_t R = 32, uint32_t C = 32, uint32_t DensityPercent = 25>
    inline ProfileCaseReturnType profile_case_sparse_diagonal_large() {
        // matmul params setup
        uint32_t M = 32768;
        uint32_t N = 32768;
        uint32_t K = 32768;
        // block params setup
        uint32_t block_matrix_height = M / R;
        uint32_t block_matrix_width = K / C;

        constexpr float density = DensityPercent / 100.0f;
        uint32_t divisor = uint32_t(std::round(1.0 / density));
        uint32_t nblocks = (block_matrix_height * block_matrix_width) / divisor; 

        // nz blocks fill the diagonal
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_DIAG, RAND);
        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        char buf[50];
        size_t n = sprintf(buf, "profile_case_sparse_diagonal_large_R%i_C%d", R, C);
        std::string test_name(buf, n);
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, test_name);
    }

    template <uint32_t R = 32, uint32_t C = 32, uint32_t DensityPercent = 25>
    inline ProfileCaseReturnType profile_case_sparse_fill_column_large() {
        // matmul params setup
        uint32_t M = 32768;
        uint32_t N = 32768;
        uint32_t K = 32768;
        // block params setup
        uint32_t block_matrix_height = M / R;
        uint32_t block_matrix_width = K / C;

        constexpr float density = DensityPercent / 100.0f;
        uint32_t divisor = uint32_t(std::round(1.0 / density));
        uint32_t nblocks = (block_matrix_height * block_matrix_width) / divisor; 

        // nz blocks fill the first column
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_COL, RAND);
        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        char buf[50];
        size_t n = sprintf(buf, "profile_case_sparse_fill_column_large_R%i_C%d", R, C);
        std::string test_name(buf, n);
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, test_name);
    }

    template <uint32_t R = 32, uint32_t C = 32, uint32_t DensityPercent = 25>
    inline ProfileCaseReturnType profile_case_sparse_fill_row_large() {
        // matmul params setup
        uint32_t M = 32768;
        uint32_t N = 32768;
        uint32_t K = 32768;
        // block params setup
        uint32_t block_matrix_height = M / R;
        uint32_t block_matrix_width = K / C;

        constexpr float density = DensityPercent / 100.0f;
        uint32_t divisor = uint32_t(std::round(1.0 / density));
        uint32_t nblocks = (block_matrix_height * block_matrix_width) / divisor; 

        // nz blocks fill the first row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_ROW, RAND);
        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        char buf[50];
        size_t n = sprintf(buf, "profile_case_sparse_fill_row_large_R%i_C%d", R, C);
        std::string test_name(buf, n);
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, test_name);
    }

    template <uint32_t R = 32, uint32_t C = 32, uint32_t DensityPercent = 25>
    inline ProfileCaseReturnType profile_case_sparse_fill_random_large() {
        // matmul params setup
        uint32_t M = 32768;
        uint32_t N = 32768;
        uint32_t K = 32768;
        // block params setup
        uint32_t block_matrix_height = M / R;
        uint32_t block_matrix_width = K / C;

        constexpr float density = DensityPercent / 100.0f;
        uint32_t divisor = uint32_t(std::round(1.0 / density));
        uint32_t nblocks = (block_matrix_height * block_matrix_width) / divisor; 

        // nz blocks placed randomly
        bsr_matrix<float> bsr(M, K, R, C, nblocks, RAND);
        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        char buf[50];
        size_t n = sprintf(buf, "profile_case_sparse_fill_random_large_R%i_C%d", R, C);
        std::string test_name(buf, n);
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, test_name);
    }


    ////////////////////////////////////////////////////////////////////////////
    ///////// Dense Cases //////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    template <uint32_t K = 512>
    inline ProfileCaseReturnType profile_case_dense_square() {
        uint32_t M = 32768;
        uint32_t N = 32768;

        dense_matrix<float> tmp(M, K, RAND);
        bsr_matrix<float> src0(tmp, N);
        dense_matrix<float> src1(K, N, RAND);

        bsr_matrix<bfloat16> src0_bfoat16 = src0.bfloat16_cast();
        dense_matrix<bfloat16> src1_bfloat16 = src1.bfloat16_cast();

        char buf[50];
        size_t n = sprintf(buf, "profile_case_dense_square_K%i", K);
        std::string test_name(buf, n);
        return std::make_tuple(src0_bfoat16, src1_bfloat16, test_name);
    }

    template <uint32_t K = 512>
    inline ProfileCaseReturnType profile_case_dense_tall() {
        uint32_t M = 32768;
        uint32_t N = 1024;
        dense_matrix<float> tmp(M, K, RAND);
        bsr_matrix<float> src0(tmp, N);
        dense_matrix<float> src1(K, N, RAND);

        bsr_matrix<bfloat16> src0_bfoat16 = src0.bfloat16_cast();
        dense_matrix<bfloat16> src1_bfloat16 = src1.bfloat16_cast();
        return std::make_tuple(src0_bfoat16, src1_bfloat16, "profile_case_dense_tall");
    }
    template <uint32_t K = 512>
    inline ProfileCaseReturnType profile_case_dense_wide() {
        uint32_t M = 1024;
        uint32_t N = 32768;

        dense_matrix<float> tmp(M, K, RAND);
        bsr_matrix<float> src0(tmp, N);
        dense_matrix<float> src1(K, N, RAND);

        bsr_matrix<bfloat16> src0_bfoat16 = src0.bfloat16_cast();
        dense_matrix<bfloat16> src1_bfloat16 = src1.bfloat16_cast();
        return std::make_tuple(src0_bfoat16, src1_bfloat16, "profile_case_dense_wide");
    }

    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    ///////// Sparse Cases /////////////////////////////////////////////////////
    ///////// p < 0.1 //////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////


    template <uint32_t R = 32, uint32_t C = 32>
    inline ProfileCaseReturnType profile_case_sparse_single_input_block() {
        // matmul params setup
        uint32_t M = 1024;
        uint32_t N = 1024;
        uint32_t K = 1024;
        // block params setup
        uint32_t nblocks = 1;
        uint32_t block_matrix_height = M / R;

        // nz block is in the first position
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_ROW, RAND);
        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        char buf[50];
        size_t n = sprintf(buf, "profile_case_sparse_single_block_R%i_C%d", R, C);
        std::string test_name(buf, n);
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, test_name);
    }

    template <uint32_t R = 32, uint32_t C = 32>
    inline ProfileCaseReturnType profile_case_sparse_diagonal() {
        // matmul params setup
        uint32_t M = 1024;
        uint32_t N = 1024;
        uint32_t K = 1024;
        // block params setup
        uint32_t block_matrix_height = M / R;
        uint32_t nblocks = block_matrix_height;

        // nz blocks fill the diagonal
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_DIAG, RAND);
        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        char buf[50];
        size_t n = sprintf(buf, "profile_case_sparse_diagonal_R%i_C%d", R, C);
        std::string test_name(buf, n);
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, test_name);
    }

    template <uint32_t R = 32, uint32_t C = 32>
    inline ProfileCaseReturnType profile_case_sparse_fill_column() {
        // matmul params setup
        uint32_t M = 1024;
        uint32_t N = 1024;
        uint32_t K = 1024;
        // block params setup
        uint32_t block_matrix_height = M / R;
        uint32_t nblocks = block_matrix_height;

        // nz blocks fill the first column
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_COL, RAND);
        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        char buf[50];
        size_t n = sprintf(buf, "profile_case_sparse_fill_column_R%i_C%d", R, C);
        std::string test_name(buf, n);
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, test_name);
    }

    template <uint32_t R = 32, uint32_t C = 32>
    inline ProfileCaseReturnType profile_case_sparse_fill_row() {
        // matmul params setup
        uint32_t M = 1024;
        uint32_t N = 1024;
        uint32_t K = 1024;
        // block params setup
        uint32_t block_matrix_height = M / R;
        uint32_t nblocks = block_matrix_height;

        // nz blocks fill the first row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_ROW, RAND);
        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        char buf[50];
        size_t n = sprintf(buf, "profile_case_sparse_fill_row_R%i_C%d", R, C);
        std::string test_name(buf, n);
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, test_name);
    }

    template <uint32_t R = 32, uint32_t C = 32>
    inline ProfileCaseReturnType profile_case_sparse_fill_random() {
        // matmul params setup
        uint32_t M = 1024;
        uint32_t N = 1024;
        uint32_t K = 1024;
        // block params setup
        uint32_t block_matrix_height = M / R;
        uint32_t nblocks = block_matrix_height;

        // nz blocks placed randomly
        bsr_matrix<float> bsr(M, K, R, C, nblocks, RAND);
        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        char buf[50];
        size_t n = sprintf(buf, "profile_case_sparse_fill_random_R%i_C%d", R, C);
        std::string test_name(buf, n);
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, test_name);
    }

} // namespace profiling_suite
