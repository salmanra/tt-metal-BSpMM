#pragma once

#include "include_me.hpp"

using namespace tt;

using CoreSpec = std::variant<CoreCoord, CoreRange, CoreRangeSet>;

namespace bsr_test_suite {

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_big_zero_rows_more() {
        // matmul params setup
        uint32_t M = 256;
        uint32_t N = 256;
        uint32_t K = 256;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 8;
        uint32_t block_matrix_height = M / R;
        std::vector<float> data(R*C*nblocks);
        for (int k = 0; k < nblocks; k++){
            for (int i = 0; i < R*C; i++){
                data[k*nblocks + i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            }
        }
        std::vector<int> indptr = {0, 4, 4, 8, 8};
        std::vector<int> indices = {0, 1, 2, 3, 0, 1, 2, 3};

        // all nz on one row
        bsr_matrix<float> bsr(data, indptr, indices, M, K, R, C, nblocks);
        dense_matrix<float> dense(K, N, RAND);

        

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_big_zero_rows_more");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_big_zero_rows() {
        // matmul params setup
        uint32_t M = 256;
        uint32_t N = 256;
        uint32_t K = 256;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 4;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_ROW, RAND);
        dense_matrix<float> dense(K, N, RAND);

        

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_big_zero_rows");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_big_diag() {
        // matmul params setup
        uint32_t M = 256;
        uint32_t N = 256;
        uint32_t K = 256;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 4;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_DIAG, RAND);
        dense_matrix<float> dense(K, N, RAND);

        

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_big_diag");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_random() {
        // matmul params setup
        uint32_t M = 256;
        uint32_t N = 256;
        uint32_t K = 256;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 4;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, RAND);
        dense_matrix<float> dense(K, N, RAND);

        

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_random");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_random_tall_blocks() {
        // matmul params setup
        uint32_t M = 256;
        uint32_t N = 256;
        uint32_t K = 256;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 64;
        uint32_t nblocks = 4;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, RAND);
        dense_matrix<float> dense(K, N, RAND);

        

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_random_tall_blocks");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_random_wide_blocks() {
        // matmul params setup
        uint32_t M = 256;
        uint32_t N = 256;
        uint32_t K = 256;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 32;
        uint32_t nblocks = 4;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, RAND);
        dense_matrix<float> dense(K, N, RAND);

        

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_random_wide_blocks");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_random_wide() {
        // matmul params setup
        uint32_t M = 256;
        uint32_t N = 1024;
        uint32_t K = 256;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 8;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, RAND);
        dense_matrix<float> dense(K, N, RAND);

        

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_random_wide");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_random_tall() {
        // matmul params setup
        uint32_t M = 1024;
        uint32_t N = 256;
        uint32_t K = 256;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 8;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, RAND);
        dense_matrix<float> dense(K, N, RAND);

        

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_random_tall");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_checkerboard() {
        // matmul params setup
        uint32_t M = 256;
        uint32_t N = 256;
        uint32_t K = 256;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 8;
        uint32_t block_matrix_height = M / R;
        
        std::vector<float> data(R*C*nblocks);
        for (int k = 0; k < nblocks; k++){
            for (int i = 0; i < R*C; i++){
                data[k*nblocks + i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            }
        }
        std::vector<int> indptr = {0, 2, 4, 6, 8};
        std::vector<int> indices = {0, 2, 1, 3, 0, 2, 1, 3};

        bsr_matrix<float> bsr(data, indptr, indices, M, K, R, C, nblocks);
        dense_matrix<float> dense(K, N, RAND);

        

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_checkerboard");
    }

    // TODO: see below. It's an output block thing, not input block
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_big_random() {
        // matmul params setup
        uint32_t M = 1024;
        uint32_t N = 1024;
        uint32_t K = 1024;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 32;
        uint32_t block_matrix_height = M / R;
        
        bsr_matrix<float> bsr(M, K, R, C, nblocks, RAND);
        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_big_random");
    }

    // TODO: this example is prime for a next-steps sort of thinking for the impl. What to do 
    //          with nblocks > ncores?
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_big_dense() {
        // matmul params setup
        uint32_t M = 1024;
        uint32_t N = 1024;
        uint32_t K = 1024;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 256;
        uint32_t block_matrix_height = M / R;
        
        bsr_matrix<float> bsr(M, K, R, C, nblocks, RAND);
        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_big_dense");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_basic() {
        // matmul params setup
        uint32_t M = 64;
        uint32_t N = 64;
        uint32_t K = 64;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 1;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_ROW, RAND);

        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_basic");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_2_blocks() {
        // matmul params setup
        uint32_t M = 64;
        uint32_t N = 64;
        uint32_t K = 128;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 2;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_ROW, RAND);
        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_2_blocks");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_2_blocks_diag() {
        // matmul params setup
        uint32_t M = 128;
        uint32_t N = 64;
        uint32_t K = 128;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 2;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_DIAG, RAND);
        dense_matrix<float> dense(K, N, RAND);

        

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_2_blocks_diag");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_diag_times_wide() {
        // matmul params setup
        uint32_t M = 128;
        uint32_t N = 128;
        uint32_t K = 128;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 2;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_DIAG, RAND);
        dense_matrix<float> dense(K, N, RAND);

        

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_diag_times_wide");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_diag_first_row_times_wide() {
        // matmul params setup
        uint32_t M = 64;
        uint32_t N = 128;
        uint32_t K = 128;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 1;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_ROW, RAND);
        dense_matrix<float> dense(K, N, RAND);

        

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_diag_first_row_times_wide");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_diag_second_row_times_wide() {
        // matmul params setup
        uint32_t M = 64;
        uint32_t N = 128;
        uint32_t K = 128;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 1;
        uint32_t block_matrix_height = M / R;
        
        std::vector<float> data(R*C*nblocks);
        for (int k = 0; k < nblocks; k++){
            for (int i = 0; i < R*C; i++){
                data[k*nblocks + i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            }
        }
        std::vector<int> indptr = {0, 1};
        std::vector<int> indices = {1};

        bsr_matrix<float> bsr(data, indptr, indices, M, K, R, C, nblocks);
        dense_matrix<float> dense(K, N, RAND);

        

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_diag_second_row_times_wide");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_simplified_times_wide() {
        // matmul params setup
        uint32_t M = 32;
        uint32_t N = 64;
        uint32_t K = 32;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 32;
        uint32_t nblocks = 1;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_ROW, RAND);
        dense_matrix<float> dense(K, N, RAND);

        

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_simplified_times_wide");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_simplified_tall_times_wide() {
        // matmul params setup
        uint32_t M = 64;
        uint32_t N = 64;
        uint32_t K = 32;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 32;
        uint32_t nblocks = 1;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_ROW, RAND);
        dense_matrix<float> dense(K, N, RAND);

        

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_simplified_tall_times_wide");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_simplified_tall_times_wide_v2() {
        // matmul params setup
        uint32_t M = 64;
        uint32_t N = 64;
        uint32_t K = 32;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 32;
        uint32_t nblocks = 1;
        uint32_t block_matrix_height = M / R;

        std::vector<float> data(R*C*nblocks);
        for (int k = 0; k < nblocks; k++){
            for (int i = 0; i < R*C; i++){
                data[k*nblocks + i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            }
        }
        std::vector<int> indptr = {0, 0, 1};
        std::vector<int> indices = {0};

        // all nz on one row
        bsr_matrix<float> bsr(data, indptr, indices, M, K, R, C, nblocks);

        dense_matrix<float> dense(K, N, RAND);

        

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_simplified_tall_times_wide_v2");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_big_block_times_wide() {
        // matmul params setup
        uint32_t M = 64;
        uint32_t N = 1024;
        uint32_t K = 64;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 1;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_ROW, RAND);
        dense_matrix<float> dense(K, N, RAND);

        

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_big_block_times_wide");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_2_blocks_fill_row() {
        // matmul params setup
        uint32_t M = 128;
        uint32_t N = 64;
        uint32_t K = 128;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 2;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_ROW, RAND);
        dense_matrix<float> dense(K, N, RAND);

        

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_2_blocks_fill_row");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_2_blocks_fill_col() {
        // matmul params setup
        uint32_t M = 128;
        uint32_t N = 64;
        uint32_t K = 128;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 2;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_COL, RAND);
        dense_matrix<float> dense(K, N, RAND);

        

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_2_blocks_fill_col");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_4_blocks() {
        // matmul params setup
        uint32_t M = 128;
        uint32_t N = 64;
        uint32_t K = 128;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 4;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_COL, RAND);
        dense_matrix<float> dense(K, N, RAND);

        

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_4_blocks");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_off_diag_first_row() {
        // matmul params setup
        uint32_t M = 64;
        uint32_t N = 64;
        uint32_t K = 128;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 1;
        uint32_t block_matrix_height = M / R;

        std::vector<float> data(R*C*nblocks);
        for (int k = 0; k < nblocks; k++){
            for (int i = 0; i < R*C; i++){
                data[k*nblocks + i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            }
        }
        std::vector<int> indptr = {0, 1};
        std::vector<int> indices = {1};

        bsr_matrix<float> bsr(data, indptr, indices, M, K, R, C, nblocks);

        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_off_diag_first_row");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_diag_first_row() {
        // matmul params setup
        uint32_t M = 64;
        uint32_t N = 64;
        uint32_t K = 128;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 1;
        uint32_t block_matrix_height = M / R;

        std::vector<float> data(R*C*nblocks);
        for (int k = 0; k < nblocks; k++){
            for (int i = 0; i < R*C; i++){
                data[k*nblocks + i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            }
        }
        std::vector<int> indptr = {0, 1};
        std::vector<int> indices = {0};

        bsr_matrix<float> bsr(data, indptr, indices, M, K, R, C, nblocks);

        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_diag_first_row");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_nonsquare() {
        // matmul params setup
        uint32_t M = 32;
        uint32_t N = 32;
        uint32_t K = 64;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 64;
        uint32_t nblocks = 1;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_ROW, RAND);

        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_nonsquare");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_2_blocks_nonsquare() {
        // matmul params setup
        uint32_t M = 32;
        uint32_t N = 32;
        uint32_t K = 128;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 64;
        uint32_t nblocks = 2;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_ROW, RAND);

        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_2_blocks_nonsquare");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_nonsquare_tall() {
        // matmul params setup
        uint32_t M = 64;
        uint32_t N = 32;
        uint32_t K = 32;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 32;
        uint32_t nblocks = 1;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_ROW, RAND);
        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_nonsquare_tall");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_2_blocks_nonsquare_tall() {
        // matmul params setup
        uint32_t M = 64;
        uint32_t N = 32;
        uint32_t K = 64;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 32;
        uint32_t nblocks = 2;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_ROW, RAND);
        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_2_blocks_nonsquare_tall");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_nonsquare_diag_blocks() {
        // matmul params setup
        uint32_t M = 64;
        uint32_t N = 32;
        uint32_t K = 128;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 64;
        uint32_t nblocks = 2;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_DIAG, RAND);

        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_nonsquare_diag_blocks");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_nonsquare_diag_first_row() {
        // matmul params setup
        uint32_t M = 32;
        uint32_t N = 32;
        uint32_t K = 128;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 64;
        uint32_t nblocks = 1;
        uint32_t block_matrix_height = M / R;

        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_DIAG, RAND);

        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_nonsquare_diag_first_row");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_nonsquare_off_diag_first_row() {
        // matmul params setup
        uint32_t M = 32;
        uint32_t N = 32;
        uint32_t K = 128;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 64;
        uint32_t nblocks = 1;
        uint32_t block_matrix_height = M / R;

        std::vector<float> data(R*C*nblocks);
        for (int k = 0; k < nblocks; k++){
            for (int i = 0; i < R*C; i++){
                data[k*nblocks + i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
                // data[k*nblocks + i] = i;
            }
        }
        std::vector<int> indptr = {0, 1};
        std::vector<int> indices = {1};

        bsr_matrix<float> bsr(data, indptr, indices, M, K, R, C, nblocks);

        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_nonsquare_off_diag_first_row");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_simplified_off_diag_first_row() {
        // matmul params setup
        uint32_t M = 32;
        uint32_t N = 32;
        uint32_t K = 64;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 32;
        uint32_t nblocks = 1;
        uint32_t block_matrix_height = M / R;

        std::vector<float> data(R*C*nblocks);
        for (int k = 0; k < nblocks; k++){
            for (int i = 0; i < R*C; i++){
                data[k*nblocks + i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
                // data[k*nblocks + i] = i;
            }
        }
        std::vector<int> indptr = {0, 1};
        std::vector<int> indices = {1};

        bsr_matrix<float> bsr(data, indptr, indices, M, K, R, C, nblocks);

        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_simplified_off_diag_first_row");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_nonsquare_diag_tall() {
        // matmul params setup
        uint32_t M = 128;
        uint32_t N = 32;
        uint32_t K = 64;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 32;
        uint32_t nblocks = 2;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_DIAG, RAND);
        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_nonsquare_diag_tall");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_many_nonsquare() {
        // matmul params setup
        uint32_t M = 64;
        uint32_t N = 32;
        uint32_t K = 128;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 64;
        uint32_t nblocks = 4;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_ROW, RAND);
        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_many_nonsquare");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_nonsquare_stacked() {
        // matmul params setup
        uint32_t M = 64;
        uint32_t N = 32;
        uint32_t K = 64;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 64;
        uint32_t nblocks = 2;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_ROW, RAND);
        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_nonsquare_stacked");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_2_blocks_col() {
        // matmul params setup
        uint32_t M = 128;
        uint32_t N = 64;
        uint32_t K = 64;
        // block params setup
        uint32_t R = 64;
        uint32_t C = 64;
        uint32_t nblocks = 2;
        uint32_t block_matrix_height = M / R;

        // all nz on one col
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_COL, RAND);
        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_2_blocks_col");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_2_blocks_row_simplified() {
        // matmul params setup
        uint32_t M = 32;
        uint32_t N = 32;
        uint32_t K = 64;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 32;
        uint32_t nblocks = 2;
        uint32_t block_matrix_height = M / R;

        // all nz on one col
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_ROW, RAND);
        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_2_blocks_row_simplified");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_2_blocks_col_simplified() {
        // matmul params setup
        uint32_t M = 64;
        uint32_t N = 32;
        uint32_t K = 32;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 32;
        uint32_t nblocks = 2;
        uint32_t block_matrix_height = M / R;

        // all nz on one col
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_COL, RAND);
        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_2_blocks_col_simplified");
    }

} // namespace bsr_test_suite
