#pragma once

#include <cstdint>
#include "include_me.hpp"

using namespace tt;

using CoreSpec = std::variant<CoreCoord, CoreRange, CoreRangeSet>;



namespace bsr_test_suite {
    // I want the registry and the alias to come with this header file. 
    // Maybe I do need to add declarations :(
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_big_zero_rows_more();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_big_zero_rows();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_big_diag();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_random();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_random_tall_blocks();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_random_wide_blocks();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_random_wide();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_random_tall();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_checkerboard();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_big_random();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_big_dense();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_basic();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_2_blocks();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_2_blocks_diag();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_diag_times_wide();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_diag_first_row_times_wide();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_diag_second_row_times_wide();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_simplified_times_wide();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_simplified_tall_times_wide();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_simplified_tall_times_wide_v2();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_big_block_times_wide();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_2_blocks_fill_row();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_2_blocks_fill_col();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_4_blocks();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_off_diag_first_row();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_diag_first_row();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_nonsquare();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_2_blocks_nonsquare();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_nonsquare_tall();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_2_blocks_nonsquare_tall();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_nonsquare_diag_blocks();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_nonsquare_diag_first_row();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_nonsquare_off_diag_first_row();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_simplified_off_diag_first_row();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_nonsquare_diag_tall();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_many_nonsquare();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_nonsquare_stacked();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_2_blocks_col();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_2_blocks_row_simplified();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_2_blocks_col_simplified();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_dense();
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_many_empty_rows();

    using TestFunctionPtr = std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> (*)();

    static TestFunctionPtr TestRegistry[] = {
        test_basic, // 0
        test_2_blocks, // 1
        test_2_blocks_col, // 2
        test_2_blocks_col_simplified, // 3
        test_2_blocks_row_simplified, // 4
        test_2_blocks_nonsquare, // 5
        test_many_nonsquare, // 6
        test_nonsquare_diag_blocks, // 7
        test_nonsquare_tall, // 8
        test_2_blocks_nonsquare_tall, // 9
        test_nonsquare, // 10
        test_nonsquare_diag_tall, // 11
        test_nonsquare_stacked, // 12
        test_nonsquare_diag_first_row, // 13
        test_nonsquare_off_diag_first_row, // 14
        test_simplified_off_diag_first_row, // 15
        test_2_blocks_diag, // 16
        test_off_diag_first_row, // 17
        test_diag_first_row, // 18
        test_2_blocks_fill_col, // 19
        test_2_blocks_fill_row, // 20
        test_4_blocks, // 21
        test_diag_times_wide, // 22
        test_simplified_times_wide, // 23
        test_simplified_tall_times_wide, // 24
        test_simplified_tall_times_wide_v2, // 25
        test_big_block_times_wide, // 26
        test_diag_first_row_times_wide, // 27
        test_diag_second_row_times_wide, // 28
        test_big_diag, // 29
        test_checkerboard, // 30
        test_random, // 31
        test_random_wide, // 32
        test_random_tall, // 33
        test_random_tall_blocks, // 34
        test_random_wide_blocks, // 35
        test_big_zero_rows, // 36
        test_big_zero_rows_more, // 37
        test_dense, // 38
        test_many_empty_rows, // 39
    };

    // this case should expose the performance difference between the naive and new versions
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_many_empty_rows() {
        // matmul params setup
        uint32_t M = 8192;
        uint32_t N = 256;
        uint32_t K = 256;
        // block params setup
        uint32_t R = 256;
        uint32_t C = 256;
        uint32_t nblocks = 1;
        uint32_t block_matrix_height = M / R;

        // all nz on one row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_ROW, RAND);
        dense_matrix<float> dense(K, N, RAND);

        

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_many_empty_rows");
    }    

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

    // TODO: this example is prime for a next-steps sort of thinking for the impl. What to do 
    //          with nblocks > ncores?
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_dense_2_blocks_per_core() {
        // matmul params setup

        uint32_t num_cores = 64; // given our wormhole arch, this is how many Tensix tiles we have

        uint32_t M = 1024;
        uint32_t N = 128;
        uint32_t K = 128;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 32;
        uint32_t nblocks = 128;
        uint32_t block_matrix_height = M / R;
        
        bsr_matrix<float> bsr(M, K, R, C, nblocks, RAND);
        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_dense_2_blocks_per_core");
    }

    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> test_dense() {
        // One output block per core!
        uint32_t num_cores = 64; // given our wormhole arch, this is how many Tensix tiles we have

        uint32_t M = 1024;
        uint32_t N = 256;
        uint32_t K = 64;
        // block params setup
        uint32_t R = 32;
        uint32_t C = 32;
        uint32_t nblocks = 64; 
        uint32_t block_matrix_height = M / R;
        
        bsr_matrix<float> bsr(M, K, R, C, nblocks, RAND);
        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_dense");
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
