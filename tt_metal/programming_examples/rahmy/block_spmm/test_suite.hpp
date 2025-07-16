#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/test_tiles.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <../matmul_common/bmm_op.hpp>
#include <tt-metalium/tilize_untilize.hpp>


#include <vector>
#include <tuple>
#include <filesystem> // for emitting test output

#include "bsr_matrix.hpp"

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
    // dense_matrix<float> dense(K, N, 2.0f); // scaling matrix
    // for (int i = 0; i < K; i++){
    //     for (int j = 0; j < N; j++) {
    //         if (i != j)
    //             dense.data[i*N + j] = 0.0f;
    //     }
    // }

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
    // dense_matrix<float> dense(K, N, 2.0f); // scaling matrix
    // for (int i = 0; i < K; i++){
    //     for (int j = 0; j < N; j++) {
    //         if (i != j)
    //             dense.data[i*N + j] = 0.0f;
    //     }
    // }


    dense_matrix<float> dense(K, N, RAND);


    bsr.pretty_print();

    bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
    dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
    return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_2_blocks");
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
    // dense_matrix<float> dense(K, N, 2.0f); // scaling matrix
    // for (int i = 0; i < K; i++){
    //     for (int j = 0; j < N; j++) {
    //         if (i != j)
    //             dense.data[i*N + j] = 0.0f;
    //     }
    // }

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
    // dense_matrix<float> dense(K, N, 2.0f); // scaling matrix
    // for (int i = 0; i < K; i++){
    //     for (int j = 0; j < N; j++) {
    //         if (i != j)
    //             dense.data[i*N + j] = 0.0f;
    //     }
    // }

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
    // dense_matrix<float> dense(K, N, 2.0f); // scaling matrix
    // for (int i = 0; i < K; i++){
    //     for (int j = 0; j < N; j++) {
    //         if (i != j)
    //             dense.data[i*N + j] = 0.0f;
    //     }
    // }

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
    // dense_matrix<float> dense(K, N, 2.0f); // scaling matrix
    // for (int i = 0; i < K; i++){
    //     for (int j = 0; j < N; j++) {
    //         if (i != j)
    //             dense.data[i*N + j] = 0.0f;
    //     }
    // }

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
    // all nz on one row
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
    // dense_matrix<float> dense(K, N, 2.0f); // scaling matrix
    // for (int i = 0; i < K; i++){
    //     for (int j = 0; j < N; j++) {
    //         if (i != j)
    //             dense.data[i*N + j] = 0.0f;
    //     }
    // }

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
    // dense_matrix<float> dense(K, N, 2.0f); // scaling matrix
    // for (int i = 0; i < K; i++){
    //     for (int j = 0; j < N; j++) {
    //         if (i != j)
    //             dense.data[i*N + j] = 0.0f;
    //     }
    // }

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
    // dense_matrix<float> dense(K, N, 2.0f); // scaling matrix
    // for (int i = 0; i < K; i++){
    //     for (int j = 0; j < N; j++) {
    //         if (i != j)
    //             dense.data[i*N + j] = 0.0f;
    //     }
    // }

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
    // dense_matrix<float> dense(K, N, 1.0f); // ID matrix
    // for (int i = 0; i < K; i++){
    //     for (int j = 0; j < N; j++) {
    //         if (i != j)
    //             dense.data[i*N + j] = 0.0f;
    //     }
    // }

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
    // dense_matrix<float> dense(K, N, 2.0f); // scaling matrix
    // for (int i = 0; i < K; i++){
    //     for (int j = 0; j < N; j++) {
    //         if (i != j)
    //             dense.data[i*N + j] = 0.0f;
    //     }
    // }

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
    // dense_matrix<float> dense(K, N, 2.0f); // scaling matrix
    // for (int i = 0; i < K; i++){
    //     for (int j = 0; j < N; j++) {
    //         if (i != j)
    //             dense.data[i*N + j] = 0.0f;
    //     }
    // }
    dense_matrix<float> dense(K, N, RAND);


    bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
    dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

    return std::make_tuple(bsr_bfloat16, dense_bfloat16, "test_2_blocks_col_simplified");
}
