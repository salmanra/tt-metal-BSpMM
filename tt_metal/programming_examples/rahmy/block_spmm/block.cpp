


#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/test_tiles.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/tt_metal.hpp>
// #include <matmul_common/bmm_op.hpp>
#include <tt-metalium/tilize_untilize.hpp>

#include "bsr_matrix.hpp"
#include "bmm_op.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;

void test_sequential_bsr_spmm() {
    bsr_matrix<float> bsr(2048, 2048, 128, 128, 3, RAND);
    dense_matrix<float> dense(2048, 16, RAND);

    bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
    dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

    dense_matrix<bfloat16> expected = bsr_bfloat16.to_dense().gemm_bfloat16(dense_bfloat16);
    dense_matrix<bfloat16> result = bsr_bfloat16.spmm_bfloat16(dense_bfloat16);

    float pearson = check_bfloat16_vector_pcc(expected.data, result.data);
    log_info(tt::LogVerif, "BSR vs Golden -- PCC = {}", pearson);
    TT_FATAL(pearson > 0.99, "PCC not high enough. Result PCC: {}, Expected PCC: 0.99", pearson);
}

void bsr_spmm_multicore_reuse(
    std::bsr_bfloat16<bfloat16>& a,
    std::dense_matrix<bfloat16>& b,
    std::dense_matrix<bfloat16>& output,
    bool bcast_batch,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t R,
    uint32_t C,
    uint32_t B,
    IDevice* device) {

    // compare to multicore_reuse
    /*
    * lines 68-160 are almost identical, except that the numblocks compute kernel arg becomes a runtime arg
    * This means that CB size, block size, and subblock size can all be used the same way as the dense example.
    *
    * DRAM setup is also almost identical, except we need to allocate the correct number of bytes for the BSR matrix.
    *
    * comptime args setup is almost identical
    *
    * Now the fun part: runtime args!
    *   1. in0_start_tile_id is the tile which starts the row of interest of A
    *   2. in1               is the tile which starts the column of interest of B
    *   3. in0_next_block_stride. Are we using this?
    *       I have overloaded the word "block" and now I'm reaping the rotten melons.
    *       block should be the same, with width 2 and height per_core_M.
    *       So we need to be able to iterate over blocks within a Block like it's a dense matrix.
    *       Then we need to iterate over Blocks like it's sparse (with the col indices).
    *       My reader kernel does not think about this. Let's think about this.
    *       Number 1: The kernel should have the Block dims RxC. is each output block many Blocks, or a fraction of a Block?
    *                   Assume R=C=128. (Rt=Ct=4). block width is 2, and we have to determine per_core_M and per_core_N
    *                   Ah I see. per_core_M/N = 16. So a single input block will take a slice of many Blocks (2x16 slice of 4 4x4 Blocks)
    *                   Bad assumption! the util function decides per_core_M/N. We have to think about how that gets decided and why.
    *       Good. So how do we load a single input block of A to SRAM?
    *                 numBlocksPerblock(height) = per_core_M / Rt (== 4)
    *
    *
    */

}

int main(int argc, char** argv) {
    bool pass = true;

    if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
        TT_THROW("Test not supported w/ slow dispatch, exiting");
    }

    try {
        // Device setup
        constexpr int device_id = 0;
        IDevice* device = CreateDevice(device_id);

        // matmul params setup
        uint32_t M = 2048;
        uint32_t N = 2048;
        uint32_t K = 2048;
        // block params setup
        uint32_t R = 128;
        uint32_t C = 128;
        uint32_t nblocks = 7;

        uint32_t Rt = R / TILE_HEIGHT;
        uint32_t Ct = C / TILE_WIDTH;

        // create (or read) source data
        bsr_matrix<float> bsr(M, K, R, C, nblocks, RAND);
        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        // run golden bsr_spmm
        dense_matrix<bfloat16> golden = bsr_bfloat16.spmm_bfloat16(dense_bfloat16);

        // tilize input data
        tilize(bsr_bfloat16.data, R, C);
        tilize(dense_bfloat16.data, K, N);

        // run bsr_spmm_multicore_reuse
        bsr_spmm_multicore_reuse(bsr_bfloat16, dense_bfloat16, false, M, N, K, R, C, 1, device);

        // untile output data

        // check all close

        // close device and check for errors
        pass &= CloseDevice(device);

    }
    catch (const std::exception& e) {
        tt::log_error(tt::LogTest, "Test failed with excpetion!");
        tt::log_error(tt::LogTest, "{}", e.what());

        throw;
    }

    if (pass) {
        tt::log_info(tt::LogTest, "Test Passed");
    }
    else {
        TT_THROW("Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
