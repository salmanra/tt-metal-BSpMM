


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
    * Once we figure out why the per_core sizes are what they are, we have to decide whether this task
    * of implementing BSR SpMM is more or less the task of many-matrices-times matrix in a single program.
    * ... okay.
    *
    *
    *
    *
    *//*
     * Setup program to execute along with its buffers and kernels to use
     * Core range is just single core
     */
    CommandQueue& cq = device->command_queue();
    Program program{};

    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    uint32_t single_tile_size = detail::TileSize(cb_data_format);
    // uint32_t single_tile_size = 2 * 1024;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    uint32_t Mt = M / TILE_HEIGHT;
    uint32_t Kt = K / TILE_WIDTH;
    uint32_t Nt = N / TILE_WIDTH;


    uint32_t in0_block_w = 2;

    // Get large matmul params
    auto matmul_params = bmm_op_utils::get_large_matmul_params(Mt, Nt, num_cores_y, num_cores_x, in0_block_w);
    uint32_t per_core_M = std::get<0>(matmul_params);
    uint32_t per_core_N = std::get<1>(matmul_params);
    uint32_t out_subblock_h = std::get<2>(matmul_params);
    uint32_t out_subblock_w = std::get<3>(matmul_params);

    log_info(tt::LogVerif, " -- Metalium Core Sizing --");
    log_info(
        tt::LogVerif,
        " -- per_core_M= {} -- per_core_N= {} -- out_subblock_h= {} -- out_subblock_w= {} --",
        per_core_M,
        per_core_N,
        out_subblock_h,
        out_subblock_w);

    TT_ASSERT(Mt % per_core_M == 0);
    TT_ASSERT(Nt % per_core_N == 0);
    TT_ASSERT(Kt % in0_block_w == 0);

    uint32_t in0_block_tiles = per_core_M * in0_block_w;
    uint32_t in0_CB_tiles = in0_block_tiles * 2;  // double buffer
    uint32_t in0_CB_size = in0_CB_tiles * single_tile_size;
    uint32_t in1_block_tiles = per_core_N * in0_block_w;
    uint32_t in1_CB_tiles = in1_block_tiles * 2;  // double buffer
    uint32_t in1_CB_size = in1_CB_tiles * single_tile_size;
    uint32_t out_block_tiles = per_core_M * per_core_N;
    uint32_t out_CB_tiles = out_block_tiles;  // No double buffer
    uint32_t out_CB_size = out_CB_tiles * single_tile_size;

    // Compute kernel compile time args
    // TODO: make this a variable runtime arg
    uint32_t num_blocks = (Kt / in0_block_w);

    uint32_t in0_num_subblocks = (per_core_M / out_subblock_h);
    uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;
    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;

    uint32_t in1_num_subblocks = (per_core_N / out_subblock_w);
    uint32_t in1_block_num_tiles = out_subblock_w * in0_block_w * in1_num_subblocks;
    uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;

    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;


    // TODO: compute kernel runtime args

    /*
     * Multi-Core prep
     */
    // auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    // uint32_t num_cores_x = compute_with_storage_grid_size.x;
    // uint32_t num_cores_y = compute_with_storage_grid_size.y;

    uint32_t num_blocks_y = Mt / per_core_M;
    uint32_t num_blocks_x = Nt / per_core_N;
    uint32_t num_blocks_total = num_blocks_y * num_blocks_x;
    TT_ASSERT(num_blocks_total <= num_cores_x * num_cores_y);
    CoreRangeSet all_cores(
        tt::tt_metal::num_cores_to_corerangeset(num_blocks_x * num_blocks_y, compute_with_storage_grid_size, true));

    log_info(tt::LogVerif, " -- Metalium Grid Sizing AFTER --");
    log_info(
        tt::LogVerif,
        "Mt= {} -- Nt= {} -- num_blocks_x= {} -- num_blocks_y= {} --",
        Mt,
        Nt,
        num_blocks_x,
        num_blocks_y);

    //////////////////////////////////////////////////
    /*
     * Create DRAM Buffers for input and output vectors
     * Writing data from input vectors to source buffers
     */

     // TODO: correct DRAM buffer sizing (dense matrices are the same as old example, bsr is different)


         /*
     * Config of Circular Buffer in the device L1
     * input tiles count is = 2 because it's single tile process, and double-buffer
     */

     // TODO: for this first, naive impl, keep all the CBs the same size, the maximum size


     // TODO: create kernel objects

     // TODO: all the runtime args :)

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
