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

#include <tuple>
#include <filesystem> // for emitting test output

#include "bsr_matrix.hpp"
#include "bmm_op.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;

uint32_t _get_maximum_block_dim_with_NoC_args(int32_t block_dim, int32_t in0_block_w, int32_t num_tiles_in_NoC_args) {
    int32_t num_available_tiles_in_SRAM = 400; // as provided by TT code. roughly: SRAM size in bytes divided by tile size in bytes
    num_available_tiles_in_SRAM -= num_tiles_in_NoC_args;
    int32_t other_dim = (num_available_tiles_in_SRAM - 2 * in0_block_w * block_dim) / (2 * in0_block_w + block_dim);
    if (other_dim > 0) {
        return other_dim;
    }
    return 0;
}

std::tuple<bsr_matrix, dense_matrix, std::string> test_basic() {
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
    bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_ROW, NO_RAND);
    dense_matrix<float> dense(K, N, 2.0f); // scaling matrix
    for (int i = 0; i < K; i++){
        for (int j = 0; j < N; j++) {
            if (i != j)
                dense.data[i*N + j] = 0.0f;
        }
    }
    return std::make_tuple(bsr, dense, "test_basic");
} 


// TODO: put this in its own file somewhere and include it. 
void bsr_spmm_multicore_reuse(
    bsr_matrix<bfloat16>& a,
    dense_matrix<bfloat16>& b,
    dense_matrix<bfloat16>& output,
    bool bcast_batch,
    uint32_t nnz_blocks,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t R,
    uint32_t C,
    uint32_t B,
    IDevice* device,
    bool verbose = false) {

    CommandQueue& cq = device->command_queue();
    Program program{};

    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    uint32_t single_tile_size = detail::TileSize(cb_data_format);
    // uint32_t single_tile_size = 2 * 1024;

    tt::DataFormat col_indices_data_format = tt::DataFormat::Int32;
    uint32_t col_indices_single_tile_size = detail::TileSize(col_indices_data_format);

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    uint32_t Mt = M / TILE_HEIGHT;
    uint32_t Kt = K / TILE_WIDTH;
    uint32_t Nt = N / TILE_WIDTH;

    uint32_t Rt = R / TILE_HEIGHT;
    uint32_t Ct = C / TILE_WIDTH;




    // Get large matmul params
    //
    // auto matmul_params = bmm_op_utils::get_large_matmul_params(Mt, Nt, num_cores_y, num_cores_x, in0_block_w);
    // uint32_t per_core_M = std::get<0>(matmul_params);
    // uint32_t per_core_N = std::get<1>(matmul_params);
    // uint32_t out_subblock_h = std::get<2>(matmul_params);
    // uint32_t out_subblock_w = std::get<3>(matmul_params);
    //
    // NAIVE: these should adapt to per core workload later. So we have to understand the util function and why it works!
    //          short idea: let the tt block size be the nz block size, then take the largest of the 20 subblock choices which fits.
    //          what breaks when in0_block_w = 2??
    uint32_t per_core_M = Rt;
    uint32_t in0_block_w = Ct;

    int32_t num_tiles_for_col_indices = (col_indices_single_tile_size - 1 + sizeof(int) * nnz_blocks) / col_indices_single_tile_size;
    uint32_t per_core_N = _get_maximum_block_dim_with_NoC_args(per_core_M, in0_block_w, num_tiles_for_col_indices);
    per_core_N = std::min(per_core_N, Ct); // TODO: this is a bit contrived and will always be Ct. idk what to do about it tho

    uint32_t out_subblock_h = 1; // TODO: figure out the correctness issue here.
    uint32_t out_subblock_w = 1;

    if (verbose) {
        log_info(tt::LogVerif, " -- Metalium Core Sizing --");
        log_info(
            tt::LogVerif,
            " -- per_core_M= {} -- per_core_N= {} -- out_subblock_h= {} -- out_subblock_w= {} --",
            per_core_M,
            per_core_N,
            out_subblock_h,
            out_subblock_w);
    }

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

    uint32_t in0_num_subblocks = (per_core_M / out_subblock_h);
    uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;
    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;

    uint32_t in1_num_subblocks = (per_core_N / out_subblock_w);
    uint32_t in1_block_num_tiles = out_subblock_w * in0_block_w * in1_num_subblocks;
    uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;

    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

    /*
     * Multi-Core prep
     */
    // auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    // uint32_t num_cores_x = compute_with_storage_grid_size.x;
    // uint32_t num_cores_y = compute_with_storage_grid_size.y;
    //
    // NAIVE: these should adapt to per core workload later.
    // uint32_t num_blocks_y = Mt / per_core_M;
    uint32_t num_blocks_y = M / R; // block_matrix_height, how many blocks tall the input matrix is.
    uint32_t num_blocks_x = Nt / per_core_N;
    uint32_t num_blocks_total = num_blocks_y * num_blocks_x;
    TT_ASSERT(num_blocks_total <= num_cores_x * num_cores_y);
    CoreRangeSet all_cores(
        tt::tt_metal::num_cores_to_corerangeset(num_blocks_x * num_blocks_y, compute_with_storage_grid_size, true));


    if (verbose) {
        log_info(tt::LogVerif, " -- Metalium Grid Sizing --");
        log_info(
            tt::LogVerif,
            " -- Mt= {} -- Nt= {} -- num_blocks_x= {} -- num_blocks_y= {} -- num_cores_x={} -- num_cores_y={} --",
            Mt,
            Nt,
            num_blocks_x,
            num_blocks_y,
            num_cores_x,
            num_cores_y);
    }

    //////////////////////////////////////////////////
    /*
     * Create DRAM Buffers for input and output vectors
     * Writing data from input vectors to source buffers
     */

    uint32_t dram_buffer_A_size =
        single_tile_size * Rt * Ct * nnz_blocks;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    uint32_t dram_buffer_B_size =
        single_tile_size * Nt * Kt;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    uint32_t dram_buffer_C_size =
        single_tile_size * Mt * Nt;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels

    // In fact let's pad this to fill a tile at least
    uint32_t dram_buffer_D_size =
        sizeof(int) * nnz_blocks; //
    dram_buffer_D_size = col_indices_single_tile_size * ((col_indices_single_tile_size - 1 + dram_buffer_D_size) / (col_indices_single_tile_size));
    tt_metal::InterleavedBufferConfig dram_config_A{
        .device = device,
        .size = dram_buffer_A_size,
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    tt_metal::InterleavedBufferConfig dram_config_B{
        .device = device,
        .size = dram_buffer_B_size,
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    tt_metal::InterleavedBufferConfig dram_config_C{
        .device = device,
        .size = dram_buffer_C_size,
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM};


    tt_metal::InterleavedBufferConfig dram_config_D{
        .device = device,
        .size = dram_buffer_D_size,
        .page_size = col_indices_single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    auto src0_dram_buffer = CreateBuffer(dram_config_A);
    auto src1_dram_buffer = CreateBuffer(dram_config_B);
    auto dst_dram_buffer = CreateBuffer(dram_config_C);
    auto column_indices_dram_buffer = CreateBuffer(dram_config_D);
    uint32_t src0_addr = src0_dram_buffer->address();
    uint32_t src1_addr = src1_dram_buffer->address();
    uint32_t dst_addr = dst_dram_buffer->address();
    uint32_t column_indices_addr = column_indices_dram_buffer->address();

    // logically we want this but the cpu can't directly manage device memory like this
    // memset((void *)dst_addr, 0, sizeof(bfloat16) * M * N);
    //
    // marty's suggestion: use the SFPU to load a CB with zeros, then NoC to DRAM.
    /*
     * Config of Circular Buffer in the device L1
     * input tiles count is = 2 because it's single tile process, and double-buffer
     */

    // NAIVE: for this first, naive impl, keep all the CBs the same size, the maximum size
    uint32_t src0_cb_index = CBIndex::c_0;  // 0
    CircularBufferConfig cb_src0_config = CircularBufferConfig(in0_CB_size, {{src0_cb_index, cb_data_format}})
                                              .set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t src1_cb_index = CBIndex::c_1;  // 1
    CircularBufferConfig cb_src1_config = CircularBufferConfig(in1_CB_size, {{src1_cb_index, cb_data_format}})
                                              .set_page_size(src1_cb_index, single_tile_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);



    uint32_t output_cb_index = tt::CBIndex::c_16;
    uint32_t interm0_cb_index = tt::CBIndex::c_24;
    std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec{
        {output_cb_index, cb_data_format}, {interm0_cb_index, cb_data_format}};
        CircularBufferConfig cb_output_config = CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
        .set_page_size(output_cb_index, single_tile_size)
        .set_page_size(interm0_cb_index, single_tile_size);
        auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    uint32_t column_indices_cb_index = CBIndex::c_2;  // 2
    CircularBufferConfig cb_column_indices_config = CircularBufferConfig(
        dram_buffer_D_size, {{column_indices_cb_index, tt::DataFormat::Int32}})
                                                .set_page_size(column_indices_cb_index, col_indices_single_tile_size);
    auto cb_column_indices = tt_metal::CreateCircularBuffer(program, all_cores, cb_column_indices_config);

     /*
     * Compile time arguments
     */
    bool src0_is_dram = src0_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool src1_is_dram = src1_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool Noc_args_is_dram = column_indices_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_is_dram, (uint32_t)src1_is_dram, (uint32_t)Noc_args_is_dram};

    bool dst_is_dram = dst_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {(uint32_t)dst_is_dram};

    /*
     * Create Kernels (Reader, Writer, Compute)
     */
    // Create reader and writer kernels per core
    auto reader_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/rahmy/block_spmm/kernels/dataflow/reader_block.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = reader_compile_time_args});

    auto writer_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/rahmy/block_spmm/kernels/dataflow/writer_block.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = writer_compile_time_args});

    // Create compute kernel
    auto mm_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/rahmy/block_spmm/kernels/compute/bmm.cpp",
        all_cores,
        tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = {}});

    uint32_t num_nnz_blocks_read = 0;
    uint32_t num_blocks_read = 0;
    for (int output_idx_y = 0; output_idx_y < num_blocks_y; output_idx_y++) {
        for (int output_idx_x = 0; output_idx_x < num_blocks_x; output_idx_x++) {
            // TODO: test and reason
            int core_idx_x = num_blocks_read % num_cores_x;
            int core_idx_y = num_blocks_read / num_cores_x;
            CoreCoord core = {(std::size_t)core_idx_x, (std::size_t)core_idx_y};

            // What is the translation between core index and output tiles?

            uint32_t num_blocks = a.indptr[output_idx_y + 1] - a.indptr[output_idx_y];
            // Write runtime args to device
            std::vector<uint32_t> mm_reader_args = {
                (std::uint32_t)src0_dram_buffer->address(),     // in0_tensor_addr
                (std::uint32_t)output_idx_y * Rt * Ct,          // in0_tensor_start_tile_id
                (std::uint32_t)1,                               // in0_tensor_stride_w
                (std::uint32_t)Ct,                              // in0_tensor_stride_h

                (std::uint32_t)in0_block_w,               // in0_block_w
                (std::uint32_t)per_core_M,                // in0_block_h
                (std::uint32_t)in0_block_w * per_core_M,  // in0_block_num_tiles

                (std::uint32_t)src1_dram_buffer->address(),  // in1_tensor_addr
                (std::uint32_t)per_core_N * output_idx_x,    // in1_tensor_start_tile_id
                (std::uint32_t)1,                            // in1_tensor_stride_w
                (std::uint32_t)Nt,                           // in1_tensor_stride_h

                (std::uint32_t)per_core_N,                // in1_block_w
                (std::uint32_t)in0_block_w,               // in1_block_h
                (std::uint32_t)per_core_N * in0_block_w,  // in1_block_num_tiles

                (std::uint32_t) a.indptr[output_idx_y],    // col indices start of row
                (std::uint32_t) a.indptr[output_idx_y + 1],// col indices end of row
                (std::uint32_t) output_idx_y, // row index into bsr matrix

                (std::uint32_t)column_indices_dram_buffer->address(), // NoC args, column indices
            };

            std::vector<uint32_t> writer_args = {
                (std::uint32_t)dst_dram_buffer->address(),                                  // out_buffer_addr
                (std::uint32_t)output_idx_x * per_core_N + output_idx_y * per_core_M * Nt,  // out_tensor_start_tile_id
                (std::uint32_t)1,                                                           // out_tensor_stride_w
                (std::uint32_t)Nt,                                                          // out_tensor_stride_h
                (std::uint32_t)out_subblock_w,       // out_tensor_next_subblock_stride_w
                (std::uint32_t)out_subblock_h * Nt,  // out_tensor_next_subblock_stride_h

                (std::uint32_t)out_subblock_w,                     // out_subblock_w
                (std::uint32_t)out_subblock_h,                     // out_subblock_h
                (std::uint32_t)(out_subblock_w * out_subblock_h),  // out_subblocks_w * out_subblocks_h
                (std::uint32_t)(per_core_N / out_subblock_w),      // out_num_subblocks_w
                (std::uint32_t)(per_core_M / out_subblock_h),      // out_num_subblocks_h

                (std::uint32_t)Mt * Nt,  // MtNt
                (std::uint32_t)B,        // batch
                (std::uint32_t)1*(num_blocks != 0) // nonzero, tells writer whether it has work to do
            };

            std::vector<uint32_t> compute_args = {
                (std::uint32_t)in0_block_w,
                (std::uint32_t)in0_num_subblocks,
                (std::uint32_t)in0_block_w * per_core_M, // in0_block_num_tiles
                (std::uint32_t)in0_subblock_num_tiles,
                (std::uint32_t)in1_num_subblocks,
                (std::uint32_t)in1_block_num_tiles,
                (std::uint32_t)in1_per_core_w,
                (std::uint32_t)num_blocks, // num_blocks
                (std::uint32_t)out_subblock_h,
                (std::uint32_t)out_subblock_w,
                (std::uint32_t)out_subblock_num_tiles,
                (std::uint32_t)B
            };

            tt_metal::SetRuntimeArgs(program, reader_id, core, mm_reader_args);
            tt_metal::SetRuntimeArgs(program, writer_id, core, writer_args);
            tt_metal::SetRuntimeArgs(program, mm_kernel_id, core, compute_args);
            num_blocks_read++;
            if (num_blocks > 0)
                num_nnz_blocks_read++;
        }
    }

    if (verbose){
        log_info(tt::LogVerif, " -- Runtime Args set --");
        log_info(
            tt::LogVerif,
            " -- nnz blocks read= {}",
            num_nnz_blocks_read);
    }


    EnqueueWriteBuffer(cq, src0_dram_buffer, a.data.data(), false);
    EnqueueWriteBuffer(cq, src1_dram_buffer, b.data.data(), false);
    EnqueueWriteBuffer(cq, column_indices_dram_buffer, a.indices.data(), true);

    if (verbose)
        log_info(tt::LogVerif, " -- All data moved to DRAM --");

    EnqueueProgram(cq, program, false);

    if (verbose)
        log_info(tt::LogVerif, " -- Program enqueued --");

    EnqueueReadBuffer(cq, dst_dram_buffer, output.data.data(), true);

    Finish(cq);

}

std::pair<std::string, float> run_test(
    bsr_matrix<bfloat16>& a,
    dense_matrix<bfloat16>& b,
    std::string test_name,
    bool verbose = false,
    bool emit_output = false) {

    /*
    Requires: a, b to be initialized on CPU
    Modifies: can modifiy output files and log data 
    Effects: 

    Returns the PCC between the sequential matmul of a and b and the multicore matmul of a and b. 
    */

    // TODO: should we put all this in a try-catch block?

    // device setup
    constexpr int device_id = 0;
    IDevice* device = CreateDevice(device_id);


    // matmul params setup
    uint32_t M = a.H;
    uint32_t N = b.W;
    uint32_t K = a.W;
    // block params setup
    uint32_t R = a.R;
    uint32_t C = a.C;
    uint32_t nblocks = a.nblocks;
    uint32_t block_matrix_height = M / R;

    uint32_t Rt = R / TILE_HEIGHT;
    uint32_t Ct = C / TILE_WIDTH;


    // initialize output_data
    dense_matrix<float> tmp(M, N);
    dense_matrix<bfloat16> output = tmp.bfloat16_cast();

    // run sequential spmm
    dense_matrix<bfloat16> golden = a.spmm_bfloat16(b);

    // tilize input data
    tilize(bsr_bfloat16.data, R, C);
    tilize(dense_bfloat16.data, K, N);

    // run bsr_spmm_multicore_reuse
    bsr_spmm_multicore_reuse(a, b, output, false, nblocks, M, N, K, R, C, 1, device, verbose);

    // untile output data
    untilize(output.data, M, N);

    float pearson = check_bfloat16_vector_pcc(golden.data, output.data);

    if (emit_output) {
        // let's write the output vectors to a file
        std::string local_path = "/home/user/tt-metal/tt_metal/programming_examples/rahmy/block_spmm/" + test_name;

        // TODO: will a failure here get caught in the larger try catch block?
        fs::create_directory(local_path);
        std::string output_file = local_path + "/output.txt";
        std::ofstream out(output_file);
        if (!out.is_open()) {
            TT_THROW("Failed to open output file: {}", output_file);
        }
        for (size_t i = 0; i < output.data.size(); i++) {
            out << output.data[i].to_float() << "\n";
        }
        out.close();

        log_info(tt::LogVerif, "Output written to {}", output_file);
        // let's write the golden vector to a file
        std::string golden_file = local_path + "/golden.txt";
        std::ofstream golden_out(golden_file);
        if (!golden_out.is_open()) {
            TT_THROW("Failed to open golden file: {}", golden_file);
        }
        for (size_t i = 0; i < golden.data.size(); i++) {
            golden_out << golden.data[i].to_float() << "\n";
        }
        golden_out.close();
    }    

    CloseDevice(device);

    return pearson; 

}

int main(int argc, char** argv) {
    /*
    1. Reserve a vector of <test_name, PCC> pairs. 
    2. call run_test(test_func(), verbose, emit_output) for each test, adding to the vector
    3. iter over vector and pretty print passes and fails to the console
    */

    std::vector<std::pair<std::string, float>> test_results;

    test_results.push_back(run_test(std::tie(test_basic()), false, false));

    uint32_t count = 0;
    for (auto &p : test_results) {
        std::cout << "Test #" << count << ": " << p.first << ' ';
        std::string result = p.second > 0.99 ? "PASS ✅" : "FAIL ❌"; 
        count++;
    }
    /*
    For later: add args for only running certain tests
    */

}