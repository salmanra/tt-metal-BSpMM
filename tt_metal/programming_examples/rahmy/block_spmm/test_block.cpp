#include "test_suite.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;

using TestFunctionPtr = std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> (*)();

TestFunctionPtr TestRegistry[] = {
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
};


struct TestResult {
    std::string test_name;
    float pearson;
    bool all_close;
};

struct TestSignature{
    std::string test_name;
    std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> (*function_ptr)();
    bool emit_output;
    bool verbose;
};


uint32_t _get_maximum_block_dim_with_NoC_args(int32_t block_dim, int32_t in0_block_w, int32_t num_tiles_in_NoC_args) {
    int32_t num_available_tiles_in_SRAM = 400; // as provided by TT code. roughly: SRAM size in bytes divided by tile size in bytes
    num_available_tiles_in_SRAM -= num_tiles_in_NoC_args;
    int32_t other_dim = (num_available_tiles_in_SRAM - 2 * in0_block_w * block_dim) / (2 * in0_block_w + block_dim);
    if (other_dim > 0) {
        return other_dim;
    }
    return 0;
}


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
    per_core_N = std::min(std::min(per_core_N, Ct), Nt); // TODO: this is a bit contrived. idk what to do about it tho

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
    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w; // this is named weird but it's correct. 

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

    uint32_t num_nnz_output_blocks = 0;
    uint32_t num_blocks_read = 0;
    for (int output_idx_y = 0; output_idx_y < num_blocks_y; output_idx_y++) {
        for (int output_idx_x = 0; output_idx_x < num_blocks_x; output_idx_x++) {
            int core_idx_x = num_blocks_read % num_cores_x;
            int core_idx_y = num_blocks_read / num_cores_x;
            CoreCoord core = {(std::size_t)core_idx_x, (std::size_t)core_idx_y};

            uint32_t num_blocks = a.indptr[output_idx_y + 1] - a.indptr[output_idx_y];
            // Write runtime args to device
            std::vector<uint32_t> mm_reader_args = {
                (std::uint32_t)src0_dram_buffer->address(),     // in0_tensor_addr
                (std::uint32_t)a.indptr[output_idx_y] * Rt * Ct,// in0_tensor_start_tile_id 
                (std::uint32_t)1,                               // in0_tensor_stride_w
                (std::uint32_t)Ct,                              // in0_tensor_stride_h

                (std::uint32_t)in0_block_w,               // in0_block_w
                (std::uint32_t)per_core_M,                // in0_block_h
                (std::uint32_t)in0_block_w * per_core_M,  // in0_block_num_tiles

                (std::uint32_t)src1_dram_buffer->address(),  // in1_tensor_addr
                (std::uint32_t)per_core_N * output_idx_x,    // in1_tensor_start_tile_id TODO: you tried, you really did...
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
                (std::uint32_t)1*(num_blocks != 0) // nonzero, tells writer whether it has values to read
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
                num_nnz_output_blocks++;
            
            if (verbose && output_idx_y == 0 && output_idx_x == 1) {
                a.pretty_print();
                log_info(tt::LogVerif, " -- Reader Args --");
                const char* reader_arg_names[] = {
                    "in0_tensor_addr",
                    "in0_tensor_start_tile_id",
                    "in0_tensor_stride_w",
                    "in0_tensor_stride_h",
                    "in0_block_w",
                    "in0_block_h",
                    "in0_block_num_tiles",
                    "in1_tensor_addr",
                    "in1_tensor_start_tile_id",
                    "in1_tensor_stride_w",
                    "in1_tensor_stride_h",
                    "in1_block_w",
                    "in1_block_h",
                    "in1_block_num_tiles",
                    "col_indices_start_of_row",
                    "col_indices_end_of_row",
                    "row_index_into_bsr_matrix",
                    "column_indices_addr"
                };
                for (size_t i = 0; i < mm_reader_args.size(); ++i) {
                    log_info(tt::LogVerif, "reader_arg[{}] ({}) = {}", i, reader_arg_names[i], mm_reader_args[i]);
                }
                log_info(tt::LogVerif, " -- Writer Args --");
                const char* writer_arg_names[] = {
                    "out_buffer_addr",
                    "out_tensor_start_tile_id",
                    "out_tensor_stride_w",
                    "out_tensor_stride_h",
                    "out_tensor_next_subblock_stride_w",
                    "out_tensor_next_subblock_stride_h",
                    "out_subblock_w",
                    "out_subblock_h",
                    "out_subblock_w * out_subblock_h",
                    "out_num_subblocks_w",
                    "out_num_subblocks_h",
                    "MtNt",
                    "batch",
                    "nonzero"
                };
                for (size_t i = 0; i < writer_args.size(); ++i) {
                    log_info(tt::LogVerif, "writer_arg[{}] ({}) = {}", i, writer_arg_names[i], writer_args[i]);
                }
                log_info(tt::LogVerif, " -- Compute Args --");
                const char* compute_arg_names[] = {
                    "in0_block_w",
                    "in0_num_subblocks",
                    "in0_block_num_tiles",
                    "in0_subblock_num_tiles",
                    "in1_num_subblocks",
                    "in1_block_num_tiles",
                    "in1_per_core_w",
                    "num_blocks",
                    "out_subblock_h",
                    "out_subblock_w",
                    "out_subblock_num_tiles",
                    "B"
                };
                for (size_t i = 0; i < compute_args.size(); ++i) {
                    log_info(tt::LogVerif, "compute_arg[{}] ({}) = {}", i, compute_arg_names[i], compute_args[i]);
                }
            }
        }
    }

    if (verbose){
        log_info(tt::LogVerif, " -- Runtime Args set --");
        log_info(
            tt::LogVerif,
            " -- nnz output blocks= {}",
            num_nnz_output_blocks);
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

TestResult run_test(
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
    tilize(a.data, R, C);
    tilize(b.data, K, N);

    // run bsr_spmm_multicore_reuse
    bsr_spmm_multicore_reuse(a, b, output, false, nblocks, M, N, K, R, C, 1, device, verbose);


    if (emit_output) {

        // it makes 1000x more sense to print the tilized result. That's what these are!... bruh moment
        // let's write the output vectors to a file
        std::string local_path = "/home/user/tt-metal/tt_metal/programming_examples/rahmy/block_spmm/" + test_name;

        // TODO: will a failure here get caught in the larger try catch block?
        std::filesystem::create_directory(local_path);
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

        tilize(golden.data, M, N);
        for (size_t i = 0; i < golden.data.size(); i++) {
            golden_out << golden.data[i].to_float() << "\n";
        }
        untilize(golden.data, M, N);
        golden_out.close();


        // // print bsr matrix. should i tilize?
        // std::string bsr_file = local_path + "/bsr.txt";
        // std::ofstream bsr_out(bsr_file);
        // if (!bsr_out.is_open()) {
        //     TT_THROW("Failed to open bsr file: {}", bsr_file);
        // }
        // untilize(a.data, R, C);
        // for (size_t i = 0; i < a.data.size(); i++) {
        //     bsr_out << a.data[i].to_float() << "\n";
        // }
        // bsr_out.close();
    }

    // untile output data
    untilize(output.data, M, N);

    float pearson = check_bfloat16_vector_pcc(golden.data, output.data);

    // this is useless when matrices are not tiny with tiny elements. I get it now.
    // PCC is faulty and gives false positives for say, equality up to scaling, but
    // all_close is simply not suitable for bfloat16. 
    // surely there is a version of all_close which bases its tolerance on the norm of the input matrices?
    bool all_close = golden.all_close_bfloat16(output);

    CloseDevice(device);

    return TestResult{test_name, pearson, all_close};
}

void add_and_run_test(
        TestFunctionPtr function_ptr,
        vector<TestResult> &results,
        bool verbose = false,
        bool emit_output = false) {
    auto [a, b, test_name] = function_ptr();
    results.push_back(run_test(a, b, test_name, verbose, emit_output));
}

bool print_and_assess_results(std::vector<TestResult> &test_results){
    std::cout << "---------------------------------------------------------------------------------" << std::endl;
    std::cout << "--- Test results ----------------------------------------------------------------" << std::endl;
    std::cout << "---------------------------------------------------------------------------------" << std::endl;

    // assume there are <1000 tests.
    std::string spacing = "  ";
    bool all_pass = true;
    char buf[12];
    uint32_t count = 0;
    for (auto &p : test_results) {
        bool pass = true;
        if (!p.all_close){
            pass = false;
            all_pass = false;
        }

        // counting digits for spacing
        if (count >= 10 && count < 100)
            spacing = " ";
        if (count >= 100)
            spacing = "";

        std::string result = pass ? "✅ PASS " : "❌ FAIL ";
        sprintf(buf, "w/ PCC=%.2f", p.pearson);
        result += std::string(buf);
        result += p.pearson > 0.99 ? " ✅ ": " ❌ ";
        std::cout << "Test #" << count << ": " << spacing << result << " " << count << ' ' << p.test_name << std::endl;
        count++;
    }

    std::string result = all_pass ? "✅✅✅ PASS ✅✅✅" : "❌❌❌ FAIL ❌❌❌";

    std::cout << "---------------------------------------------------------------------------------" << std::endl;
    std::cout << result << std::endl;
    std::cout << "---------------------------------------------------------------------------------" << std::endl;


    return all_pass;
}

void test_suite(){
    /*
    1. Reserve a vector of <test_name, PCC> pairs.
    2. call run_test(test_func(), verbose, emit_output) for each test, adding to the vector
    3. iter over vector and pretty print passes and fails to the console
    */
    std::vector<TestResult> test_results;
    size_t num_tests = sizeof(TestRegistry) / sizeof(TestRegistry[0]);
    for (size_t i = 0; i < num_tests; i++) {
        add_and_run_test(TestRegistry[i], test_results);
    }
    bool pass = print_and_assess_results(test_results);

    /*
    For later, 
    can we add tests at runtime, ie not having to rebuild this function each time we add a test?
    */
}

void run_verbose_test(int test_num){
    auto [a, b, test_name] = TestRegistry[test_num]();
    TestResult res = run_test(a, b, test_name, true, true);

    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "--- Single Test results --------------------------------" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;

    bool pass = true;
    if (!res.all_close){
        pass = false;
    }

    char buf[12];
    std::string result = pass ? "✅ PASS " : "❌ FAIL ";
    sprintf(buf, "w/ PCC=%.2f", res.pearson);
    result += std::string(buf);
    result += res.pearson > 0.99 ? " ✅ ": " ❌ ";
    std::cout << "Test #" << test_num << ": " << result << " " << test_num << ' ' << res.test_name << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
}

int main(int argc, char** argv) {
    bool test = true;
    if (argc >= 2) {
        test = std::string(argv[1]) == "1";
    }
    if (test) {
        test_suite();
    }
    else {
        int test_num = argc > 2 ? std::stoi(argv[2]) : -1;
        if (test_num == -1) {
            std::cout << "No test specified. Returning." << std::endl;
            return 0;
        }
        // TODO: maybe refactor this so there is more control over verbosity and such
        run_verbose_test(test_num);
    }
}
