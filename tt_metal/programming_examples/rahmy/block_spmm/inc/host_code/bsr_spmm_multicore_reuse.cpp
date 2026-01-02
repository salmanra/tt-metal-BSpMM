#include "../host_code.hpp"

namespace bsr_host_code {

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
    // uint32_t per_core_N = _get_maximum_block_dim_with_NoC_args(per_core_M, in0_block_w, num_tiles_for_col_indices);
    // per_core_N = std::min({per_core_N, Ct, Nt});
    uint32_t per_core_N = get_Npc_from_BSR_block_size(Nt, per_core_M, in0_block_w, num_cores_x, num_tiles_for_col_indices);

    // pick the largest subblock size that fits within the block size
    uint32_t out_subblock_h = 0, out_subblock_w = 0;
    for (auto& subblock_hw : bmm_op_utils::SUBBLOCK_HW_CHOICES) {
        out_subblock_h = std::get<0>(subblock_hw);
        out_subblock_w = std::get<1>(subblock_hw);
        if (per_core_M % out_subblock_h == 0 and per_core_N % out_subblock_w == 0) {
            break;
        }
    }


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

    uint32_t dram_buffer_dst_row_size =
        single_tile_size * Rt * Nt;

    std::unordered_map<uint32_t, std::shared_ptr<Buffer>> dst_dram_buffers;
    for (uint32_t i = 0; i < a.indptr.size() - 1; i++) {
        if (a.indptr[i+1] - a.indptr[i] > 0)
            dst_dram_buffers[i] = MakeBuffer(device, dram_buffer_dst_row_size, single_tile_size);
    }

    uint32_t nnz_output_blocks_total = num_blocks_x * dst_dram_buffers.size(); // blocks per row * nnz rows

    CoreRangeSet all_cores(
        tt::tt_metal::num_cores_to_corerangeset(nnz_output_blocks_total, compute_with_storage_grid_size, true));


    if (verbose) {
        log_info(tt::LogVerif, " -- Metalium Grid Sizing --");
        log_info(
            tt::LogVerif,
            " -- Mt= {} -- Nt= {} -- nnz_output_blocks= {} -- num_cores_used={} -- num_cores_available_x={} -- num_cores_available_y={} --",
            Mt,
            Nt,
            nnz_output_blocks_total,
            all_cores,
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


    // In fact let's pad this to fill a tile at least
    uint32_t dram_buffer_D_size =
        sizeof(int) * nnz_blocks; //
    dram_buffer_D_size = col_indices_single_tile_size * ((col_indices_single_tile_size - 1 + dram_buffer_D_size) / (col_indices_single_tile_size));


    auto src0_dram_buffer = MakeBuffer(device, dram_buffer_A_size, single_tile_size);
    auto src1_dram_buffer = MakeBuffer(device, dram_buffer_B_size, single_tile_size);
    auto column_indices_dram_buffer = MakeBuffer(device, dram_buffer_D_size, col_indices_single_tile_size);



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

    bool dst_is_dram = true;
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

    // instead of iterating over cores, first determine how many output blocks to a core (get the core range set i guess)
    // and iterate over output blocks, keeping track of which nz output block you are in by taking
    // into account the zero rows.
    // ^^ This was good, but  now we have to go back to thinking about cores since we want to be able to assign many output blocks to a single core.
    //    ... it's not clear whether doing this first and then working on the multicast impl is better
    //          Yes it is clear: i'm already thinking about this, and the depth of reasoning I'm practicing will help
    //          me with whatever I decide to do afterwards.

    TT_ASSERT(nnz_output_blocks_total <= num_cores_x * num_cores_y);

    uint32_t nnz_output_blocks_read = 0;
    for (int output_idx_y = 0; output_idx_y < a.indptr.size() - 1; output_idx_y++) {
        uint32_t nnz_blocks_in_row = a.indptr[output_idx_y + 1] - a.indptr[output_idx_y];
        if (nnz_blocks_in_row == 0)
            continue;
        // else, we are in a nonzero row
        for (uint32_t output_idx_x = 0; output_idx_x < num_blocks_x; output_idx_x++){
            int core_idx_x = nnz_output_blocks_read % num_cores_x;
            int core_idx_y = nnz_output_blocks_read / num_cores_x;
            CoreCoord core = {(std::size_t)core_idx_x, (std::size_t)core_idx_y};

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
                (std::uint32_t)dst_dram_buffers[output_idx_y]->address(),      // out_buffer_addr
                (std::uint32_t)output_idx_x * per_core_N,  // out_tensor_start_tile_id
                (std::uint32_t)1,                           // out_tensor_stride_w
                (std::uint32_t)Nt,                         // out_tensor_stride_h
                (std::uint32_t)out_subblock_w,       // out_tensor_next_subblock_stride_w
                (std::uint32_t)out_subblock_h * Nt,  // out_tensor_next_subblock_stride_h

                (std::uint32_t)out_subblock_w,                     // out_subblock_w
                (std::uint32_t)out_subblock_h,                     // out_subblock_h
                (std::uint32_t)(out_subblock_w * out_subblock_h),  // out_subblocks_w * out_subblocks_h
                (std::uint32_t)(per_core_N / out_subblock_w),      // out_num_subblocks_w
                (std::uint32_t)(per_core_M / out_subblock_h),      // out_num_subblocks_h

                (std::uint32_t)Mt * Nt,  // MtNt... only used in the batch impl...
                (std::uint32_t)B,        // batch
                (std::uint32_t)1*(nnz_blocks_in_row != 0) // nonzero, tells writer whether it has values to read
            };

            std::vector<uint32_t> compute_args = {
                (std::uint32_t)in0_block_w,
                (std::uint32_t)in0_num_subblocks,
                (std::uint32_t)in0_block_w * per_core_M, // in0_block_num_tiles
                (std::uint32_t)in0_subblock_num_tiles,
                (std::uint32_t)in1_num_subblocks,
                (std::uint32_t)in1_block_num_tiles,
                (std::uint32_t)in1_per_core_w,
                (std::uint32_t)nnz_blocks_in_row, // num_blocks
                (std::uint32_t)out_subblock_h,
                (std::uint32_t)out_subblock_w,
                (std::uint32_t)out_subblock_num_tiles,
                (std::uint32_t)B
            };

            tt_metal::SetRuntimeArgs(program, reader_id, core, mm_reader_args);
            tt_metal::SetRuntimeArgs(program, writer_id, core, writer_args);
            tt_metal::SetRuntimeArgs(program, mm_kernel_id, core, compute_args);
            nnz_output_blocks_read++;

            if (verbose && output_idx_y == 0 && output_idx_x == 0) {
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
                    "nnz_blocks_in_row",
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
            nnz_output_blocks_read);
    }


    EnqueueWriteBuffer(cq, src0_dram_buffer, a.data.data(), true);
    EnqueueWriteBuffer(cq, src1_dram_buffer, b.data.data(), true);
    EnqueueWriteBuffer(cq, column_indices_dram_buffer, a.indices.data(), true);

    if (verbose)
        log_info(tt::LogVerif, " -- All data moved to DRAM --");

    EnqueueProgram(cq, program, true);

    if (verbose)
        log_info(tt::LogVerif, " -- Program enqueued --");

    for (auto & pair : dst_dram_buffers){
        uintptr_t row_index = pair.first;
        std::shared_ptr<Buffer> buffer = pair.second;
        EnqueueReadBuffer(cq, buffer, output.data.data() + (row_index * R * N), true);
    }

    Finish(cq);
}

}