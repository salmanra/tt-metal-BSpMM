#include "../host_code.hpp"
namespace bsr_host_code {

void bsr_spmm_multicore_reuse_iteration(
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
    bool verbose){

    /*
    Host code considerations:
        - call Mpc, Npc the tile counts of CB sizes
        - Mpc = Rt
        - BSR Block size informs maximum Npc
        - Mt, Nt, Mpc, max Npc inform Npc, num_iters_y, num_iters_x
            - Ah! the num_iters_y/x split kinda informs which dimension
                we should be biased on...
                If we only iter on y, one core gets many rows, each with
                a sparsity pattern. IDEA: "Mcast sharing"
                    - Which means we could let some other core concurrently
                        get the same row and mcast share.
                If we only iter on x, one core gets one input row, with
                only one sparsity pattern. And is guaranteed to reuse
                all the blocks in that row. IDEA: "self-circulation"
            - So we bias on num_iters_y (ie, bias for large Npc).
            - Generally let's say num_iters_y and num_iters_x are gt 1.
                - Like in test 41 with core grid {2, 2}
                - IDEA: can analyze pattern ahead of time to see which of the two above ideas is optimal
        - RK ->
             -> for each num_iter_y, needs brs&bre into indptr, indices
             -> for each num_iter_x, just needs the count num_iter_x
             -> needs num_tiles for col_indices, indptr
        - CK ->
             -> for each num_iter_y, needs num_blocks (bre-brs)
             -> for each num_iter_x, just needs the count num_iter_x
        - WK ->
             -> just needs start tile of output region and iter counts

        RK -> (brs&bre)*num_iter_y - runtime_args
           -> num_iter_x - compiletime

        CK -> nblocks - runtime
           -> num_iter_x - compiletime

        WK -> both compile_time


    */
    /*
    some unstructured thoughts

    The set up all woked great actually. The folded matrix, the runtime args, the compiletime args, the work distribution.
    It's just the set up had some incorretct assumptions.

    Under the new, more relaxed assumptions, we need a few more things
        - num_iter_x as a common compiletime arg (unlike ycoords, we can rely on xcoords being a range)
        - that's it?

    It is being pointed out to me in 510 that I should be testing each "thing"
        I've already decided it's too hard to test each kernel specifcally
        I've already tested my bsr_matrix code
        I have NOT tested the folding, and othe rthings like that (what other things?)
    */

    // program
    // command queue
    // data format constants

    // TT-Metal CommandQueue and Program setup
    CommandQueue& cq = device->command_queue();
    Program program{};

    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    uint32_t single_tile_size = detail::TileSize(cb_data_format);

    tt::DataFormat indexing_data_format = tt::DataFormat::Int32;
    uint32_t indexing_data_single_tile_size = detail::TileSize(indexing_data_format);
    uint32_t num_tiles_for_col_indices = (indexing_data_single_tile_size - 1 + sizeof(int) * nnz_blocks) / indexing_data_single_tile_size;
    uint32_t num_tiles_for_indptr = (indexing_data_single_tile_size - 1 + sizeof(int) * (M / R + 1)) / indexing_data_single_tile_size;
    uint32_t num_tiles_indexing = num_tiles_for_col_indices + num_tiles_for_indptr;

    // Core Grid detection
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    // auto compute_with_storage_grid_size = CoreCoord(3, 1);
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;



    // Per-core tiling and blocking args
    uint32_t Mt = M / TILE_HEIGHT;
    uint32_t Kt = K / TILE_WIDTH;
    uint32_t Nt = N / TILE_WIDTH;

    uint32_t Rt = R / TILE_HEIGHT;
    uint32_t Ct = C / TILE_WIDTH;

    uint32_t in0_block_h = Rt;
    uint32_t in0_block_w = Ct;
    uint32_t in1_block_w = get_Npc_from_BSR_block_size(Nt, in0_block_h, in0_block_w, num_cores_x, num_tiles_indexing);

    TT_ASSERT(Mt % in0_block_h == 0);
    TT_ASSERT(Nt % in1_block_w == 0);
    TT_ASSERT(Kt % in0_block_w == 0);

    // Core grid assignment
   std::vector<uint32_t> folded_bsr_matrix_indices;
    uint32_t nnz_rows = 0;
    uint32_t folded_index = 0;
    for (uint32_t i = 0; i < a.indptr.size() - 1; i++) {
        if (a.indptr[i+1] - a.indptr[i] > 0){
            folded_bsr_matrix_indices.push_back(folded_index);
            nnz_rows++;
        }
        folded_index++;
    }
    uint32_t height_of_folded_matrix = Rt * nnz_rows;

    uint32_t num_blocks_x = Nt / in1_block_w;
    uint32_t num_blocks_y = nnz_rows;
    uint32_t num_blocks_total = num_blocks_x * num_blocks_y;

    uint32_t num_iters_x = (num_blocks_x + num_cores_x - 1) / num_cores_x;
    uint32_t num_iters_y = (num_blocks_y + num_cores_y - 1) / num_cores_y;

    uint32_t num_work_regions = (num_blocks_total + num_iters_x * num_iters_y - 1)/ (num_iters_x * num_iters_y);
    uint32_t target_num_cores;
    if (num_work_regions < num_cores_total)
        target_num_cores = num_work_regions;
    else
        target_num_cores = num_cores_total;


    CoreRangeSet all_cores(
        tt::tt_metal::num_cores_to_corerangeset(target_num_cores, compute_with_storage_grid_size, true));

    // CoreCoord all_cores(0, 0);
    // all_cores.x = std::min(num_blocks_x, num_cores_x);
    // all_cores.y = std::min(num_blocks_y, num_cores_y);
    uint32_t out_subblock_h = 0, out_subblock_w = 0;
    for (auto& subblock_hw : bmm_op_utils::SUBBLOCK_HW_CHOICES) {
        out_subblock_h = std::get<0>(subblock_hw);
        out_subblock_w = std::get<1>(subblock_hw);
        if (in0_block_h % out_subblock_h == 0 and in1_block_w % out_subblock_w == 0) {
            break;
        }
    }

    // Circural Buffer sizing
    uint32_t in0_CB_num_tiles = in0_block_h * in0_block_w * 2; // double buffer
    uint32_t in0_CB_size = in0_CB_num_tiles * single_tile_size;
    uint32_t in1_CB_num_tiles = in0_block_w * in1_block_w * 2; // double buffer
    uint32_t in1_CB_size = in1_CB_num_tiles * single_tile_size;
    uint32_t out_CB_num_tiles = in0_block_h * in1_block_w; // single buffer
    uint32_t out_CB_size = out_CB_num_tiles * single_tile_size;

    uint32_t in0_num_subblocks = (in0_block_h / out_subblock_h);
    uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;
    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w; // this is named weird but it's correct.

    uint32_t in1_num_subblocks = (in1_block_w / out_subblock_w);
    uint32_t in1_block_num_tiles = out_subblock_w * in0_block_w * in1_num_subblocks;
    uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;

    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;


    // DRAM buffers initialiation
    uint32_t dram_buffer_dst_row_size =
        single_tile_size * Rt * Nt;

    uint32_t dram_buffer_dst_total_size = dram_buffer_dst_row_size * nnz_rows;

    uint32_t dram_buffer_A_size =
        single_tile_size * Rt * Ct * nnz_blocks;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    uint32_t dram_buffer_B_size =
        single_tile_size * Nt * Kt;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels

    uint32_t dram_buffer_col_indices_size =
        sizeof(indexing_data_format) * nnz_blocks;
    // Round up to tile size
    dram_buffer_col_indices_size = indexing_data_single_tile_size * ((indexing_data_single_tile_size - 1 + dram_buffer_col_indices_size) / (indexing_data_single_tile_size));

    uint32_t dram_buffer_indptr_size =
        sizeof(indexing_data_format) * (M / R + 1);
    // Round up to tile size
    dram_buffer_indptr_size = indexing_data_single_tile_size * ((indexing_data_single_tile_size - 1 + dram_buffer_indptr_size) / (indexing_data_single_tile_size));

    auto dst_dram_buffer = MakeBuffer(device, dram_buffer_dst_total_size, single_tile_size);
    auto src0_dram_buffer = MakeBuffer(device, dram_buffer_A_size, single_tile_size);
    auto src1_dram_buffer = MakeBuffer(device, dram_buffer_B_size, single_tile_size);
    auto column_indices_dram_buffer = MakeBuffer(device, dram_buffer_col_indices_size, dram_buffer_col_indices_size);
    auto indptr_dram_buffer = MakeBuffer(device, dram_buffer_indptr_size, dram_buffer_indptr_size);


    if (verbose) {
        log_info(tt::LogVerif, " -- Metalium Block and subblock sizing --");
        log_info(
            tt::LogVerif,
            " -- per_core_M={} -- per_block_M={} -- per_core_N={} -- out_subblock_h={} -- out_subblock_w={} --",
            num_iters_y * in0_block_h,
            in0_block_h,
            num_iters_x * in1_block_w,
            out_subblock_h,
            out_subblock_w);
    }

    if (verbose) {
        log_info(tt::LogVerif, " -- Core Grid Allocaiton Information --");
        log_info(
            tt::LogVerif,
            " -- num_cores_y={} -- num_cores_x={} -- num_iters_y={} -- num_iters_x={} -- nnz_rows={} --",
            num_cores_y,
            num_cores_x,
            num_iters_y,
            num_iters_x,
            nnz_rows);
    }

    if (verbose) {
        log_info(tt::LogVerif, " -- Metalium Core Grid Sizing --");
        log_info(
            tt::LogVerif,
            " -- Mt= {} -- Nt= {} -- num_output_blocks= {} -- num_cores_used={} -- num_cores_available_x={} -- num_cores_available_y={} --",
            Mt,
            Nt,
            num_blocks_total,
            all_cores.num_cores(),
            num_cores_x,
            num_cores_y);
    }


    /*
    SRAM Circular Buffers
    */
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
        dram_buffer_col_indices_size, {{column_indices_cb_index, tt::DataFormat::Int32}})
                                                .set_page_size(column_indices_cb_index, dram_buffer_col_indices_size);
    auto cb_column_indices = tt_metal::CreateCircularBuffer(program, all_cores, cb_column_indices_config);

    auto indptr_cb_index = CBIndex::c_3; // 3
    auto cb_indptr = MakeCircularBuffer(program, all_cores, indptr_cb_index, dram_buffer_indptr_size, dram_buffer_indptr_size, indexing_data_format);


    // Compiletime arguments
    bool src0_is_dram = src0_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool src1_is_dram = src1_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool col_indices_is_dram = column_indices_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool indptr_is_dram = indptr_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)src0_is_dram,
        (std::uint32_t)src1_is_dram,
        (std::uint32_t)col_indices_is_dram,
        (std::uint32_t)indptr_is_dram,

        (std::uint32_t)src0_dram_buffer->address(),     // in0_tensor_addr
        (std::uint32_t)1,                               // in0_tensor_stride_w
        (std::uint32_t)Ct,                              // in0_tensor_stride_h

        (std::uint32_t)in0_block_w,               // in0_block_w
        (std::uint32_t)Rt,                         // in0_block_h
        (std::uint32_t)in0_block_w * Rt,  // in0_block_num_tiles

        (std::uint32_t)src1_dram_buffer->address(),  // in1_tensor_addr
        (std::uint32_t)1,                            // in1_tensor_stride_w
        (std::uint32_t)Nt,                           // in1_tensor_stride_h

        (std::uint32_t)in1_block_w,                // in1_block_w
        (std::uint32_t)in0_block_w,               // in1_block_h
        (std::uint32_t)in1_block_w * in0_block_w,  // in1_block_num_tiles


        (std::uint32_t)column_indices_dram_buffer->address(), // NoC args, column indices
        (std::uint32_t)indptr_dram_buffer->address(), // NoC args, indptr

        (std::uint32_t)num_tiles_for_col_indices,
        (std::uint32_t)num_tiles_for_indptr,

        // in0_tensor_start_tile_id obtained by // a.indptr[output_idx_y] * Rt * Ct,
        // in1_tensor_start_tile_id obtained by // per_core_N * output_idx_x
        // col indices start of row obtained by // a.indptr[output_idx_y],
        // col indices end of row obtained by //  a.indptr[output_idx_y + 1],
    };

    std::vector<uint32_t> compute_kernel_compile_time_args = {
        (std::uint32_t)in0_block_w,
        (std::uint32_t)in0_num_subblocks,
        (std::uint32_t)in0_block_num_tiles,
        (std::uint32_t)in0_subblock_num_tiles,
        (std::uint32_t)in1_num_subblocks,
        (std::uint32_t)in1_block_num_tiles,
        (std::uint32_t)in1_per_core_w,
        (std::uint32_t)out_subblock_h,
        (std::uint32_t)out_subblock_w,
        (std::uint32_t)out_subblock_num_tiles,
        (std::uint32_t)num_iters_x,
    };

    bool out_is_dram = dst_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)out_is_dram,
        (std::uint32_t)dst_dram_buffer->address(),      // out_buffer_addr
        (std::uint32_t)1,                           // out_tensor_stride_w
        (std::uint32_t)Nt,                         // out_tensor_stride_h
        (std::uint32_t)out_subblock_w,       // out_tensor_next_subblock_stride_w
        (std::uint32_t)out_subblock_h * Nt,  // out_tensor_next_subblock_stride_h

        (std::uint32_t)out_subblock_w,                     // out_subblock_w
        (std::uint32_t)out_subblock_h,                     // out_subblock_h
        (std::uint32_t)(out_subblock_w * out_subblock_h),  // out_subblocks_w * out_subblocks_h
        (std::uint32_t)(in1_block_w / out_subblock_w),      // out_num_subblocks_w
        (std::uint32_t)(in0_block_h / out_subblock_h),      // out_num_subblocks_h

        (std::uint32_t)Rt * Nt,  // Size of output row, used to index into next output block
        (std::uint32_t)num_iters_x,
    };

    if (verbose) {
        auto print_args = [](const std::string& name, const std::vector<uint32_t>& args, const std::vector<std::string>& arg_names) {
            std::cout << "==== " << name << " ====" << std::endl;
            for (size_t i = 0; i < args.size(); ++i) {
                if (i < arg_names.size())
                    std::cout << "  [" << i << "] " << arg_names[i] << " = " << args[i] << std::endl;
                else
                    std::cout << "  [" << i << "] = " << args[i] << std::endl;
            }
            std::cout << std::endl;
        };

        print_args("reader_compile_time_args", reader_compile_time_args, {
            "src0_is_dram",
            "src1_is_dram",
            "col_indices_is_dram",
            "indptr_is_dram",
            "src0_dram_buffer->address()",
            "in0_tensor_stride_w",
            "in0_tensor_stride_h",
            "in0_block_w",
            "in0_block_h",
            "in0_block_num_tiles",
            "src1_dram_buffer->address()",
            "in1_tensor_stride_w",
            "in1_tensor_stride_h",
            "in1_block_w",
            "in1_block_h",
            "in1_block_num_tiles",
            "column_indices_dram_buffer->address()",
            "indptr_dram_buffer->address()",
            "num_tiles_for_col_indices",
            "num_tiles_for_indptr"
        });

        print_args("compute_kernel_compile_time_args", compute_kernel_compile_time_args, {
            "in0_block_w",
            "in0_num_subblocks",
            "in0_block_num_tiles",
            "in0_subblock_num_tiles",
            "in1_num_subblocks",
            "in1_block_num_tiles",
            "in1_per_core_w",
            "out_subblock_h",
            "out_subblock_w",
            "out_subblock_num_tiles",
            "num_iters_x",
        });

        print_args("writer_compile_time_args", writer_compile_time_args, {
            "out_is_dram",
            "dst_dram_buffer->address()",
            "out_tensor_stride_w",
            "out_tensor_stride_h",
            "out_tensor_next_subblock_stride_w",
            "out_tensor_next_subblock_stride_h",
            "out_subblock_w",
            "out_subblock_h",
            "out_subblock_w * out_subblock_h",
            "out_num_subblocks_w",
            "out_num_subblocks_h",
            "Rt * Nt",
            "num_iters_x",
        });
    }

    // Create Kernels
    auto reader_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/rahmy/block_spmm/kernels/dataflow/reader_block_iter.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = reader_compile_time_args});

    auto writer_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/rahmy/block_spmm/kernels/dataflow/writer_block_iter.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = writer_compile_time_args});

    // Create compute kernel
    auto mm_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/rahmy/block_spmm/kernels/compute/bmm_iter.cpp",
        all_cores,
        tt_metal::ComputeConfig{.math_fidelity = math_fidelity,
                                       .compile_args = compute_kernel_compile_time_args});
    // Runtime arguments
    auto core_coords_vec = corerange_to_cores(all_cores);
    uint32_t work_region = 0;
    for (auto & core : core_coords_vec){
        uint32_t core_idx_x = core.x;
        uint32_t core_idx_y = core.y;

        if (verbose)
            log_info(tt::LogVerif, "Core x {} y {}", core_idx_x, core_idx_y);

        int output_idx_x_start = (work_region * num_iters_x) % num_blocks_x;
        int folded_output_idx_y_start = (((work_region * num_iters_x) / num_blocks_x) * num_iters_y) % num_blocks_y;
        work_region++;

        std::vector<uint32_t> reader_runtime_args;
        std::vector<uint32_t> compute_runtime_args;
        std::vector<uint32_t> writer_runtime_args;

        uint32_t num_iters_y_remaining = num_blocks_y - folded_output_idx_y_start;
        uint32_t num_iters_y_this_core = std::min(num_iters_y, num_iters_y_remaining);
        uint32_t num_iters_x_this_core = std::min(num_iters_x, num_blocks_x - output_idx_x_start); // TODO: make a test case that actually tests this
        reader_runtime_args.push_back(num_iters_x_this_core);
        reader_runtime_args.push_back(num_iters_y_this_core);
        compute_runtime_args.push_back(num_iters_y_this_core);
        reader_runtime_args.push_back(output_idx_x_start);
        for (int iter_y = 0; iter_y < num_iters_y_this_core; iter_y++) {
            uint32_t output_idx_y = folded_bsr_matrix_indices[folded_output_idx_y_start + iter_y];
            reader_runtime_args.push_back(output_idx_y);
            compute_runtime_args.push_back(a.indptr[output_idx_y + 1] - a.indptr[output_idx_y]);
        }

        writer_runtime_args.push_back((folded_output_idx_y_start * Rt * Nt) + (output_idx_x_start * in1_block_w));
        writer_runtime_args.push_back(num_iters_y_this_core);

        tt_metal::SetRuntimeArgs(program, reader_id, core, reader_runtime_args);
        tt_metal::SetRuntimeArgs(program, mm_kernel_id, core, compute_runtime_args);
        tt_metal::SetRuntimeArgs(program, writer_id, core, writer_runtime_args);

        if (verbose && core_idx_x == 0 && core_idx_y == 0) {
            a.pretty_print();
            log_info(tt::LogVerif, " -- Reader Args --");
            log_info(tt::LogVerif, "reader_arg[0] (num_iters_x) = {}", reader_runtime_args[0]);
            log_info(tt::LogVerif, "reader_arg[1] (num_iters_y) = {}",  reader_runtime_args[1]);
            log_info(tt::LogVerif, "reader_arg[2] (output_idx_x_start) = {}",  reader_runtime_args[2]);
            for (size_t i = 0; i < num_iters_y_this_core; ++i) {
                log_info(tt::LogVerif, "reader_arg[{}] (y_coord) = {}", i + 3, reader_runtime_args[i+3]);
            }

            log_info(tt::LogVerif, " -- Writer Args --");
            const char* writer_arg_names[] = {
                "out_tensor_start_tile_id",
                "num_iters_y_this_core"
            };
            for (size_t i = 0; i < writer_runtime_args.size(); ++i) {
                log_info(tt::LogVerif, "writer_arg[{}] ({}) = {}", i, writer_arg_names[i], writer_runtime_args[i]);
            }
            log_info(tt::LogVerif, " -- Compute Args --");
            log_info(tt::LogVerif, "compute_arg[{}] (num_iters_y) = {}", 0, compute_runtime_args[0]);
            for (size_t i = 0; i < num_iters_y_this_core; ++i) {
                log_info(tt::LogVerif, "compute_arg[{}] (row_size) = {}", i+1, compute_runtime_args[i+1]);
            }
        }
    }

    // EnqueueWriteBuffers
    EnqueueWriteBuffer(cq, src0_dram_buffer, a.data.data(), true);
    EnqueueWriteBuffer(cq, src1_dram_buffer, b.data.data(), true);
    EnqueueWriteBuffer(cq, column_indices_dram_buffer, a.indices.data(), true);
    EnqueueWriteBuffer(cq, indptr_dram_buffer, a.indptr.data(), true);
    // EnqueueProgram
    EnqueueProgram(cq, program, true);

    if (verbose)
        log_info(tt::LogVerif, " -- Program returned --");
    // EnqueueReadSubBuffers
    uint32_t nonzero_row_index = 0;
    for (size_t row_index = 0; row_index < a.indptr.size() - 1; row_index++) {
        if (a.indptr[row_index+1] - a.indptr[row_index] == 0)
            continue;
        BufferRegion DRAM_row(nonzero_row_index * dram_buffer_dst_row_size, dram_buffer_dst_row_size);
        EnqueueReadSubBuffer(cq, dst_dram_buffer, output.data.data() + (row_index * R * N), DRAM_row, true);
        nonzero_row_index++;
    }

    if (verbose)
        log_info(tt::LogVerif, " -- Finished reading output --");
    Finish(cq);
}

}