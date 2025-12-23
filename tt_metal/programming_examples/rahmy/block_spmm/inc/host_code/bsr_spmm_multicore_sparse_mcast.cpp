#include "../host_code.hpp"

namespace bsr_host_code {


void bsr_spmm_multicore_sparse_mcast(
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
    bool verbose)
    {
    // nothing really changes from iteration version.
    // Create two semaphores.
    // Split into 2 reader kernels.
    // everything else is identical.

    // get core grid. Is the CoreRangeSet we create in the previous examples guaranteed to be
    // a rectangle? Let's assume it is.

    // Define all cores, left column, and all but left column
    // Compile CK, WK to all cores
    // Compile custom RK to left column, all but left column

    // Create two semaphores (in0_mcast_sender/receiver) on all cores

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
   std::deque<uint32_t> folded_bsr_matrix_indices;
    uint32_t nnz_rows = 0;
    uint32_t folded_index = 0;
    for (uint32_t i = 0; i < a.indptr.size() - 1; i++) {
        if (a.indptr[i+1] - a.indptr[i] > 0){
            folded_bsr_matrix_indices.push_back(folded_index);
            nnz_rows++;
        }
        folded_index++;
    }
    folded_bsr_matrix_indices.push_back(folded_index);
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

    uint32_t out_subblock_h = 0, out_subblock_w = 0;
    for (auto& subblock_hw : bmm_op_utils::SUBBLOCK_HW_CHOICES) {
        out_subblock_h = std::get<0>(subblock_hw);
        out_subblock_w = std::get<1>(subblock_hw);
        if (in0_block_h % out_subblock_h == 0 and in1_block_w % out_subblock_w == 0) {
            break;
        }
    }

    CoreCoord start_core = {0, 0};
    // CoreCoord core_range = bmm_op_utils::get_core_range(num_blocks_y / num_iters_y, num_blocks_x / num_iters_x, num_cores_y, num_cores_x);
    CoreCoord core_range(0, 0);
    if ( (num_blocks_y / num_iters_y) <= num_cores_y &&
        (num_blocks_x / num_iters_x) <= num_cores_x) {
        core_range.x = (num_blocks_x + num_iters_x - 1) / num_iters_x;
        core_range.y = (num_blocks_y + num_iters_y - 1) / num_iters_y;
    }
    uint32_t start_core_x = start_core.x;
    uint32_t start_core_y = start_core.y;
    uint32_t num_cores_c = core_range.x;
    uint32_t num_cores_r = core_range.y;

    // TODO: when only using one core, all_except_left_column is bad and throws a runtime error.
    CoreRange all_cores(
        {(std::size_t)start_core_x, (std::size_t)start_core_y},
        {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y + num_cores_r - 1});

    CoreRange left_column(
        {(std::size_t)start_core_x, (std::size_t)start_core_y},
        {(std::size_t)start_core_x, (std::size_t)start_core_y + num_cores_r - 1});


    // this variable exists purely to allow the constructor to CoreRange to not fail when we only use one column of the core grid
    uint32_t column_offset = num_cores_c > 1 ? num_cores_c : num_cores_c + 1;
    CoreRange all_except_left_column(
        {(std::size_t)start_core_x + 1, (std::size_t)start_core_y},
        {(std::size_t)start_core_x + column_offset - 1, (std::size_t)start_core_y + num_cores_r - 1});


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
            " -- available_cores_y={} -- available_cores_x={} -- num_iters_y={} -- num_iters_x={} -- nnz_rows={} --",
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
            " -- Mt= {} -- Nt= {} -- num_output_blocks= {} -- cores_used={} -- num_blocks_x={} -- num_blocks_y={} --",
            Mt,
            Nt,
            num_blocks_total,
            all_cores,
            num_blocks_x,
            num_blocks_y);
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
    // Two unique reader kernels, one compute kernel, one unique writer kernel with 2 compile configs
    //
    auto reader_in0_sender_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/rahmy/block_spmm/kernels/dataflow/reader_block_iter_in0_sender.cpp",
        left_column,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = reader_compile_time_args});


    KernelHandle reader_in0_receiver_id = 0;
    if (num_cores_c > 1) {
        reader_in0_receiver_id = tt_metal::CreateKernel(
            program,
            "tt_metal/programming_examples/rahmy/block_spmm/kernels/dataflow/reader_block_iter_in0_receiver.cpp",
            all_except_left_column,
            tt_metal::DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = reader_compile_time_args});
    }

    auto writer_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/rahmy/block_spmm/kernels/dataflow/writer_block_load_balanced.cpp",
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
                                        // .fp32_dest_acc_en = true,
                                        .compile_args = compute_kernel_compile_time_args});

    auto in0_mcast_sender_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, INVALID);
    auto in0_mcast_receiver_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, INVALID);


    // Runtime arguments
    uint32_t work_region = 0;

    // Scanning for load-balancing
    // 0. Sort block rows by number of nonzero blocks, get perm vector
    //  get folded indices first, then sort and perm
    uint32_t num_empty_rows = (M / R) - nnz_rows;
    std::vector<int> row_diffs;

    for (int i = 0; i < folded_bsr_matrix_indices.size() - 1; i++){
        row_diffs.push_back(folded_bsr_matrix_indices[i+1] - folded_bsr_matrix_indices[i]);
    }
    std::vector<int> perm(row_diffs.size());
    sortingPermutation(row_diffs, perm);

    // remove last num_empty_rows elements from perm
    perm.resize(nnz_rows);

    if (verbose){
        std::cout << "row diffs: ";
        for (int i = 0; i < row_diffs.size(); i++){
            std::cout << row_diffs[i] << ' ';
        }
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << "perm: ";
        for (int i = 0; i < perm.size(); i++){
            std::cout << perm[i] << ' ';
        }
    }
    // 1. initialize a vector for each row of cores
    std::vector<std::vector<uint32_t>> output_y_indices(num_cores_r, std::vector<uint32_t>());
    // 2. While count is less than num output blocks
    uint32_t num_rows_assigned = 0;
    uint32_t iter_count = 1;
    uint32_t subarray_iter = 0;
    while (num_rows_assigned < nnz_rows) {
        uint32_t num_rows_to_assign = std::min(num_cores_r, nnz_rows - num_rows_assigned);
        subarray_iter = 0;
        for (uint32_t core_row = 0; core_row < num_rows_to_assign; core_row++){
            if (num_rows_assigned++ >= nnz_rows)
                break;
            output_y_indices[core_row].push_back(perm[num_cores_r * (iter_count - 1) + subarray_iter]);
            subarray_iter++;
        }
        num_rows_to_assign = std::min(num_cores_r, nnz_rows - num_rows_assigned);
        subarray_iter = num_cores_r - num_rows_to_assign;
        for (uint32_t core_row = num_cores_r; core_row > num_cores_r - num_rows_to_assign; core_row--){
            if (num_rows_assigned++ >= nnz_rows)
                break;
            output_y_indices[core_row - 1].push_back(perm[nnz_rows - num_cores_r * iter_count + subarray_iter]);
            subarray_iter++;
        }
        iter_count++;
    }

    for (uint32_t core_idx_y = 0; core_idx_y < num_cores_r; core_idx_y++) {
        for (uint32_t core_idx_x = 0; core_idx_x < num_cores_c; core_idx_x++) {
            CoreCoord core(core_idx_x, core_idx_y);
            if (verbose)
              log_info(tt::LogVerif, "Core x {} y {}", core_idx_x, core_idx_y);

            CoreCoord left_core = {(std::size_t)start_core_x, (std::size_t)core.y};
            CoreCoord left_core_plus_one = {(std::size_t)start_core_x + 1, (std::size_t)core.y};
            CoreCoord right_core = {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)core.y};
            CoreCoord top_core = {(std::size_t)core.x, (std::size_t)start_core_y};
            CoreCoord top_core_plus_one = {(std::size_t)core.x, (std::size_t)start_core_y + 1};
            CoreCoord bottom_core = {(std::size_t)core.x, (std::size_t)start_core_y + num_cores_r - 1};

            auto left_core_physical = device->worker_core_from_logical_core(left_core);
            auto left_core_plus_one_physical = device->worker_core_from_logical_core(left_core_plus_one);
            auto right_core_physical = device->worker_core_from_logical_core(right_core);
            auto top_core_physical = device->worker_core_from_logical_core(top_core);
            auto top_core_plus_one_physical = device->worker_core_from_logical_core(top_core_plus_one);
            auto bottom_core_physical = device->worker_core_from_logical_core(bottom_core);


            int output_idx_x_start = (core_idx_x * num_iters_x) % num_blocks_x;
            work_region++;

            std::vector<uint32_t> reader_runtime_args = {
                (std::uint32_t)right_core_physical.x,          // in0_mcast_dest_noc_start_x
                (std::uint32_t)right_core_physical.y,          // in0_mcast_dest_noc_start_y
                (std::uint32_t)left_core_plus_one_physical.x,  // in0_mcast_dest_noc_end_x
                (std::uint32_t)left_core_plus_one_physical.y,  // in0_mcast_dest_noc_end_y
                (std::uint32_t)(num_cores_c - 1),              // in0_mcast_num_dests
                (std::uint32_t)left_core_physical.x,           // in0_mcast_sender_noc_x
                (std::uint32_t)left_core_physical.y,           // in0_mcast_sender_noc_y
                (std::uint32_t)in0_mcast_sender_semaphore_id,
                (std::uint32_t)in0_mcast_receiver_semaphore_id,
            };
            std::vector<uint32_t> compute_runtime_args;
            std::vector<uint32_t> writer_runtime_args;

            uint32_t num_iters_y_this_core = output_y_indices[core_idx_y].size();
            uint32_t num_iters_x_this_core = std::min(num_iters_x, num_blocks_x - output_idx_x_start + 1);
            reader_runtime_args.push_back(num_iters_x_this_core);
            reader_runtime_args.push_back(num_iters_y_this_core);
            compute_runtime_args.push_back(num_iters_y_this_core);
            reader_runtime_args.push_back(output_idx_x_start);
            writer_runtime_args.push_back(output_idx_x_start * in1_block_w);
            writer_runtime_args.push_back(num_iters_y_this_core);
            writer_runtime_args.push_back(num_cores_r);
            for (int iter_y = 0; iter_y < num_iters_y_this_core; iter_y++) {
                uint32_t folded_output_idx_y = output_y_indices[core_idx_y][iter_y];
                uint32_t output_idx_y = folded_bsr_matrix_indices[folded_output_idx_y];
                reader_runtime_args.push_back(output_idx_y);
                writer_runtime_args.push_back(folded_output_idx_y);
                compute_runtime_args.push_back(a.indptr[output_idx_y + 1] - a.indptr[output_idx_y]);
            }

            if (core_idx_x == 0){
                tt_metal::SetRuntimeArgs(program, reader_in0_sender_id, core, reader_runtime_args);
            }
            else {
                tt_metal::SetRuntimeArgs(program, reader_in0_receiver_id, core, reader_runtime_args);
            }
            tt_metal::SetRuntimeArgs(program, mm_kernel_id, core, compute_runtime_args);
            tt_metal::SetRuntimeArgs(program, writer_id, core, writer_runtime_args);

            if (verbose && core_idx_x == 0 && core_idx_y == 0) {
                a.pretty_print();
                // Reader args
                log_info(tt::LogVerif, " -- Reader Args --");
                const char* reader_arg_names[] = {
                    "in0_mcast_dest_noc_start_x",  // [0]
                    "in0_mcast_dest_noc_start_y",  // [1]
                    "in0_mcast_dest_noc_end_x",    // [2]
                    "in0_mcast_dest_noc_end_y",    // [3]
                    "in0_mcast_num_dests",         // [4]
                    "in0_mcast_sender_noc_x",      // [5]
                    "in0_mcast_sender_noc_y",      // [6]
                    "in0_mcast_sender_semaphore",  // [7]
                    "in0_mcast_receiver_semaphore" // [8]
                    // [9].. dynamic: num_iters_x, num_iters_y, output_idx_x_start, then output y coords
                };
                for (size_t i = 0; i < reader_runtime_args.size(); ++i) {
                    if (i < std::size(reader_arg_names)) {
                        log_info(tt::LogVerif, "reader_arg[{}] ({}) = {}", i, reader_arg_names[i], reader_runtime_args[i]);
                    } else {
                        // dynamic region: provide semantic names depending on position
                        if (i == 9)
                            log_info(tt::LogVerif, "reader_arg[{}] (num_iters_x) = {}", i, reader_runtime_args[i]);
                        else if (i == 10)
                            log_info(tt::LogVerif, "reader_arg[{}] (num_iters_y) = {}", i, reader_runtime_args[i]);
                        else if (i == 11)
                            log_info(tt::LogVerif, "reader_arg[{}] (output_idx_x_start) = {}", i, reader_runtime_args[i]);
                        else
                            log_info(tt::LogVerif, "reader_arg[{}] (output_idx_y[{}]) = {}", i, i - 12, reader_runtime_args[i]);
                    }
                }

                // Writer args
                log_info(tt::LogVerif, " -- Writer Args --");
                const char* writer_arg_names[] = {
                    "out_tensor_start_tile_id", // [0]
                    "num_iters_y_this_core",    // [1]
                    "num_cores_y"               // [2]
                    // [3].. folded_output_idx_y entries
                };
                for (size_t i = 0; i < writer_runtime_args.size(); ++i) {
                    if (i < std::size(writer_arg_names)) {
                        log_info(tt::LogVerif, "writer_arg[{}] ({}) = {}", i, writer_arg_names[i], writer_runtime_args[i]);
                    } else {
                        log_info(tt::LogVerif, "writer_arg[{}] (folded_output_idx_y[{}]) = {}", i, i - 3, writer_runtime_args[i]);
                    }
                }

                // Compute args
                log_info(tt::LogVerif, " -- Compute Args --");
                if (!compute_runtime_args.empty()) {
                    log_info(tt::LogVerif, "compute_arg[0] (num_iters_y) = {}", compute_runtime_args[0]);
                    for (size_t i = 1; i < compute_runtime_args.size(); ++i) {
                        log_info(tt::LogVerif, "compute_arg[{}] (row_size[{}]) = {}", i, i - 1, compute_runtime_args[i]);
                    }
                } else {
                    log_info(tt::LogVerif, "compute_args empty");
                }
            }
        }
    }

    // EnqueueWriteBuffers
    EnqueueWriteBuffer(cq, src0_dram_buffer, a.data.data(), true);
    EnqueueWriteBuffer(cq, src1_dram_buffer, b.data.data(), true);
    EnqueueWriteBuffer(cq, column_indices_dram_buffer, a.indices.data(), true);
    EnqueueWriteBuffer(cq, indptr_dram_buffer, a.indptr.data(), true);

    // TODO: is there a macro for build_Tracy we can invoke here to wrap in a loop and get cooking?
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