#pragma once

#include <sys/types.h>
#include <algorithm>
#include <cstdint>

#include "include_me.hpp"
#include "tt-metalium/assert.hpp"
#include "tt-metalium/bfloat16.hpp"
// #include "tt-metalium/buffer_constants.hpp"
// #include "tt-metalium/circular_buffer_types.hpp"
#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/host_api.hpp"

using namespace tt;
using namespace tt::tt_metal;

using CoreSpec = std::variant<CoreCoord, CoreRange, CoreRangeSet>;

namespace bsr_host_code {

// list of host code function declarations
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
    bool verbose);

void bsr_spmm_multicore_reuse_naive(
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
    bool verbose);

void bsr_spmm_multicore_reuse_many_blocks_per_core(
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
    bool verbose);


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
    bool verbose);

    void bsr_spmm_multicore_load_balanced(
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
    bool verbose);

// TEST: reuse host code with iter device code (iters set to 1)
void bsr_spmm_multicore_host_reuse_device_iter(
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
    bool verbose);

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
    bool verbose);


using HostCodeFunctionPtr = void (*)(
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
    bool verbose);


static std::pair<HostCodeFunctionPtr, std::string> HostCodeRegistry[] = {
    {bsr_spmm_multicore_sparse_mcast, "bsr_spmm_multicore_sparse_mcast"}, 
    // {bsr_spmm_multicore_load_balanced, "bsr_spmm_multicore_load_balanced"},
    // {bsr_spmm_multicore_reuse_iteration, "bsr_spmm_multicore_reuse_iteration"},
    // {bsr_spmm_multicore_reuse_many_blocks_per_core, "bsr_spmm_multicore_reuse_many_blocks_per_core"}, // Defunct!
    // {bsr_spmm_multicore_reuse, "bsr_spmm_multicore_reuse"},
    // {bsr_spmm_multicore_reuse_naive, "bsr_spmm_multicore_reuse_naive"},
    // {bsr_spmm_multicore_host_reuse_device_iter, "bsr_spmm_multicore_host_reuse_device_iter"} // TEST
};


std::shared_ptr<Buffer> MakeBuffer(IDevice* device, uint32_t size, uint32_t page_size, bool sram = false) {
    InterleavedBufferConfig config{
        .device = device,
        .size = size,
        .page_size = page_size,
        .buffer_type = (sram ? BufferType::L1 : BufferType::DRAM)};
    return CreateBuffer(config);
}

std::shared_ptr<Buffer> MakeBuffer(IDevice* device, uint32_t n_tiles, size_t element_size, bool sram = false) {
    const uint32_t tile_size = element_size * TILE_WIDTH * TILE_HEIGHT;
    const uint32_t page_tiles = sram ? n_tiles : 1;
    return MakeBuffer(device, tile_size * n_tiles, page_tiles * tile_size, sram);
}

CBHandle MakeCircularBuffer(
    Program& program, const CoreSpec& core, tt::CBIndex cb, uint32_t size, uint32_t page_size, tt::DataFormat format) {
    CircularBufferConfig cb_src0_config = CircularBufferConfig(size, {{cb, format}}).set_page_size(cb, page_size);
    return CreateCircularBuffer(program, core, cb_src0_config);
}

CBHandle MakeCircularBufferFP32(Program& program, const CoreSpec& core, tt::CBIndex cb, uint32_t n_tiles) {
    constexpr uint32_t tile_size = sizeof(float) * TILE_WIDTH * TILE_HEIGHT;
    return MakeCircularBuffer(program, core, cb, n_tiles * tile_size, tile_size, tt::DataFormat::Float32);
}


uint32_t _get_maximum_block_dim_with_NoC_args(int32_t block_dim, int32_t in0_block_w, int32_t num_tiles_in_NoC_args) {
    int32_t num_available_tiles_in_SRAM = 400; // as provided by TT code. roughly: SRAM size in bytes divided by tile size in bytes
                                               // but i think this is the Grayskull number. not important for now
    num_available_tiles_in_SRAM -= num_tiles_in_NoC_args;
    int32_t other_dim = (num_available_tiles_in_SRAM - 2 * in0_block_w * block_dim) / (2 * in0_block_w + block_dim);
    if (other_dim > 0) {
        return other_dim;
    }
    return 0;
}

uint32_t get_Npc_from_BSR_block_size(uint32_t Nt, uint32_t Mpc, uint32_t in0_block_w, uint32_t num_cores_x, uint32_t num_tiles_for_indexing) {
    auto Nt_fac = get_prime_factors(Nt);
    uint32_t Npc_min = 1;
    for (auto it = Nt_fac.begin(); it != Nt_fac.end(); ++it) {
        auto ele = *it;
        if (ele > num_cores_x) {
            Npc_min *= ele;
            Nt_fac.erase(it);
            --it;
        }
    }
    uint32_t Npc = Npc_min;
    auto Npc_choices = get_possible_products(Nt_fac);
    auto Npc_max = _get_maximum_block_dim_with_NoC_args(Mpc, in0_block_w, num_tiles_for_indexing);
    for (auto& ele : Npc_choices) {
        if (ele * Npc_min <= Npc_max) {
            Npc = ele * Npc_min;
        } else {
            break;
        }
    }

    return Npc;
}

template<class Vals>
void sortingPermutation(const Vals& values, std::vector<int>& v){
    int size = values.size(); 
    v.clear(); v.reserve(size);
    for(int i=0; i < size; ++i)
        v.push_back(i);

    std::sort(v.begin(), v.end(), [&values](int a, int b) -> bool { 
        return values[a] > values[b];
    });
}

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
 
// void bsr_spmm_multicore_load_balanced(
//     bsr_matrix<bfloat16>& a,
//     dense_matrix<bfloat16>& b,
//     dense_matrix<bfloat16>& output,
//     bool bcast_batch,
//     uint32_t nnz_blocks,
//     uint32_t M,
//     uint32_t N,
//     uint32_t K,
//     uint32_t R,
//     uint32_t C,
//     uint32_t B,
//     IDevice* device,
//     bool verbose){
//     // TT-Metal CommandQueue and Program setup
//     CommandQueue& cq = device->command_queue();
//     Program program{};

//     tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
//     MathFidelity math_fidelity = MathFidelity::HiFi4;
//     uint32_t single_tile_size = detail::TileSize(cb_data_format);

//     tt::DataFormat indexing_data_format = tt::DataFormat::Int32;
//     uint32_t indexing_data_single_tile_size = detail::TileSize(indexing_data_format);
//     uint32_t num_tiles_for_col_indices = (indexing_data_single_tile_size - 1 + sizeof(int) * nnz_blocks) / indexing_data_single_tile_size;
//     uint32_t num_tiles_for_indptr = (indexing_data_single_tile_size - 1 + sizeof(int) * (M / R + 1)) / indexing_data_single_tile_size;
//     uint32_t num_tiles_indexing = num_tiles_for_col_indices + num_tiles_for_indptr;

//     // Core Grid detection
//     auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
//     // auto compute_with_storage_grid_size = CoreCoord(3, 1);
//     uint32_t num_cores_x = compute_with_storage_grid_size.x;
//     uint32_t num_cores_y = compute_with_storage_grid_size.y;
//     uint32_t num_cores_total = num_cores_x * num_cores_y;



//     // Per-core tiling and blocking args
//     uint32_t Mt = M / TILE_HEIGHT;
//     uint32_t Kt = K / TILE_WIDTH;
//     uint32_t Nt = N / TILE_WIDTH;

//     uint32_t Rt = R / TILE_HEIGHT;
//     uint32_t Ct = C / TILE_WIDTH;

//     uint32_t in0_block_h = Rt;
//     uint32_t in0_block_w = Ct;
//     uint32_t in1_block_w = get_Npc_from_BSR_block_size(Nt, in0_block_h, in0_block_w, num_cores_x, num_tiles_indexing);

//     TT_ASSERT(Mt % in0_block_h == 0);
//     TT_ASSERT(Nt % in1_block_w == 0);
//     TT_ASSERT(Kt % in0_block_w == 0);

//     // Core grid assignment
//    std::deque<uint32_t> folded_bsr_matrix_indices;
//     uint32_t nnz_rows = 0;
//     uint32_t folded_index = 0;
//     for (uint32_t i = 0; i < a.indptr.size() - 1; i++) {
//         if (a.indptr[i+1] - a.indptr[i] > 0){
//             folded_bsr_matrix_indices.push_back(folded_index);
//             nnz_rows++;
//         }
//         folded_index++;
//     }
//     folded_bsr_matrix_indices.push_back(folded_index);
//     uint32_t height_of_folded_matrix = Rt * nnz_rows;

//     uint32_t num_blocks_x = Nt / in1_block_w;
//     uint32_t num_blocks_y = nnz_rows;
//     uint32_t num_blocks_total = num_blocks_x * num_blocks_y;

//     uint32_t num_iters_x = (num_blocks_x + num_cores_x - 1) / num_cores_x;
//     uint32_t num_iters_y = (num_blocks_y + num_cores_y - 1) / num_cores_y;


//     uint32_t num_work_regions = (num_blocks_total + num_iters_x * num_iters_y - 1)/ (num_iters_x * num_iters_y);
//     uint32_t target_num_cores;
//     if (num_work_regions < num_cores_total)
//         target_num_cores = num_work_regions;
//     else 
//         target_num_cores = num_cores_total;

//     uint32_t out_subblock_h, out_subblock_w;
//     for (auto& subblock_hw : bmm_op_utils::SUBBLOCK_HW_CHOICES) {
//         out_subblock_h = std::get<0>(subblock_hw);
//         out_subblock_w = std::get<1>(subblock_hw);
//         if (in0_block_h % out_subblock_h == 0 and in1_block_w % out_subblock_w == 0) {
//             break;
//         }
//     }

//     CoreCoord start_core = {0, 0};
//     // CoreCoord core_range = bmm_op_utils::get_core_range(num_blocks_y / num_iters_y, num_blocks_x / num_iters_x, num_cores_y, num_cores_x);
//     CoreCoord core_range(0, 0);
//     if ( (num_blocks_y / num_iters_y) <= num_cores_y &&
//         (num_blocks_x / num_iters_x) <= num_cores_x) {
//         core_range.x = (num_blocks_x + num_iters_x - 1) / num_iters_x;
//         core_range.y = (num_blocks_y + num_iters_y - 1) / num_iters_y;
//     }
//     uint32_t start_core_x = start_core.x;
//     uint32_t start_core_y = start_core.y;
//     uint32_t num_cores_c = core_range.x;
//     uint32_t num_cores_r = core_range.y;

//     CoreRange all_cores(
//         {(std::size_t)start_core_x, (std::size_t)start_core_y},
//         {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y + num_cores_r - 1});
    

//     // Circural Buffer sizing
//     uint32_t in0_CB_num_tiles = in0_block_h * in0_block_w * 2; // double buffer
//     uint32_t in0_CB_size = in0_CB_num_tiles * single_tile_size;
//     uint32_t in1_CB_num_tiles = in0_block_w * in1_block_w * 2; // double buffer
//     uint32_t in1_CB_size = in1_CB_num_tiles * single_tile_size;
//     uint32_t out_CB_num_tiles = in0_block_h * in1_block_w; // single buffer
//     uint32_t out_CB_size = out_CB_num_tiles * single_tile_size;

//     uint32_t in0_num_subblocks = (in0_block_h / out_subblock_h);
//     uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;
//     uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w; // this is named weird but it's correct.

//     uint32_t in1_num_subblocks = (in1_block_w / out_subblock_w);
//     uint32_t in1_block_num_tiles = out_subblock_w * in0_block_w * in1_num_subblocks;
//     uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;

//     uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;  


//     // DRAM buffers initialiation
//     uint32_t dram_buffer_dst_row_size =
//         single_tile_size * Rt * Nt;

//     uint32_t dram_buffer_dst_total_size = dram_buffer_dst_row_size * nnz_rows;

//     uint32_t dram_buffer_A_size =
//         single_tile_size * Rt * Ct * nnz_blocks;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels
//     uint32_t dram_buffer_B_size =
//         single_tile_size * Nt * Kt;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    
//     uint32_t dram_buffer_col_indices_size = 
//         sizeof(indexing_data_format) * nnz_blocks;
//     // Round up to tile size
//     dram_buffer_col_indices_size = indexing_data_single_tile_size * ((indexing_data_single_tile_size - 1 + dram_buffer_col_indices_size) / (indexing_data_single_tile_size));

//     uint32_t dram_buffer_indptr_size = 
//         sizeof(indexing_data_format) * (M / R + 1);
//     // Round up to tile size
//     dram_buffer_indptr_size = indexing_data_single_tile_size * ((indexing_data_single_tile_size - 1 + dram_buffer_indptr_size) / (indexing_data_single_tile_size));
    
//     auto dst_dram_buffer = MakeBuffer(device, dram_buffer_dst_total_size, single_tile_size);
//     auto src0_dram_buffer = MakeBuffer(device, dram_buffer_A_size, single_tile_size);
//     auto src1_dram_buffer = MakeBuffer(device, dram_buffer_B_size, single_tile_size);
//     auto column_indices_dram_buffer = MakeBuffer(device, dram_buffer_col_indices_size, dram_buffer_col_indices_size);
//     auto indptr_dram_buffer = MakeBuffer(device, dram_buffer_indptr_size, dram_buffer_indptr_size);


//     if (verbose) {
//         log_info(tt::LogVerif, " -- Metalium Block and subblock sizing --");
//         log_info(
//             tt::LogVerif,
//             " -- per_core_M={} -- per_block_M={} -- per_core_N={} -- out_subblock_h={} -- out_subblock_w={} --",
//             num_iters_y * in0_block_h,
//             in0_block_h,
//             num_iters_x * in1_block_w,
//             out_subblock_h,
//             out_subblock_w);
//     }

//     if (verbose) {
//         log_info(tt::LogVerif, " -- Core Grid Allocaiton Information --");
//         log_info(
//             tt::LogVerif,
//             " -- available_cores_y={} -- available_cores_x={} -- num_iters_y={} -- num_iters_x={} -- nnz_rows={} --",
//             num_cores_y,
//             num_cores_x,
//             num_iters_y,
//             num_iters_x, 
//             nnz_rows);
//     }

//     if (verbose) {
//         log_info(tt::LogVerif, " -- Metalium Core Grid Sizing --");
//         log_info(
//             tt::LogVerif,
//             " -- Mt= {} -- Nt= {} -- num_output_blocks= {} -- cores_used={} -- num_blocks_x={} -- num_blocks_y={} --",
//             Mt,
//             Nt,
//             num_blocks_total,
//             all_cores,
//             num_blocks_x,
//             num_blocks_y);
//     }


//     /*
//     SRAM Circular Buffers
//     */
//     uint32_t src0_cb_index = CBIndex::c_0;  // 0
//     CircularBufferConfig cb_src0_config = CircularBufferConfig(in0_CB_size, {{src0_cb_index, cb_data_format}})
//                                               .set_page_size(src0_cb_index, single_tile_size);
//     auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

//     uint32_t src1_cb_index = CBIndex::c_1;  // 1
//     CircularBufferConfig cb_src1_config = CircularBufferConfig(in1_CB_size, {{src1_cb_index, cb_data_format}})
//                                               .set_page_size(src1_cb_index, single_tile_size);
//     auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

//     uint32_t output_cb_index = tt::CBIndex::c_16;
//     uint32_t interm0_cb_index = tt::CBIndex::c_24;
//     std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec{
//         {output_cb_index, cb_data_format}, {interm0_cb_index, cb_data_format}};
//         CircularBufferConfig cb_output_config = CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
//         .set_page_size(output_cb_index, single_tile_size)
//         .set_page_size(interm0_cb_index, single_tile_size);
//         auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

//     uint32_t column_indices_cb_index = CBIndex::c_2;  // 2
//     CircularBufferConfig cb_column_indices_config = CircularBufferConfig(
//         dram_buffer_col_indices_size, {{column_indices_cb_index, tt::DataFormat::Int32}})
//                                                 .set_page_size(column_indices_cb_index, dram_buffer_col_indices_size);
//     auto cb_column_indices = tt_metal::CreateCircularBuffer(program, all_cores, cb_column_indices_config);

//     auto indptr_cb_index = CBIndex::c_3; // 3
//     auto cb_indptr = MakeCircularBuffer(program, all_cores, indptr_cb_index, dram_buffer_indptr_size, dram_buffer_indptr_size, indexing_data_format);


//     // Compiletime arguments
//     bool src0_is_dram = src0_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
//     bool src1_is_dram = src1_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
//     bool col_indices_is_dram = column_indices_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
//     bool indptr_is_dram = indptr_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
//     std::vector<uint32_t> reader_compile_time_args = {
//         (std::uint32_t)src0_is_dram,
//         (std::uint32_t)src1_is_dram,
//         (std::uint32_t)col_indices_is_dram,
//         (std::uint32_t)indptr_is_dram,

//         (std::uint32_t)src0_dram_buffer->address(),     // in0_tensor_addr
//         (std::uint32_t)1,                               // in0_tensor_stride_w
//         (std::uint32_t)Ct,                              // in0_tensor_stride_h

//         (std::uint32_t)in0_block_w,               // in0_block_w
//         (std::uint32_t)Rt,                         // in0_block_h
//         (std::uint32_t)in0_block_w * Rt,  // in0_block_num_tiles

//         (std::uint32_t)src1_dram_buffer->address(),  // in1_tensor_addr
//         (std::uint32_t)1,                            // in1_tensor_stride_w
//         (std::uint32_t)Nt,                           // in1_tensor_stride_h

//         (std::uint32_t)in1_block_w,                // in1_block_w
//         (std::uint32_t)in0_block_w,               // in1_block_h
//         (std::uint32_t)in1_block_w * in0_block_w,  // in1_block_num_tiles


//         (std::uint32_t)column_indices_dram_buffer->address(), // NoC args, column indices
//         (std::uint32_t)indptr_dram_buffer->address(), // NoC args, indptr

//         (std::uint32_t)num_tiles_for_col_indices, 
//         (std::uint32_t)num_tiles_for_indptr,

//         // in0_tensor_start_tile_id obtained by // a.indptr[output_idx_y] * Rt * Ct,
//         // in1_tensor_start_tile_id obtained by // per_core_N * output_idx_x
//         // col indices start of row obtained by // a.indptr[output_idx_y],
//         // col indices end of row obtained by //  a.indptr[output_idx_y + 1],
//     };

//     std::vector<uint32_t> compute_kernel_compile_time_args = {
//         (std::uint32_t)in0_block_w,
//         (std::uint32_t)in0_num_subblocks,
//         (std::uint32_t)in0_block_num_tiles,
//         (std::uint32_t)in0_subblock_num_tiles,
//         (std::uint32_t)in1_num_subblocks,
//         (std::uint32_t)in1_block_num_tiles,
//         (std::uint32_t)in1_per_core_w,
//         (std::uint32_t)out_subblock_h,
//         (std::uint32_t)out_subblock_w,
//         (std::uint32_t)out_subblock_num_tiles,
//         (std::uint32_t)num_iters_x,
//     };

//     bool out_is_dram = dst_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
//     std::vector<uint32_t> writer_compile_time_args = {
//         (std::uint32_t)out_is_dram,
//         (std::uint32_t)dst_dram_buffer->address(),      // out_buffer_addr
//         (std::uint32_t)1,                           // out_tensor_stride_w
//         (std::uint32_t)Nt,                         // out_tensor_stride_h
//         (std::uint32_t)out_subblock_w,       // out_tensor_next_subblock_stride_w
//         (std::uint32_t)out_subblock_h * Nt,  // out_tensor_next_subblock_stride_h

//         (std::uint32_t)out_subblock_w,                     // out_subblock_w
//         (std::uint32_t)out_subblock_h,                     // out_subblock_h
//         (std::uint32_t)(out_subblock_w * out_subblock_h),  // out_subblocks_w * out_subblocks_h
//         (std::uint32_t)(in1_block_w / out_subblock_w),      // out_num_subblocks_w
//         (std::uint32_t)(in0_block_h / out_subblock_h),      // out_num_subblocks_h

//         (std::uint32_t)Rt * Nt,  // Size of output row, used to index into next output block
//         (std::uint32_t)num_iters_x,
//     };

//     if (verbose) {
//         auto print_args = [](const std::string& name, const std::vector<uint32_t>& args, const std::vector<std::string>& arg_names) {
//             std::cout << "==== " << name << " ====" << std::endl;
//             for (size_t i = 0; i < args.size(); ++i) {
//                 if (i < arg_names.size())
//                     std::cout << "  [" << i << "] " << arg_names[i] << " = " << args[i] << std::endl;
//                 else
//                     std::cout << "  [" << i << "] = " << args[i] << std::endl;
//             }
//             std::cout << std::endl;
//         };

//         print_args("reader_compile_time_args", reader_compile_time_args, {
//             "src0_is_dram",
//             "src1_is_dram",
//             "col_indices_is_dram",
//             "indptr_is_dram",
//             "src0_dram_buffer->address()",
//             "in0_tensor_stride_w",
//             "in0_tensor_stride_h",
//             "in0_block_w",
//             "in0_block_h",
//             "in0_block_num_tiles",
//             "src1_dram_buffer->address()",
//             "in1_tensor_stride_w",
//             "in1_tensor_stride_h",
//             "in1_block_w",
//             "in1_block_h",
//             "in1_block_num_tiles",
//             "column_indices_dram_buffer->address()",
//             "indptr_dram_buffer->address()",
//             "num_tiles_for_col_indices",
//             "num_tiles_for_indptr"
//         });

//         print_args("compute_kernel_compile_time_args", compute_kernel_compile_time_args, {
//             "in0_block_w",
//             "in0_num_subblocks",
//             "in0_block_num_tiles",
//             "in0_subblock_num_tiles",
//             "in1_num_subblocks",
//             "in1_block_num_tiles",
//             "in1_per_core_w",
//             "out_subblock_h",
//             "out_subblock_w",
//             "out_subblock_num_tiles",
//             "num_iters_x",
//         });

//         print_args("writer_compile_time_args", writer_compile_time_args, {
//             "out_is_dram",
//             "dst_dram_buffer->address()",
//             "out_tensor_stride_w",
//             "out_tensor_stride_h",
//             "out_tensor_next_subblock_stride_w",
//             "out_tensor_next_subblock_stride_h",
//             "out_subblock_w",
//             "out_subblock_h",
//             "out_subblock_w * out_subblock_h",
//             "out_num_subblocks_w",
//             "out_num_subblocks_h",
//             "Rt * Nt",
//             "num_iters_x",
//         });
//     }

//     // Create Kernels
//     auto reader_id = tt_metal::CreateKernel(
//         program,
//         "tt_metal/programming_examples/rahmy/block_spmm/kernels/dataflow/reader_block_iter.cpp",
//         all_cores,
//         tt_metal::DataMovementConfig{
//             .processor = DataMovementProcessor::RISCV_0,
//             .noc = NOC::RISCV_0_default,
//             .compile_args = reader_compile_time_args});

//     auto writer_id = tt_metal::CreateKernel(
//         program,
//         "tt_metal/programming_examples/rahmy/block_spmm/kernels/dataflow/writer_block_load_balanced.cpp",
//         all_cores,
//         tt_metal::DataMovementConfig{
//             .processor = DataMovementProcessor::RISCV_1,
//             .noc = NOC::RISCV_1_default,
//             .compile_args = writer_compile_time_args});

//     // Create compute kernel
//     auto mm_kernel_id = tt_metal::CreateKernel(
//         program,
//         "tt_metal/programming_examples/rahmy/block_spmm/kernels/compute/bmm_iter.cpp",
//         all_cores,
//         tt_metal::ComputeConfig{.math_fidelity = math_fidelity,
//                                         // .fp32_dest_acc_en = true,
//                                         .compile_args = compute_kernel_compile_time_args});
//     // Runtime arguments
//     uint32_t work_region = 0;

//     // Scanning for load-balancing
//     // 0. Sort block rows by number of nonzero blocks, get perm vector
//     // TODO: get folded indices first, then sort and perm
//     uint32_t num_empty_rows = (M / R) - nnz_rows;
//     std::vector<int> row_diffs;
//     // for (int i = 0; i < a.indptr.size() - 1; i++){
//     //     row_diffs.push_back(a.indptr[i+1] - a.indptr[i]);
//     // }
//     // std::vector<int> perm(row_diffs.size());
//     // sortingPermutation(row_diffs, perm);

//     // // remove last num_empty_rows elements from perm
//     // perm.resize(nnz_rows);

//     for (int i = 0; i < folded_bsr_matrix_indices.size() - 1; i++){
//         row_diffs.push_back(folded_bsr_matrix_indices[i+1] - folded_bsr_matrix_indices[i]);
//     }
//     std::vector<int> perm(row_diffs.size());
//     sortingPermutation(row_diffs, perm);

//     // remove last num_empty_rows elements from perm
//     perm.resize(nnz_rows);

//     if (verbose){
//         std::cout << "row diffs: ";
//         for (int i = 0; i < row_diffs.size(); i++){
//             std::cout << row_diffs[i] << ' ';
//         }
//         std::cout << std::endl;
//         std::cout << std::endl;
//         std::cout << "perm: ";
//         for (int i = 0; i < perm.size(); i++){
//             std::cout << perm[i] << ' ';
//         }
//     }
//     // 1. initialize a vector for each row of cores
//     std::vector<std::vector<uint32_t>> output_y_indices(num_cores_r, std::vector<uint32_t>());
//     // 2. While count is less than num output blocks 
//     uint32_t num_rows_assigned = 0;
//     uint32_t iter_count = 1;
//     uint32_t subarray_iter = 0;
//     while (num_rows_assigned < nnz_rows) {
//         uint32_t num_rows_to_assign = std::min(num_cores_r, nnz_rows - num_rows_assigned);
//         subarray_iter = 0;
//         for (uint32_t core_row = 0; core_row < num_rows_to_assign; core_row++){
//             if (num_rows_assigned++ >= nnz_rows)
//                 break;
//             output_y_indices[core_row].push_back(perm[num_cores_r * (iter_count - 1) + subarray_iter]);
//             subarray_iter++;
//         }
//         num_rows_to_assign = std::min(num_cores_r, nnz_rows - num_rows_assigned);
//         subarray_iter = num_cores_r - num_rows_to_assign;
//         for (uint32_t core_row = num_cores_r; core_row > num_cores_r - num_rows_to_assign; core_row--){
//             if (num_rows_assigned++ >= nnz_rows)
//                 break;
//             output_y_indices[core_row - 1].push_back(perm[nnz_rows - num_cores_r * iter_count + subarray_iter]);
//             subarray_iter++;
//         }
//         iter_count++;
//     }

//     for (uint32_t core_idx_y = 0; core_idx_y < num_cores_r; core_idx_y++) {
//         for (uint32_t core_idx_x = 0; core_idx_x < num_cores_c; core_idx_x++) {
//             CoreCoord core(core_idx_x, core_idx_y);
//             if (verbose)
//               log_info(tt::LogVerif, "Core x {} y {}", core_idx_x, core_idx_y);
                
//             int output_idx_x_start = (core_idx_x * num_iters_x) % num_blocks_x;
//             work_region++;

//             std::vector<uint32_t> reader_runtime_args;
//             std::vector<uint32_t> compute_runtime_args;
//             std::vector<uint32_t> writer_runtime_args;

//             uint32_t num_iters_y_this_core = output_y_indices[core_idx_y].size();
//             uint32_t num_iters_x_this_core = std::min(num_iters_x, num_blocks_x - output_idx_x_start + 1);
//             reader_runtime_args.push_back(num_iters_x_this_core);
//             reader_runtime_args.push_back(num_iters_y_this_core);
//             compute_runtime_args.push_back(num_iters_y_this_core);
//             reader_runtime_args.push_back(output_idx_x_start);
//             writer_runtime_args.push_back(output_idx_x_start * in1_block_w);
//             writer_runtime_args.push_back(num_iters_y_this_core);
//             writer_runtime_args.push_back(num_cores_r);
//             for (int iter_y = 0; iter_y < num_iters_y_this_core; iter_y++) {
//                 uint32_t folded_output_idx_y = output_y_indices[core_idx_y][iter_y];
//                 uint32_t output_idx_y = folded_bsr_matrix_indices[folded_output_idx_y];
//                 reader_runtime_args.push_back(output_idx_y);
//                 writer_runtime_args.push_back(folded_output_idx_y);
//                 compute_runtime_args.push_back(a.indptr[output_idx_y + 1] - a.indptr[output_idx_y]);
//             }


//             tt_metal::SetRuntimeArgs(program, reader_id, core, reader_runtime_args);
//             tt_metal::SetRuntimeArgs(program, mm_kernel_id, core, compute_runtime_args);
//             tt_metal::SetRuntimeArgs(program, writer_id, core, writer_runtime_args);

//             if (verbose){
//                 if (num_iters_x_this_core < num_iters_x || num_iters_y_this_core < num_iters_y){
//                     log_info(tt::LogVerif, " -- Num iters diverged! --");
//                     log_info(tt::LogVerif, "(num_iters_x_this_core) = {}", num_iters_x_this_core);
//                     log_info(tt::LogVerif, "(num_iters_y_this_core) = {}", num_iters_y_this_core);
//                 }
//             }

//             if (verbose && core_idx_x == 0 && core_idx_y == 7) {
//                 a.pretty_print();
//                 log_info(tt::LogVerif, " -- Reader Args --");
//                 log_info(tt::LogVerif, "reader_arg[0] (num_iters_x) = {}", reader_runtime_args[0]);
//                 log_info(tt::LogVerif, "reader_arg[1] (num_iters_y) = {}",  reader_runtime_args[1]);
//                 log_info(tt::LogVerif, "reader_arg[2] (output_idx_x_start) = {}",  reader_runtime_args[2]);
//                 for (size_t i = 0; i < num_iters_y_this_core; ++i) {
//                     log_info(tt::LogVerif, "reader_arg[{}] (y_coord) = {}", i + 3, reader_runtime_args[i+3]);
//                 }

//                 log_info(tt::LogVerif, " -- Writer Args --");
//                 const char* writer_arg_names[] = {
//                     "out_tensor_start_tile_id",
//                     "num_iters_y_this_core",
//                     "num_cores_y"
//                 };
//                 for (size_t i = 0; i < 3; ++i) {
//                     log_info(tt::LogVerif, "writer_arg[{}] ({}) = {}", i, writer_arg_names[i], writer_runtime_args[i]);
//                 }
//                 for (size_t i = 3; i < writer_runtime_args.size(); ++i) {
//                     log_info(tt::LogVerif, "writer_arg[{}] (output_idx_y) = {}", i, writer_runtime_args[i]);
//                 }
//                 log_info(tt::LogVerif, " -- Compute Args --");
//                 log_info(tt::LogVerif, "compute_arg[0] (num_iters_y) = {}", compute_runtime_args[0]);

//                 for (size_t i = 1; i < compute_runtime_args.size(); ++i) {
//                     log_info(tt::LogVerif, "compute_arg[{}] (row_size) = {}", i, compute_runtime_args[i]);
//                 }
//             }
//         }
//     } 

//     // EnqueueWriteBuffers
//     EnqueueWriteBuffer(cq, src0_dram_buffer, a.data.data(), true);
//     EnqueueWriteBuffer(cq, src1_dram_buffer, b.data.data(), true);
//     EnqueueWriteBuffer(cq, column_indices_dram_buffer, a.indices.data(), true);
//     EnqueueWriteBuffer(cq, indptr_dram_buffer, a.indptr.data(), true);
    
//     // TODO: is there a macro for build_Tracy we can invoke here to wrap in a loop and get cooking?
//     EnqueueProgram(cq, program, true);

//     if (verbose)
//         log_info(tt::LogVerif, " -- Program returned --");
//     // EnqueueReadSubBuffers
//     uint32_t nonzero_row_index = 0;
//     for (size_t row_index = 0; row_index < a.indptr.size() - 1; row_index++) {
//         if (a.indptr[row_index+1] - a.indptr[row_index] == 0)
//             continue;
//         BufferRegion DRAM_row(nonzero_row_index * dram_buffer_dst_row_size, dram_buffer_dst_row_size);
//         EnqueueReadSubBuffer(cq, dst_dram_buffer, output.data.data() + (row_index * R * N), DRAM_row, true);
//         nonzero_row_index++;
//     }

//     if (verbose)
//         log_info(tt::LogVerif, " -- Finished reading output --");
//     Finish(cq);
// }

// void bsr_spmm_multicore_reuse_iteration(
//     bsr_matrix<bfloat16>& a,
//     dense_matrix<bfloat16>& b,
//     dense_matrix<bfloat16>& output,
//     bool bcast_batch,
//     uint32_t nnz_blocks,
//     uint32_t M,
//     uint32_t N,
//     uint32_t K,
//     uint32_t R,
//     uint32_t C,
//     uint32_t B,
//     IDevice* device,
//     bool verbose){

//     /*
//     Host code considerations:
//         - call Mpc, Npc the tile counts of CB sizes
//         - Mpc = Rt
//         - BSR Block size informs maximum Npc
//         - Mt, Nt, Mpc, max Npc inform Npc, num_iters_y, num_iters_x
//             - Ah! the num_iters_y/x split kinda informs which dimension 
//                 we should be biased on...
//                 If we only iter on y, one core gets many rows, each with 
//                 a sparsity pattern. IDEA: "Mcast sharing"
//                     - Which means we could let some other core concurrently
//                         get the same row and mcast share.
//                 If we only iter on x, one core gets one input row, with
//                 only one sparsity pattern. And is guaranteed to reuse 
//                 all the blocks in that row. IDEA: "self-circulation"
//             - So we bias on num_iters_y (ie, bias for large Npc). 
//             - Generally let's say num_iters_y and num_iters_x are gt 1.
//                 - Like in test 41 with core grid {2, 2}
//                 - IDEA: can analyze pattern ahead of time to see which of the two above ideas is optimal
//         - RK -> 
//              -> for each num_iter_y, needs brs&bre into indptr, indices
//              -> for each num_iter_x, just needs the count num_iter_x
//              -> needs num_tiles for col_indices, indptr
//         - CK -> 
//              -> for each num_iter_y, needs num_blocks (bre-brs)
//              -> for each num_iter_x, just needs the count num_iter_x
//         - WK -> 
//              -> just needs start tile of output region and iter counts

//         RK -> (brs&bre)*num_iter_y - runtime_args
//            -> num_iter_x - compiletime
        
//         CK -> nblocks - runtime
//            -> num_iter_x - compiletime

//         WK -> both compile_time
        

//     */
//     /*
//     some unstructured thoughts

//     The set up all woked great actually. The folded matrix, the runtime args, the compiletime args, the work distribution.
//     It's just the set up had some incorretct assumptions. 

//     Under the new, more relaxed assumptions, we need a few more things
//         - num_iter_x as a common compiletime arg (unlike ycoords, we can rely on xcoords being a range)
//         - that's it? 

//     It is being pointed out to me in 510 that I should be testing each "thing"
//         I've already decided it's too hard to test each kernel specifcally
//         I've already tested my bsr_matrix code
//         I have NOT tested the folding, and othe rthings like that (what other things?)
//     */

//     // program
//     // command queue
//     // data format constants

//     // TT-Metal CommandQueue and Program setup
//     CommandQueue& cq = device->command_queue();
//     Program program{};

//     tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
//     MathFidelity math_fidelity = MathFidelity::HiFi4;
//     uint32_t single_tile_size = detail::TileSize(cb_data_format);

//     tt::DataFormat indexing_data_format = tt::DataFormat::Int32;
//     uint32_t indexing_data_single_tile_size = detail::TileSize(indexing_data_format);
//     uint32_t num_tiles_for_col_indices = (indexing_data_single_tile_size - 1 + sizeof(int) * nnz_blocks) / indexing_data_single_tile_size;
//     uint32_t num_tiles_for_indptr = (indexing_data_single_tile_size - 1 + sizeof(int) * (M / R + 1)) / indexing_data_single_tile_size;
//     uint32_t num_tiles_indexing = num_tiles_for_col_indices + num_tiles_for_indptr;

//     // Core Grid detection
//     auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
//     // auto compute_with_storage_grid_size = CoreCoord(3, 1);
//     uint32_t num_cores_x = compute_with_storage_grid_size.x;
//     uint32_t num_cores_y = compute_with_storage_grid_size.y;
//     uint32_t num_cores_total = num_cores_x * num_cores_y;



//     // Per-core tiling and blocking args
//     uint32_t Mt = M / TILE_HEIGHT;
//     uint32_t Kt = K / TILE_WIDTH;
//     uint32_t Nt = N / TILE_WIDTH;

//     uint32_t Rt = R / TILE_HEIGHT;
//     uint32_t Ct = C / TILE_WIDTH;

//     uint32_t in0_block_h = Rt;
//     uint32_t in0_block_w = Ct;
//     uint32_t in1_block_w = get_Npc_from_BSR_block_size(Nt, in0_block_h, in0_block_w, num_cores_x, num_tiles_indexing);

//     TT_ASSERT(Mt % in0_block_h == 0);
//     TT_ASSERT(Nt % in1_block_w == 0);
//     TT_ASSERT(Kt % in0_block_w == 0);

//     // Core grid assignment
//    std::vector<uint32_t> folded_bsr_matrix_indices;
//     uint32_t nnz_rows = 0;
//     uint32_t folded_index = 0;
//     for (uint32_t i = 0; i < a.indptr.size() - 1; i++) {
//         if (a.indptr[i+1] - a.indptr[i] > 0){
//             folded_bsr_matrix_indices.push_back(folded_index);
//             nnz_rows++;
//         }
//         folded_index++;
//     }
//     uint32_t height_of_folded_matrix = Rt * nnz_rows;

//     uint32_t num_blocks_x = Nt / in1_block_w;
//     uint32_t num_blocks_y = nnz_rows;
//     uint32_t num_blocks_total = num_blocks_x * num_blocks_y;

//     uint32_t num_iters_x = (num_blocks_x + num_cores_x - 1) / num_cores_x;
//     uint32_t num_iters_y = (num_blocks_y + num_cores_y - 1) / num_cores_y;

//     uint32_t num_work_regions = (num_blocks_total + num_iters_x * num_iters_y - 1)/ (num_iters_x * num_iters_y);
//     uint32_t target_num_cores;
//     if (num_work_regions < num_cores_total)
//         target_num_cores = num_work_regions;
//     else 
//         target_num_cores = num_cores_total;

    
//     CoreRangeSet all_cores(
//         tt::tt_metal::num_cores_to_corerangeset(target_num_cores, compute_with_storage_grid_size, true));
    
//     // CoreCoord all_cores(0, 0);
//     // all_cores.x = std::min(num_blocks_x, num_cores_x);
//     // all_cores.y = std::min(num_blocks_y, num_cores_y);
//     uint32_t out_subblock_h, out_subblock_w;
//     for (auto& subblock_hw : bmm_op_utils::SUBBLOCK_HW_CHOICES) {
//         out_subblock_h = std::get<0>(subblock_hw);
//         out_subblock_w = std::get<1>(subblock_hw);
//         if (in0_block_h % out_subblock_h == 0 and in1_block_w % out_subblock_w == 0) {
//             break;
//         }
//     }

//     // Circural Buffer sizing
//     uint32_t in0_CB_num_tiles = in0_block_h * in0_block_w * 2; // double buffer
//     uint32_t in0_CB_size = in0_CB_num_tiles * single_tile_size;
//     uint32_t in1_CB_num_tiles = in0_block_w * in1_block_w * 2; // double buffer
//     uint32_t in1_CB_size = in1_CB_num_tiles * single_tile_size;
//     uint32_t out_CB_num_tiles = in0_block_h * in1_block_w; // single buffer
//     uint32_t out_CB_size = out_CB_num_tiles * single_tile_size;

//     uint32_t in0_num_subblocks = (in0_block_h / out_subblock_h);
//     uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;
//     uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w; // this is named weird but it's correct.

//     uint32_t in1_num_subblocks = (in1_block_w / out_subblock_w);
//     uint32_t in1_block_num_tiles = out_subblock_w * in0_block_w * in1_num_subblocks;
//     uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;

//     uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;  


//     // DRAM buffers initialiation
//     uint32_t dram_buffer_dst_row_size =
//         single_tile_size * Rt * Nt;

//     uint32_t dram_buffer_dst_total_size = dram_buffer_dst_row_size * nnz_rows;

//     uint32_t dram_buffer_A_size =
//         single_tile_size * Rt * Ct * nnz_blocks;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels
//     uint32_t dram_buffer_B_size =
//         single_tile_size * Nt * Kt;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    
//     uint32_t dram_buffer_col_indices_size = 
//         sizeof(indexing_data_format) * nnz_blocks;
//     // Round up to tile size
//     dram_buffer_col_indices_size = indexing_data_single_tile_size * ((indexing_data_single_tile_size - 1 + dram_buffer_col_indices_size) / (indexing_data_single_tile_size));

//     uint32_t dram_buffer_indptr_size = 
//         sizeof(indexing_data_format) * (M / R + 1);
//     // Round up to tile size
//     dram_buffer_indptr_size = indexing_data_single_tile_size * ((indexing_data_single_tile_size - 1 + dram_buffer_indptr_size) / (indexing_data_single_tile_size));
    
//     auto dst_dram_buffer = MakeBuffer(device, dram_buffer_dst_total_size, single_tile_size);
//     auto src0_dram_buffer = MakeBuffer(device, dram_buffer_A_size, single_tile_size);
//     auto src1_dram_buffer = MakeBuffer(device, dram_buffer_B_size, single_tile_size);
//     auto column_indices_dram_buffer = MakeBuffer(device, dram_buffer_col_indices_size, dram_buffer_col_indices_size);
//     auto indptr_dram_buffer = MakeBuffer(device, dram_buffer_indptr_size, dram_buffer_indptr_size);


//     if (verbose) {
//         log_info(tt::LogVerif, " -- Metalium Block and subblock sizing --");
//         log_info(
//             tt::LogVerif,
//             " -- per_core_M={} -- per_block_M={} -- per_core_N={} -- out_subblock_h={} -- out_subblock_w={} --",
//             num_iters_y * in0_block_h,
//             in0_block_h,
//             num_iters_x * in1_block_w,
//             out_subblock_h,
//             out_subblock_w);
//     }

//     if (verbose) {
//         log_info(tt::LogVerif, " -- Core Grid Allocaiton Information --");
//         log_info(
//             tt::LogVerif,
//             " -- num_cores_y={} -- num_cores_x={} -- num_iters_y={} -- num_iters_x={} -- nnz_rows={} --",
//             num_cores_y,
//             num_cores_x,
//             num_iters_y,
//             num_iters_x, 
//             nnz_rows);
//     }

//     if (verbose) {
//         log_info(tt::LogVerif, " -- Metalium Core Grid Sizing --");
//         log_info(
//             tt::LogVerif,
//             " -- Mt= {} -- Nt= {} -- num_output_blocks= {} -- num_cores_used={} -- num_cores_available_x={} -- num_cores_available_y={} --",
//             Mt,
//             Nt,
//             num_blocks_total,
//             all_cores.num_cores(),
//             num_cores_x,
//             num_cores_y);
//     }


//     /*
//     SRAM Circular Buffers
//     */
//     uint32_t src0_cb_index = CBIndex::c_0;  // 0
//     CircularBufferConfig cb_src0_config = CircularBufferConfig(in0_CB_size, {{src0_cb_index, cb_data_format}})
//                                               .set_page_size(src0_cb_index, single_tile_size);
//     auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

//     uint32_t src1_cb_index = CBIndex::c_1;  // 1
//     CircularBufferConfig cb_src1_config = CircularBufferConfig(in1_CB_size, {{src1_cb_index, cb_data_format}})
//                                               .set_page_size(src1_cb_index, single_tile_size);
//     auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

//     uint32_t output_cb_index = tt::CBIndex::c_16;
//     uint32_t interm0_cb_index = tt::CBIndex::c_24;
//     std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec{
//         {output_cb_index, cb_data_format}, {interm0_cb_index, cb_data_format}};
//         CircularBufferConfig cb_output_config = CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
//         .set_page_size(output_cb_index, single_tile_size)
//         .set_page_size(interm0_cb_index, single_tile_size);
//         auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

//     uint32_t column_indices_cb_index = CBIndex::c_2;  // 2
//     CircularBufferConfig cb_column_indices_config = CircularBufferConfig(
//         dram_buffer_col_indices_size, {{column_indices_cb_index, tt::DataFormat::Int32}})
//                                                 .set_page_size(column_indices_cb_index, dram_buffer_col_indices_size);
//     auto cb_column_indices = tt_metal::CreateCircularBuffer(program, all_cores, cb_column_indices_config);

//     auto indptr_cb_index = CBIndex::c_3; // 3
//     auto cb_indptr = MakeCircularBuffer(program, all_cores, indptr_cb_index, dram_buffer_indptr_size, dram_buffer_indptr_size, indexing_data_format);


//     // Compiletime arguments
//     bool src0_is_dram = src0_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
//     bool src1_is_dram = src1_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
//     bool col_indices_is_dram = column_indices_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
//     bool indptr_is_dram = indptr_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
//     std::vector<uint32_t> reader_compile_time_args = {
//         (std::uint32_t)src0_is_dram,
//         (std::uint32_t)src1_is_dram,
//         (std::uint32_t)col_indices_is_dram,
//         (std::uint32_t)indptr_is_dram,

//         (std::uint32_t)src0_dram_buffer->address(),     // in0_tensor_addr
//         (std::uint32_t)1,                               // in0_tensor_stride_w
//         (std::uint32_t)Ct,                              // in0_tensor_stride_h

//         (std::uint32_t)in0_block_w,               // in0_block_w
//         (std::uint32_t)Rt,                         // in0_block_h
//         (std::uint32_t)in0_block_w * Rt,  // in0_block_num_tiles

//         (std::uint32_t)src1_dram_buffer->address(),  // in1_tensor_addr
//         (std::uint32_t)1,                            // in1_tensor_stride_w
//         (std::uint32_t)Nt,                           // in1_tensor_stride_h

//         (std::uint32_t)in1_block_w,                // in1_block_w
//         (std::uint32_t)in0_block_w,               // in1_block_h
//         (std::uint32_t)in1_block_w * in0_block_w,  // in1_block_num_tiles


//         (std::uint32_t)column_indices_dram_buffer->address(), // NoC args, column indices
//         (std::uint32_t)indptr_dram_buffer->address(), // NoC args, indptr

//         (std::uint32_t)num_tiles_for_col_indices, 
//         (std::uint32_t)num_tiles_for_indptr,

//         // in0_tensor_start_tile_id obtained by // a.indptr[output_idx_y] * Rt * Ct,
//         // in1_tensor_start_tile_id obtained by // per_core_N * output_idx_x
//         // col indices start of row obtained by // a.indptr[output_idx_y],
//         // col indices end of row obtained by //  a.indptr[output_idx_y + 1],
//     };

//     std::vector<uint32_t> compute_kernel_compile_time_args = {
//         (std::uint32_t)in0_block_w,
//         (std::uint32_t)in0_num_subblocks,
//         (std::uint32_t)in0_block_num_tiles,
//         (std::uint32_t)in0_subblock_num_tiles,
//         (std::uint32_t)in1_num_subblocks,
//         (std::uint32_t)in1_block_num_tiles,
//         (std::uint32_t)in1_per_core_w,
//         (std::uint32_t)out_subblock_h,
//         (std::uint32_t)out_subblock_w,
//         (std::uint32_t)out_subblock_num_tiles,
//         (std::uint32_t)num_iters_x,
//     };

//     bool out_is_dram = dst_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
//     std::vector<uint32_t> writer_compile_time_args = {
//         (std::uint32_t)out_is_dram,
//         (std::uint32_t)dst_dram_buffer->address(),      // out_buffer_addr
//         (std::uint32_t)1,                           // out_tensor_stride_w
//         (std::uint32_t)Nt,                         // out_tensor_stride_h
//         (std::uint32_t)out_subblock_w,       // out_tensor_next_subblock_stride_w
//         (std::uint32_t)out_subblock_h * Nt,  // out_tensor_next_subblock_stride_h

//         (std::uint32_t)out_subblock_w,                     // out_subblock_w
//         (std::uint32_t)out_subblock_h,                     // out_subblock_h
//         (std::uint32_t)(out_subblock_w * out_subblock_h),  // out_subblocks_w * out_subblocks_h
//         (std::uint32_t)(in1_block_w / out_subblock_w),      // out_num_subblocks_w
//         (std::uint32_t)(in0_block_h / out_subblock_h),      // out_num_subblocks_h

//         (std::uint32_t)Rt * Nt,  // Size of output row, used to index into next output block
//         (std::uint32_t)num_iters_x,
//     };

//     if (verbose) {
//         auto print_args = [](const std::string& name, const std::vector<uint32_t>& args, const std::vector<std::string>& arg_names) {
//             std::cout << "==== " << name << " ====" << std::endl;
//             for (size_t i = 0; i < args.size(); ++i) {
//                 if (i < arg_names.size())
//                     std::cout << "  [" << i << "] " << arg_names[i] << " = " << args[i] << std::endl;
//                 else
//                     std::cout << "  [" << i << "] = " << args[i] << std::endl;
//             }
//             std::cout << std::endl;
//         };

//         print_args("reader_compile_time_args", reader_compile_time_args, {
//             "src0_is_dram",
//             "src1_is_dram",
//             "col_indices_is_dram",
//             "indptr_is_dram",
//             "src0_dram_buffer->address()",
//             "in0_tensor_stride_w",
//             "in0_tensor_stride_h",
//             "in0_block_w",
//             "in0_block_h",
//             "in0_block_num_tiles",
//             "src1_dram_buffer->address()",
//             "in1_tensor_stride_w",
//             "in1_tensor_stride_h",
//             "in1_block_w",
//             "in1_block_h",
//             "in1_block_num_tiles",
//             "column_indices_dram_buffer->address()",
//             "indptr_dram_buffer->address()",
//             "num_tiles_for_col_indices",
//             "num_tiles_for_indptr"
//         });

//         print_args("compute_kernel_compile_time_args", compute_kernel_compile_time_args, {
//             "in0_block_w",
//             "in0_num_subblocks",
//             "in0_block_num_tiles",
//             "in0_subblock_num_tiles",
//             "in1_num_subblocks",
//             "in1_block_num_tiles",
//             "in1_per_core_w",
//             "out_subblock_h",
//             "out_subblock_w",
//             "out_subblock_num_tiles",
//             "num_iters_x",
//         });

//         print_args("writer_compile_time_args", writer_compile_time_args, {
//             "out_is_dram",
//             "dst_dram_buffer->address()",
//             "out_tensor_stride_w",
//             "out_tensor_stride_h",
//             "out_tensor_next_subblock_stride_w",
//             "out_tensor_next_subblock_stride_h",
//             "out_subblock_w",
//             "out_subblock_h",
//             "out_subblock_w * out_subblock_h",
//             "out_num_subblocks_w",
//             "out_num_subblocks_h",
//             "Rt * Nt",
//             "num_iters_x",
//         });
//     }

//     // Create Kernels
//     auto reader_id = tt_metal::CreateKernel(
//         program,
//         "tt_metal/programming_examples/rahmy/block_spmm/kernels/dataflow/reader_block_iter.cpp",
//         all_cores,
//         tt_metal::DataMovementConfig{
//             .processor = DataMovementProcessor::RISCV_0,
//             .noc = NOC::RISCV_0_default,
//             .compile_args = reader_compile_time_args});

//     auto writer_id = tt_metal::CreateKernel(
//         program,
//         "tt_metal/programming_examples/rahmy/block_spmm/kernels/dataflow/writer_block_iter.cpp",
//         all_cores,
//         tt_metal::DataMovementConfig{
//             .processor = DataMovementProcessor::RISCV_1,
//             .noc = NOC::RISCV_1_default,
//             .compile_args = writer_compile_time_args});

//     // Create compute kernel
//     auto mm_kernel_id = tt_metal::CreateKernel(
//         program,
//         "tt_metal/programming_examples/rahmy/block_spmm/kernels/compute/bmm_iter.cpp",
//         all_cores,
//         tt_metal::ComputeConfig{.math_fidelity = math_fidelity,
//                                        .compile_args = compute_kernel_compile_time_args});
//     // Runtime arguments
//     auto core_coords_vec = corerange_to_cores(all_cores);
//     uint32_t work_region = 0;
//     for (auto & core : core_coords_vec){
//         uint32_t core_idx_x = core.x;
//         uint32_t core_idx_y = core.y;
        
//         if (verbose)
//             log_info(tt::LogVerif, "Core x {} y {}", core_idx_x, core_idx_y);
            
//         int output_idx_x_start = (work_region * num_iters_x) % num_blocks_x;
//         int folded_output_idx_y_start = (((work_region * num_iters_x) / num_blocks_x) * num_iters_y) % num_blocks_y;
//         work_region++;

//         std::vector<uint32_t> reader_runtime_args;
//         std::vector<uint32_t> compute_runtime_args;
//         std::vector<uint32_t> writer_runtime_args;

//         uint32_t num_iters_y_remaining = num_blocks_y - folded_output_idx_y_start;
//         uint32_t num_iters_y_this_core = std::min(num_iters_y, num_iters_y_remaining);
//         uint32_t num_iters_x_this_core = std::min(num_iters_x, num_blocks_x - output_idx_x_start); // TODO: make a test case that actually tests this
//         reader_runtime_args.push_back(num_iters_x_this_core);
//         reader_runtime_args.push_back(num_iters_y_this_core);
//         compute_runtime_args.push_back(num_iters_y_this_core);
//         reader_runtime_args.push_back(output_idx_x_start);
//         for (int iter_y = 0; iter_y < num_iters_y_this_core; iter_y++) {
//             uint32_t output_idx_y = folded_bsr_matrix_indices[folded_output_idx_y_start + iter_y];
//             reader_runtime_args.push_back(output_idx_y);
//             compute_runtime_args.push_back(a.indptr[output_idx_y + 1] - a.indptr[output_idx_y]);
//         }

//         writer_runtime_args.push_back((folded_output_idx_y_start * Rt * Nt) + (output_idx_x_start * in1_block_w));
//         writer_runtime_args.push_back(num_iters_y_this_core);

//         tt_metal::SetRuntimeArgs(program, reader_id, core, reader_runtime_args);
//         tt_metal::SetRuntimeArgs(program, mm_kernel_id, core, compute_runtime_args);
//         tt_metal::SetRuntimeArgs(program, writer_id, core, writer_runtime_args);

//         if (verbose && core_idx_x == 0 && core_idx_y == 0) {
//             a.pretty_print();
//             log_info(tt::LogVerif, " -- Reader Args --");
//             log_info(tt::LogVerif, "reader_arg[0] (num_iters_x) = {}", reader_runtime_args[0]);
//             log_info(tt::LogVerif, "reader_arg[1] (num_iters_y) = {}",  reader_runtime_args[1]);
//             log_info(tt::LogVerif, "reader_arg[2] (output_idx_x_start) = {}",  reader_runtime_args[2]);
//             for (size_t i = 0; i < num_iters_y_this_core; ++i) {
//                 log_info(tt::LogVerif, "reader_arg[{}] (y_coord) = {}", i + 3, reader_runtime_args[i+3]);
//             }

//             log_info(tt::LogVerif, " -- Writer Args --");
//             const char* writer_arg_names[] = {
//                 "out_tensor_start_tile_id",
//                 "num_iters_y_this_core"
//             };
//             for (size_t i = 0; i < writer_runtime_args.size(); ++i) {
//                 log_info(tt::LogVerif, "writer_arg[{}] ({}) = {}", i, writer_arg_names[i], writer_runtime_args[i]);
//             }
//             log_info(tt::LogVerif, " -- Compute Args --");
//             log_info(tt::LogVerif, "compute_arg[{}] (num_iters_y) = {}", 0, compute_runtime_args[0]);
//             for (size_t i = 0; i < num_iters_y_this_core; ++i) {
//                 log_info(tt::LogVerif, "compute_arg[{}] (row_size) = {}", i+1, compute_runtime_args[i+1]);
//             }
//         }
//     }

//     // EnqueueWriteBuffers
//     EnqueueWriteBuffer(cq, src0_dram_buffer, a.data.data(), true);
//     EnqueueWriteBuffer(cq, src1_dram_buffer, b.data.data(), true);
//     EnqueueWriteBuffer(cq, column_indices_dram_buffer, a.indices.data(), true);
//     EnqueueWriteBuffer(cq, indptr_dram_buffer, a.indptr.data(), true);
//     // EnqueueProgram
//     EnqueueProgram(cq, program, true);

//     if (verbose)
//         log_info(tt::LogVerif, " -- Program returned --");
//     // EnqueueReadSubBuffers
//     uint32_t nonzero_row_index = 0;
//     for (size_t row_index = 0; row_index < a.indptr.size() - 1; row_index++) {
//         if (a.indptr[row_index+1] - a.indptr[row_index] == 0)
//             continue;
//         BufferRegion DRAM_row(nonzero_row_index * dram_buffer_dst_row_size, dram_buffer_dst_row_size);
//         EnqueueReadSubBuffer(cq, dst_dram_buffer, output.data.data() + (row_index * R * N), DRAM_row, true);
//         nonzero_row_index++;
//     }

//     if (verbose)
//         log_info(tt::LogVerif, " -- Finished reading output --");
//     Finish(cq);
// }

// void bsr_spmm_multicore_reuse_many_blocks_per_core(
//     bsr_matrix<bfloat16>& a,
//     dense_matrix<bfloat16>& b,
//     dense_matrix<bfloat16>& output,
//     bool bcast_batch,
//     uint32_t nnz_blocks,
//     uint32_t M,
//     uint32_t N,
//     uint32_t K,
//     uint32_t R,
//     uint32_t C,
//     uint32_t B,
//     IDevice* device,
//     bool verbose = false) {

//     CommandQueue& cq = device->command_queue();
//     Program program{};

//     tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
//     MathFidelity math_fidelity = MathFidelity::HiFi4;
//     uint32_t single_tile_size = detail::TileSize(cb_data_format);
//     // uint32_t single_tile_size = 2 * 1024;

//     tt::DataFormat indexing_data_format = tt::DataFormat::Int32;
//     uint32_t indexing_data_single_tile_size = detail::TileSize(indexing_data_format);

//     auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
//     uint32_t num_cores_x = compute_with_storage_grid_size.x;
//     uint32_t num_cores_y = compute_with_storage_grid_size.y;

//     uint32_t Mt = M / TILE_HEIGHT;
//     uint32_t Kt = K / TILE_WIDTH;
//     uint32_t Nt = N / TILE_WIDTH;

//     uint32_t Rt = R / TILE_HEIGHT;
//     uint32_t Ct = C / TILE_WIDTH;


//     struct output_block_args {
//         // reader args
//         uint32_t in0_tensor_start_tile_id;
//         uint32_t in1_tensor_start_tile_id;
//         uint32_t block_row_start;
//         uint32_t block_row_end;
//         uint32_t block_row_index;

//         // writer args
//         uint32_t out_buffer_addr;
//         uint32_t out_tensor_start_tile_id;
//     };

//     //


//     // okay i get it: getting the matmul params is easy in the dense case because you need params in terms of tiles, and tiles have
//     // a fixed size for every possible matrix pair. But for us, we want the matmul params in terms of BSR blocks, which have variable size.
//     // But it's also still tiles. Let's focus on our current test suite (with block size a multiple of tile size). This is still very useful in
//     // the dense case, and in prototyping the overhead analysis.
//     // Ohh... OPPORTUNITY: we can max out on Npc, then iterate over Mpc{Rt}.
//     //                      This lets us reuse the SRAM buffers, meaning we can
//     //                      actually handle more data per core than the dense case!!!
//     //                      we peakin.

//     // We admit flexible per_core_M while maintaining rigid per_block_M
//     uint32_t per_block_M = Rt;
//     uint32_t in0_block_w = Ct;

//     // TODO: get the number of tiles used for the new NoC args and add to this calc
//     int32_t num_tiles_for_col_indices = (indexing_data_single_tile_size - 1 + sizeof(int) * nnz_blocks) / indexing_data_single_tile_size;
//     uint32_t per_core_N = get_Npc_from_BSR_block_size(Nt, per_block_M, in0_block_w, num_cores_x, num_tiles_for_col_indices);
//     uint32_t per_core_M = _get_maximum_block_dim_with_NoC_args(per_core_N, Ct, num_tiles_for_col_indices);
//     per_core_M = std::min(Mt, Rt * (per_core_M / Rt));
//     // pick the largest subblock size that fits within the block size
//     uint32_t out_subblock_h, out_subblock_w;
//     for (auto& subblock_hw : bmm_op_utils::SUBBLOCK_HW_CHOICES) {
//         out_subblock_h = std::get<0>(subblock_hw);
//         out_subblock_w = std::get<1>(subblock_hw);
//         if (per_block_M % out_subblock_h == 0 and per_core_N % out_subblock_w == 0) {
//             break;
//         }
//     }


//     if (verbose) {
//         log_info(tt::LogVerif, " -- Metalium Core Sizing --");
//         log_info(
//             tt::LogVerif,
//             " -- per_core_M={} -- per_block_M= {} -- per_core_N= {} -- out_subblock_h= {} -- out_subblock_w= {} --",
//             per_core_M,
//             per_block_M,
//             per_core_N,
//             out_subblock_h,
//             out_subblock_w);
//     }

//     TT_ASSERT(Mt % per_block_M == 0);
//     TT_ASSERT(Nt % per_core_N == 0);
//     TT_ASSERT(Kt % in0_block_w == 0);

//     uint32_t in0_core_tiles = per_block_M * in0_block_w;
//     uint32_t in0_CB_tiles = in0_core_tiles * 2;  // double buffer
//     uint32_t in0_CB_size = in0_CB_tiles * single_tile_size;
//     uint32_t in1_block_tiles = per_core_N * in0_block_w;
//     uint32_t in1_CB_tiles = in1_block_tiles * 2;  // double buffer
//     uint32_t in1_CB_size = in1_CB_tiles * single_tile_size;
//     uint32_t out_block_tiles = per_block_M * per_core_N;
//     uint32_t out_CB_tiles = out_block_tiles;  // No double buffer
//     uint32_t out_CB_size = out_CB_tiles * single_tile_size;

//     uint32_t in0_num_subblocks = (per_block_M / out_subblock_h);
//     uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;
//     uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w; // this is named weird but it's correct.

//     uint32_t in1_num_subblocks = (per_core_N / out_subblock_w);
//     uint32_t in1_block_num_tiles = out_subblock_w * in0_block_w * in1_num_subblocks;
//     uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;

//     uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

//     //////////////////////////////////////////////////
//     /*
//      * Create DRAM Buffers for input and output vectors
//      * Writing data from input vectors to source buffers
//      */
//     uint32_t num_blocks_x = Nt / per_core_N;
//     uint32_t dram_buffer_dst_row_size =
//         single_tile_size * Rt * Nt;
//     uint32_t dram_buffer_dst_total_size = 0;
//     uint32_t nnz_rows = 0;

//     for (uint32_t i = 0; i < a.indptr.size() - 1; i++) {
//         if (a.indptr[i+1] - a.indptr[i] > 0)
//             nnz_rows++;
//     }
//     dram_buffer_dst_total_size = dram_buffer_dst_row_size * nnz_rows;
//     auto dst_dram_buffer = MakeBuffer(device, dram_buffer_dst_total_size, single_tile_size);


//     uint32_t dram_buffer_A_size =
//         single_tile_size * Rt * Ct * nnz_blocks;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels
//     uint32_t dram_buffer_B_size =
//         single_tile_size * Nt * Kt;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels

//     uint32_t dram_buffer_D_size =
//         sizeof(int) * nnz_blocks; //
//     if (dram_buffer_D_size > indexing_data_single_tile_size)
//         dram_buffer_D_size = indexing_data_single_tile_size * ((indexing_data_single_tile_size - 1 + dram_buffer_D_size) / (indexing_data_single_tile_size));

//     uint32_t dram_buffer_indptr_size =
//         sizeof(int) * (M / R);
//     if (dram_buffer_indptr_size > indexing_data_single_tile_size)
//         dram_buffer_indptr_size = indexing_data_single_tile_size * ((indexing_data_single_tile_size - 1 + dram_buffer_indptr_size) / (indexing_data_single_tile_size));


//     auto src0_dram_buffer = MakeBuffer(device, dram_buffer_A_size, single_tile_size);
//     auto src1_dram_buffer = MakeBuffer(device, dram_buffer_B_size, single_tile_size);
//     auto column_indices_dram_buffer = MakeBuffer(device, dram_buffer_D_size, std::min(indexing_data_single_tile_size, dram_buffer_D_size));
//     auto indptr_dram_buffer = MakeBuffer(device, dram_buffer_indptr_size, std::min(indexing_data_single_tile_size, dram_buffer_indptr_size));

//     uint32_t nnz_output_blocks_total = num_blocks_x * nnz_rows; // blocks per row * nnz rows


//     CoreRangeSet all_cores(
//         tt::tt_metal::num_cores_to_corerangeset(nnz_output_blocks_total, compute_with_storage_grid_size, true));


//     if (verbose) {
//         log_info(tt::LogVerif, " -- Metalium Grid Sizing --");
//         log_info(
//             tt::LogVerif,
//             " -- Mt= {} -- Nt= {} -- nnz_output_blocks= {} -- num_cores_used={} -- num_cores_available_x={} -- num_cores_available_y={} --",
//             Mt,
//             Nt,
//             nnz_output_blocks_total,
//             all_cores.num_cores(),
//             num_cores_x,
//             num_cores_y);
//     }

//     /*
//     SRAM Circular Buffers
//     */
//     uint32_t src0_cb_index = CBIndex::c_0;  // 0
//     CircularBufferConfig cb_src0_config = CircularBufferConfig(in0_CB_size, {{src0_cb_index, cb_data_format}})
//                                               .set_page_size(src0_cb_index, single_tile_size);
//     auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

//     uint32_t src1_cb_index = CBIndex::c_1;  // 1
//     CircularBufferConfig cb_src1_config = CircularBufferConfig(in1_CB_size, {{src1_cb_index, cb_data_format}})
//                                               .set_page_size(src1_cb_index, single_tile_size);
//     auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

//     uint32_t output_cb_index = tt::CBIndex::c_16;
//     uint32_t interm0_cb_index = tt::CBIndex::c_24;
//     std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec{
//         {output_cb_index, cb_data_format}, {interm0_cb_index, cb_data_format}};
//         CircularBufferConfig cb_output_config = CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
//         .set_page_size(output_cb_index, single_tile_size)
//         .set_page_size(interm0_cb_index, single_tile_size);
//         auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);


//     // TODO: uh this is nasty and wastes memory and may even be incorrect for buffer sizes larger than a single tile.
//     uint32_t column_indices_cb_index = CBIndex::c_2;  // 2
//     CircularBufferConfig cb_column_indices_config = CircularBufferConfig(
//         dram_buffer_D_size, {{column_indices_cb_index, tt::DataFormat::Int32}})
//                                                 .set_page_size(column_indices_cb_index, std::min(indexing_data_single_tile_size, dram_buffer_D_size));
//     auto cb_column_indices = tt_metal::CreateCircularBuffer(program, all_cores, cb_column_indices_config);

//     auto indptr_cb_index = CBIndex::c_3; // 3
//     auto cb_indptr = MakeCircularBuffer(program, all_cores, indptr_cb_index, dram_buffer_indptr_size, std::min(indexing_data_single_tile_size, dram_buffer_indptr_size), tt::DataFormat::Int32);

//      /*
//      * Compile time arguments
//      */
//     bool src0_is_dram = src0_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
//     bool src1_is_dram = src1_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
//     bool col_indices_is_dram = column_indices_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
//     // std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_is_dram, (uint32_t)src1_is_dram, (uint32_t)Noc_args_is_dram};
//     bool indptr_is_dram = indptr_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
//     std::vector<uint32_t> reader_compile_time_args = {
//         (std::uint32_t)src0_is_dram,
//         (std::uint32_t)src1_is_dram,
//         (std::uint32_t)col_indices_is_dram,
//         (std::uint32_t)indptr_is_dram,

//         (std::uint32_t)src0_dram_buffer->address(),     // in0_tensor_addr
//         (std::uint32_t)1,                               // in0_tensor_stride_w
//         (std::uint32_t)Ct,                              // in0_tensor_stride_h

//         (std::uint32_t)in0_block_w,               // in0_block_w
//         (std::uint32_t)Rt,                         // in0_block_h
//         (std::uint32_t)in0_block_w * Rt,  // in0_block_num_tiles

//         (std::uint32_t)src1_dram_buffer->address(),  // in1_tensor_addr
//         (std::uint32_t)1,                            // in1_tensor_stride_w
//         (std::uint32_t)Nt,                           // in1_tensor_stride_h

//         (std::uint32_t)per_core_N,                // in1_block_w
//         (std::uint32_t)in0_block_w,               // in1_block_h
//         (std::uint32_t)per_core_N * in0_block_w,  // in1_block_num_tiles


//         (std::uint32_t)column_indices_dram_buffer->address(), // NoC args, column indices
//         (std::uint32_t)indptr_dram_buffer->address(), // NoC args, indptr

//         (std::uint32_t)num_tiles_for_col_indices, 
//         (std::uint32_t)1,

//         // in0_tensor_start_tile_id obtained by // a.indptr[output_idx_y] * Rt * Ct,
//         // in1_tensor_start_tile_id obtained by // per_core_N * output_idx_x
//         // col indices start of row obtained by // a.indptr[output_idx_y],
//         // col indices end of row obtained by //  a.indptr[output_idx_y + 1],
//     };

//         std::vector<uint32_t> compute_kernel_compile_time_args = {
//         (std::uint32_t)in0_block_w,
//         (std::uint32_t)in0_num_subblocks,
//         (std::uint32_t)in0_block_num_tiles,
//         (std::uint32_t)in0_subblock_num_tiles,
//         (std::uint32_t)in1_num_subblocks,
//         (std::uint32_t)in1_block_num_tiles,
//         (std::uint32_t)in1_per_core_w,
//         (std::uint32_t)out_subblock_h,
//         (std::uint32_t)out_subblock_w,
//         (std::uint32_t)out_subblock_num_tiles,
//         (std::uint32_t)1,
//     };
//     // bool dst_is_dram = true;
//     // std::vector<uint32_t> writer_compile_time_args = {(uint32_t)dst_is_dram};
//     bool out_is_dram = dst_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
//     std::vector<uint32_t> writer_compile_time_args = {
//         (std::uint32_t)out_is_dram,
//         (std::uint32_t)dst_dram_buffer->address(),      // out_buffer_addr
//         (std::uint32_t)1,                           // out_tensor_stride_w
//         (std::uint32_t)Nt,                         // out_tensor_stride_h
//         (std::uint32_t)out_subblock_w,       // out_tensor_next_subblock_stride_w
//         (std::uint32_t)out_subblock_h * Nt,  // out_tensor_next_subblock_stride_h

//         (std::uint32_t)out_subblock_w,                     // out_subblock_w
//         (std::uint32_t)out_subblock_h,                     // out_subblock_h
//         (std::uint32_t)(out_subblock_w * out_subblock_h),  // out_subblocks_w * out_subblocks_h
//         (std::uint32_t)(per_core_N / out_subblock_w),      // out_num_subblocks_w
//         (std::uint32_t)(per_block_M / out_subblock_h),      // out_num_subblocks_h

//         (std::uint32_t)Rt * Nt,  // Size of output row, used to index into next output block
//         (std::uint32_t)1,
//     };

//     /*
//      * Create Kernels (Reader, Writer, Compute)
//      */
//     // Create reader and writer kernels per core

//     // TESTING: let's let this funtion use the iter kernels with iter=1
//     auto reader_id = tt_metal::CreateKernel(
//         program,
//         "tt_metal/programming_examples/rahmy/block_spmm/kernels/dataflow/reader_block_iter.cpp",
//         all_cores,
//         tt_metal::DataMovementConfig{
//             .processor = DataMovementProcessor::RISCV_0,
//             .noc = NOC::RISCV_0_default,
//             .compile_args = reader_compile_time_args});

//     auto writer_id = tt_metal::CreateKernel(
//         program,
//         "tt_metal/programming_examples/rahmy/block_spmm/kernels/dataflow/writer_block_iter.cpp",
//         all_cores,
//         tt_metal::DataMovementConfig{
//             .processor = DataMovementProcessor::RISCV_1,
//             .noc = NOC::RISCV_1_default,
//             .compile_args = writer_compile_time_args});

//     // Create compute kernel
//     auto mm_kernel_id = tt_metal::CreateKernel(
//         program,
//         "tt_metal/programming_examples/rahmy/block_spmm/kernels/compute/bmm_iter.cpp",
//         all_cores,
//         tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = compute_kernel_compile_time_args});

//     uint32_t nnz_output_blocks_read = 0;
//     uint32_t num_empty_rows_so_far = 0;
//     for (int output_idx_y = 0; output_idx_y < a.indptr.size() - 1; output_idx_y++) {
//         uint32_t nnz_blocks_in_row = a.indptr[output_idx_y + 1] - a.indptr[output_idx_y];
//         if (nnz_blocks_in_row == 0){
//             num_empty_rows_so_far++;
//             continue;
//         }
//         // else, we are in a nonzero row
//         for (uint32_t output_idx_x = 0; output_idx_x < num_blocks_x; output_idx_x++){
//             int core_idx_x = nnz_output_blocks_read % num_cores_x;
//             int core_idx_y = nnz_output_blocks_read / num_cores_x;
//             CoreCoord core = {(std::size_t)core_idx_x, (std::size_t)core_idx_y};


//             // Write runtime args to device
//             std::vector<uint32_t> mm_reader_args = {
//                 (std::uint32_t)1,                               // iters x
//                 (std::uint32_t)1,                               // iters y
//                 (std::uint32_t)output_idx_x,                    // out idx x start
//                 (std::uint32_t)output_idx_y - num_empty_rows_so_far,                    // out idx y start
//             };

//             // output tile isn't about output_idx_y becase that's treating the dense matrix (no folds).
//             // but we have folds!!!
//             std::vector<uint32_t> writer_args = {
//                 (std::uint32_t)((output_idx_y - num_empty_rows_so_far) * Rt * Nt) + output_idx_x * per_core_N,  // out_tensor_start_tile_id
//                 (std::uint32_t)1,                           // iters y
//             };

//             std::vector<uint32_t> compute_args = {
//                 (std::uint32_t)1, // iters y
//                 (std::uint32_t)nnz_blocks_in_row, // row size
//             };

//             tt_metal::SetRuntimeArgs(program, reader_id, core, mm_reader_args);
//             tt_metal::SetRuntimeArgs(program, writer_id, core, writer_args);
//             tt_metal::SetRuntimeArgs(program, mm_kernel_id, core, compute_args);
//             nnz_output_blocks_read++;

//             if (verbose && output_idx_y == 0 && output_idx_x == 0) {
//                 a.pretty_print();
//                 log_info(tt::LogVerif, " -- Reader Args --");
//                 const char* reader_arg_names[] = {
//                     "in0_tensor_addr",
//                     "in0_tensor_start_tile_id",
//                     "in0_tensor_stride_w",
//                     "in0_tensor_stride_h",
//                     "in0_block_w",
//                     "in0_block_h",
//                     "in0_block_num_tiles",
//                     "in1_tensor_addr",
//                     "in1_tensor_start_tile_id",
//                     "in1_tensor_stride_w",
//                     "in1_tensor_stride_h",
//                     "in1_block_w",
//                     "in1_block_h",
//                     "in1_block_num_tiles",
//                     "col_indices_start_of_row",
//                     "col_indices_end_of_row",
//                     "row_index_into_bsr_matrix",
//                     "column_indices_addr",
//                     "indptr_dram_buffer->address()",
//                 };
//                 for (size_t i = 0; i < mm_reader_args.size(); ++i) {
//                     log_info(tt::LogVerif, "reader_arg[{}] ({}) = {}", i, reader_arg_names[i], mm_reader_args[i]);
//                 }
//                 log_info(tt::LogVerif, " -- Writer Args --");
//                 const char* writer_arg_names[] = {
//                     "out_buffer_addr",
//                     "out_tensor_start_tile_id",
//                     "out_tensor_stride_w",
//                     "out_tensor_stride_h",
//                     "out_tensor_next_subblock_stride_w",
//                     "out_tensor_next_subblock_stride_h",
//                     "out_subblock_w",
//                     "out_subblock_h",
//                     "out_subblock_w * out_subblock_h",
//                     "out_num_subblocks_w",
//                     "out_num_subblocks_h",
//                     "MtNt",
//                     "batch",
//                     "nonzero"
//                 };
//                 for (size_t i = 0; i < writer_args.size(); ++i) {
//                     log_info(tt::LogVerif, "writer_arg[{}] ({}) = {}", i, writer_arg_names[i], writer_args[i]);
//                 }
//                 log_info(tt::LogVerif, " -- Compute Args --");
//                 const char* compute_arg_names[] = {
//                     "in0_block_w",
//                     "in0_num_subblocks",
//                     "in0_block_num_tiles",
//                     "in0_subblock_num_tiles",
//                     "in1_num_subblocks",
//                     "in1_block_num_tiles",
//                     "in1_per_core_w",
//                     "nnz_blocks_in_row",
//                     "out_subblock_h",
//                     "out_subblock_w",
//                     "out_subblock_num_tiles",
//                     "B"
//                 };
//                 for (size_t i = 0; i < compute_args.size(); ++i) {
//                     log_info(tt::LogVerif, "compute_arg[{}] ({}) = {}", i, compute_arg_names[i], compute_args[i]);
//                 }
//             }
//         }
//     }



//     if (verbose){
//         log_info(tt::LogVerif, " -- Runtime Args set --");
//         log_info(
//             tt::LogVerif,
//             " -- nnz output blocks= {}",
//             nnz_output_blocks_read);
//     }


//     EnqueueWriteBuffer(cq, src0_dram_buffer, a.data.data(), true);
//     EnqueueWriteBuffer(cq, src1_dram_buffer, b.data.data(), true);
//     EnqueueWriteBuffer(cq, column_indices_dram_buffer, a.indices.data(), true);
//     EnqueueWriteBuffer(cq, indptr_dram_buffer, a.indptr.data(), true);

//     if (verbose)
//         log_info(tt::LogVerif, " -- All data moved to DRAM --");

//     EnqueueProgram(cq, program, false);

//     if (verbose)
//         log_info(tt::LogVerif, " -- Program enqueued --");


//     // we index into host data by row_index,
//     // and we index into DRAM data by "folded" row index
//     uint32_t nonzero_row_index = 0;
//     for (size_t row_index = 0; row_index < a.indptr.size() - 1; row_index++) {
//         if (a.indptr[row_index+1] - a.indptr[row_index] == 0)
//             continue;
//         BufferRegion DRAM_row(nonzero_row_index * dram_buffer_dst_row_size, dram_buffer_dst_row_size);
//         EnqueueReadSubBuffer(cq, dst_dram_buffer, output.data.data() + (row_index * R * N), DRAM_row, true);
//         nonzero_row_index++;
//     }

//     Finish(cq);
// }

// void bsr_spmm_multicore_host_reuse_device_iter(
//     bsr_matrix<bfloat16>& a,
//     dense_matrix<bfloat16>& b,
//     dense_matrix<bfloat16>& output,
//     bool bcast_batch,
//     uint32_t nnz_blocks,
//     uint32_t M,
//     uint32_t N,
//     uint32_t K,
//     uint32_t R,
//     uint32_t C,
//     uint32_t B,
//     IDevice* device,
//     bool verbose = false) {

//     CommandQueue& cq = device->command_queue();
//     Program program{};

//     tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
//     MathFidelity math_fidelity = MathFidelity::HiFi4;
//     uint32_t single_tile_size = detail::TileSize(cb_data_format);
//     // uint32_t single_tile_size = 2 * 1024;

//     tt::DataFormat col_indices_data_format = tt::DataFormat::Int32;
//     uint32_t col_indices_single_tile_size = detail::TileSize(col_indices_data_format);

//     auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
//     uint32_t num_cores_x = compute_with_storage_grid_size.x;
//     uint32_t num_cores_y = compute_with_storage_grid_size.y;

//     uint32_t Mt = M / TILE_HEIGHT;
//     uint32_t Kt = K / TILE_WIDTH;
//     uint32_t Nt = N / TILE_WIDTH;

//     uint32_t Rt = R / TILE_HEIGHT;
//     uint32_t Ct = C / TILE_WIDTH;

//     uint32_t per_core_M = Rt;
//     uint32_t in0_block_w = Ct;

//     int32_t num_tiles_for_col_indices = (col_indices_single_tile_size - 1 + sizeof(int) * nnz_blocks) / col_indices_single_tile_size;
//     uint32_t per_core_N = get_Npc_from_BSR_block_size(Nt, per_core_M, in0_block_w, num_cores_x, num_tiles_for_col_indices);
    
//     // pick the largest subblock size that fits within the block size
//     uint32_t out_subblock_h, out_subblock_w;
//     for (auto& subblock_hw : bmm_op_utils::SUBBLOCK_HW_CHOICES) {
//         out_subblock_h = std::get<0>(subblock_hw);
//         out_subblock_w = std::get<1>(subblock_hw);
//         if (per_core_M % out_subblock_h == 0 and per_core_N % out_subblock_w == 0) {
//             break;
//         }
//     }


//     if (verbose) {
//         log_info(tt::LogVerif, " -- Metalium Core Sizing --");
//         log_info(
//             tt::LogVerif,
//             " -- per_core_M= {} -- per_core_N= {} -- out_subblock_h= {} -- out_subblock_w= {} --",
//             per_core_M,
//             per_core_N,
//             out_subblock_h,
//             out_subblock_w);
//     }

//     TT_ASSERT(Mt % per_core_M == 0);
//     TT_ASSERT(Nt % per_core_N == 0);
//     TT_ASSERT(Kt % in0_block_w == 0);

//     uint32_t in0_block_tiles = per_core_M * in0_block_w;
//     uint32_t in0_CB_tiles = in0_block_tiles * 2;  // double buffer
//     uint32_t in0_CB_size = in0_CB_tiles * single_tile_size;
//     uint32_t in1_block_tiles = per_core_N * in0_block_w;
//     uint32_t in1_CB_tiles = in1_block_tiles * 2;  // double buffer
//     uint32_t in1_CB_size = in1_CB_tiles * single_tile_size;
//     uint32_t out_block_tiles = per_core_M * per_core_N;
//     uint32_t out_CB_tiles = out_block_tiles;  // No double buffer
//     uint32_t out_CB_size = out_CB_tiles * single_tile_size;

//     uint32_t in0_num_subblocks = (per_core_M / out_subblock_h);
//     uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;
//     uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w; // this is named weird but it's correct.

//     uint32_t in1_num_subblocks = (per_core_N / out_subblock_w);
//     uint32_t in1_block_num_tiles = out_subblock_w * in0_block_w * in1_num_subblocks;
//     uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;

//     uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

//     uint32_t num_blocks_y = M / R; // block_matrix_height, how many blocks tall the input matrix is.
//     uint32_t num_blocks_x = Nt / per_core_N;

//     //////////////////////////////////////////////////
//     /*
//      * Create DRAM Buffers for input and output vectors
//      * Writing data from input vectors to source buffers
//      */

//     uint32_t dram_buffer_dst_row_size =
//         single_tile_size * Rt * Nt;
//     uint32_t dram_buffer_dst_total_size = 0;
//     uint32_t nnz_rows = 0;

//     for (uint32_t i = 0; i < a.indptr.size() - 1; i++) {
//         if (a.indptr[i+1] - a.indptr[i] > 0)
//             nnz_rows++;
//     }
//     dram_buffer_dst_total_size = dram_buffer_dst_row_size * nnz_rows;
//     auto dst_dram_buffer = MakeBuffer(device, dram_buffer_dst_total_size, single_tile_size);
//     uint32_t nnz_output_blocks_total = num_blocks_x * nnz_rows; // blocks per row * nnz rows

//     CoreRangeSet all_cores(
//         tt::tt_metal::num_cores_to_corerangeset(nnz_output_blocks_total, compute_with_storage_grid_size, true));


//     if (verbose) {
//         log_info(tt::LogVerif, " -- Metalium Grid Sizing --");
//         log_info(
//             tt::LogVerif,
//             " -- Mt= {} -- Nt= {} -- nnz_output_blocks= {} -- num_cores_used={} -- num_cores_available_x={} -- num_cores_available_y={} --",
//             Mt,
//             Nt,
//             nnz_output_blocks_total,
//             all_cores.size(),
//             num_cores_x,
//             num_cores_y);
//     }

//     uint32_t dram_buffer_A_size =
//         single_tile_size * Rt * Ct * nnz_blocks;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels
//     uint32_t dram_buffer_B_size =
//         single_tile_size * Nt * Kt;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels


//     // In fact let's pad this to fill a tile at least
//     uint32_t dram_buffer_D_size =
//         sizeof(int) * nnz_blocks; //
//     dram_buffer_D_size = col_indices_single_tile_size * ((col_indices_single_tile_size - 1 + dram_buffer_D_size) / (col_indices_single_tile_size));



//     tt::DataFormat indexing_data_format = tt::DataFormat::Int32;
//     uint32_t indexing_data_single_tile_size = detail::TileSize(indexing_data_format);

//     uint32_t dram_buffer_indptr_size =
//         sizeof(int) * (M / R);
//     if (dram_buffer_indptr_size > indexing_data_single_tile_size)
//         dram_buffer_indptr_size = indexing_data_single_tile_size * ((indexing_data_single_tile_size - 1 + dram_buffer_indptr_size) / (indexing_data_single_tile_size));


//     auto src0_dram_buffer = MakeBuffer(device, dram_buffer_A_size, single_tile_size);
//     auto src1_dram_buffer = MakeBuffer(device, dram_buffer_B_size, single_tile_size);
//     auto column_indices_dram_buffer = MakeBuffer(device, dram_buffer_D_size, std::min(indexing_data_single_tile_size, dram_buffer_D_size));
//     auto indptr_dram_buffer = MakeBuffer(device, dram_buffer_indptr_size, std::min(indexing_data_single_tile_size, dram_buffer_indptr_size));



//     // NAIVE: for this first, naive impl, keep all the CBs the same size, the maximum size
//     uint32_t src0_cb_index = CBIndex::c_0;  // 0
//     CircularBufferConfig cb_src0_config = CircularBufferConfig(in0_CB_size, {{src0_cb_index, cb_data_format}})
//                                               .set_page_size(src0_cb_index, single_tile_size);
//     auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

//     uint32_t src1_cb_index = CBIndex::c_1;  // 1
//     CircularBufferConfig cb_src1_config = CircularBufferConfig(in1_CB_size, {{src1_cb_index, cb_data_format}})
//                                               .set_page_size(src1_cb_index, single_tile_size);
//     auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);



//     uint32_t output_cb_index = tt::CBIndex::c_16;
//     uint32_t interm0_cb_index = tt::CBIndex::c_24;
//     std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec{
//         {output_cb_index, cb_data_format}, {interm0_cb_index, cb_data_format}};
//         CircularBufferConfig cb_output_config = CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
//         .set_page_size(output_cb_index, single_tile_size)
//         .set_page_size(interm0_cb_index, single_tile_size);
//         auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

//     uint32_t column_indices_cb_index = CBIndex::c_2;  // 2
//     CircularBufferConfig cb_column_indices_config = CircularBufferConfig(
//         dram_buffer_D_size, {{column_indices_cb_index, tt::DataFormat::Int32}})
//                                                 .set_page_size(column_indices_cb_index, col_indices_single_tile_size);
//     auto cb_column_indices = tt_metal::CreateCircularBuffer(program, all_cores, cb_column_indices_config);

//     auto indptr_cb_index = CBIndex::c_3; // 3
//     auto cb_indptr = MakeCircularBuffer(program, all_cores, indptr_cb_index, dram_buffer_indptr_size, std::min(indexing_data_single_tile_size, dram_buffer_indptr_size), tt::DataFormat::Int32);

//      /*
//      * Compile time arguments
//      */
//    bool src0_is_dram = src0_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
//     bool src1_is_dram = src1_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
//     bool col_indices_is_dram = column_indices_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
//     // std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_is_dram, (uint32_t)src1_is_dram, (uint32_t)Noc_args_is_dram};
//     bool indptr_is_dram = indptr_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
//     std::vector<uint32_t> reader_compile_time_args = {
//         (std::uint32_t)src0_is_dram,
//         (std::uint32_t)src1_is_dram,
//         (std::uint32_t)col_indices_is_dram,
//         (std::uint32_t)indptr_is_dram,

//         (std::uint32_t)src0_dram_buffer->address(),     // in0_tensor_addr
//         (std::uint32_t)1,                               // in0_tensor_stride_w
//         (std::uint32_t)Ct,                              // in0_tensor_stride_h

//         (std::uint32_t)in0_block_w,               // in0_block_w
//         (std::uint32_t)Rt,                         // in0_block_h
//         (std::uint32_t)in0_block_w * Rt,  // in0_block_num_tiles

//         (std::uint32_t)src1_dram_buffer->address(),  // in1_tensor_addr
//         (std::uint32_t)1,                            // in1_tensor_stride_w
//         (std::uint32_t)Nt,                           // in1_tensor_stride_h

//         (std::uint32_t)per_core_N,                // in1_block_w
//         (std::uint32_t)in0_block_w,               // in1_block_h
//         (std::uint32_t)per_core_N * in0_block_w,  // in1_block_num_tiles


//         (std::uint32_t)column_indices_dram_buffer->address(), // NoC args, column indices
//         (std::uint32_t)indptr_dram_buffer->address(), // NoC args, indptr

//         (std::uint32_t)num_tiles_for_col_indices, 
//         (std::uint32_t)1,

//         // in0_tensor_start_tile_id obtained by // a.indptr[output_idx_y] * Rt * Ct,
//         // in1_tensor_start_tile_id obtained by // per_core_N * output_idx_x
//         // col indices start of row obtained by // a.indptr[output_idx_y],
//         // col indices end of row obtained by //  a.indptr[output_idx_y + 1],
//     };

//         std::vector<uint32_t> compute_kernel_compile_time_args = {
//         (std::uint32_t)in0_block_w,
//         (std::uint32_t)in0_num_subblocks,
//         (std::uint32_t)in0_block_num_tiles,
//         (std::uint32_t)in0_subblock_num_tiles,
//         (std::uint32_t)in1_num_subblocks,
//         (std::uint32_t)in1_block_num_tiles,
//         (std::uint32_t)in1_per_core_w,
//         (std::uint32_t)out_subblock_h,
//         (std::uint32_t)out_subblock_w,
//         (std::uint32_t)out_subblock_num_tiles,
//         (std::uint32_t)1,
//     };
//     bool out_is_dram = true;
//     std::vector<uint32_t> writer_compile_time_args = {
//         (std::uint32_t)out_is_dram,
//         (std::uint32_t)dst_dram_buffer->address(),      // out_buffer_addr
//         (std::uint32_t)1,                           // out_tensor_stride_w
//         (std::uint32_t)Nt,                         // out_tensor_stride_h
//         (std::uint32_t)out_subblock_w,       // out_tensor_next_subblock_stride_w
//         (std::uint32_t)out_subblock_h * Nt,  // out_tensor_next_subblock_stride_h

//         (std::uint32_t)out_subblock_w,                     // out_subblock_w
//         (std::uint32_t)out_subblock_h,                     // out_subblock_h
//         (std::uint32_t)(out_subblock_w * out_subblock_h),  // out_subblocks_w * out_subblocks_h
//         (std::uint32_t)(per_core_N / out_subblock_w),      // out_num_subblocks_w
//         (std::uint32_t)(per_core_M / out_subblock_h),      // out_num_subblocks_h

//         (std::uint32_t)Rt * Nt,  // Size of output row, used to index into next output block
//         (std::uint32_t)1,
//     };

//     /*
//      * Create Kernels (Reader, Writer, Compute)
//      */
//     // Create reader and writer kernels per core
//     auto reader_id = tt_metal::CreateKernel(
//         program,
//         "tt_metal/programming_examples/rahmy/block_spmm/kernels/dataflow/reader_block_iter.cpp",
//         all_cores,
//         tt_metal::DataMovementConfig{
//             .processor = DataMovementProcessor::RISCV_0,
//             .noc = NOC::RISCV_0_default,
//             .compile_args = reader_compile_time_args});

//     auto writer_id = tt_metal::CreateKernel(
//         program,
//         "tt_metal/programming_examples/rahmy/block_spmm/kernels/dataflow/writer_block_iter.cpp",
//         all_cores,
//         tt_metal::DataMovementConfig{
//             .processor = DataMovementProcessor::RISCV_1,
//             .noc = NOC::RISCV_1_default,
//             .compile_args = writer_compile_time_args});

//     // Create compute kernel
//     auto mm_kernel_id = tt_metal::CreateKernel(
//         program,
//         "tt_metal/programming_examples/rahmy/block_spmm/kernels/compute/bmm_iter.cpp",
//         all_cores,
//         tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = compute_kernel_compile_time_args});

//     // instead of iterating over cores, first determine how many output blocks to a core (get the core range set i guess)
//     // and iterate over output blocks, keeping track of which nz output block you are in by taking
//     // into account the zero rows.
//     // ^^ This was good, but  now we have to go back to thinking about cores since we want to be able to assign many output blocks to a single core.
//     //    ... it's not clear whether doing this first and then working on the multicast impl is better
//     //          Yes it is clear: i'm already thinking about this, and the depth of reasoning I'm practicing will help
//     //          me with whatever I decide to do afterwards.

//     TT_ASSERT(nnz_output_blocks_total <= num_cores_x * num_cores_y);
//     uint32_t nnz_output_blocks_read = 0;
//     uint32_t num_empty_rows_so_far = 0;
//     for (int output_idx_y = 0; output_idx_y < a.indptr.size() - 1; output_idx_y++) {
//         uint32_t nnz_blocks_in_row = a.indptr[output_idx_y + 1] - a.indptr[output_idx_y];
//         if (nnz_blocks_in_row == 0){
//             num_empty_rows_so_far++;
//             continue;
//         }
//         // else, we are in a nonzero row
//         for (uint32_t output_idx_x = 0; output_idx_x < num_blocks_x; output_idx_x++){
//             int core_idx_x = nnz_output_blocks_read % num_cores_x;
//             int core_idx_y = nnz_output_blocks_read / num_cores_x;
//             CoreCoord core = {(std::size_t)core_idx_x, (std::size_t)core_idx_y};
//             if (verbose)
//               log_info(tt::LogVerif, "Core x {} y {}", core_idx_x, core_idx_y);
//             // Write runtime args to device
//             std::vector<uint32_t> mm_reader_args = {
//                 (std::uint32_t)1,                               // iters x
//                 (std::uint32_t)1,                               // iters y
//                 (std::uint32_t)output_idx_x,                    // out idx x start
//                 (std::uint32_t)output_idx_y - num_empty_rows_so_far,                    // out idx y start
//             };

//             // output tile isn't about output_idx_y becase that's treating the dense matrix (no folds).
//             // but we have folds!!!
//             std::vector<uint32_t> writer_args = {
//                 (std::uint32_t)((output_idx_y - num_empty_rows_so_far) * Rt * Nt) + output_idx_x * per_core_N,  // out_tensor_start_tile_id
//                 (std::uint32_t)1,                           // iters y
//             };

//             std::vector<uint32_t> compute_args = {
//                 (std::uint32_t)1, // iters y
//                 (std::uint32_t)nnz_blocks_in_row, // row size
//             };

//             tt_metal::SetRuntimeArgs(program, reader_id, core, mm_reader_args);
//             tt_metal::SetRuntimeArgs(program, writer_id, core, writer_args);
//             tt_metal::SetRuntimeArgs(program, mm_kernel_id, core, compute_args);
//             nnz_output_blocks_read++;

//             if (verbose && output_idx_y == 0 && output_idx_x == 0) {
//                 a.pretty_print();
//                 log_info(tt::LogVerif, " -- Reader Args --");
//                 const char* reader_arg_names[] = {
//                     "in0_tensor_addr",
//                     "in0_tensor_start_tile_id",
//                     "in0_tensor_stride_w",
//                     "in0_tensor_stride_h",
//                     "in0_block_w",
//                     "in0_block_h",
//                     "in0_block_num_tiles",
//                     "in1_tensor_addr",
//                     "in1_tensor_start_tile_id",
//                     "in1_tensor_stride_w",
//                     "in1_tensor_stride_h",
//                     "in1_block_w",
//                     "in1_block_h",
//                     "in1_block_num_tiles",
//                     "col_indices_start_of_row",
//                     "col_indices_end_of_row",
//                     "row_index_into_bsr_matrix",
//                     "column_indices_addr"
//                 };
//                 for (size_t i = 0; i < mm_reader_args.size(); ++i) {
//                     log_info(tt::LogVerif, "reader_arg[{}] ({}) = {}", i, reader_arg_names[i], mm_reader_args[i]);
//                 }
//                 log_info(tt::LogVerif, " -- Writer Args --");
//                 const char* writer_arg_names[] = {
//                     "out_buffer_addr",
//                     "out_tensor_start_tile_id",
//                     "out_tensor_stride_w",
//                     "out_tensor_stride_h",
//                     "out_tensor_next_subblock_stride_w",
//                     "out_tensor_next_subblock_stride_h",
//                     "out_subblock_w",
//                     "out_subblock_h",
//                     "out_subblock_w * out_subblock_h",
//                     "out_num_subblocks_w",
//                     "out_num_subblocks_h",
//                     "MtNt",
//                     "batch",
//                     "nonzero"
//                 };
//                 for (size_t i = 0; i < writer_args.size(); ++i) {
//                     log_info(tt::LogVerif, "writer_arg[{}] ({}) = {}", i, writer_arg_names[i], writer_args[i]);
//                 }
//                 log_info(tt::LogVerif, " -- Compute Args --");
//                 const char* compute_arg_names[] = {
//                     "in0_block_w",
//                     "in0_num_subblocks",
//                     "in0_block_num_tiles",
//                     "in0_subblock_num_tiles",
//                     "in1_num_subblocks",
//                     "in1_block_num_tiles",
//                     "in1_per_core_w",
//                     "nnz_blocks_in_row",
//                     "out_subblock_h",
//                     "out_subblock_w",
//                     "out_subblock_num_tiles",
//                     "B"
//                 };
//                 for (size_t i = 0; i < compute_args.size(); ++i) {
//                     log_info(tt::LogVerif, "compute_arg[{}] ({}) = {}", i, compute_arg_names[i], compute_args[i]);
//                 }
//             }
//         }
//     }



//     if (verbose){
//         log_info(tt::LogVerif, " -- Runtime Args set --");
//         log_info(
//             tt::LogVerif,
//             " -- nnz output blocks= {}",
//             nnz_output_blocks_read);
//     }

//     EnqueueWriteBuffer(cq, src0_dram_buffer, a.data.data(), true);
//     EnqueueWriteBuffer(cq, src1_dram_buffer, b.data.data(), true);
//     EnqueueWriteBuffer(cq, column_indices_dram_buffer, a.indices.data(), true);
//     EnqueueWriteBuffer(cq, indptr_dram_buffer, a.indptr.data(), true);

//     if (verbose)
//         log_info(tt::LogVerif, " -- All data moved to DRAM --");

//     EnqueueProgram(cq, program, true);

//     if (verbose)
//         log_info(tt::LogVerif, " -- Program enqueued --");

//     uint32_t nonzero_row_index = 0;
//     for (size_t row_index = 0; row_index < a.indptr.size() - 1; row_index++) {
//         if (a.indptr[row_index+1] - a.indptr[row_index] == 0)
//             continue;
//         BufferRegion DRAM_row(nonzero_row_index * dram_buffer_dst_row_size, dram_buffer_dst_row_size);
//         EnqueueReadSubBuffer(cq, dst_dram_buffer, output.data.data() + (row_index * R * N), DRAM_row, true);
//         nonzero_row_index++;
//     }

//     Finish(cq);
// }


// void bsr_spmm_multicore_reuse(
//     bsr_matrix<bfloat16>& a,
//     dense_matrix<bfloat16>& b,
//     dense_matrix<bfloat16>& output,
//     bool bcast_batch,
//     uint32_t nnz_blocks,
//     uint32_t M,
//     uint32_t N,
//     uint32_t K,
//     uint32_t R,
//     uint32_t C,
//     uint32_t B,
//     IDevice* device,
//     bool verbose = false) {

//     CommandQueue& cq = device->command_queue();
//     Program program{};

//     tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
//     MathFidelity math_fidelity = MathFidelity::HiFi4;
//     uint32_t single_tile_size = detail::TileSize(cb_data_format);
//     // uint32_t single_tile_size = 2 * 1024;

//     tt::DataFormat col_indices_data_format = tt::DataFormat::Int32;
//     uint32_t col_indices_single_tile_size = detail::TileSize(col_indices_data_format);

//     auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
//     uint32_t num_cores_x = compute_with_storage_grid_size.x;
//     uint32_t num_cores_y = compute_with_storage_grid_size.y;

//     uint32_t Mt = M / TILE_HEIGHT;
//     uint32_t Kt = K / TILE_WIDTH;
//     uint32_t Nt = N / TILE_WIDTH;

//     uint32_t Rt = R / TILE_HEIGHT;
//     uint32_t Ct = C / TILE_WIDTH;




//     // Get large matmul params
//     //
//     // auto matmul_params = bmm_op_utils::get_large_matmul_params(Mt, Nt, num_cores_y, num_cores_x, in0_block_w);
//     // uint32_t per_core_M = std::get<0>(matmul_params);
//     // uint32_t per_core_N = std::get<1>(matmul_params);
//     // uint32_t out_subblock_h = std::get<2>(matmul_params);
//     // uint32_t out_subblock_w = std::get<3>(matmul_params);
//     //
//     // NAIVE: these should adapt to per core workload later. So we have to understand the util function and why it works!
//     //          short idea: let the tt block size be the nz block size, then take the largest of the 20 subblock choices which fits.
//     //          what breaks when in0_block_w = 2??
//     uint32_t per_core_M = Rt;
//     uint32_t in0_block_w = Ct;

//     int32_t num_tiles_for_col_indices = (col_indices_single_tile_size - 1 + sizeof(int) * nnz_blocks) / col_indices_single_tile_size;
//     // uint32_t per_core_N = _get_maximum_block_dim_with_NoC_args(per_core_M, in0_block_w, num_tiles_for_col_indices);
//     // per_core_N = std::min({per_core_N, Ct, Nt});
//     uint32_t per_core_N = get_Npc_from_BSR_block_size(Nt, per_core_M, in0_block_w, num_cores_x, num_tiles_for_col_indices);
    
//     // pick the largest subblock size that fits within the block size
//     uint32_t out_subblock_h, out_subblock_w;
//     for (auto& subblock_hw : bmm_op_utils::SUBBLOCK_HW_CHOICES) {
//         out_subblock_h = std::get<0>(subblock_hw);
//         out_subblock_w = std::get<1>(subblock_hw);
//         if (per_core_M % out_subblock_h == 0 and per_core_N % out_subblock_w == 0) {
//             break;
//         }
//     }


//     if (verbose) {
//         log_info(tt::LogVerif, " -- Metalium Core Sizing --");
//         log_info(
//             tt::LogVerif,
//             " -- per_core_M= {} -- per_core_N= {} -- out_subblock_h= {} -- out_subblock_w= {} --",
//             per_core_M,
//             per_core_N,
//             out_subblock_h,
//             out_subblock_w);
//     }

//     TT_ASSERT(Mt % per_core_M == 0);
//     TT_ASSERT(Nt % per_core_N == 0);
//     TT_ASSERT(Kt % in0_block_w == 0);

//     uint32_t in0_block_tiles = per_core_M * in0_block_w;
//     uint32_t in0_CB_tiles = in0_block_tiles * 2;  // double buffer
//     uint32_t in0_CB_size = in0_CB_tiles * single_tile_size;
//     uint32_t in1_block_tiles = per_core_N * in0_block_w;
//     uint32_t in1_CB_tiles = in1_block_tiles * 2;  // double buffer
//     uint32_t in1_CB_size = in1_CB_tiles * single_tile_size;
//     uint32_t out_block_tiles = per_core_M * per_core_N;
//     uint32_t out_CB_tiles = out_block_tiles;  // No double buffer
//     uint32_t out_CB_size = out_CB_tiles * single_tile_size;

//     uint32_t in0_num_subblocks = (per_core_M / out_subblock_h);
//     uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;
//     uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w; // this is named weird but it's correct.

//     uint32_t in1_num_subblocks = (per_core_N / out_subblock_w);
//     uint32_t in1_block_num_tiles = out_subblock_w * in0_block_w * in1_num_subblocks;
//     uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;

//     uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

//     /*
//      * Multi-Core prep
//      */
//     // auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
//     // uint32_t num_cores_x = compute_with_storage_grid_size.x;
//     // uint32_t num_cores_y = compute_with_storage_grid_size.y;
//     //
//     // NAIVE: these should adapt to per core workload later.
//     // uint32_t num_blocks_y = Mt / per_core_M;
//     uint32_t num_blocks_y = M / R; // block_matrix_height, how many blocks tall the input matrix is.
//     uint32_t num_blocks_x = Nt / per_core_N;
//     uint32_t num_blocks_total = num_blocks_y * num_blocks_x;
//     TT_ASSERT(num_blocks_total <= num_cores_x * num_cores_y);

//     uint32_t dram_buffer_dst_row_size =
//         single_tile_size * Rt * Nt;

//     std::unordered_map<uint32_t, std::shared_ptr<Buffer>> dst_dram_buffers;
//     for (uint32_t i = 0; i < a.indptr.size() - 1; i++) {
//         if (a.indptr[i+1] - a.indptr[i] > 0)
//             dst_dram_buffers[i] = MakeBuffer(device, dram_buffer_dst_row_size, single_tile_size);
//     }

//     uint32_t nnz_output_blocks_total = num_blocks_x * dst_dram_buffers.size(); // blocks per row * nnz rows

//     CoreRangeSet all_cores(
//         tt::tt_metal::num_cores_to_corerangeset(nnz_output_blocks_total, compute_with_storage_grid_size, true));


//     if (verbose) {
//         log_info(tt::LogVerif, " -- Metalium Grid Sizing --");
//         log_info(
//             tt::LogVerif,
//             " -- Mt= {} -- Nt= {} -- nnz_output_blocks= {} -- num_cores_used={} -- num_cores_available_x={} -- num_cores_available_y={} --",
//             Mt,
//             Nt,
//             nnz_output_blocks_total,
//             all_cores,
//             num_cores_x,
//             num_cores_y);
//     }

//     //////////////////////////////////////////////////
//     /*
//      * Create DRAM Buffers for input and output vectors
//      * Writing data from input vectors to source buffers
//      */

//     uint32_t dram_buffer_A_size =
//         single_tile_size * Rt * Ct * nnz_blocks;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels
//     uint32_t dram_buffer_B_size =
//         single_tile_size * Nt * Kt;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels


//     // In fact let's pad this to fill a tile at least
//     uint32_t dram_buffer_D_size =
//         sizeof(int) * nnz_blocks; //
//     dram_buffer_D_size = col_indices_single_tile_size * ((col_indices_single_tile_size - 1 + dram_buffer_D_size) / (col_indices_single_tile_size));


//     auto src0_dram_buffer = MakeBuffer(device, dram_buffer_A_size, single_tile_size);
//     auto src1_dram_buffer = MakeBuffer(device, dram_buffer_B_size, single_tile_size);
//     auto column_indices_dram_buffer = MakeBuffer(device, dram_buffer_D_size, col_indices_single_tile_size);



//     // NAIVE: for this first, naive impl, keep all the CBs the same size, the maximum size
//     uint32_t src0_cb_index = CBIndex::c_0;  // 0
//     CircularBufferConfig cb_src0_config = CircularBufferConfig(in0_CB_size, {{src0_cb_index, cb_data_format}})
//                                               .set_page_size(src0_cb_index, single_tile_size);
//     auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

//     uint32_t src1_cb_index = CBIndex::c_1;  // 1
//     CircularBufferConfig cb_src1_config = CircularBufferConfig(in1_CB_size, {{src1_cb_index, cb_data_format}})
//                                               .set_page_size(src1_cb_index, single_tile_size);
//     auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);



//     uint32_t output_cb_index = tt::CBIndex::c_16;
//     uint32_t interm0_cb_index = tt::CBIndex::c_24;
//     std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec{
//         {output_cb_index, cb_data_format}, {interm0_cb_index, cb_data_format}};
//         CircularBufferConfig cb_output_config = CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
//         .set_page_size(output_cb_index, single_tile_size)
//         .set_page_size(interm0_cb_index, single_tile_size);
//         auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

//     uint32_t column_indices_cb_index = CBIndex::c_2;  // 2
//     CircularBufferConfig cb_column_indices_config = CircularBufferConfig(
//         dram_buffer_D_size, {{column_indices_cb_index, tt::DataFormat::Int32}})
//                                                 .set_page_size(column_indices_cb_index, col_indices_single_tile_size);
//     auto cb_column_indices = tt_metal::CreateCircularBuffer(program, all_cores, cb_column_indices_config);

//      /*
//      * Compile time arguments
//      */
//     bool src0_is_dram = src0_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
//     bool src1_is_dram = src1_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
//     bool Noc_args_is_dram = column_indices_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
//     std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_is_dram, (uint32_t)src1_is_dram, (uint32_t)Noc_args_is_dram};

//     bool dst_is_dram = true;
//     std::vector<uint32_t> writer_compile_time_args = {(uint32_t)dst_is_dram};

//     /*
//      * Create Kernels (Reader, Writer, Compute)
//      */
//     // Create reader and writer kernels per core
//     auto reader_id = tt_metal::CreateKernel(
//         program,
//         "tt_metal/programming_examples/rahmy/block_spmm/kernels/dataflow/reader_block.cpp",
//         all_cores,
//         tt_metal::DataMovementConfig{
//             .processor = DataMovementProcessor::RISCV_0,
//             .noc = NOC::RISCV_0_default,
//             .compile_args = reader_compile_time_args});

//     auto writer_id = tt_metal::CreateKernel(
//         program,
//         "tt_metal/programming_examples/rahmy/block_spmm/kernels/dataflow/writer_block.cpp",
//         all_cores,
//         tt_metal::DataMovementConfig{
//             .processor = DataMovementProcessor::RISCV_1,
//             .noc = NOC::RISCV_1_default,
//             .compile_args = writer_compile_time_args});

//     // Create compute kernel
//     auto mm_kernel_id = tt_metal::CreateKernel(
//         program,
//         "tt_metal/programming_examples/rahmy/block_spmm/kernels/compute/bmm.cpp",
//         all_cores,
//         tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = {}});

//     // instead of iterating over cores, first determine how many output blocks to a core (get the core range set i guess)
//     // and iterate over output blocks, keeping track of which nz output block you are in by taking
//     // into account the zero rows.
//     // ^^ This was good, but  now we have to go back to thinking about cores since we want to be able to assign many output blocks to a single core.
//     //    ... it's not clear whether doing this first and then working on the multicast impl is better
//     //          Yes it is clear: i'm already thinking about this, and the depth of reasoning I'm practicing will help
//     //          me with whatever I decide to do afterwards.

//     TT_ASSERT(nnz_output_blocks_total <= num_cores_x * num_cores_y);

//     uint32_t nnz_output_blocks_read = 0;
//     for (int output_idx_y = 0; output_idx_y < a.indptr.size() - 1; output_idx_y++) {
//         uint32_t nnz_blocks_in_row = a.indptr[output_idx_y + 1] - a.indptr[output_idx_y];
//         if (nnz_blocks_in_row == 0)
//             continue;
//         // else, we are in a nonzero row
//         for (uint32_t output_idx_x = 0; output_idx_x < num_blocks_x; output_idx_x++){
//             int core_idx_x = nnz_output_blocks_read % num_cores_x;
//             int core_idx_y = nnz_output_blocks_read / num_cores_x;
//             CoreCoord core = {(std::size_t)core_idx_x, (std::size_t)core_idx_y};

//             // Write runtime args to device
//             std::vector<uint32_t> mm_reader_args = {
//                 (std::uint32_t)src0_dram_buffer->address(),     // in0_tensor_addr
//                 (std::uint32_t)a.indptr[output_idx_y] * Rt * Ct,// in0_tensor_start_tile_id
//                 (std::uint32_t)1,                               // in0_tensor_stride_w
//                 (std::uint32_t)Ct,                              // in0_tensor_stride_h

//                 (std::uint32_t)in0_block_w,               // in0_block_w
//                 (std::uint32_t)per_core_M,                // in0_block_h
//                 (std::uint32_t)in0_block_w * per_core_M,  // in0_block_num_tiles

//                 (std::uint32_t)src1_dram_buffer->address(),  // in1_tensor_addr
//                 (std::uint32_t)per_core_N * output_idx_x,    // in1_tensor_start_tile_id
//                 (std::uint32_t)1,                            // in1_tensor_stride_w
//                 (std::uint32_t)Nt,                           // in1_tensor_stride_h

//                 (std::uint32_t)per_core_N,                // in1_block_w
//                 (std::uint32_t)in0_block_w,               // in1_block_h
//                 (std::uint32_t)per_core_N * in0_block_w,  // in1_block_num_tiles

//                 (std::uint32_t) a.indptr[output_idx_y],    // col indices start of row
//                 (std::uint32_t) a.indptr[output_idx_y + 1],// col indices end of row
//                 (std::uint32_t) output_idx_y, // row index into bsr matrix

//                 (std::uint32_t)column_indices_dram_buffer->address(), // NoC args, column indices
//             };

//             std::vector<uint32_t> writer_args = {
//                 (std::uint32_t)dst_dram_buffers[output_idx_y]->address(),      // out_buffer_addr
//                 (std::uint32_t)output_idx_x * per_core_N,  // out_tensor_start_tile_id
//                 (std::uint32_t)1,                           // out_tensor_stride_w
//                 (std::uint32_t)Nt,                         // out_tensor_stride_h
//                 (std::uint32_t)out_subblock_w,       // out_tensor_next_subblock_stride_w
//                 (std::uint32_t)out_subblock_h * Nt,  // out_tensor_next_subblock_stride_h

//                 (std::uint32_t)out_subblock_w,                     // out_subblock_w
//                 (std::uint32_t)out_subblock_h,                     // out_subblock_h
//                 (std::uint32_t)(out_subblock_w * out_subblock_h),  // out_subblocks_w * out_subblocks_h
//                 (std::uint32_t)(per_core_N / out_subblock_w),      // out_num_subblocks_w
//                 (std::uint32_t)(per_core_M / out_subblock_h),      // out_num_subblocks_h

//                 (std::uint32_t)Mt * Nt,  // MtNt... only used in the batch impl...
//                 (std::uint32_t)B,        // batch
//                 (std::uint32_t)1*(nnz_blocks_in_row != 0) // nonzero, tells writer whether it has values to read
//             };

//             std::vector<uint32_t> compute_args = {
//                 (std::uint32_t)in0_block_w,
//                 (std::uint32_t)in0_num_subblocks,
//                 (std::uint32_t)in0_block_w * per_core_M, // in0_block_num_tiles
//                 (std::uint32_t)in0_subblock_num_tiles,
//                 (std::uint32_t)in1_num_subblocks,
//                 (std::uint32_t)in1_block_num_tiles,
//                 (std::uint32_t)in1_per_core_w,
//                 (std::uint32_t)nnz_blocks_in_row, // num_blocks
//                 (std::uint32_t)out_subblock_h,
//                 (std::uint32_t)out_subblock_w,
//                 (std::uint32_t)out_subblock_num_tiles,
//                 (std::uint32_t)B
//             };

//             tt_metal::SetRuntimeArgs(program, reader_id, core, mm_reader_args);
//             tt_metal::SetRuntimeArgs(program, writer_id, core, writer_args);
//             tt_metal::SetRuntimeArgs(program, mm_kernel_id, core, compute_args);
//             nnz_output_blocks_read++;

//             if (verbose && output_idx_y == 0 && output_idx_x == 0) {
//                 a.pretty_print();
//                 log_info(tt::LogVerif, " -- Reader Args --");
//                 const char* reader_arg_names[] = {
//                     "in0_tensor_addr",
//                     "in0_tensor_start_tile_id",
//                     "in0_tensor_stride_w",
//                     "in0_tensor_stride_h",
//                     "in0_block_w",
//                     "in0_block_h",
//                     "in0_block_num_tiles",
//                     "in1_tensor_addr",
//                     "in1_tensor_start_tile_id",
//                     "in1_tensor_stride_w",
//                     "in1_tensor_stride_h",
//                     "in1_block_w",
//                     "in1_block_h",
//                     "in1_block_num_tiles",
//                     "col_indices_start_of_row",
//                     "col_indices_end_of_row",
//                     "row_index_into_bsr_matrix",
//                     "column_indices_addr"
//                 };
//                 for (size_t i = 0; i < mm_reader_args.size(); ++i) {
//                     log_info(tt::LogVerif, "reader_arg[{}] ({}) = {}", i, reader_arg_names[i], mm_reader_args[i]);
//                 }
//                 log_info(tt::LogVerif, " -- Writer Args --");
//                 const char* writer_arg_names[] = {
//                     "out_buffer_addr",
//                     "out_tensor_start_tile_id",
//                     "out_tensor_stride_w",
//                     "out_tensor_stride_h",
//                     "out_tensor_next_subblock_stride_w",
//                     "out_tensor_next_subblock_stride_h",
//                     "out_subblock_w",
//                     "out_subblock_h",
//                     "out_subblock_w * out_subblock_h",
//                     "out_num_subblocks_w",
//                     "out_num_subblocks_h",
//                     "MtNt",
//                     "batch",
//                     "nonzero"
//                 };
//                 for (size_t i = 0; i < writer_args.size(); ++i) {
//                     log_info(tt::LogVerif, "writer_arg[{}] ({}) = {}", i, writer_arg_names[i], writer_args[i]);
//                 }
//                 log_info(tt::LogVerif, " -- Compute Args --");
//                 const char* compute_arg_names[] = {
//                     "in0_block_w",
//                     "in0_num_subblocks",
//                     "in0_block_num_tiles",
//                     "in0_subblock_num_tiles",
//                     "in1_num_subblocks",
//                     "in1_block_num_tiles",
//                     "in1_per_core_w",
//                     "nnz_blocks_in_row",
//                     "out_subblock_h",
//                     "out_subblock_w",
//                     "out_subblock_num_tiles",
//                     "B"
//                 };
//                 for (size_t i = 0; i < compute_args.size(); ++i) {
//                     log_info(tt::LogVerif, "compute_arg[{}] ({}) = {}", i, compute_arg_names[i], compute_args[i]);
//                 }
//             }
//         }
//     }



//     if (verbose){
//         log_info(tt::LogVerif, " -- Runtime Args set --");
//         log_info(
//             tt::LogVerif,
//             " -- nnz output blocks= {}",
//             nnz_output_blocks_read);
//     }


//     EnqueueWriteBuffer(cq, src0_dram_buffer, a.data.data(), true);
//     EnqueueWriteBuffer(cq, src1_dram_buffer, b.data.data(), true);
//     EnqueueWriteBuffer(cq, column_indices_dram_buffer, a.indices.data(), true);

//     if (verbose)
//         log_info(tt::LogVerif, " -- All data moved to DRAM --");

//     EnqueueProgram(cq, program, true);

//     if (verbose)
//         log_info(tt::LogVerif, " -- Program enqueued --");

//     for (auto & pair : dst_dram_buffers){
//         uintptr_t row_index = pair.first;
//         std::shared_ptr<Buffer> buffer = pair.second;
//         EnqueueReadBuffer(cq, buffer, output.data.data() + (row_index * R * N), true);
//     }

//     Finish(cq);
// }

// void bsr_spmm_multicore_reuse_naive(
//     bsr_matrix<bfloat16>& a,
//     dense_matrix<bfloat16>& b,
//     dense_matrix<bfloat16>& output,
//     bool bcast_batch,
//     uint32_t nnz_blocks,
//     uint32_t M,
//     uint32_t N,
//     uint32_t K,
//     uint32_t R,
//     uint32_t C,
//     uint32_t B,
//     IDevice* device,
//     bool verbose = false) {

//     CommandQueue& cq = device->command_queue();
//     Program program{};

//     tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
//     MathFidelity math_fidelity = MathFidelity::HiFi4;
//     uint32_t single_tile_size = detail::TileSize(cb_data_format);
//     // uint32_t single_tile_size = 2 * 1024;

//     tt::DataFormat col_indices_data_format = tt::DataFormat::Int32;
//     uint32_t col_indices_single_tile_size = detail::TileSize(col_indices_data_format);

//     auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
//     uint32_t num_cores_x = compute_with_storage_grid_size.x;
//     uint32_t num_cores_y = compute_with_storage_grid_size.y;

//     uint32_t Mt = M / TILE_HEIGHT;
//     uint32_t Kt = K / TILE_WIDTH;
//     uint32_t Nt = N / TILE_WIDTH;

//     uint32_t Rt = R / TILE_HEIGHT;
//     uint32_t Ct = C / TILE_WIDTH;




//     // Get large matmul params
//     //
//     // auto matmul_params = bmm_op_utils::get_large_matmul_params(Mt, Nt, num_cores_y, num_cores_x, in0_block_w);
//     // uint32_t per_core_M = std::get<0>(matmul_params);
//     // uint32_t per_core_N = std::get<1>(matmul_params);
//     // uint32_t out_subblock_h = std::get<2>(matmul_params);
//     // uint32_t out_subblock_w = std::get<3>(matmul_params);
//     //
//     // NAIVE: these should adapt to per core workload later. So we have to understand the util function and why it works!
//     //          short idea: let the tt block size be the nz block size, then take the largest of the 20 subblock choices which fits.
//     //          what breaks when in0_block_w = 2??
//     uint32_t per_core_M = Rt;
//     uint32_t in0_block_w = Ct;

//     int32_t num_tiles_for_col_indices = (col_indices_single_tile_size - 1 + sizeof(int) * nnz_blocks) / col_indices_single_tile_size;
//     uint32_t per_core_N = _get_maximum_block_dim_with_NoC_args(per_core_M, in0_block_w, num_tiles_for_col_indices);
//     per_core_N = std::min(std::min(per_core_N, Ct), Nt);


//     // pick the largest subblock size that fits within the block size
//     uint32_t out_subblock_h, out_subblock_w;
//     for (auto& subblock_hw : bmm_op_utils::SUBBLOCK_HW_CHOICES) {
//         out_subblock_h = std::get<0>(subblock_hw);
//         out_subblock_w = std::get<1>(subblock_hw);
//         if (per_core_M % out_subblock_h == 0 and per_core_N % out_subblock_w == 0) {
//             break;
//         }
//     }


//     if (verbose) {
//         log_info(tt::LogVerif, " -- Metalium Core Sizing --");
//         log_info(
//             tt::LogVerif,
//             " -- per_core_M= {} -- per_core_N= {} -- out_subblock_h= {} -- out_subblock_w= {} --",
//             per_core_M,
//             per_core_N,
//             out_subblock_h,
//             out_subblock_w);
//     }

//     TT_ASSERT(Mt % per_core_M == 0);
//     TT_ASSERT(Nt % per_core_N == 0);
//     TT_ASSERT(Kt % in0_block_w == 0);

//     uint32_t in0_block_tiles = per_core_M * in0_block_w;
//     uint32_t in0_CB_tiles = in0_block_tiles * 2;  // double buffer
//     uint32_t in0_CB_size = in0_CB_tiles * single_tile_size;
//     uint32_t in1_block_tiles = per_core_N * in0_block_w;
//     uint32_t in1_CB_tiles = in1_block_tiles * 2;  // double buffer
//     uint32_t in1_CB_size = in1_CB_tiles * single_tile_size;
//     uint32_t out_block_tiles = per_core_M * per_core_N;
//     uint32_t out_CB_tiles = out_block_tiles;  // No double buffer
//     uint32_t out_CB_size = out_CB_tiles * single_tile_size;

//     uint32_t in0_num_subblocks = (per_core_M / out_subblock_h);
//     uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;
//     uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w; // this is named weird but it's correct.

//     uint32_t in1_num_subblocks = (per_core_N / out_subblock_w);
//     uint32_t in1_block_num_tiles = out_subblock_w * in0_block_w * in1_num_subblocks;
//     uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;

//     uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

//     /*
//      * Multi-Core prep
//      */
//     // auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
//     // uint32_t num_cores_x = compute_with_storage_grid_size.x;
//     // uint32_t num_cores_y = compute_with_storage_grid_size.y;
//     //
//     // NAIVE: these should adapt to per core workload later.
//     // uint32_t num_blocks_y = Mt / per_core_M;
//     uint32_t num_blocks_y = M / R; // block_matrix_height, how many blocks tall the input matrix is.
//     uint32_t num_blocks_x = Nt / per_core_N;
//     uint32_t num_blocks_total = num_blocks_y * num_blocks_x;
//     TT_ASSERT(num_blocks_total <= num_cores_x * num_cores_y);
//     CoreRangeSet all_cores(
//         tt::tt_metal::num_cores_to_corerangeset(num_blocks_x * num_blocks_y, compute_with_storage_grid_size, true));


//     if (verbose) {
//         log_info(tt::LogVerif, " -- Metalium Grid Sizing --");
//         log_info(
//             tt::LogVerif,
//             " -- Mt= {} -- Nt= {} -- num_blocks_x= {} -- num_blocks_y= {} -- num_cores_used={}-- num_cores_x={} -- num_cores_y={} --",
//             Mt,
//             Nt,
//             num_blocks_x,
//             num_blocks_y,
//             all_cores.size(),
//             num_cores_x,
//             num_cores_y);
//     }

//     //////////////////////////////////////////////////
//     /*
//      * Create DRAM Buffers for input and output vectors
//      * Writing data from input vectors to source buffers
//      */

//     uint32_t dram_buffer_A_size =
//         single_tile_size * Rt * Ct * nnz_blocks;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels
//     uint32_t dram_buffer_B_size =
//         single_tile_size * Nt * Kt;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels
//     uint32_t dram_buffer_C_size =
//         single_tile_size * Mt * Nt;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels

//     // In fact let's pad this to fill a tile at least
//     uint32_t dram_buffer_D_size =
//         sizeof(int) * nnz_blocks; //
//     dram_buffer_D_size = col_indices_single_tile_size * ((col_indices_single_tile_size - 1 + dram_buffer_D_size) / (col_indices_single_tile_size));
//     tt_metal::InterleavedBufferConfig dram_config_A{
//         .device = device,
//         .size = dram_buffer_A_size,
//         .page_size = single_tile_size,
//         .buffer_type = tt_metal::BufferType::DRAM};

//     tt_metal::InterleavedBufferConfig dram_config_B{
//         .device = device,
//         .size = dram_buffer_B_size,
//         .page_size = single_tile_size,
//         .buffer_type = tt_metal::BufferType::DRAM};

//     tt_metal::InterleavedBufferConfig dram_config_C{
//         .device = device,
//         .size = dram_buffer_C_size,
//         .page_size = single_tile_size,
//         .buffer_type = tt_metal::BufferType::DRAM};


//     tt_metal::InterleavedBufferConfig dram_config_D{
//         .device = device,
//         .size = dram_buffer_D_size,
//         .page_size = col_indices_single_tile_size,
//         .buffer_type = tt_metal::BufferType::DRAM};

//     auto src0_dram_buffer = CreateBuffer(dram_config_A);
//     auto src1_dram_buffer = CreateBuffer(dram_config_B);
//     auto dst_dram_buffer = CreateBuffer(dram_config_C);
//     auto column_indices_dram_buffer = CreateBuffer(dram_config_D);
//     uint32_t src0_addr = src0_dram_buffer->address();
//     uint32_t src1_addr = src1_dram_buffer->address();
//     uint32_t dst_addr = dst_dram_buffer->address();
//     uint32_t column_indices_addr = column_indices_dram_buffer->address();

//     // logically we want this but the cpu can't directly manage device memory like this
//     // memset((void *)dst_addr, 0, sizeof(bfloat16) * M * N);
//     //
//     // marty's suggestion: use the SFPU to load a CB with zeros, then NoC to DRAM.
//     /*
//      * Config of Circular Buffer in the device L1
//      * input tiles count is = 2 because it's single tile process, and double-buffer
//      */

//     // NAIVE: for this first, naive impl, keep all the CBs the same size, the maximum size
//     uint32_t src0_cb_index = CBIndex::c_0;  // 0
//     CircularBufferConfig cb_src0_config = CircularBufferConfig(in0_CB_size, {{src0_cb_index, cb_data_format}})
//                                               .set_page_size(src0_cb_index, single_tile_size);
//     auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

//     uint32_t src1_cb_index = CBIndex::c_1;  // 1
//     CircularBufferConfig cb_src1_config = CircularBufferConfig(in1_CB_size, {{src1_cb_index, cb_data_format}})
//                                               .set_page_size(src1_cb_index, single_tile_size);
//     auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);



//     uint32_t output_cb_index = tt::CBIndex::c_16;
//     uint32_t interm0_cb_index = tt::CBIndex::c_24;
//     std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec{
//         {output_cb_index, cb_data_format}, {interm0_cb_index, cb_data_format}};
//         CircularBufferConfig cb_output_config = CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
//         .set_page_size(output_cb_index, single_tile_size)
//         .set_page_size(interm0_cb_index, single_tile_size);
//         auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

//     uint32_t column_indices_cb_index = CBIndex::c_2;  // 2
//     CircularBufferConfig cb_column_indices_config = CircularBufferConfig(
//         dram_buffer_D_size, {{column_indices_cb_index, tt::DataFormat::Int32}})
//                                                 .set_page_size(column_indices_cb_index, col_indices_single_tile_size);
//     auto cb_column_indices = tt_metal::CreateCircularBuffer(program, all_cores, cb_column_indices_config);

//      /*
//      * Compile time arguments
//      */
//     bool src0_is_dram = src0_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
//     bool src1_is_dram = src1_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
//     bool Noc_args_is_dram = column_indices_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
//     std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_is_dram, (uint32_t)src1_is_dram, (uint32_t)Noc_args_is_dram};

//     bool dst_is_dram = dst_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
//     std::vector<uint32_t> writer_compile_time_args = {(uint32_t)dst_is_dram};

//     /*
//      * Create Kernels (Reader, Writer, Compute)
//      */
//     // Create reader and writer kernels per core
//     auto reader_id = tt_metal::CreateKernel(
//         program,
//         "tt_metal/programming_examples/rahmy/block_spmm/kernels/dataflow/reader_block.cpp",
//         all_cores,
//         tt_metal::DataMovementConfig{
//             .processor = DataMovementProcessor::RISCV_0,
//             .noc = NOC::RISCV_0_default,
//             .compile_args = reader_compile_time_args});

//     auto writer_id = tt_metal::CreateKernel(
//         program,
//         "tt_metal/programming_examples/rahmy/block_spmm/kernels/dataflow/writer_block.cpp",
//         all_cores,
//         tt_metal::DataMovementConfig{
//             .processor = DataMovementProcessor::RISCV_1,
//             .noc = NOC::RISCV_1_default,
//             .compile_args = writer_compile_time_args});

//     // Create compute kernel
//     auto mm_kernel_id = tt_metal::CreateKernel(
//         program,
//         "tt_metal/programming_examples/rahmy/block_spmm/kernels/compute/bmm.cpp",
//         all_cores,
//         tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = {}});

//     uint32_t num_nnz_output_blocks = 0;
//     uint32_t num_blocks_read = 0;
//     for (int output_idx_y = 0; output_idx_y < num_blocks_y; output_idx_y++) {
//         for (int output_idx_x = 0; output_idx_x < num_blocks_x; output_idx_x++) {
//             int core_idx_x = num_blocks_read % num_cores_x;
//             int core_idx_y = num_blocks_read / num_cores_x;
//             CoreCoord core = {(std::size_t)core_idx_x, (std::size_t)core_idx_y};

//             uint32_t num_blocks = a.indptr[output_idx_y + 1] - a.indptr[output_idx_y];
//             // Write runtime args to device
//             std::vector<uint32_t> mm_reader_args = {
//                 (std::uint32_t)src0_dram_buffer->address(),     // in0_tensor_addr
//                 (std::uint32_t)a.indptr[output_idx_y] * Rt * Ct,// in0_tensor_start_tile_id
//                 (std::uint32_t)1,                               // in0_tensor_stride_w
//                 (std::uint32_t)Ct,                              // in0_tensor_stride_h

//                 (std::uint32_t)in0_block_w,               // in0_block_w
//                 (std::uint32_t)per_core_M,                // in0_block_h
//                 (std::uint32_t)in0_block_w * per_core_M,  // in0_block_num_tiles

//                 (std::uint32_t)src1_dram_buffer->address(),  // in1_tensor_addr
//                 (std::uint32_t)per_core_N * output_idx_x,    // in1_tensor_start_tile_id
//                 (std::uint32_t)1,                            // in1_tensor_stride_w
//                 (std::uint32_t)Nt,                           // in1_tensor_stride_h

//                 (std::uint32_t)per_core_N,                // in1_block_w
//                 (std::uint32_t)in0_block_w,               // in1_block_h
//                 (std::uint32_t)per_core_N * in0_block_w,  // in1_block_num_tiles

//                 (std::uint32_t) a.indptr[output_idx_y],    // col indices start of row
//                 (std::uint32_t) a.indptr[output_idx_y + 1],// col indices end of row
//                 (std::uint32_t) output_idx_y, // row index into bsr matrix

//                 (std::uint32_t)column_indices_dram_buffer->address(), // NoC args, column indices
//             };

//             std::vector<uint32_t> writer_args = {
//                 (std::uint32_t)dst_dram_buffer->address(),                                  // out_buffer_addr
//                 (std::uint32_t)output_idx_x * per_core_N + output_idx_y * per_core_M * Nt,  // out_tensor_start_tile_id
//                 (std::uint32_t)1,                                                           // out_tensor_stride_w
//                 (std::uint32_t)Nt,                                                          // out_tensor_stride_h
//                 (std::uint32_t)out_subblock_w,       // out_tensor_next_subblock_stride_w
//                 (std::uint32_t)out_subblock_h * Nt,  // out_tensor_next_subblock_stride_h

//                 (std::uint32_t)out_subblock_w,                     // out_subblock_w
//                 (std::uint32_t)out_subblock_h,                     // out_subblock_h
//                 (std::uint32_t)(out_subblock_w * out_subblock_h),  // out_subblocks_w * out_subblocks_h
//                 (std::uint32_t)(per_core_N / out_subblock_w),      // out_num_subblocks_w
//                 (std::uint32_t)(per_core_M / out_subblock_h),      // out_num_subblocks_h

//                 (std::uint32_t)Mt * Nt,  // MtNt
//                 (std::uint32_t)B,        // batch
//                 (std::uint32_t)1*(num_blocks != 0) // nonzero, tells writer whether it has values to read
//             };

//             std::vector<uint32_t> compute_args = {
//                 (std::uint32_t)in0_block_w,
//                 (std::uint32_t)in0_num_subblocks,
//                 (std::uint32_t)in0_block_w * per_core_M, // in0_block_num_tiles
//                 (std::uint32_t)in0_subblock_num_tiles,
//                 (std::uint32_t)in1_num_subblocks,
//                 (std::uint32_t)in1_block_num_tiles,
//                 (std::uint32_t)in1_per_core_w,
//                 (std::uint32_t)num_blocks, // num_blocks
//                 (std::uint32_t)out_subblock_h,
//                 (std::uint32_t)out_subblock_w,
//                 (std::uint32_t)out_subblock_num_tiles,
//                 (std::uint32_t)B
//             };

//             tt_metal::SetRuntimeArgs(program, reader_id, core, mm_reader_args);
//             tt_metal::SetRuntimeArgs(program, writer_id, core, writer_args);
//             tt_metal::SetRuntimeArgs(program, mm_kernel_id, core, compute_args);
//             num_blocks_read++;
//             if (num_blocks > 0)
//                 num_nnz_output_blocks++;

//             if (verbose && output_idx_y == 0 && output_idx_x == 1) {
//                 a.pretty_print();
//                 log_info(tt::LogVerif, " -- Reader Args --");
//                 const char* reader_arg_names[] = {
//                     "in0_tensor_addr",
//                     "in0_tensor_start_tile_id",
//                     "in0_tensor_stride_w",
//                     "in0_tensor_stride_h",
//                     "in0_block_w",
//                     "in0_block_h",
//                     "in0_block_num_tiles",
//                     "in1_tensor_addr",
//                     "in1_tensor_start_tile_id",
//                     "in1_tensor_stride_w",
//                     "in1_tensor_stride_h",
//                     "in1_block_w",
//                     "in1_block_h",
//                     "in1_block_num_tiles",
//                     "col_indices_start_of_row",
//                     "col_indices_end_of_row",
//                     "row_index_into_bsr_matrix",
//                     "column_indices_addr"
//                 };
//                 for (size_t i = 0; i < mm_reader_args.size(); ++i) {
//                     log_info(tt::LogVerif, "reader_arg[{}] ({}) = {}", i, reader_arg_names[i], mm_reader_args[i]);
//                 }
//                 log_info(tt::LogVerif, " -- Writer Args --");
//                 const char* writer_arg_names[] = {
//                     "out_buffer_addr",
//                     "out_tensor_start_tile_id",
//                     "out_tensor_stride_w",
//                     "out_tensor_stride_h",
//                     "out_tensor_next_subblock_stride_w",
//                     "out_tensor_next_subblock_stride_h",
//                     "out_subblock_w",
//                     "out_subblock_h",
//                     "out_subblock_w * out_subblock_h",
//                     "out_num_subblocks_w",
//                     "out_num_subblocks_h",
//                     "MtNt",
//                     "batch",
//                     "nonzero"
//                 };
//                 for (size_t i = 0; i < writer_args.size(); ++i) {
//                     log_info(tt::LogVerif, "writer_arg[{}] ({}) = {}", i, writer_arg_names[i], writer_args[i]);
//                 }
//                 log_info(tt::LogVerif, " -- Compute Args --");
//                 const char* compute_arg_names[] = {
//                     "in0_block_w",
//                     "in0_num_subblocks",
//                     "in0_block_num_tiles",
//                     "in0_subblock_num_tiles",
//                     "in1_num_subblocks",
//                     "in1_block_num_tiles",
//                     "in1_per_core_w",
//                     "num_blocks",
//                     "out_subblock_h",
//                     "out_subblock_w",
//                     "out_subblock_num_tiles",
//                     "B"
//                 };
//                 for (size_t i = 0; i < compute_args.size(); ++i) {
//                     log_info(tt::LogVerif, "compute_arg[{}] ({}) = {}", i, compute_arg_names[i], compute_args[i]);
//                 }
//             }
//         }
//     }

//     if (verbose){
//         log_info(tt::LogVerif, " -- Runtime Args set --");
//         log_info(
//             tt::LogVerif,
//             " -- nnz output blocks= {} -- num output blocks total {} --",
//             num_nnz_output_blocks,
//             num_blocks_read);
//     }


//     EnqueueWriteBuffer(cq, src0_dram_buffer, a.data.data(), true);
//     EnqueueWriteBuffer(cq, src1_dram_buffer, b.data.data(), true);
//     EnqueueWriteBuffer(cq, column_indices_dram_buffer, a.indices.data(), true);

//     if (verbose)
//         log_info(tt::LogVerif, " -- All data moved to DRAM --");

//     EnqueueProgram(cq, program, true);

//     if (verbose)
//         log_info(tt::LogVerif, " -- Program enqueued --");

//     EnqueueReadBuffer(cq, dst_dram_buffer, output.data.data(), true);

//     Finish(cq);

// }
}

namespace dense_host_code {

    void matmul_multicore_reuse(
        std::vector<bfloat16>& a,
        std::vector<bfloat16>& b,
        std::vector<bfloat16>& output,
        bool bcast_batch, uint32_t M, uint32_t N, uint32_t K, uint32_t B, IDevice* device);

    void matmul_multicore_reuse_mcast(
        std::vector<bfloat16>& a,
        std::vector<bfloat16>& b,
        std::vector<bfloat16>& output,
        bool bcast_batch, uint32_t M, uint32_t N, uint32_t K, uint32_t B, IDevice* device);

    using DenseHostCodeFunctionPtr = void (*)(
        std::vector<bfloat16>& a,
        std::vector<bfloat16>& b,
        std::vector<bfloat16>& output,
        bool bcast_batch,
        uint32_t M,
        uint32_t N,
        uint32_t K,
        uint32_t B,
        IDevice* device);


// static std::pair<DenseHostCodeFunctionPtr, std::string> DenseHostCodeRegistry[] = {
//     {matmul_multicore_reuse_mcast, "matmul_multicore_reuse_mcast"}, // 0
//     {matmul_multicore_reuse, "matmul_multicore_reuse"}, // 1
// };
//     void matmul_multicore_reuse(
//         std::vector<bfloat16>& a,
//         std::vector<bfloat16>& b,
//         std::vector<bfloat16>& output,
//         bool bcast_batch,
//         uint32_t M,
//         uint32_t N,
//         uint32_t K,
//         uint32_t B,
//         IDevice* device) {

//         ZoneScoped;
//         /*
//         * Setup program to execute along with its buffers and kernels to use
//         * Core range is just single core
//         */
//         CommandQueue& cq = device->command_queue();
//         Program program{};

//         tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
//         MathFidelity math_fidelity = MathFidelity::HiFi4;
//         uint32_t single_tile_size = detail::TileSize(cb_data_format);
//         // uint32_t single_tile_size = 2 * 1024;

//         auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
//         uint32_t num_cores_x = compute_with_storage_grid_size.x;
//         uint32_t num_cores_y = compute_with_storage_grid_size.y;


//         /*
//         * EXtracting Matrix dimensions from input/output vectors
//         */
//         // C = A*B
//         // MN = MK*KN
//         uint32_t Mt = M / TILE_HEIGHT;
//         uint32_t Kt = K / TILE_WIDTH;
//         uint32_t Nt = N / TILE_WIDTH;
//         uint32_t KtNt = Kt * Nt;
//         uint32_t MtKt = Mt * Kt;
//         uint32_t MtNt = Mt * Nt;


//         // We are profiling, don't want this output now
//         // log_info(tt::LogVerif, " -- Metalium Grid Sizing --");
//         // log_info(
//         //     tt::LogVerif,
//         //     "Mt= {} -- Nt= {} -- num_cores_x= {} -- num_cores_y= {} --",
//         //     Mt,
//         //     Nt,
//         //     num_cores_x,
//         //     num_cores_y);

//         // NOTE: Only supports matmuls where output is blocks of 16 x 16 tiles (ie. multiples of 16*32 x 16*32)
//         // NOTE: Maximum number of tiles in output is 120 * 16^2 = 30,720 (eg. [1, 1, 5120, 6144])2
//         uint32_t in0_block_w = 2;

//         // uint32_t out_subblock_h = 4;
//         // uint32_t out_subblock_w = 2;
//         // uint32_t per_core_M = 16;
//         // uint32_t per_core_N = 16;

//         // Get large matmul params
//         auto matmul_params = bmm_op_utils::get_large_matmul_params(Mt, Nt, num_cores_y, num_cores_x, in0_block_w);
//         uint32_t per_core_M = std::get<0>(matmul_params);
//         uint32_t per_core_N = std::get<1>(matmul_params);
//         uint32_t out_subblock_h = std::get<2>(matmul_params);
//         uint32_t out_subblock_w = std::get<3>(matmul_params);

//         // We are profiling, don't want this output now
//         // log_info(tt::LogVerif, " -- Metalium Core Sizing --");
//         // log_info(
//         //     tt::LogVerif,
//         //     " -- per_core_M= {} -- per_core_N= {} -- out_subblock_h= {} -- out_subblock_w= {} --",
//         //     per_core_M,
//         //     per_core_N,
//         //     out_subblock_h,
//         //     out_subblock_w);

//         TT_ASSERT(Mt % per_core_M == 0);
//         TT_ASSERT(Nt % per_core_N == 0);
//         TT_ASSERT(Kt % in0_block_w == 0);

//         uint32_t in0_block_tiles = per_core_M * in0_block_w;
//         uint32_t in0_CB_tiles = in0_block_tiles * 2;  // double buffer
//         uint32_t in0_CB_size = in0_CB_tiles * single_tile_size;
//         uint32_t in1_block_tiles = per_core_N * in0_block_w;
//         uint32_t in1_CB_tiles = in1_block_tiles * 2;  // double buffer
//         uint32_t in1_CB_size = in1_CB_tiles * single_tile_size;
//         uint32_t out_block_tiles = per_core_M * per_core_N;
//         uint32_t out_CB_tiles = out_block_tiles;  // No double buffer
//         uint32_t out_CB_size = out_CB_tiles * single_tile_size;

//         // Compute kernel compile time args
//         uint32_t num_blocks = (Kt / in0_block_w);

//         uint32_t in0_num_subblocks = (per_core_M / out_subblock_h);
//         uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;
//         uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;

//         uint32_t in1_num_subblocks = (per_core_N / out_subblock_w);
//         uint32_t in1_block_num_tiles = out_subblock_w * in0_block_w * in1_num_subblocks;
//         uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;

//         uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

//         std::vector<uint32_t> compute_kernel_args = {
//             in0_block_w,             // in0_block_w
//             in0_num_subblocks,       // in0_num_subblocks
//             in0_block_num_tiles,     // in0_block_num_tiles
//             in0_subblock_num_tiles,  // in0_subblock_num_tiles

//             in1_num_subblocks,    // in1_num_subblocks
//             in1_block_num_tiles,  // in1_block_num_tiles
//             in1_per_core_w,       // in1_per_core_w

//             num_blocks,  // num_blocks

//             out_subblock_h,          // out_subblock_h
//             out_subblock_w,          // out_subblock_w
//             out_subblock_num_tiles,  // out_subblock_num_tiles
//             B                        // batch
//         };

//         /*
//         * Multi-Core prep
//         */
//         // auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
//         // uint32_t num_cores_x = compute_with_storage_grid_size.x;
//         // uint32_t num_cores_y = compute_with_storage_grid_size.y;

//         uint32_t num_blocks_y = Mt / per_core_M;
//         uint32_t num_blocks_x = Nt / per_core_N;
//         uint32_t num_blocks_total = num_blocks_y * num_blocks_x;
//         TT_ASSERT(num_blocks_total <= num_cores_x * num_cores_y);
//         CoreRangeSet all_cores(
//             tt::tt_metal::num_cores_to_corerangeset(num_blocks_x * num_blocks_y, compute_with_storage_grid_size, true));


//         //////////////////////////////////////////////////
//         /*
//         * Create DRAM Buffers for input and output vectors
//         * Writing data from input vectors to source buffers
//         */

//         uint32_t dram_buffer_A_size =
//             single_tile_size * Mt * Kt;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels
//         uint32_t dram_buffer_B_size =
//             single_tile_size * Nt * Kt;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels
//         uint32_t dram_buffer_C_size =
//             single_tile_size * Mt * Nt;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels

//         tt_metal::InterleavedBufferConfig dram_config_A{
//             .device = device,
//             .size = dram_buffer_A_size,
//             .page_size = single_tile_size,
//             .buffer_type = tt_metal::BufferType::DRAM};

//         tt_metal::InterleavedBufferConfig dram_config_B{
//             .device = device,
//             .size = dram_buffer_B_size,
//             .page_size = single_tile_size,
//             .buffer_type = tt_metal::BufferType::DRAM};

//         tt_metal::InterleavedBufferConfig dram_config_C{
//             .device = device,
//             .size = dram_buffer_C_size,
//             .page_size = single_tile_size,
//             .buffer_type = tt_metal::BufferType::DRAM};

//         auto src0_dram_buffer = CreateBuffer(dram_config_A);
//         auto src1_dram_buffer = CreateBuffer(dram_config_B);
//         auto dst_dram_buffer = CreateBuffer(dram_config_C);
//         uint32_t src0_addr = src0_dram_buffer->address();
//         uint32_t src1_addr = src1_dram_buffer->address();
//         uint32_t dst_addr = dst_dram_buffer->address();

//         /*
//         * Config of Circular Buffer in the device L1
//         * input tiles count is = 2 because it's single tile process, and double-buffer
//         */
//         uint32_t src0_cb_index = CBIndex::c_0;  // 0
//         CircularBufferConfig cb_src0_config = CircularBufferConfig(in0_CB_size, {{src0_cb_index, cb_data_format}})
//                                                 .set_page_size(src0_cb_index, single_tile_size);
//         auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

//         uint32_t src1_cb_index = CBIndex::c_1;  // 1
//         CircularBufferConfig cb_src1_config = CircularBufferConfig(in1_CB_size, {{src1_cb_index, cb_data_format}})
//                                                 .set_page_size(src1_cb_index, single_tile_size);
//         auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

//         uint32_t output_cb_index = tt::CBIndex::c_16;
//         uint32_t interm0_cb_index = 24;
//         std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec{
//             {output_cb_index, cb_data_format}, {interm0_cb_index, cb_data_format}};
//         CircularBufferConfig cb_output_config = CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
//                                                     .set_page_size(output_cb_index, single_tile_size)
//                                                     .set_page_size(interm0_cb_index, single_tile_size);
//         auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

//         /*
//         * Compile time arguments
//         */
//         bool src0_is_dram = src0_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
//         bool src1_is_dram = src1_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
//         std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_is_dram, (uint32_t)src1_is_dram};

//         bool dst_is_dram = dst_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
//         // std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t) output_cb_index, (uint32_t)dst_is_dram};
//         std::vector<uint32_t> writer_compile_time_args = {(uint32_t)dst_is_dram};

//         /*
//         * Create Kernels (Reader, Writer, Compute)
//         */
//         // Create reader and writer kernels per core
//         auto reader_id = tt_metal::CreateKernel(
//             program,
//             "tt_metal/programming_examples/matmul_common/kernels/dataflow/reader_bmm_tile_layout.cpp",
//             all_cores,
//             tt_metal::DataMovementConfig{
//                 .processor = DataMovementProcessor::RISCV_1,
//                 .noc = NOC::RISCV_1_default,
//                 .compile_args = reader_compile_time_args});

//         auto writer_id = tt_metal::CreateKernel(
//             program,
//             "tt_metal/programming_examples/matmul_common/kernels/dataflow/writer_bmm_tile_layout.cpp",
//             all_cores,
//             tt_metal::DataMovementConfig{
//                 .processor = DataMovementProcessor::RISCV_0,
//                 .noc = NOC::RISCV_0_default,
//                 .compile_args = writer_compile_time_args});

//         // Create compute kernel
//         auto mm_kernel_id = tt_metal::CreateKernel(
//             program,
//             "tt_metal/programming_examples/matmul_common/kernels/compute/bmm_large_block_zm.cpp",
//             all_cores,
//             tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = compute_kernel_args});

//         /*
//         * Kernels - Runtime arguments
//         */
//         uint32_t num_blocks_read = 0;
//         for (int output_idx_y = 0; output_idx_y < num_blocks_y; output_idx_y++) {
//             for (int output_idx_x = 0; output_idx_x < num_blocks_x; output_idx_x++) {
//                 int core_idx_x = num_blocks_read % num_cores_x;
//                 int core_idx_y = num_blocks_read / num_cores_x;
//                 CoreCoord core = {(std::size_t)core_idx_x, (std::size_t)core_idx_y};

//                 // Write runtime args to device
//                 std::vector<uint32_t> mm_reader_args = {
//                     (std::uint32_t)src0_dram_buffer->address(),     // in0_tensor_addr
//                     (std::uint32_t)Kt * per_core_M * output_idx_y,  // in0_tensor_start_tile_id
//                     (std::uint32_t)1,                               // in0_tensor_stride_w
//                     (std::uint32_t)Kt,                              // in0_tensor_stride_h
//                     (std::uint32_t)in0_block_w,                     // in0_tensor_next_block_stride

//                     (std::uint32_t)in0_block_w,               // in0_block_w
//                     (std::uint32_t)per_core_M,                // in0_block_h
//                     (std::uint32_t)in0_block_w * per_core_M,  // in0_block_num_tiles

//                     (std::uint32_t)src1_dram_buffer->address(),  // in1_tensor_addr
//                     (std::uint32_t)per_core_N * output_idx_x,    // in1_tensor_start_tile_id
//                     (std::uint32_t)1,                            // in1_tensor_stride_w
//                     (std::uint32_t)Nt,                           // in1_tensor_stride_h
//                     (std::uint32_t)in0_block_w * Nt,             // in1_tensor_next_block_stride

//                     (std::uint32_t)per_core_N,                // in1_block_w
//                     (std::uint32_t)in0_block_w,               // in1_block_h
//                     (std::uint32_t)per_core_N * in0_block_w,  // in1_block_num_tiles

//                     (std::uint32_t)Kt / in0_block_w,  // num_blocks

//                     (std::uint32_t)Mt * Kt,     // MtKt
//                     (std::uint32_t)Kt * Nt,     // KtNt
//                     (std::uint32_t)B,           // batch
//                     (std::uint32_t)bcast_batch  // bcast_B
//                 };

//                 std::vector<uint32_t> writer_args = {
//                     (std::uint32_t)dst_dram_buffer->address(),                                  // out_buffer_addr
//                     (std::uint32_t)output_idx_x * per_core_N + output_idx_y * per_core_M * Nt,  // out_tensor_start_tile_id
//                     (std::uint32_t)1,                                                           // out_tensor_stride_w
//                     (std::uint32_t)Nt,                                                          // out_tensor_stride_h
//                     (std::uint32_t)out_subblock_w,       // out_tensor_next_subblock_stride_w
//                     (std::uint32_t)out_subblock_h * Nt,  // out_tensor_next_subblock_stride_h

//                     (std::uint32_t)out_subblock_w,                     // out_subblock_w
//                     (std::uint32_t)out_subblock_h,                     // out_subblock_h
//                     (std::uint32_t)(out_subblock_w * out_subblock_h),  // out_subblocks_w * out_subblocks_h
//                     (std::uint32_t)(per_core_N / out_subblock_w),      // out_num_subblocks_w
//                     (std::uint32_t)(per_core_M / out_subblock_h),      // out_num_subblocks_h

//                     (std::uint32_t)Mt * Nt,  // MtNt
//                     (std::uint32_t)B         // batch
//                 };

//                 tt_metal::SetRuntimeArgs(program, reader_id, core, mm_reader_args);
//                 tt_metal::SetRuntimeArgs(program, writer_id, core, writer_args);

//                 num_blocks_read++;
//             }
//         }

//         /* Launch program & read in output buffer result into the host vector */
//         // LaunchProgram(device, program);
//         // ReadFromBuffer(dst_dram_buffer, output);
//         // ReadFromBuffer(src0_dram_buffer, output);

//         // if we actually want this zone to mean anything, it has to capture the time
//         // between beginning the Writes and the first succesful reader kernel NoC.
//         /*
//         Could the behavior be:
//             1. The reader kernel blocks on its NoC call until the entire write buffer is filled
//             OR
//             2. The reader kernel can NoC tiles as they are ready individually
//         I have a feeling it's the second one, in which case the meaningful times will be a
//         little harder to squeeze out. If it is the second one, then the nonblocking h2d/d2h will
//         result in runtime which is somewhat resilient to input matrix size (compute can begin at the same
//         time regardless of input matrix size)
//         Then there are deeper possibilities;
//         1. nonblocking write writes contiguously, thus the first cores to get data are those
//             asking for the first tiles of data, thus the cores begin their computations in a
//             cascading manner.
//         2. nonblocking write writes "randomly", or in some other less predictable manner, and
//             the cores follow some other pattern of beginning their computation, or none at all.
//         But DRAM is banked, and the core placement is potentially unoptimized for the banking,
//         so the first available DRAM tiles may or may not be far away from their target Tensix cores,
//         which introduces a second layer of difficulty in assessing the behavior of non-blocking dram writes.

//         TODO: figure out how to relate the core ids in Tracy's (x, y) pairs to the metalium grid (logical and physical).

//         */

//         // for learning how the code operates, let's  make all these calls block.
//         {
//             ZoneScopedNC("Data Movement and Device code.", tracy::Color::Brown4);
//             EnqueueWriteBuffer(cq, src0_dram_buffer, a.data(), true);
//             EnqueueWriteBuffer(cq, src1_dram_buffer, b.data(), true);
//             EnqueueProgram(cq, program, true);
//             EnqueueReadBuffer(cq, dst_dram_buffer, output.data(), true);
//         }
//     }

//     void matmul_multicore_reuse_mcast(
//         std::vector<bfloat16>& a,
//         std::vector<bfloat16>& b,
//         std::vector<bfloat16>& output,
//         bool bcast_batch,
//         uint32_t M,
//         uint32_t N,
//         uint32_t K,
//         uint32_t B,
//         IDevice* device) {
//         /*
//         * Setup program to execute along with its buffers and kernels to use
//         * Core range is just single core
//         */
//         CommandQueue& cq = device->command_queue();
//         Program program{};

//         tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
//         MathFidelity math_fidelity = MathFidelity::HiFi4;
//         uint32_t single_tile_size = detail::TileSize(cb_data_format);
//         // uint32_t single_tile_size = 2 * 1024;

//         auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
//         uint32_t num_cores_x = compute_with_storage_grid_size.x;
//         uint32_t num_cores_y = compute_with_storage_grid_size.y;

//         /*
//         * EXtracting Matrix dimensions from input/output vectors
//         */
//         // C = A*B
//         // MN = MK*KN
//         uint32_t Mt = M / TILE_HEIGHT;
//         uint32_t Kt = K / TILE_WIDTH;
//         uint32_t Nt = N / TILE_WIDTH;
//         uint32_t KtNt = Kt * Nt;
//         uint32_t MtKt = Mt * Kt;
//         uint32_t MtNt = Mt * Nt;

//         // NOTE: Only supports matmuls where output is blocks of 16 x 16 tiles (ie. multiples of 16*32 x 16*32)
//         // NOTE: Maximum number of tiles in output is 120 * 16^2 = 30,720 (eg. [1, 1, 5120, 6144])2
//         uint32_t in0_block_w = 2;
//         // uint32_t out_subblock_h = 4;
//         // uint32_t out_subblock_w = 2;
//         // uint32_t per_core_M = 16;
//         // uint32_t per_core_N = 16;

//         // Get large matmul params
//         auto matmul_params = bmm_op_utils::get_large_matmul_params(Mt, Nt, num_cores_y, num_cores_x, in0_block_w);
//         uint32_t per_core_M = std::get<0>(matmul_params);
//         uint32_t per_core_N = std::get<1>(matmul_params);
//         uint32_t out_subblock_h = std::get<2>(matmul_params);
//         uint32_t out_subblock_w = std::get<3>(matmul_params);

//         TT_ASSERT(Mt % per_core_M == 0);
//         TT_ASSERT(Nt % per_core_N == 0);
//         TT_ASSERT(Kt % in0_block_w == 0);

//         uint32_t in0_block_tiles = per_core_M * in0_block_w;
//         uint32_t in0_CB_tiles = in0_block_tiles * 2;  // double buffer
//         uint32_t in0_CB_size = in0_CB_tiles * single_tile_size;
//         uint32_t in1_block_tiles = per_core_N * in0_block_w;
//         uint32_t in1_CB_tiles = in1_block_tiles * 2;  // double buffer
//         uint32_t in1_CB_size = in1_CB_tiles * single_tile_size;
//         uint32_t out_block_tiles = per_core_M * per_core_N;
//         uint32_t out_CB_tiles = out_block_tiles;  // No double buffer
//         uint32_t out_CB_size = out_CB_tiles * single_tile_size;

//         // Compute kernel compile time args
//         uint32_t num_blocks = (Kt / in0_block_w);

//         uint32_t in0_num_subblocks = (per_core_M / out_subblock_h);
//         uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;
//         uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;

//         uint32_t in1_num_subblocks = (per_core_N / out_subblock_w);
//         uint32_t in1_block_num_tiles = out_subblock_w * in0_block_w * in1_num_subblocks;
//         uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;

//         uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

//         std::vector<uint32_t> compute_kernel_args = {
//             in0_block_w,             // in0_block_w
//             in0_num_subblocks,       // in0_num_subblocks
//             in0_block_num_tiles,     // in0_block_num_tiles
//             in0_subblock_num_tiles,  // in0_subblock_num_tiles

//             in1_num_subblocks,    // in1_num_subblocks
//             in1_block_num_tiles,  // in1_block_num_tiles
//             in1_per_core_w,       // in1_per_core_w

//             num_blocks,  // num_blocks

//             out_subblock_h,          // out_subblock_h
//             out_subblock_w,          // out_subblock_w
//             out_subblock_num_tiles,  // out_subblock_num_tiles
//             B                        // batch
//         };

//         /*
//         * Multi-Core prep
//         */
//         uint32_t num_blocks_y = Mt / per_core_M;
//         uint32_t num_blocks_x = Nt / per_core_N;
//         uint32_t num_blocks_total = num_blocks_y * num_blocks_x;
//         TT_ASSERT(num_blocks_total <= num_cores_x * num_cores_y);
//         CoreCoord start_core = {0, 0};
//         CoreCoord core_range = bmm_op_utils::get_core_range(num_blocks_y, num_blocks_x, num_cores_y, num_cores_x);

//         uint32_t start_core_x = start_core.x;
//         uint32_t start_core_y = start_core.y;
//         uint32_t num_cores_c = core_range.x;
//         uint32_t num_cores_r = core_range.y;

//         CoreRange all_cores(
//             {(std::size_t)start_core_x, (std::size_t)start_core_y},
//             {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y + num_cores_r - 1});

//         CoreRange left_column(
//             {(std::size_t)start_core_x, (std::size_t)start_core_y},
//             {(std::size_t)start_core_x, (std::size_t)start_core_y + num_cores_r - 1});

//         CoreRange all_except_left_column(
//             {(std::size_t)start_core_x + 1, (std::size_t)start_core_y},
//             {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y + num_cores_r - 1});

//         CoreRange in0_sender_in1_sender(
//             {(std::size_t)start_core_x, (std::size_t)start_core_y}, {(std::size_t)start_core_x, (std::size_t)start_core_y});

//         CoreRange in0_sender_in1_receiver(
//             {(std::size_t)start_core_x, (std::size_t)start_core_y + 1},
//             {(std::size_t)start_core_x, (std::size_t)start_core_y + num_cores_r - 1});

//         CoreRange in0_receiver_in1_sender(
//             {(std::size_t)start_core_x + 1, (std::size_t)start_core_y},
//             {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y});

//         CoreRange in0_receiver_in1_receiver(
//             {(std::size_t)start_core_x + 1, (std::size_t)start_core_y + 1},
//             {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y + num_cores_r - 1});

//         //////////////////////////////////////////////////
//         /*
//         * Create DRAM Buffers for input and output vectors
//         * Writing data from input vectors to source buffers
//         */

//         uint32_t dram_buffer_A_size =
//             single_tile_size * Mt * Kt;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels
//         uint32_t dram_buffer_B_size =
//             single_tile_size * Nt * Kt;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels
//         uint32_t dram_buffer_C_size =
//             single_tile_size * Mt * Nt;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels
//         tt_metal::InterleavedBufferConfig dram_config_A{
//             .device = device,
//             .size = dram_buffer_A_size,
//             .page_size = single_tile_size,
//             .buffer_type = tt_metal::BufferType::DRAM};

//         tt_metal::InterleavedBufferConfig dram_config_B{
//             .device = device,
//             .size = dram_buffer_B_size,
//             .page_size = single_tile_size,
//             .buffer_type = tt_metal::BufferType::DRAM};

//         tt_metal::InterleavedBufferConfig dram_config_C{
//             .device = device,
//             .size = dram_buffer_C_size,
//             .page_size = single_tile_size,
//             .buffer_type = tt_metal::BufferType::DRAM};

//         auto src0_dram_buffer = CreateBuffer(dram_config_A);
//         auto src1_dram_buffer = CreateBuffer(dram_config_B);
//         auto dst_dram_buffer = CreateBuffer(dram_config_C);
//         uint32_t src0_addr = src0_dram_buffer->address();
//         uint32_t src1_addr = src1_dram_buffer->address();
//         uint32_t dst_addr = dst_dram_buffer->address();

//         /*
//         * Config of Circular Buffer in the device L1
//         * input tiles count is = 2 because it's single tile process, and double-buffer
//         */
//         uint32_t src0_cb_index = CBIndex::c_0;  // 0
//         CircularBufferConfig cb_src0_config = CircularBufferConfig(in0_CB_size, {{src0_cb_index, cb_data_format}})
//                                                 .set_page_size(src0_cb_index, single_tile_size);
//         auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

//         uint32_t src1_cb_index = CBIndex::c_1;  // 1
//         CircularBufferConfig cb_src1_config = CircularBufferConfig(in1_CB_size, {{src1_cb_index, cb_data_format}})
//                                                 .set_page_size(src1_cb_index, single_tile_size);
//         auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

//         uint32_t output_cb_index = tt::CBIndex::c_16;
//         uint32_t interm0_cb_index = 24;
//         std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec{
//             {output_cb_index, cb_data_format}, {interm0_cb_index, cb_data_format}};
//         CircularBufferConfig cb_output_config = CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
//                                                     .set_page_size(output_cb_index, single_tile_size)
//                                                     .set_page_size(interm0_cb_index, single_tile_size);
//         auto cb_output = tt_metal::CreateCircularBuffer(program, CoreRangeSet({all_cores}), cb_output_config);

//         ////////////////////////////
//         /*
//         * Compile time arguments
//         */
//         bool src0_is_dram = src0_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
//         bool src1_is_dram = src1_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
//         std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_is_dram, (uint32_t)src1_is_dram};

//         bool dst_is_dram = dst_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
//         // std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t) output_cb_index, (uint32_t)dst_is_dram};
//         std::vector<uint32_t> writer_compile_time_args = {(uint32_t)dst_is_dram};

//         /*
//         * Create Kernels (Reader, Writer, Compute)
//         */
//         // Create reader and writer kernels per core group

//         auto mm_reader_kernel_in0_sender_in1_sender_id = tt_metal::CreateKernel(
//             program,
//             "tt_metal/programming_examples/matmul_common/kernels/dataflow/reader_bmm_tile_layout_in0_sender_in1_sender.cpp",
//             in0_sender_in1_sender,
//             tt_metal::DataMovementConfig{
//                 .processor = tt_metal::DataMovementProcessor::RISCV_1,
//                 .noc = tt_metal::NOC::RISCV_0_default,
//                 .compile_args = reader_compile_time_args});

//         auto mm_reader_kernel_in0_sender_in1_receiver_id = tt_metal::CreateKernel(
//             program,
//             "tt_metal/programming_examples/matmul_common/kernels/dataflow/"
//             "reader_bmm_tile_layout_in0_sender_in1_receiver.cpp",
//             in0_sender_in1_receiver,
//             tt_metal::DataMovementConfig{
//                 .processor = tt_metal::DataMovementProcessor::RISCV_1,
//                 .noc = tt_metal::NOC::RISCV_0_default,
//                 .compile_args = reader_compile_time_args});

//         auto mm_reader_kernel_in0_receiver_in1_sender_id = tt_metal::CreateKernel(
//             program,
//             "tt_metal/programming_examples/matmul_common/kernels/dataflow/"
//             "reader_bmm_tile_layout_in0_receiver_in1_sender.cpp",
//             in0_receiver_in1_sender,
//             tt_metal::DataMovementConfig{
//                 .processor = tt_metal::DataMovementProcessor::RISCV_1,
//                 .noc = tt_metal::NOC::RISCV_1_default,
//                 .compile_args = reader_compile_time_args});

//         auto mm_reader_kernel_in0_receiver_in1_receiver_id = tt_metal::CreateKernel(
//             program,
//             "tt_metal/programming_examples/matmul_common/kernels/dataflow/"
//             "reader_bmm_tile_layout_in0_receiver_in1_receiver.cpp",
//             in0_receiver_in1_receiver,
//             tt_metal::DataMovementConfig{
//                 .processor = tt_metal::DataMovementProcessor::RISCV_1,
//                 .noc = tt_metal::NOC::RISCV_1_default,
//                 .compile_args = reader_compile_time_args});

//         auto unary_writer_kernel_noc0_id = tt_metal::CreateKernel(
//             program,
//             "tt_metal/programming_examples/matmul_common/kernels/dataflow/writer_bmm_tile_layout.cpp",
//             all_except_left_column,
//             tt_metal::DataMovementConfig{
//                 .processor = tt_metal::DataMovementProcessor::RISCV_0,
//                 .noc = tt_metal::NOC::RISCV_0_default,
//                 .compile_args = writer_compile_time_args});

//         auto unary_writer_kernel_noc1_id = tt_metal::CreateKernel(
//             program,
//             "tt_metal/programming_examples/matmul_common/kernels/dataflow/writer_bmm_tile_layout.cpp",
//             left_column,
//             tt_metal::DataMovementConfig{
//                 .processor = tt_metal::DataMovementProcessor::RISCV_0,
//                 .noc = tt_metal::NOC::RISCV_1_default,
//                 .compile_args = writer_compile_time_args});

//         // Create compute kernel
//         auto mm_kernel_id = tt_metal::CreateKernel(
//             program,
//             "tt_metal/programming_examples/matmul_common/kernels/compute/bmm_large_block_zm.cpp",
//             all_cores,
//             tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = compute_kernel_args});

//         auto in0_mcast_sender_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, INVALID);
//         auto in0_mcast_receiver_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, INVALID);
//         auto in1_mcast_sender_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, INVALID);
//         auto in1_mcast_receiver_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, INVALID);

//         /*
//         * Kernels - Runtime arguments
//         */
//         std::vector<KernelHandle> reader_kernel_ids;
//         std::vector<KernelHandle> writer_kernel_ids;
//         for (int core_idx_y = 0; core_idx_y < num_cores_r; core_idx_y++) {
//             for (int core_idx_x = 0; core_idx_x < num_cores_c; core_idx_x++) {
//                 CoreCoord core = {(std::size_t)start_core_x + core_idx_x, (std::size_t)start_core_y + core_idx_y};

//                 CoreCoord left_core = {(std::size_t)start_core_x, (std::size_t)core.y};
//                 CoreCoord left_core_plus_one = {(std::size_t)start_core_x + 1, (std::size_t)core.y};
//                 CoreCoord right_core = {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)core.y};
//                 CoreCoord top_core = {(std::size_t)core.x, (std::size_t)start_core_y};
//                 CoreCoord top_core_plus_one = {(std::size_t)core.x, (std::size_t)start_core_y + 1};
//                 CoreCoord bottom_core = {(std::size_t)core.x, (std::size_t)start_core_y + num_cores_r - 1};

//                 auto left_core_physical = device->worker_core_from_logical_core(left_core);
//                 auto left_core_plus_one_physical = device->worker_core_from_logical_core(left_core_plus_one);
//                 auto right_core_physical = device->worker_core_from_logical_core(right_core);
//                 auto top_core_physical = device->worker_core_from_logical_core(top_core);
//                 auto top_core_plus_one_physical = device->worker_core_from_logical_core(top_core_plus_one);
//                 auto bottom_core_physical = device->worker_core_from_logical_core(bottom_core);

//                 std::vector<uint32_t> mm_reader_args = {
//                     (std::uint32_t)src0_dram_buffer->address(),   // in0_buffer_addr
//                     (std::uint32_t)Kt * per_core_M * core_idx_y,  // in0_buffer_start_tile_id
//                     (std::uint32_t)1,                             // in0_buffer_stride_w
//                     (std::uint32_t)Kt,                            // in0_buffer_stride_h
//                     (std::uint32_t)in0_block_w,                   // in0_buffer_next_block_stride

//                     (std::uint32_t)in0_block_w,               // in0_block_w
//                     (std::uint32_t)per_core_M,                // in0_block_h
//                     (std::uint32_t)in0_block_w * per_core_M,  // in0_block_num_tiles

//                     (std::uint32_t)src1_dram_buffer->address(),  // in1_buffer_addr
//                     (std::uint32_t)per_core_N * core_idx_x,      // in1_buffer_start_tile_id
//                     (std::uint32_t)1,                            // in1_buffer_stride_w
//                     (std::uint32_t)Nt,                           // in1_buffer_stride_h
//                     (std::uint32_t)in0_block_w * Nt,             // in1_buffer_next_block_stride

//                     (std::uint32_t)per_core_N,                // in1_block_w
//                     (std::uint32_t)in0_block_w,               // in1_block_h
//                     (std::uint32_t)per_core_N * in0_block_w,  // in1_block_num_tiles

//                     (std::uint32_t)Kt / in0_block_w,  // num_blocks

//                     (std::uint32_t)right_core_physical.x,          // in0_mcast_dest_noc_start_x
//                     (std::uint32_t)right_core_physical.y,          // in0_mcast_dest_noc_start_y
//                     (std::uint32_t)left_core_plus_one_physical.x,  // in0_mcast_dest_noc_end_x
//                     (std::uint32_t)left_core_plus_one_physical.y,  // in0_mcast_dest_noc_end_y
//                     (std::uint32_t)(num_cores_c - 1),              // in0_mcast_num_dests
//                     (std::uint32_t)left_core_physical.x,           // in0_mcast_sender_noc_x
//                     (std::uint32_t)left_core_physical.y,           // in0_mcast_sender_noc_y
//                     (std::uint32_t)in0_mcast_sender_semaphore_id,
//                     (std::uint32_t)in0_mcast_receiver_semaphore_id,

//                     (std::uint32_t)bottom_core_physical.x,        // in0_mcast_dest_noc_start_x
//                     (std::uint32_t)bottom_core_physical.y,        // in0_mcast_dest_noc_start_y
//                     (std::uint32_t)top_core_plus_one_physical.x,  // in0_mcast_dest_noc_end_x
//                     (std::uint32_t)top_core_plus_one_physical.y,  // in0_mcast_dest_noc_end_y
//                     (std::uint32_t)(num_cores_r - 1),             // in0_mcast_num_dests
//                     (std::uint32_t)top_core_physical.x,           // in0_mcast_sender_noc_x
//                     (std::uint32_t)top_core_physical.y,           // in0_mcast_sender_noc_y
//                     (std::uint32_t)in1_mcast_sender_semaphore_id,
//                     (std::uint32_t)in1_mcast_receiver_semaphore_id,

//                     (std::uint32_t)Mt * Kt,     // MtKt
//                     (std::uint32_t)Kt * Nt,     // KtNt
//                     (std::uint32_t)B,           // batch
//                     (std::uint32_t)bcast_batch  // bcast_B
//                 };

//                 std::vector<uint32_t> writer_args = {
//                     (std::uint32_t)dst_dram_buffer->address(),                              // out_buffer_addr
//                     (std::uint32_t)core_idx_x * per_core_N + core_idx_y * per_core_M * Nt,  // out_buffer_start_tile_id
//                     (std::uint32_t)1,                                                       // out_buffer_stride_w
//                     (std::uint32_t)Nt,                                                      // out_buffer_stride_h
//                     (std::uint32_t)out_subblock_w,       // out_buffer_next_subblock_stride_w
//                     (std::uint32_t)out_subblock_h * Nt,  // out_buffer_next_subblock_stride_h

//                     (std::uint32_t)out_subblock_w,                     // out_subblock_w
//                     (std::uint32_t)out_subblock_h,                     // out_subblock_h
//                     (std::uint32_t)(out_subblock_w * out_subblock_h),  // out_subblocks_w * out_subblocks_h
//                     (std::uint32_t)(per_core_N / out_subblock_w),      // out_num_subblocks_w
//                     (std::uint32_t)(per_core_M / out_subblock_h),      // out_num_subblocks_h

//                     (std::uint32_t)Mt * Nt,  // MtNt
//                     (std::uint32_t)B         // batch
//                 };

//                 if (core_idx_x == 0 and core_idx_y == 0) {
//                     tt_metal::SetRuntimeArgs(
//                         program, mm_reader_kernel_in0_sender_in1_sender_id, core, mm_reader_args);      // RISCV_0_default
//                     tt_metal::SetRuntimeArgs(program, unary_writer_kernel_noc1_id, core, writer_args);  // RISCV_1_default
//                     reader_kernel_ids.push_back(mm_reader_kernel_in0_sender_in1_sender_id);
//                     writer_kernel_ids.push_back(unary_writer_kernel_noc1_id);
//                 } else if (core_idx_x == 0 and core_idx_y != 0) {
//                     tt_metal::SetRuntimeArgs(
//                         program, mm_reader_kernel_in0_sender_in1_receiver_id, core, mm_reader_args);    // RISCV_0_default
//                     tt_metal::SetRuntimeArgs(program, unary_writer_kernel_noc1_id, core, writer_args);  // RISCV_1_default
//                     reader_kernel_ids.push_back(mm_reader_kernel_in0_sender_in1_receiver_id);
//                     writer_kernel_ids.push_back(unary_writer_kernel_noc1_id);
//                 } else if (core_idx_x != 0 and core_idx_y == 0) {
//                     tt_metal::SetRuntimeArgs(
//                         program, mm_reader_kernel_in0_receiver_in1_sender_id, core, mm_reader_args);    // RISCV_1_default
//                     tt_metal::SetRuntimeArgs(program, unary_writer_kernel_noc0_id, core, writer_args);  // RISCV_0_default
//                     reader_kernel_ids.push_back(mm_reader_kernel_in0_receiver_in1_sender_id);
//                     writer_kernel_ids.push_back(unary_writer_kernel_noc0_id);
//                 } else {
//                     tt_metal::SetRuntimeArgs(
//                         program, mm_reader_kernel_in0_receiver_in1_receiver_id, core, mm_reader_args);  // RISCV_1_default
//                     tt_metal::SetRuntimeArgs(program, unary_writer_kernel_noc0_id, core, writer_args);  // RISCV_0_default
//                     reader_kernel_ids.push_back(mm_reader_kernel_in0_receiver_in1_receiver_id);
//                     writer_kernel_ids.push_back(unary_writer_kernel_noc0_id);
//                 }
//             }
//         }

//         /* Launch program & read in output buffer result into the host vector */

//         EnqueueWriteBuffer(cq, src0_dram_buffer, a.data(), false);
//         EnqueueWriteBuffer(cq, src1_dram_buffer, b.data(), false);
//         EnqueueProgram(cq, program, false);
//         EnqueueReadBuffer(cq, dst_dram_buffer, output.data(), true);
//     }
}