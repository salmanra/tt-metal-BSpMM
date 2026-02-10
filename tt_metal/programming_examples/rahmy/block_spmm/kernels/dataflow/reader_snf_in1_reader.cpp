#include <stdint.h>
#include <cstdint>
#include "dataflow_api.h"
#include "hostdevcommon/kernel_structs.h"
#include <tools/profiler/kernel_profiler.hpp>
#include "tt_metal/programming_examples/rahmy/block_spmm/kernels/common/spmm_reader_common.hpp"
#include "tt_metal/programming_examples/rahmy/block_spmm/kernels/common/spmm_tile_ops.hpp"
#include "tt_metal/programming_examples/rahmy/block_spmm/kernels/common/spmm_indexing.hpp"


void kernel_main(){
    ///////////////////////////////////////////////////////////////////////
    /// COMPILETIME ARGS //////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    constexpr bool in0_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool in1_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr bool col_indices_is_dram = get_compile_time_arg_val(2) == 1;
    constexpr bool indptr_is_dram = get_compile_time_arg_val(3) == 1;

    constexpr uint32_t in0_tensor_addr = get_compile_time_arg_val(4);
    constexpr uint32_t in0_tensor_stride_w = get_compile_time_arg_val(5);
    constexpr uint32_t in0_tensor_stride_h = get_compile_time_arg_val(6);

    constexpr uint32_t in0_block_w = get_compile_time_arg_val(7);
    constexpr uint32_t in0_block_h = get_compile_time_arg_val(8);
    constexpr uint32_t in0_block_num_tiles = get_compile_time_arg_val(9);

    constexpr uint32_t in1_tensor_addr = get_compile_time_arg_val(10);
    constexpr uint32_t in1_tensor_stride_w = get_compile_time_arg_val(11);
    constexpr uint32_t in1_tensor_stride_h = get_compile_time_arg_val(12);

    constexpr uint32_t in1_block_w = get_compile_time_arg_val(13);
    constexpr uint32_t in1_block_h = get_compile_time_arg_val(14);
    constexpr uint32_t in1_block_num_tiles = get_compile_time_arg_val(15);

    constexpr uint32_t col_indices_addr = get_compile_time_arg_val(16);
    constexpr uint32_t indptr_addr = get_compile_time_arg_val(17);

    constexpr uint32_t col_indices_num_tiles = get_compile_time_arg_val(18);
    constexpr uint32_t indptr_num_tiles = get_compile_time_arg_val(19);

    ///////////////////////////////////////////////////////////////////////
    /// END COMPILETIME ARGS //////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////
    /// RUNTIME ARGS //////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    uint32_t arg_index = 0;
    const uint32_t num_iters_x = get_arg_val<uint32_t>(arg_index++);
    const uint32_t num_iters_y = get_arg_val<uint32_t>(arg_index++);
    const uint32_t output_idx_x_start = get_arg_val<uint32_t>(arg_index++);
    uint32_t y_coords[num_iters_y];
    for (uint32_t i = 0; i < num_iters_y; i++){
        y_coords[i] = get_arg_val<uint32_t>(arg_index++);
    }

    ///////////////////////////////////////////////////////////////////////
    /// END RUNTIME ARGS //////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    const auto ti = spmm::get_tile_info();

    const InterleavedAddrGenFast<in1_is_dram> s1 = {
        .bank_base_address = in1_tensor_addr, .page_size = ti.in1_tile_size, .data_format = ti.in1_format};

    // Wait for indexing data loaded by in0_reader on the other RISC
    uint32_t* indptr = spmm::wait_for_indexing(spmm::cb_id_indptr);
    uint32_t* col_indices = spmm::wait_for_indexing(spmm::cb_id_col_indices);

    ///////////////////////////////////////////////////////////////////////
    /// PROGRAM BODY //////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    uint32_t output_idx_y, output_idx_x;
    for (uint32_t iter_y = 0; iter_y < num_iters_y; iter_y++){
        // Get y_coord for this iter
        output_idx_y = y_coords[iter_y];
        uint32_t block_row_start = indptr[output_idx_y];
        uint32_t block_row_end = indptr[output_idx_y + 1];

        uint32_t in0_tensor_start_tile_id = block_row_start * in0_block_num_tiles;
        for (uint32_t iter_x = 0; iter_x < num_iters_x; iter_x++){
            output_idx_x = output_idx_x_start + iter_x;
            uint32_t in1_tensor_start_tile_id = in1_block_w * output_idx_x;
            for (uint32_t reduction_iter = block_row_start; reduction_iter < block_row_end; reduction_iter++){
                cb_reserve_back(spmm::cb_id_in1, in1_block_num_tiles);

                uint32_t l1_write_addr_in1 = get_write_ptr(spmm::cb_id_in1);

                // Read in1 block (row selected by BSR col_indices)
                uint32_t bsr_col_index = col_indices[reduction_iter];
                uint32_t in1_block_stride = in1_block_h * in1_tensor_stride_h;
                spmm::read_block_by_tile(
                    in1_tensor_start_tile_id + bsr_col_index * in1_block_stride,
                    s1, l1_write_addr_in1,
                    ti.in1_tile_size, in1_block_h, in1_block_w,
                    in1_tensor_stride_h, in1_tensor_stride_w);

                noc_async_read_barrier();
                DeviceZoneScopedN("in1 Block Pushed to CB");
                cb_push_back(spmm::cb_id_in1, in1_block_num_tiles);
            }
        }
    }
    DPRINT_DATA1(DPRINT << "in1 kernel complete" << ENDL());

}
