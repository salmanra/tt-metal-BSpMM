#include <stdint.h>
#include <cstdint>
#include "dataflow_api.h"
#include "hostdevcommon/kernel_structs.h"
#include "debug/dprint.h"  // required in all kernels using DPRINT

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

    DPRINT_DATA0(DPRINT << "Comptime args obtained: "  << ENDL());

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
    DPRINT_DATA0(DPRINT << "Runtime args obtained. num output blocks to compute: " << num_iters_x * num_iters_y << ENDL());

    ///////////////////////////////////////////////////////////////////////
    /// END RUNTIME ARGS //////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    
    const uint32_t cb_id_in0 = tt::CBIndex::c_0;
    const uint32_t cb_id_in1 = tt::CBIndex::c_1;
    const uint32_t cb_id_col_indices = tt::CBIndex::c_2;
    const uint32_t cb_id_indptr = tt::CBIndex::c_3;

    // input data format will probably by bfloat16
    const uint32_t in0_single_tile_size_bytes = get_tile_size(cb_id_in0);
    const DataFormat in0_data_format = get_dataformat(cb_id_in0);

    const uint32_t in1_single_tile_size_bytes = get_tile_size(cb_id_in1);
    const DataFormat in1_data_format = get_dataformat(cb_id_in1);

    const uint32_t col_indices_single_tile_size_bytes = get_tile_size(cb_id_col_indices);
    const DataFormat col_indices_data_format = get_dataformat(cb_id_col_indices);

    const uint32_t indptr_single_tile_size_bytes = get_tile_size(cb_id_indptr);
    const DataFormat indptr_data_format = get_dataformat(cb_id_indptr);

    uint32_t l1_write_addr_in0;
    uint32_t l1_write_addr_in1;
    uint32_t l1_write_addr_col_indices;
    uint32_t l1_write_addr_indptr;

    const InterleavedAddrGenFast<in0_is_dram> s0 = {
        .bank_base_address = in0_tensor_addr, .page_size = in0_single_tile_size_bytes, .data_format = in0_data_format};
    const InterleavedAddrGenFast<in1_is_dram> s1 = {
        .bank_base_address = in1_tensor_addr, .page_size = in1_single_tile_size_bytes, .data_format = in1_data_format};
    const InterleavedAddrGenFast<col_indices_is_dram> s2 = {
        .bank_base_address = col_indices_addr,
        .page_size = col_indices_single_tile_size_bytes,
        .data_format = col_indices_data_format};
    const InterleavedAddrGenFast<indptr_is_dram> s3 = {
        .bank_base_address = indptr_addr,
        .page_size = indptr_single_tile_size_bytes,
        .data_format = indptr_data_format};

    cb_reserve_back(cb_id_col_indices, col_indices_num_tiles);
    l1_write_addr_col_indices = get_write_ptr(cb_id_col_indices);
    uint32_t col_indices_dram_start_id = 0;
    for (uint32_t i = 0; i < col_indices_num_tiles; i++){
        noc_async_read_tile(col_indices_dram_start_id, s2, l1_write_addr_col_indices);
        col_indices_dram_start_id++;
        l1_write_addr_col_indices += col_indices_single_tile_size_bytes;
    }
    l1_write_addr_col_indices -= col_indices_single_tile_size_bytes * col_indices_num_tiles;
    noc_async_read_barrier();
    cb_push_back(cb_id_col_indices, col_indices_num_tiles);

    cb_reserve_back(cb_id_indptr, indptr_num_tiles);
    l1_write_addr_indptr = get_write_ptr(cb_id_indptr);
    uint32_t indptr_dram_start_id = 0;
    for (uint32_t i = 0; i < indptr_num_tiles; i++){
        noc_async_read_tile(indptr_dram_start_id, s3, l1_write_addr_indptr);
        indptr_dram_start_id++;
        l1_write_addr_indptr += indptr_single_tile_size_bytes;
    }
    l1_write_addr_indptr -= indptr_single_tile_size_bytes * indptr_num_tiles;
    noc_async_read_barrier();
    cb_push_back(cb_id_indptr, indptr_num_tiles);
    
    uint32_t* col_indices = (uint32_t*) l1_write_addr_col_indices;
    uint32_t* indptr = (uint32_t*) l1_write_addr_indptr;
    ///////////////////////////////////////////////////////////////////////
    /// PROGRAM BODY //////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    uint32_t output_idx_y, output_idx_x;
    for (uint32_t iter_y = 0; iter_y < num_iters_y; iter_y++){
        // Get y_coord for this iter
        output_idx_y = y_coords[iter_y];
        uint32_t block_row_start = indptr[output_idx_y];
        uint32_t block_row_end = indptr[output_idx_y + 1];
        DPRINT_DATA0(DPRINT << "output_idx_y: " << output_idx_y << " block_row_start: " << block_row_start << " block_row_end: " << block_row_end << ENDL());

        uint32_t in0_tensor_start_tile_id = block_row_start * in0_block_num_tiles;
        for (uint32_t iter_x = 0; iter_x < num_iters_x; iter_x++){
            output_idx_x = iter_x;
            uint32_t in1_tensor_start_tile_id = in1_block_w * output_idx_x; 
            for (uint32_t reduction_iter = block_row_start; reduction_iter < block_row_end; reduction_iter++){

                cb_reserve_back(cb_id_in0, in0_block_num_tiles);
                cb_reserve_back(cb_id_in1, in1_block_num_tiles);

                l1_write_addr_in0 = get_write_ptr(cb_id_in0);
                l1_write_addr_in1 = get_write_ptr(cb_id_in1);

                // Read in0 block
                uint32_t num_blocks_in = reduction_iter - block_row_start;
                uint32_t in0_tensor_row_start_tile_id = in0_tensor_start_tile_id + num_blocks_in * in0_block_num_tiles;
                for (uint32_t h = 0; h < in0_block_h; h++) {
                    uint32_t in0_tensor_tile_id = in0_tensor_row_start_tile_id; 
                    for (uint32_t w = 0; w < in0_block_w; w++) {
                        DPRINT_DATA0(DPRINT << "TILE ID: " << in0_tensor_tile_id << ENDL());
                        noc_async_read_tile(in0_tensor_tile_id, s0, l1_write_addr_in0);
                        l1_write_addr_in0 += in0_single_tile_size_bytes;
                        in0_tensor_tile_id += in0_tensor_stride_w;
                    }
                    in0_tensor_row_start_tile_id += in0_tensor_stride_h;
                }

                // Read in1 block
                uint32_t bsr_col_index = col_indices[reduction_iter];
                uint32_t in1_block_stride = in1_block_h * in1_tensor_stride_h;
                uint32_t in1_tensor_row_start_tile_id = in1_tensor_start_tile_id + bsr_col_index * in1_block_stride;
                for (uint32_t h = 0; h < in1_block_h; h++) {
                    uint32_t in1_tensor_tile_id = in1_tensor_row_start_tile_id;
                    for (uint32_t w = 0; w < in1_block_w; w++) {
                        noc_async_read_tile(in1_tensor_tile_id, s1, l1_write_addr_in1);
                        l1_write_addr_in1 += in1_single_tile_size_bytes;
                        in1_tensor_tile_id += in1_tensor_stride_w;
                    }
                    in1_tensor_row_start_tile_id += in1_tensor_stride_h;
                }

                noc_async_read_barrier();

                DPRINT_DATA0(DPRINT << "block " << output_idx_y << ", " << col_indices[reduction_iter] << " read" << ENDL());

                // uint32_t* CB_values = (uint32_t*)l1_write_addr_in0;
                // for (size_t idx = 0; idx < in0_single_tile_size_bytes / 4; idx+=32){
                //     for (size_t inner = 0; inner < 32; inner++){
                //         float top_bits = (float)(CB_values[idx + inner] >> 16);
                //         float bottom_bits = (float)(CB_values[idx + inner] & 0xFFFF);
                //         //DPRINT_DATA0(DPRINT << top_bits << ' ' << bottom_bits << ' ');
                //     }
                //     //DPRINT_DATA0(DPRINT << ENDL());
                // }
                // //DPRINT_DATA0(DPRINT << ENDL());
                // //DPRINT_DATA0(DPRINT << ENDL());
                // //DPRINT_DATA0(DPRINT << ENDL());
                // uint32_t* dense_CB_values = (uint32_t*)l1_write_addr_in1;
                // for (size_t idx = 0; idx < in0_single_tile_size_bytes / 4; idx+=32){
                //     for (size_t inner = 0; inner < 32; inner++){
                //         float top_bits = (float)(dense_CB_values[idx + inner] >> 16);
                //         float bottom_bits = (float)(dense_CB_values[idx + inner] & 0xFFFF);
                //         //DPRINT_DATA0(DPRINT << top_bits << ' ' << bottom_bits << ' ');
                //     }
                //     //DPRINT_DATA0(DPRINT << ENDL());
                // }
                cb_push_back(cb_id_in0, in0_block_num_tiles);
                cb_push_back(cb_id_in1, in1_block_num_tiles);
            }
        }
    }
    cb_pop_front(cb_id_col_indices, indptr_num_tiles);
    cb_pop_front(cb_id_indptr, indptr_num_tiles);
    DPRINT_DATA0(DPRINT << "Reader kernel complete!" << ENDL());
}