/*
Match order of RK
for num_iters_x:
    for num_iters_y:
        business as usual
        out_tensor_start_tile_id += in0_stride_h
    out_tensor_start_tile_id += in1_block_w

*/

#include <cstdint>
#include <cstring>
#include "dataflow_api.h"

void kernel_main() {
    ///////////////////////////////////////////////////////////////////////
    /// COMPILETIME ARGS //////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    constexpr bool out_is_dram = get_compile_time_arg_val(0) == 1;
    // out tensor args
    uint32_t out_tensor_addr = get_compile_time_arg_val(1);
    uint32_t out_tensor_stride_w = get_compile_time_arg_val(2);
    uint32_t out_tensor_stride_h = get_compile_time_arg_val(3);
    uint32_t out_tensor_next_subblock_stride_w = get_compile_time_arg_val(4);
    uint32_t out_tensor_next_subblock_stride_h = get_compile_time_arg_val(5);

    // out subblock args
    uint32_t out_subblock_w = get_compile_time_arg_val(6);
    uint32_t out_subblock_h = get_compile_time_arg_val(7);
    uint32_t out_subblock_tile_count = get_compile_time_arg_val(8);
    uint32_t out_num_subblocks_w = get_compile_time_arg_val(9);
    uint32_t out_num_subblocks_h = get_compile_time_arg_val(10);

    // Mpc/Mpb args
    uint32_t RtNt = get_compile_time_arg_val(11);  // 
    uint32_t num_iters_x = get_compile_time_arg_val(12); //
    ///////////////////////////////////////////////////////////////////////
    /// END COMPILETIME ARGS //////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////
    /// RUNTIME ARGS //////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    uint32_t out_tensor_start_tile_id = get_arg_val<uint32_t>(0);
    uint32_t num_iters_y = get_arg_val<uint32_t>(1); //

    ///////////////////////////////////////////////////////////////////////
    /// END RUNTIME ARGS //////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    constexpr uint32_t cb_id_out0 = 16;
    // single-tile
    const uint32_t single_tile_size_bytes = get_tile_size(cb_id_out0);
    uint32_t l1_read_addr_increment = single_tile_size_bytes;
    const DataFormat data_format = get_dataformat(cb_id_out0);


    const InterleavedAddrGenFast<out_is_dram> s = {
        .bank_base_address = out_tensor_addr, .page_size = single_tile_size_bytes, .data_format = data_format};

    uint32_t out_tensor_x_coord_offset = 0;
    for (uint32_t y = 0; y < num_iters_y; y++){
        for (uint32_t x = 0; x < num_iters_x; x++){
            uint32_t out_tensor_sbh_start_tile_id = out_tensor_start_tile_id + out_tensor_x_coord_offset;
            for (uint32_t sbh = 0; sbh < out_num_subblocks_h; sbh++) {
                uint32_t out_tensor_sbw_start_tile_id = out_tensor_sbh_start_tile_id;
                for (uint32_t sbw = 0; sbw < out_num_subblocks_w; sbw++) {
                    uint32_t out_tensor_sb_row_start_tile_id = out_tensor_sbw_start_tile_id;
                    cb_wait_front(cb_id_out0, out_subblock_tile_count);
                    uint32_t l1_read_addr = get_read_ptr(cb_id_out0);

                    for (uint32_t h = 0; h < out_subblock_h; h++) {
                        uint32_t out_tensor_tile_id = out_tensor_sb_row_start_tile_id;
                        for (uint32_t w = 0; w < out_subblock_w; w++) {
                            noc_async_write_tile(out_tensor_tile_id, s, l1_read_addr);
                            l1_read_addr += l1_read_addr_increment;

                            out_tensor_tile_id += out_tensor_stride_w;
                        }
                        out_tensor_sb_row_start_tile_id += out_tensor_stride_h;
                    }

                    noc_async_write_barrier(); 

                    cb_pop_front(cb_id_out0, out_subblock_tile_count);
                    out_tensor_sbw_start_tile_id += out_tensor_next_subblock_stride_w;
                }
                out_tensor_sbh_start_tile_id += out_tensor_next_subblock_stride_h;
            }
            out_tensor_x_coord_offset += out_num_subblocks_w * out_tensor_next_subblock_stride_w;
        }
        // hop to next output row and reset output column offset
        out_tensor_start_tile_id += RtNt;
        out_tensor_x_coord_offset = 0;
    }
}