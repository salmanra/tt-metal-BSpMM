// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstring>
#include "dataflow_api.h"
#include "debug/dprint.h"  // required in all kernels using DPRINT
// #include "tt-metalium/bfloat16.hpp"



void kernel_main() {
    // out tensor args
    uint32_t out_tensor_addr = get_arg_val<uint32_t>(0);
    uint32_t out_tensor_start_tile_id = get_arg_val<uint32_t>(1);
    uint32_t out_tensor_stride_w = get_arg_val<uint32_t>(2);
    uint32_t out_tensor_stride_h = get_arg_val<uint32_t>(3);
    uint32_t out_tensor_next_subblock_stride_w = get_arg_val<uint32_t>(4);
    uint32_t out_tensor_next_subblock_stride_h = get_arg_val<uint32_t>(5);

    // out subblock args
    uint32_t out_subblock_w = get_arg_val<uint32_t>(6);
    uint32_t out_subblock_h = get_arg_val<uint32_t>(7);
    uint32_t out_subblock_tile_count = get_arg_val<uint32_t>(8);
    uint32_t out_num_subblocks_w = get_arg_val<uint32_t>(9);
    uint32_t out_num_subblocks_h = get_arg_val<uint32_t>(10);

    // Mpc/Mpb args
    uint32_t RtNt = get_arg_val<uint32_t>(11);  // 
    uint32_t num_output_blocks = get_arg_val<uint32_t>(12); //


    constexpr bool out_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t bytes_for_sync = get_compile_time_arg_val(1);

    constexpr uint32_t cb_id_out0 = 16;

    // single-tile
    const uint32_t single_tile_size_bytes = get_tile_size(cb_id_out0);
    uint32_t l1_read_addr_increment = single_tile_size_bytes;
    const DataFormat data_format = get_dataformat(cb_id_out0);


    const InterleavedAddrGenFast<out_is_dram> s = {
        .bank_base_address = out_tensor_addr, .page_size = single_tile_size_bytes, .data_format = data_format};

    const uint32_t cb_id_sync = tt::CBIndex::c_4;
    uint32_t l1_read_addr_sync;

    // cb_reserve_back(cb_id_sync, 1);
    // l1_read_addr_sync = get_read_ptr(cb_id_sync);
    // //DPRINT_DATA1(DPRINT << "Writer CB Sync Address:  " << l1_read_addr_sync << ENDL());

    bool one_time_profile = true;
    for (uint32_t b = 0; b < num_output_blocks; b++) {
        uint32_t out_tensor_sbh_start_tile_id = out_tensor_start_tile_id;
        for (uint32_t sbh = 0; sbh < out_num_subblocks_h; sbh++) {
            uint32_t out_tensor_sbw_start_tile_id = out_tensor_sbh_start_tile_id;
            for (uint32_t sbw = 0; sbw < out_num_subblocks_w; sbw++) {
                uint32_t out_tensor_sb_row_start_tile_id = out_tensor_sbw_start_tile_id;
                DPRINT_DATA1(DPRINT << "Writer waiting!" << ENDL());
                cb_wait_front(cb_id_out0, out_subblock_tile_count);
                uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
                DPRINT_DATA1(DPRINT << "Let's Get Writing!" << ENDL());


                for (uint32_t h = 0; h < out_subblock_h; h++) {
                    uint32_t out_tensor_tile_id = out_tensor_sb_row_start_tile_id;
                    for (uint32_t w = 0; w < out_subblock_w; w++) {
                        noc_async_write_tile(out_tensor_tile_id, s, l1_read_addr);
                        l1_read_addr += l1_read_addr_increment;

                        out_tensor_tile_id += out_tensor_stride_w;
                    }
                    out_tensor_sb_row_start_tile_id += out_tensor_stride_h;
                }

                noc_async_write_barrier();  // This will wait until the write is done. As
                                            // an alternative, noc_async_write_flushed()
                                            // can be faster because it waits until the
                                            // write request is sent. In that case, you
                                            // have to use noc_async_write_barrier() at
                                            // least once at the end of data movement kernel
                                            // to make sure all writes are done.
                // TODO: 
                // DPRINT the entire tile from the CB.... 
                // we need to make a char buffer and just start pushing...
                // uint32_t* CB_values = (uint32_t*)l1_read_addr;
                // for (size_t idx = 0; idx < single_tile_size_bytes / 4; idx+=32){
                //     for (size_t inner = 0; inner < 32; inner++){
                //         float top_bits = (float)(CB_values[idx + inner] >> 16);
                //         float bottom_bits = (float)(CB_values[idx + inner] & 0xFFFF);
                //         //DPRINT_DATA1(DPRINT << top_bits << ' ' << bottom_bits << ' ');
                //     }
                //     //DPRINT_DATA1(DPRINT << ENDL());
                // }

                cb_pop_front(cb_id_out0, out_subblock_tile_count);
                DPRINT_DATA1(DPRINT << "Writer popped a subblock!" << ENDL());
                out_tensor_sbw_start_tile_id += out_tensor_next_subblock_stride_w;
            }
            out_tensor_sbh_start_tile_id += out_tensor_next_subblock_stride_h;
        }
        out_tensor_start_tile_id += RtNt;
    }
    DPRINT_DATA1(DPRINT << "Writer core done" << ENDL());

}
