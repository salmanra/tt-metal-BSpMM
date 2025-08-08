// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include "dataflow_api.h"
#include "debug/dprint.h"  // required in all kernels using DPRINT



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

    // batch args
    uint32_t MtNt = get_arg_val<uint32_t>(11);  // TODO: figure out this constant. And it should be constant now!
                                                //          But only because we handle the multiple blocks serially...
    uint32_t batch = get_arg_val<uint32_t>(12); // TODO: rename to reflect num blocks M per core.

    uint32_t nonzero = get_arg_val<uint32_t>(13);

    constexpr bool out_is_dram = get_compile_time_arg_val(0) == 1;

    constexpr uint32_t cb_id_out0 = 16;

    // single-tile
    const uint32_t single_tile_size_bytes = get_tile_size(cb_id_out0);
    uint32_t l1_read_addr_increment = single_tile_size_bytes;
    const DataFormat data_format = get_dataformat(cb_id_out0);


    const InterleavedAddrGenFast<out_is_dram> s = {
        .bank_base_address = out_tensor_addr, .page_size = single_tile_size_bytes, .data_format = data_format};

    bool one_time_profile = true;
    for (uint32_t b = 0; b < batch; b++) {
        // TODO: relate this variable to indexing DSes, 
        uint32_t out_tensor_sbh_start_tile_id = out_tensor_start_tile_id;
        for (uint32_t sbh = 0; sbh < out_num_subblocks_h; sbh++) {
            uint32_t out_tensor_sbw_start_tile_id = out_tensor_sbh_start_tile_id;
            for (uint32_t sbw = 0; sbw < out_num_subblocks_w; sbw++) {
                uint32_t out_tensor_sb_row_start_tile_id = out_tensor_sbw_start_tile_id;

                // nonzero
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

                noc_async_write_barrier();  // This will wait until the write is done. As
                                            // an alternative, noc_async_write_flushed()
                                            // can be faster because it waits until the
                                            // write request is sent. In that case, you
                                            // have to use noc_async_write_barrier() at
                                            // least once at the end of data movement kernel
                                            // to make sure all writes are done.
                cb_pop_front(cb_id_out0, out_subblock_tile_count);
                out_tensor_sbw_start_tile_id += out_tensor_next_subblock_stride_w;
            }
            out_tensor_sbh_start_tile_id += out_tensor_next_subblock_stride_h;
        }
        // TODO: add runtime args for this, might need to be a NoC arg situation
        out_tensor_start_tile_id += MtNt;
    }
    DPRINT_DATA1(DPRINT << "Writer core done" << ENDL());

}
