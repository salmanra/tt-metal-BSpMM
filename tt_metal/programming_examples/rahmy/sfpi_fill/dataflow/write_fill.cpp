// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <cstring>
#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {


    // return;

    uint32_t dst_addr  = get_arg_val<uint32_t>(0);
    uint32_t num_tiles_to_write = get_compile_time_arg_val(0);

    constexpr uint32_t cb_id_out0 = tt::CBIndex::c_24;

    // single-tile ublocks
    uint32_t ublock_size_bytes = get_tile_size(cb_id_out0);
    uint32_t ublock_size_tiles = 1;

    const DataFormat data_format = get_dataformat(cb_id_out0);

    const InterleavedAddrGenFast<true> s = {
        .bank_base_address = dst_addr, 
        .page_size = ublock_size_bytes, 
        .data_format = data_format};

    // cb_wait_front a single tile
    cb_wait_front(cb_id_out0, ublock_size_tiles);
    uint32_t l1_read_addr = get_read_ptr(cb_id_out0);

    uint32_t* data = (uint32_t*)l1_read_addr;
    memset((void *)data, 0, 2*32*32); // this is evil if it works...
    // uh it worked. I don't know how to feel about this

    float top_bits = (float)(data[0] >> 16);
    float bottom_bits = (float)(data[0] & 0xFFFF);
    DPRINT_DATA0(DPRINT << "Some CB elements: " << top_bits << ' ' << bottom_bits<< ENDL());
    
    for (uint32_t i = 0; i < num_tiles_to_write; i += ublock_size_tiles) {
        // don't have to wait more than once on the CB, 
        // after the first wait we can just NoC as many times as we need.
        noc_async_write_tile(i, s, l1_read_addr);
        noc_async_write_barrier();
        // DPRINT_DATA0(DPRINT << "wrote a tile" << ENDL());

    }
}
