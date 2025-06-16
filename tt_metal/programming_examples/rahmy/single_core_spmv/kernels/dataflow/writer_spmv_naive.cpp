// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "debug/dprint.h"

void kernel_main() {
    // same arg indices as in reader_spmv_naive for reuse
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t M = get_arg_val<uint32_t>(1);
    uint32_t N = get_arg_val<uint32_t>(2);
    uint32_t nnz = get_arg_val<uint32_t>(3);

    DPRINT << "Writer heating up. M: " << M << ", N: " << N << ", nnz: " << nnz << ENDL();

    constexpr bool dst_is_dram = get_compile_time_arg_val(0) == 1;

    uint32_t intermediate_cb_index = tt::CBIndex::c_3;

    constexpr uint32_t onetile = 1;
    constexpr uint32_t cb_id_out0 = 16;
    constexpr uint32_t cb_id_inter = 3;
    const uint32_t tile_bytes = get_tile_size(cb_id_out0);
    const DataFormat data_format = get_dataformat(cb_id_out0);

    const InterleavedAddrGenFast<dst_is_dram> dst = {
        .bank_base_address = dst_addr, .page_size = tile_bytes, .data_format = data_format};

    constexpr int num_elts_per_tile = 32 * 32;
    int num_tiles = (nnz + num_elts_per_tile - 1) / num_elts_per_tile;

    for (int i = 0; i < num_tiles; i++) {
        // read tile from intermediate buffer
        DPRINT << "Writer waiting on compute" << ENDL();
        cb_wait_front(cb_id_inter, 1);
        DPRINT << "Writer received from compute" << ENDL();

        uint32_t l1_read_addr = get_read_ptr(intermediate_cb_index);

        // do some minor compute on the tile you read

        // write results of compute to output buffer
        DPRINT << "Writer writing" << ENDL();

        // TODO: get rid of this
        noc_async_write_tile(i, dst, l1_read_addr);
        noc_async_write_barrier();  // This will wait until the write is done. As an alternative,
                                    // noc_async_write_flushed() can be faster because it waits
                                    // until the write request is sent. In that case, you have to
                                    // use noc_async_write_barrier() at least once at the end of
                                    // data movement kernel to make sure all writes are done.
        cb_pop_front(cb_id_inter, 1);
    }

    // all tiles have been read and reduced, now we can write result to DRAM
    // for (uint32_t i = 0; i < num_tiles; i++){

    // }

    DPRINT << "Writer cooked. " << ENDL();
}
