// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

#include "debug/dprint.h"

void kernel_main() {
    // same arg indices as in reader_binary_diff_lenghts for compat
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(1);
    uint32_t src2_addr = get_arg_val<uint32_t>(2);
    uint32_t M = get_arg_val<uint32_t>(3);
    uint32_t N = get_arg_val<uint32_t>(4);
    uint32_t nnz = get_arg_val<uint32_t>(5);

    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool src1_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr bool src2_is_dram = get_compile_time_arg_val(2) == 1;

    DPRINT << "Reader heating up. M: " << M << ", N: " << N << ", nnz: " << nnz << ENDL();

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t cb_id_in2 = 2;

    constexpr uint32_t onepage = 1;
    constexpr uint32_t onetile = 1;

    const uint32_t src0_tile_bytes = get_tile_size(cb_id_in0);
    const DataFormat src0_data_format = get_dataformat(cb_id_in0);
    const uint32_t src1_tile_bytes = get_tile_size(cb_id_in1);
    const DataFormat src1_data_format = get_dataformat(cb_id_in1);
    // TODO: this gets passed to the InterleavedAddrGen constructor. Is it the size of a page? Is it the right size?
    const uint32_t src2_tile_bytes = get_tile_size(cb_id_in2);
    const DataFormat src2_data_format = get_dataformat(cb_id_in2);

    const InterleavedAddrGenFast<src0_is_dram> s0 = {
        .bank_base_address = src0_addr, .page_size = src0_tile_bytes, .data_format = src0_data_format};

    const InterleavedAddrGenFast<src1_is_dram> s1 = {
        .bank_base_address = src1_addr, .page_size = src1_tile_bytes, .data_format = src1_data_format};

    const InterleavedAddrGen<src2_is_dram> s2 = {.bank_base_address = src2_addr, .page_size = src2_tile_bytes};

    constexpr int num_elts_per_tile = 32 * 32;
    int num_tiles = (nnz + num_elts_per_tile - 1) / num_elts_per_tile;

    // I think we read consecutive tiles. So the indexing logic is very simple.
    for (int i = 0; i < num_tiles; i++) {
        {  // Read A_vals tile
            cb_reserve_back(cb_id_in0, onetile);
            uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
            noc_async_read_tile(i, s0, l1_write_addr_in0);
            noc_async_read_barrier();
            cb_push_back(cb_id_in0, onetile);
        }

        {  // Read X_vals tile
            cb_reserve_back(cb_id_in1, onepage);
            uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);
            noc_async_read_tile(i, s1, l1_write_addr_in1);
            noc_async_read_barrier();
            cb_push_back(cb_id_in1, onepage);
        }

        // {  // Read Rows page
        //     cb_reserve_back(cb_id_in2, onepage);
        //     uint32_t l1_write_addr_in2 = get_write_ptr(cb_id_in2);
        //     s2.noc_async_read_page(i, l1_write_addr_in2);
        //     // noc_async_read_page(i, s2, l1_write_addr_in2);
        //     noc_async_read_barrier();
        //     cb_push_back(cb_id_in2, onepage);
        // }
    }

    DPRINT << "Reader finished: " << ENDL();
}
