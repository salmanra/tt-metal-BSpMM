

/*
For
*/

// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/eltwise_binary.h"

#include "debug/dprint.h"

using std::uint32_t;

namespace NAMESPACE {
void MAIN {
    constexpr int onetile = 1;

    int dst_tile_index = 0;
    int in0_block_tile_index = 0;

    uint32_t M = get_arg_val<uint32_t>(0);
    uint32_t N = get_arg_val<uint32_t>(1);
    uint32_t nnz = get_arg_val<uint32_t>(2);

    DPRINT_MATH(DPRINT << "Compute kernel heating up. M: " << M << ", N: " << N << ", nnz: " << nnz << ENDL());

    uint32_t intermediate_cb_index = tt::CBIndex::c_3;

    constexpr int num_elts_per_tile = 32 * 32;
    int num_tiles = (nnz + num_elts_per_tile - 1) / num_elts_per_tile;
    mul_tiles_init(tt::CBIndex::c_0, tt::CBIndex::c_1);

    for (int i = 0; i < num_tiles; i++) {
        DPRINT_MATH(DPRINT << "Compute kernel has a tile to pack" << ENDL());

        cb_wait_front(tt::CBIndex::c_0, onetile);
        cb_wait_front(tt::CBIndex::c_1, onetile);

        tile_regs_acquire();

        // elt-wise mul
        mul_tiles(tt::CBIndex::c_0, tt::CBIndex::c_1, 0, 0, 0);

        tile_regs_commit();

        cb_pop_front(tt::CBIndex::c_0, onetile);
        cb_pop_front(tt::CBIndex::c_1, onetile);

        tile_regs_wait();

        cb_reserve_back(3, onetile);
        pack_tile(0, 3);
        cb_push_back(3, onetile);

        tile_regs_release();
    }
    DPRINT_MATH(DPRINT << "Compute kernel finished" << ENDL());
}
}  // namespace NAMESPACE
