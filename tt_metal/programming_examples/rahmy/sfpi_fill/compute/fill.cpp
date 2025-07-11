// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_unary/fill.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "ckernel.h"



namespace NAMESPACE {
void MAIN {

    // return; 


    fill_tile_init();
    cb_reserve_back(tt::CBIndex::c_24, 1);

    tile_regs_acquire();
    fill_tile(0, 0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(0, tt::CBIndex::c_24);
    tile_regs_release();

    DPRINT_MATH(DPRINT << "compute kernel all done filling" << ENDL()); 

    cb_push_back(tt::CBIndex::c_24, 1);

    // for (uint32_t tile_index = 0; tile_index < 1; ++tile_index) {
    //     acquire_dst();

    //     // custom llk here, when it returns the DST reg should have the tile to be packed
    //     // TODO: how many tiles does a single call with 8 iterations fill? 
    //     // eh... cause it could be 8x32 or 8x16. the latter make more sense but is less satisfying. Let's see. 
    //     ckernel::sfpu::calculate_fill<false, 64>(0);
    //     pack_tile(0, tt::CBIndex::c_24);

    //     release_dst();
    // }
}
}  // namespace NAMESPACE
