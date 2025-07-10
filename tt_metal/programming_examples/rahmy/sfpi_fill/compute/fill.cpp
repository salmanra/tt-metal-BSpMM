// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"


// only want this to run on math core
#ifdef TRISC_MATH
#define ITERATIONS (8)
inline void fill_zeros()
{
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat zeros = 0.0f;
        sfpi::dst_reg[0] = zeros;
        sfpi::dst_reg++;
    }
}

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);

    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_16);
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(tt::CBIndex::c_16, per_core_block_dim);
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            acquire_dst();

            // custom llk here, when it returns the DST reg should have the tile to be packed
            MATH();
            pack_tile(0, tt::CBIndex::c_16);

            release_dst();
        }
        cb_push_back(tt::CBIndex::c_16, per_core_block_dim);
    }
}
}  // namespace NAMESPACE
