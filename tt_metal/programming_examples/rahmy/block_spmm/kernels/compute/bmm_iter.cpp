/*
CK is neutral to order, but it's probably worth naming loop vars
    to reflect the order that RK and WK respec/demand
*/


#include <cstdint>
#include "hostdevcommon/kernel_structs.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "circular_buffer.h"


namespace NAMESPACE {
void MAIN {
    ///////////////////////////////////////////////////////////////////////
    /// COMPILETIME ARGS //////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    constexpr uint32_t in0_block_w = get_compile_time_arg_val(0);              // inner block size in tiles
    constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(1);        // outer row block size (in inner row blocks)
    constexpr uint32_t in0_block_num_tiles = get_compile_time_arg_val(2);      // out_subblock_h*in0_block_w*in0_num_subblocks;
    constexpr uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(3);   // out_subblock_h*in0_block_w
    constexpr uint32_t in1_num_subblocks = get_compile_time_arg_val(4);        // outer column block size (in inner column blocks)
    constexpr uint32_t in1_block_num_tiles = get_compile_time_arg_val(5);      // out_subblock_w*in0_block_w* in1_num_subblocks;
    constexpr uint32_t in1_per_core_w = get_compile_time_arg_val(6);           // out_subblock_w*in1_num_subblocks
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(7);           // inner row block size in tiles
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(8);           // inner column block size in tiles
    constexpr uint32_t out_subblock_num_tiles = get_compile_time_arg_val(9);  // out_subblock_h * out_subblock_w;
    constexpr uint32_t num_iters_x = get_compile_time_arg_val(10);

    ///////////////////////////////////////////////////////////////////////
    /// END COMPILETIME ARGS //////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////
    /// RUNTIME ARGS //////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    const uint32_t num_iters_y = get_arg_val<uint32_t>(0);
    uint32_t row_sizes[num_iters_y];
    for (uint32_t i = 0; i < num_iters_y; i++){
        row_sizes[i] = get_arg_val<uint32_t>(i+1);
    }

    ///////////////////////////////////////////////////////////////////////
    /// END RUNTIME ARGS //////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////
    /// PROGRAM BODY //////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    mm_init();

    for (uint32_t iter_y = 0; iter_y < num_iters_y; iter_y++){
        uint32_t num_blocks = row_sizes[iter_y];
        for (uint32_t iter_x = 0; iter_x < num_iters_x; iter_x++){
            bool enable_reload = false;
            bool spill = num_blocks > 1;
            // DPRINT_MATH(DPRINT << "Num blocks: " << num_blocks << ENDL());

            uint32_t out_num_tiles_to_wait = out_subblock_num_tiles;
            for (uint32_t input_block = 0; input_block < num_blocks; input_block++){
                bool last_out = input_block == (num_blocks - 1);

                cb_wait_front(tt::CBIndex::c_0, in0_block_num_tiles);
                cb_wait_front(tt::CBIndex::c_1, in1_block_num_tiles);

                // DPRINT_MATH(DPRINT << "in0 block num tiles:  " << in0_block_num_tiles << ENDL());
                // DPRINT_MATH(DPRINT << "in1 block num tiles:  " <<  in1_block_num_tiles << ENDL());


                int in0_index_subblock_offset = 0;
                for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
                    int in1_index_subblock_offset = 0;
                    for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
                        // acquire_dst();
                        ckernel::tile_regs_acquire();
                        // DPRINT_MATH(DPRINT << "acquired" << ENDL());

                        if (enable_reload) {
                            copy_tile_to_dst_init_short(tt::CBIndex::c_24);
                            cb_wait_front(tt::CBIndex::c_24, out_subblock_num_tiles);
                            for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                                copy_tile(tt::CBIndex::c_24, i, i);
                            }
                            cb_pop_front(tt::CBIndex::c_24, out_subblock_num_tiles);
                            mm_init_short();
                        }

                        // Compute output sub-block from in0_subblock x in1_subblock
                        int dst_index = 0;
                        int in0_index_h_offset = 0;
                        for (uint32_t h = 0; h < out_subblock_h; h++) {
                            for (uint32_t w = 0; w < out_subblock_w; w++) {
                                int in1_index_inner_dim_offset = 0;
                                for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                                    int in0_index = in0_index_subblock_offset + in0_index_h_offset + inner_dim;
                                    int in1_index = in1_index_subblock_offset + in1_index_inner_dim_offset + w;
                                    // DPRINT_MATH(DPRINT << "pre matmul tiles" << ENDL());

                                    matmul_tiles(
                                        tt::CBIndex::c_0,
                                        tt::CBIndex::c_1,
                                        in0_index,
                                        in1_index,
                                        dst_index, // DST register
                                        false /* transpose */);
                                    // DPRINT_MATH(DPRINT << "post matmul tiles" << ENDL());

                                    in1_index_inner_dim_offset += in1_per_core_w;
                                }
                                dst_index++;
                            }
                            in0_index_h_offset += in0_block_w;
                        }
                        // DPRINT_MATH(DPRINT << "mulled" << ENDL());
                        ckernel::tile_regs_commit();

                        ckernel::tile_regs_wait();
                        if (last_out) {
                            // Pack out to output buffer
                            cb_reserve_back(tt::CBIndex::c_16, out_subblock_num_tiles);
                            for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                                pack_tile(i, tt::CBIndex::c_16);
                            }
                            cb_push_back(tt::CBIndex::c_16, out_subblock_num_tiles);
                            DPRINT_MATH(DPRINT << "pushed to 16 " << ENDL());

                        } else {
                            // Wait for tiles in output buffer to be written out since interm and output share memory
                            if (input_block == 0) {
                                // DPRINT_MATH(DPRINT << "reserved 16, block == 0 " << ENDL());
                                cb_reserve_back(tt::CBIndex::c_16, out_num_tiles_to_wait);
                                out_num_tiles_to_wait += out_subblock_num_tiles;
                            }
                            // Move partial result to interm buffer
                            cb_reserve_back(tt::CBIndex::c_24, out_subblock_num_tiles);
                            // DPRINT_MATH(DPRINT << "reserved 24 " << ENDL());
                            for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                                pack_tile(i, tt::CBIndex::c_24);
                            }
                            cb_push_back(tt::CBIndex::c_24, out_subblock_num_tiles);
                            // DPRINT_MATH(DPRINT << "pushed to 24 " << ENDL());

                        }
                        ckernel::tile_regs_release();
                        // release_dst();
                        // DPRINT_MATH(DPRINT << "released" << ENDL());

                        in1_index_subblock_offset += out_subblock_w;
                    }
                    in0_index_subblock_offset += in0_subblock_num_tiles;
                }

                if (spill) {
                    enable_reload = true;
                }

                cb_pop_front(tt::CBIndex::c_0, in0_block_num_tiles);
                cb_pop_front(tt::CBIndex::c_1, in1_block_num_tiles);

                // DPRINT_MATH(DPRINT << "out " << ENDL());

            }
        }
    }
};
}