// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "hostdevcommon/kernel_structs.h"


#include "compute_kernel_api/tile_move_copy.h"
#include "circular_buffer.h"
#include "compute_kernel_api/matmul.h"
#include "debug/dprint.h"  // required in all kernels using DPRINT

#include "compute_kernel_api/eltwise_unary/fill.h"


namespace NAMESPACE {
void MAIN {
    uint32_t in0_block_w = get_compile_time_arg_val(0);              // inner block size in tiles
    uint32_t in0_num_subblocks = get_compile_time_arg_val(1);        // outer row block size (in inner row blocks)
    uint32_t in0_block_num_tiles = get_compile_time_arg_val(2);      // out_subblock_h*in0_block_w*in0_num_subblocks;
    uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(3);   // out_subblock_h*in0_block_w
    uint32_t in1_num_subblocks = get_compile_time_arg_val(4);        // outer column block size (in inner column blocks)
    uint32_t in1_block_num_tiles = get_compile_time_arg_val(5);      // out_subblock_w*in0_block_w* in1_num_subblocks;
    uint32_t in1_per_core_w = get_compile_time_arg_val(6);           // out_subblock_w*in1_num_subblocks
    uint32_t out_subblock_h = get_compile_time_arg_val(7);           // inner row block size in tiles
    uint32_t out_subblock_w = get_compile_time_arg_val(8);           // inner column block size in tiles
    uint32_t out_subblock_num_tiles = get_compile_time_arg_val(9);  // out_subblock_h * out_subblock_w;
    const uint32_t num_output_blocks = get_compile_time_arg_val(10);                   // num output blocks
    uint32_t bytes_for_sync = get_compile_time_arg_val(11);           // size of synchronization CB 


    // uint32_t in0_block_w = get_arg_val<uint32_t>(0);              // inner block size in tiles
    // uint32_t in0_num_subblocks = get_arg_val<uint32_t>(1);        // outer row block size (in inner row blocks)
    // uint32_t in0_block_num_tiles = get_arg_val<uint32_t>(2);      // out_subblock_h*in0_block_w*in0_num_subblocks;
    // uint32_t in0_subblock_num_tiles = get_arg_val<uint32_t>(3);   // out_subblock_h*in0_block_w
    // uint32_t in1_num_subblocks = get_arg_val<uint32_t>(4);        // outer column block size (in inner column blocks)
    // uint32_t in1_block_num_tiles = get_arg_val<uint32_t>(5);      // out_subblock_w*in0_block_w* in1_num_subblocks;
    // uint32_t in1_per_core_w = get_arg_val<uint32_t>(6);           // out_subblock_w*in1_num_subblocks
    // // uint32_t num_blocks = get_arg_val<uint32_t>(7);               // outer inner dim (in inner dim blocks)
    // uint32_t out_subblock_h = get_arg_val<uint32_t>(7);           // inner row block size in tiles
    // uint32_t out_subblock_w = get_arg_val<uint32_t>(8);           // inner column block size in tiles
    // uint32_t out_subblock_num_tiles = get_arg_val<uint32_t>(9);  // out_subblock_h * out_subblock_w;
    // uint32_t batch = get_arg_val<uint32_t>(10);                   // batch dim


    const uint32_t cb_id_in0 = tt::CBIndex::c_0;
    const uint32_t cb_id_in1 = tt::CBIndex::c_1;
    const uint32_t cb_id_NoC_Args = tt::CBIndex::c_2;
    const uint32_t cb_id_indptr = tt::CBIndex::c_3;
    const uint32_t cb_id_sync = tt::CBIndex::c_4;

    uint32_t l1_write_addr_in0;
    uint32_t l1_write_addr_in1;
    uint32_t l1_write_addr_NoC_Args;
    uint32_t l1_write_addr_indptr;
    uint32_t l1_write_addr_sync;


    // cb_wait_front(cb_id_NoC_Args, 1);
    // l1_write_addr_NoC_Args = get_read_ptr(cb_id_NoC_Args);
    
    // LocalCBInterface interface = get_local_cb_interface(cb_id_NoC_Args);
    
    // DPRINT_MATH(DPRINT << "Math core CB Interface read ptr " << interface.fifo_rd_ptr << ENDL());
    // DPRINT_MATH(DPRINT << "tt l1 ptr:" << tt_l1_ptr << ENDL());


    mm_init();
    // variable number of args
    uint32_t row_sizes[num_output_blocks];
    uint32_t max_row_size = 0;
    for (uint32_t i = 0; i < num_output_blocks; i++){
        row_sizes[i] = get_arg_val<uint32_t>(i);
        max_row_size = std::max(max_row_size, row_sizes[i]);
        //DPRINT_MATH(DPRINT << "Row" << i << " size:" << row_sizes[i] << ENDL());
    }
    // uint32_t max_row_size = get_arg_val<uint32_t>(num_output_blocks);
    uint32_t num_blocks = max_row_size;

    //DPRINT_MATH(DPRINT << "Math core waiting on a max of " << num_blocks << " blocks." << ENDL());
    // output CB is MpcxNpc, which means we aren't batching. 
    // The outermost loop now has a different interpretation. 
    bool enable_reload = false;
    bool spill = num_blocks > 1;
    for (uint32_t reduction_block = 0; reduction_block < max_row_size; reduction_block++){
        uint32_t out_num_tiles_to_wait = out_subblock_num_tiles;
        for (uint32_t output_block = 0; output_block < num_output_blocks; output_block++){
            if (reduction_block > row_sizes[output_block]){
                // this row is dropped
                // 1. index into next output block 
                // 2. continue
                //DPRINT_MATH(DPRINT << "Math core SKIPPING" << ENDL());

                // QUESTION: What breaks if there's no special indexing into output blocks?
                //             Test 6. Test 6 breaks. 
                continue;
            }
            bool last_out = reduction_block == (row_sizes[output_block] - 1);

            // maybe this region does a single block of matmul ...
            cb_wait_front(tt::CBIndex::c_0, in0_block_num_tiles);
            //DPRINT_MATH(DPRINT << "Math core done waiting 0" << ENDL());
            cb_wait_front(tt::CBIndex::c_1, in1_block_num_tiles);
            //DPRINT_MATH(DPRINT << "Math core done waiting 0 and 1" << ENDL());


            int in0_index_subblock_offset = 0;
            for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
                int in1_index_subblock_offset = 0;
                for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
                    acquire_dst();

                    if (enable_reload) {
                        copy_tile_to_dst_init_short(tt::CBIndex::c_24);
                        //DPRINT_MATH(DPRINT << "Reload begins: " << output_block << " " << reduction_block << ENDL());
                        // The last iter (1, 1) fails here, it deadlocks here. 
                        cb_wait_front(tt::CBIndex::c_24, out_subblock_num_tiles);
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            copy_tile(tt::CBIndex::c_24, i, i);
                        }
                        cb_pop_front(tt::CBIndex::c_24, out_subblock_num_tiles);
                        mm_init_short();
                        //DPRINT_MATH(DPRINT << "Reload done" << ENDL());
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
                                matmul_tiles(
                                    tt::CBIndex::c_0,
                                    tt::CBIndex::c_1,
                                    in0_index,
                                    in1_index,
                                    dst_index, // DST register
                                    false /* transpose */);
                                in1_index_inner_dim_offset += in1_per_core_w;
                            }
                            dst_index++;
                        }
                        in0_index_h_offset += in0_block_w;
                    }

                    if (last_out) {
                        DPRINT_MATH(DPRINT << "Last out, ob={" << output_block << "} rb={" << reduction_block << '}' << ENDL());
                        // Pack out to output buffer
                        cb_reserve_back(tt::CBIndex::c_16, out_subblock_num_tiles);
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            pack_tile(i, tt::CBIndex::c_16);
                        }
                        cb_push_back(tt::CBIndex::c_16, out_subblock_num_tiles);
                        //DPRINT_MATH(DPRINT << "Last out and it's done!" << ENDL());
                    } else {
                        //DPRINT_MATH(DPRINT << "NOT last out, ob={" << output_block << "} rb={" << reduction_block << '}' << ENDL());
                        // Wait for tiles in output buffer to be written out since interm and output share memory
                        if (reduction_block == 0) {
                            cb_reserve_back(tt::CBIndex::c_16, out_num_tiles_to_wait);
                            out_num_tiles_to_wait += out_subblock_num_tiles;
                        }
                        // Move partial result to interm buffer
                        cb_reserve_back(tt::CBIndex::c_24, out_subblock_num_tiles);
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            pack_tile(i, tt::CBIndex::c_24);
                        }
                        cb_push_back(tt::CBIndex::c_24, out_subblock_num_tiles);
                    }

                    release_dst();
                    in1_index_subblock_offset += out_subblock_w;
                }
                in0_index_subblock_offset += in0_subblock_num_tiles;
            }

            if (spill) {
                enable_reload = true;
            }
            if (last_out){
                // now, last_out refers to the end of a row, not to the end of the program
                // so when we move on, we need to disable reloading
                enable_reload = false;
            }

            cb_pop_front(tt::CBIndex::c_0, in0_block_num_tiles);
            cb_pop_front(tt::CBIndex::c_1, in1_block_num_tiles);
            //DPRINT_MATH(DPRINT << "Math core popped" << ENDL());
        }
        // ... and this region indexes into the next output block.
        //DPRINT_MATH(DPRINT << "NEXT BLOCKS PLEASE" << ENDL());
    }
    DPRINT_MATH(DPRINT << "Math core received " << num_blocks << " blocks." << ENDL());
}
}  // namespace NAMESPACE
