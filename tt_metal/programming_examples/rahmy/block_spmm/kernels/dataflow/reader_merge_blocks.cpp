#include <stdint.h>
#include <cstdint>
#include "dataflow_api.h"
#include "debug/dprint.h"  // required in all kernels using DPRINT
#include "hostdevcommon/kernel_structs.h"

void kernel_main(){
    // in0 tensor args
    // uint32_t in0_tensor_addr = get_arg_val<uint32_t>(0);
    // // uint32_t in0_tensor_start_tile_id = get_arg_val<uint32_t>(1); // TODO:
    // uint32_t in0_tensor_stride_w = get_arg_val<uint32_t>(1);
    // uint32_t in0_tensor_stride_h = get_arg_val<uint32_t>(2);
    // // in0_next_block_stride is replaced with in0_block_num_tiles

    // // in0 block args
    // uint32_t in0_block_w = get_arg_val<uint32_t>(3);
    // uint32_t in0_block_h = get_arg_val<uint32_t>(4);
    // uint32_t in0_block_num_tiles = get_arg_val<uint32_t>(5);

    // // in1 tensor args
    // uint32_t in1_tensor_addr = get_arg_val<uint32_t>(6);
    // // uint32_t in1_tensor_start_tile_id = get_arg_val<uint32_t>(8); // TODO:
    // uint32_t in1_tensor_stride_w = get_arg_val<uint32_t>(7);
    // uint32_t in1_tensor_stride_h = get_arg_val<uint32_t>(8);

    // // in1 block args
    // uint32_t in1_block_w = get_arg_val<uint32_t>(9);
    // uint32_t in1_block_h = get_arg_val<uint32_t>(10);
    // uint32_t in1_block_num_tiles = get_arg_val<uint32_t>(11);

    // in0/in1 common args
    // num_blocks is determined by the runtime args, range of nonzero blocks in that row
    // uint32_t block_row_start = get_arg_val<uint32_t>(14); // TODO:
    // uint32_t block_row_end = get_arg_val<uint32_t>(15);// TODO:
    // uint32_t block_row_index = get_arg_val<uint32_t>(16);  // TODO: // this is the row index in the sparse matrix

    // NoC Args
    // uint32_t NoC_Args_addr = get_arg_val<uint32_t>(12);
    // uint32_t indptr_addr = get_arg_val<uint32_t>(13);


    const uint32_t num_output_blocks = get_arg_val<uint32_t>(0);
    const uint32_t output_idx_x = get_arg_val<uint32_t>(1);
    // variable number of args
    uint32_t y_coords[num_output_blocks];
    for (uint32_t i = 0; i < num_output_blocks; i++){
        y_coords[i] = get_arg_val<uint32_t>(2 + i);
        // DPRINT_DATA0(DPRINT << "y_coord i: " << y_coords[i] << ENDL());

    }
    uint32_t max_row_size = get_arg_val<uint32_t>(2 + num_output_blocks);

    DPRINT_DATA0(DPRINT << "Runtime args obtained. num output blocks to compute: " << num_output_blocks << ENDL());


    constexpr bool in0_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool in1_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr bool NoC_Args_is_dram = get_compile_time_arg_val(2) == 1;

    constexpr uint32_t in0_tensor_addr = get_compile_time_arg_val(3);
    constexpr uint32_t in0_tensor_stride_w = get_compile_time_arg_val(4);
    constexpr uint32_t in0_tensor_stride_h = get_compile_time_arg_val(5);

    constexpr uint32_t in0_block_w = get_compile_time_arg_val(6);
    constexpr uint32_t in0_block_h = get_compile_time_arg_val(7);
    constexpr uint32_t in0_block_num_tiles = get_compile_time_arg_val(8);

    constexpr uint32_t in1_tensor_addr = get_compile_time_arg_val(9);
    constexpr uint32_t in1_tensor_stride_w = get_compile_time_arg_val(10);
    constexpr uint32_t in1_tensor_stride_h = get_compile_time_arg_val(11);

    constexpr uint32_t in1_block_w = get_compile_time_arg_val(12);
    constexpr uint32_t in1_block_h = get_compile_time_arg_val(13);
    constexpr uint32_t in1_block_num_tiles = get_compile_time_arg_val(14);

    constexpr uint32_t NoC_Args_addr = get_compile_time_arg_val(15);
    constexpr uint32_t indptr_addr = get_compile_time_arg_val(16);
    constexpr uint32_t bytes_for_sync = get_compile_time_arg_val(17);



    const uint32_t cb_id_in0 = tt::CBIndex::c_0;
    const uint32_t cb_id_in1 = tt::CBIndex::c_1;
    const uint32_t cb_id_NoC_Args = tt::CBIndex::c_2;
    const uint32_t cb_id_indptr = tt::CBIndex::c_3;
    const uint32_t cb_id_sync = tt::CBIndex::c_4;


    // input data format will probably by bfloat16
    const uint32_t in0_single_tile_size_bytes = get_tile_size(cb_id_in0);
    const DataFormat in0_data_format = get_dataformat(cb_id_in0);
    const uint32_t in1_single_tile_size_bytes = get_tile_size(cb_id_in1);
    const DataFormat in1_data_format = get_dataformat(cb_id_in1);
    // NoC Args dataformat will probably be uint32_t
    const uint32_t NoC_Args_single_tile_size_bytes = get_tile_size(cb_id_NoC_Args);
    const DataFormat NoC_Args_data_format = get_dataformat(cb_id_NoC_Args);
    const uint32_t indptr_single_tile_size_bytes = get_tile_size(cb_id_indptr);
    const DataFormat indptr_data_format = get_dataformat(cb_id_indptr);


    // The keys to the kingdom
    uint32_t l1_write_addr_in0;
    uint32_t l1_write_addr_in1;
    uint32_t l1_write_addr_NoC_Args;
    uint32_t l1_write_addr_indptr;
    uint32_t l1_write_addr_sync;

    const InterleavedAddrGenFast<in0_is_dram> s0 = {
        .bank_base_address = in0_tensor_addr, .page_size = in0_single_tile_size_bytes, .data_format = in0_data_format};
    const InterleavedAddrGenFast<in1_is_dram> s1 = {
        .bank_base_address = in1_tensor_addr, .page_size = in1_single_tile_size_bytes, .data_format = in1_data_format};
    const InterleavedAddrGenFast<NoC_Args_is_dram> s2 = {
        .bank_base_address = NoC_Args_addr,
        .page_size = NoC_Args_single_tile_size_bytes,
        .data_format = NoC_Args_data_format};
    const InterleavedAddrGenFast<true> s3 = {
        .bank_base_address = indptr_addr,
        .page_size = indptr_single_tile_size_bytes,
        .data_format = indptr_data_format};


    // POTENTIAL: memory save
    // we can NoC bytes from DRAM for indexing as needed (once for each output block, not too bad?)
    // saves us memory in SRAM.
    // 
    uint32_t *column_indices, *indptr;
    // NoC Args are read first, so that we can use them to read the in0 and in1 blocks.
    // The reader kernel is both the producer and consumer of the NoC Args!
    cb_reserve_back(cb_id_NoC_Args, 1); // assume col indices fit into one tile for now
    l1_write_addr_NoC_Args = get_write_ptr(cb_id_NoC_Args);
    noc_async_read_tile(0, s2, l1_write_addr_NoC_Args);
    noc_async_read_barrier();
    cb_push_back(cb_id_NoC_Args, 1);

    column_indices = (uint32_t*)l1_write_addr_NoC_Args;

    cb_reserve_back(cb_id_indptr, 1);
    l1_write_addr_indptr = get_write_ptr(cb_id_indptr);
    noc_async_read_tile(0, s3, l1_write_addr_indptr);
    noc_async_read_barrier();
    cb_push_back(cb_id_indptr, 1);

    indptr = (uint32_t*)l1_write_addr_indptr; // this might not be right... what's the datatype of indptr?

    // TODO: before reading any matrix data from DRAM, determine which blocks the kernel will 
    //       read and which are duplicates
    //       ... wait. col indices is in SRAM. the compute kernel can access that directly!!!
    // cb_reserve_back(cb_id_sync, 1);
    // l1_write_addr_sync = get_write_ptr(cb_id_sync);
    // DPRINT_DATA0(DPRINT << "Reader CB Sync Address:  " << l1_write_addr_sync << ENDL());

    // uint32_t index_into_dense_matrix = 0;
    // uint32_t max_num_blocks_per_row = 0;
    // for (uint32_t output_block = 0; output_block < num_output_blocks; output_block++){
    //     uint32_t output_idx_y = y_coords[output_block]; 
    //     uint32_t block_row_start = indptr[output_idx_y];
    //     uint32_t block_row_end = indptr[output_idx_y + 1];
    //     uint32_t num_blocks = block_row_end - block_row_start;
    //     max_num_blocks_per_row = max(max_num_blocks_per_row, num_blocks);

    //     uint32_t bsr_col_index = column_indices[output_block];
    //     // if not a duplicate, increment index
    //     // push index to cb sync
    // }

    // for (uint32_t reduction_index = 0; reduction_index < max_num_blocks_per_row; reduction_index++) {
    //     for (uint32_t output_block = 0; output_block < num_output_blocks; output_block++){
    //         uint32_t output_idx_y = y_coords[output_block]; 
    //         uint32_t block_row_start = indptr[output_idx_y];
    //         uint32_t block_row_end = indptr[output_idx_y + 1];
    //         uint32_t block_idx = block_row_start + reduction_index;

    //         // the big idea is we can fit num_output_blocks of A and num_output_blocks of B 
    //         // in the CBs at once (and double buffered on that). 
    //         // We always read num_output_blocks blocks of A in one iter until 
    //         //  rows start dropping away.
    //         // We can choose to read fewer blocks of B if some output blocks are in the same
    //         //  column.
    //         //
    //         // But the indexing logic is hard... let's just start copying tiles. (it would be a copy into a pack, 
    //         // and the CK would have to handle it. Let's just NoC). 
    //     }
    // }

    // This ordering (output_block indexing in the inner loop) lends itself to data sharing later
    for (uint32_t reduction_block = 0; reduction_block < max_row_size; reduction_block++){
        for (uint32_t output_block = 0; output_block < num_output_blocks; output_block++){
            uint32_t output_idx_y = y_coords[output_block]; 
            uint32_t block_row_start = indptr[output_idx_y];
            uint32_t block_row_end = indptr[output_idx_y + 1];

            if (block_row_start + reduction_block >= block_row_end){
                // this row is dropped
                continue;
            }

            uint32_t in0_tensor_start_tile_id = block_row_start * in0_block_num_tiles;
            uint32_t in1_tensor_start_tile_id = in1_block_w * output_idx_x; 

            cb_reserve_back(cb_id_in0, in0_block_num_tiles);
            cb_reserve_back(cb_id_in1, in1_block_num_tiles);

            l1_write_addr_in0 = get_write_ptr(cb_id_in0);
            l1_write_addr_in1 = get_write_ptr(cb_id_in1);

            // Read in0 block
            uint32_t num_blocks_in = reduction_block;
            uint32_t in0_tensor_row_start_tile_id = in0_tensor_start_tile_id + num_blocks_in * in0_block_num_tiles;
            for (uint32_t h = 0; h < in0_block_h; h++) {
                uint32_t in0_tensor_tile_id = in0_tensor_row_start_tile_id; 
                for (uint32_t w = 0; w < in0_block_w; w++) {
                    noc_async_read_tile(in0_tensor_tile_id, s0, l1_write_addr_in0);
                    l1_write_addr_in0 += in0_single_tile_size_bytes;
                    in0_tensor_tile_id += in0_tensor_stride_w;
                }
                in0_tensor_row_start_tile_id += in0_tensor_stride_h;
            }

            // Read in1 block
            uint32_t bsr_col_index = column_indices[block_row_start + reduction_block];
            uint32_t in1_block_stride = in1_block_h * in1_tensor_stride_h;
            uint32_t in1_tensor_row_start_tile_id = in1_tensor_start_tile_id + bsr_col_index * in1_block_stride;
            for (uint32_t h = 0; h < in1_block_h; h++) {
                uint32_t in1_tensor_tile_id = in1_tensor_row_start_tile_id;
                for (uint32_t w = 0; w < in1_block_w; w++) {
                    noc_async_read_tile(in1_tensor_tile_id, s1, l1_write_addr_in1);
                    l1_write_addr_in1 += in1_single_tile_size_bytes;
                    in1_tensor_tile_id += in1_tensor_stride_w;
                }
                in1_tensor_row_start_tile_id += in1_tensor_stride_h;
            }

            noc_async_read_barrier();

            DPRINT_DATA0(DPRINT << "block " << output_idx_y << ", " << column_indices[block_row_start + reduction_block] << " read" << ENDL());

            cb_push_back(cb_id_in0, in0_block_num_tiles);
            cb_push_back(cb_id_in1, in1_block_num_tiles);
        }
    }

    DPRINT_DATA0(DPRINT << "Reader kernel complete!" << ENDL());

}
