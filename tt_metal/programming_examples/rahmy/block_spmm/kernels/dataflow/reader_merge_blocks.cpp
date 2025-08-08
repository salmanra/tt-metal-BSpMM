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
    for (int i = 0; i < num_output_blocks; i++){
        y_coords[i] = get_arg_val<uint32_t>(2 + i);
    }

    DPRINT_DATA0(DPRINT << "Runtime args obtained. num output blocks to compute: " << num_output_blocks << ENDL());


    constexpr bool in0_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool in1_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr bool NoC_Args_is_dram = get_compile_time_arg_val(2) == 1;

    constexpr bool in0_tensor_addr = get_compile_time_arg_val(3) == 1;
    constexpr bool in0_tensor_stride_w = get_compile_time_arg_val(4) == 1;
    constexpr bool in0_tensor_stride_h = get_compile_time_arg_val(5) == 1;

    constexpr bool in0_block_w = get_compile_time_arg_val(6) == 1;
    constexpr bool in0_block_h = get_compile_time_arg_val(7) == 1;
    constexpr bool in0_block_num_tiles = get_compile_time_arg_val(8) == 1;

    constexpr bool in1_tensor_addr = get_compile_time_arg_val(9) == 1;
    constexpr bool in1_tensor_stride_w = get_compile_time_arg_val(10) == 1;
    constexpr bool in1_tensor_stride_h = get_compile_time_arg_val(11) == 1;

    constexpr bool in1_block_w = get_compile_time_arg_val(12) == 1;
    constexpr bool in1_block_h = get_compile_time_arg_val(13) == 1;
    constexpr bool in1_block_num_tiles = get_compile_time_arg_val(14) == 1;

    constexpr bool NoC_Args_addr = get_compile_time_arg_val(15) == 1;
    constexpr bool indptr_addr = get_compile_time_arg_val(16) == 1;



    const uint32_t cb_id_in0 = tt::CBIndex::c_0;
    const uint32_t cb_id_in1 = tt::CBIndex::c_1;
    const uint32_t cb_id_NoC_Args = tt::CBIndex::c_2;
    const uint32_t cb_id_indptr = tt::CBIndex::c_3;


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
    // we might want to explicitly define the datatype (uint16, uint32) instead of just int.

    // Now the idea is to pass the number of output blocks as a runtime arg, 
    // then let the reader grab the rows as needed since if there is more than one
    // output block assigned to a core, each output block is in its own row. 

    // Now, iterate over the blocks in the row
    // We could either:
    //    1. NoC the entire col indices array to each core, and then read the blocks indexing with the indptr values
    //    2. NoC just the col indices for each core's block row, and then read zero indexed
    // I choose 1. because it's simpler for now!
    // TODO: wrap this in a for loop over the number of output blocks being processed by this core, 
    //        accessing/indexing the sparse indexing DSes
    for (uint32_t output_block = 0; output_block < num_output_blocks; output_block++){
        uint32_t output_idx_y = y_coords[output_block]; 
        uint32_t block_row_start = indptr[output_idx_y];
        uint32_t block_row_end = indptr[output_idx_y + 1];
        uint32_t in0_tensor_start_tile_id = block_row_start * in0_block_num_tiles;
        uint32_t in1_tensor_start_tile_id = in1_block_w * output_idx_x;
        for (uint32_t block = block_row_start; block < block_row_end; block++) {
            cb_reserve_back(cb_id_in0, in0_block_num_tiles);
            cb_reserve_back(cb_id_in1, in1_block_num_tiles);

            l1_write_addr_in0 = get_write_ptr(cb_id_in0);
            l1_write_addr_in1 = get_write_ptr(cb_id_in1);

            // Read in0 block
            uint32_t num_blocks_in = block - block_row_start;
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
            // We should start however many columns deep the corresponding output block is.
            // Ah. The terms are confusing.
            // --- "column_indices" gets us the column indices of the BSR matrix, which are the row indices of the dense matrix
            // --- "in1_tensor_start_tile_id" will be the top of a column, and bsr_col_index gets us the row in the dense matrix
            uint32_t bsr_col_index = column_indices[block];
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

            // I want this to print before announcing to the compute kernel
            //DPRINT_DATA0(DPRINT << "block " << block << ", " << column_indices[block] << " read" << ENDL());

            cb_push_back(cb_id_in0, in0_block_num_tiles);
            cb_push_back(cb_id_in1, in1_block_num_tiles);
        }
    }
}
