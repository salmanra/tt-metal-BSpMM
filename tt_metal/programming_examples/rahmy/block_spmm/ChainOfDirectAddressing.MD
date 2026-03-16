# The plan for sharing reads of in1

Starter host code in /home/user/tt-metal/tt_metal/programming_examples/rahmy/block_spmm/inc/host_code/bsr_spmm_multicore_snfin0_cdain1.cpp

Starter kernel code for in1 data movement is in /home/user/tt-metal/tt_metal/programming_examples/rahmy/block_spmm/kernels/dataflow/reader_in1_chain_of_direct_addressing.cpp


For each block read to the dense matrix, a Tensix core can find the set of other Tensix cores which "incidentally" access the same block of memory at the same iter_y AND reduction_iter. This probably sucks (if two rows' access patterns differ by an offset of 1 block, this strategy fails to capture the sharing), but in denser cases can probably expose a decent bit of data sharing. And it may be a good start. 

All the major changes take place in the data movement kernel for in1. First, much like the kernel reads the indexing data, the kernel now needs to also read the output_y_indices vector and the folded_bsr_matrix_indices vector which together store the mapping of core_idx_y and iter_y to a dense matrix row. Then, when the reader kernel determines the row in which it needs to make a DRAM read request, it also now must obtain the "share set", the other Tensix tiles in its column which access this same dense matrix block (Tensix tiles in the same column will access the same dense matrix column in the same iter_x, so the share set is whichever tiles in the same column are also accessing the same dense matrix row at the same iter_x and iter_y and reduction iter). Once a Tensix tile has its row_idx into the dense matrix, it must check the other 7 core rows and obtain the share set (the set of core_idx_y which are accessing the same dense matrix row in this iter_y, iter_x, and reduction_iter). Note that reduction iter in the equation I am about to present is slightly different from reduction_iter in the kernel because this one goes from 0 to (block_row_end - block_row_start). 

my_row = col_indices[indptr[folded_bsr_matrix_indices[output_y_indices[my_core_idx_y, iter_y]]]+reduction_iter]

my_share_set = {idx_y | col_indices[indptr[folded_bsr_matrix_indices[output_y_indices[idx_y, iter_y]]]+reduction_iter] == my_row}

my_share_set_injector = max(my_share_set)

my_share_set_sink = min(my_share_set)

if (I'm not the injector) my_sender = min over the subset of my_share_set which is greater than my_core_idx_y

if (I'm not the sink) my_receiver = max over the subset of my_share_set which is less than my_core_idx_y

The above info is then enough to set up the correct semaphores, then perform the store and forward data movement of the block across the share set