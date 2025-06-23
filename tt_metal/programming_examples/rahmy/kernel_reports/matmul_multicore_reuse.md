### A look at *matmul_multicore_reuse*

First, see how we hard code the inner dimension block width in tiles. Instead of letting the block sizes of the input matrices to equal the number of tiles assigned to a single core for each matrix, we cut the block width to 2, allowing ```bmm_op_utils.hpp``` to fit the optimal subblock size according the one of 20 preset sizes (which are hardware-optmized). This means that the input matrices are subdivided four times. In units of tiles, we compute across cores (per_core_M, Kt) and (Kt, per_core_N), across blocks within a core (per_core_M, 2) and (2, per_core_N), across subblocks within a block (out_subblock_h, 2) and (2, out_subblock_w), and across tiles within a subblock (1, 1). In the unoptimized matmul example, input matrices were subdivided only twice: once across cores, and once across tiles within a core.
```C++
   // NOTE: Only supports matmuls where output is blocks of 16 x 16 tiles (ie. multiples of 16*32 x 16*32)
   // NOTE: Maximum number of tiles in output is 120 * 16^2 = 30,720 (eg. [1, 1, 5120, 6144])2
   uint32_t in0_block_w = 2;
```

Let's look at the compile-time arguments to the compute kernel. Compile-time arguments can be thought of as common across cores, and  most TT-Metal programs are designed for the compute kernel to be core-agnostic, letting the dataflow kernels to handle the indexing logic unique to each core (This core-agnostic property will NOT hold for sparse operations).

```C++
    uint32_t in0_block_w = get_compile_time_arg_val(0);              // inner block size in tiles
    uint32_t in0_num_subblocks = get_compile_time_arg_val(1);        // outer row block size
    uint32_t in0_block_num_tiles = get_compile_time_arg_val(2);      // out_subblock_h*in0_block_w*in0_num_subblocks;
    uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(3);   // out_subblock_h*in0_block_w
    uint32_t in1_num_subblocks = get_compile_time_arg_val(4);        // outer column block size
    uint32_t in1_block_num_tiles = get_compile_time_arg_val(5);      // out_subblock_w*in0_block_w* in1_num_subblocks;
    uint32_t in1_per_core_w = get_compile_time_arg_val(6);           // out_subblock_w*in1_num_subblocks
    uint32_t num_blocks = get_compile_time_arg_val(7);               // outer inner dim (in inner dim blocks)
    uint32_t out_subblock_h = get_compile_time_arg_val(8);           // inner row block size in tiles
    uint32_t out_subblock_w = get_compile_time_arg_val(9);           // inner column block size in tiles
    uint32_t out_subblock_num_tiles = get_compile_time_arg_val(10);  // out_subblock_h * out_subblock_w;
    uint32_t batch = get_compile_time_arg_val(11);                   // batch dim
```

Remember arg [0] is hardcoded and determines the inner block dimension. Args [8, 9, 10] come from the subblock size as determined by the util function. Everything else is found using `per_core_M`, `per_core_N`, and the block and subblock dimensions. Many of these values can in principle be computed in the compute kernel, but it is more resource efficient to pass them as arguments so as to save on work performed on the RISCV cores.

Now let's jump around a bit and look at the computation of a single output subblock.

```C++
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
                dst_index,
                false /* transpose */);
            in1_index_inner_dim_offset += in1_per_core_w;
        }
        dst_index++;
    }
    in0_index_h_offset += in0_block_w;
}
```

There is much to point out, but for now let's pay attention to the following.
1. We are writing tiles to the DST register in row major order.
2. Both *input blocks* associated with this *output subblock* are present in L1 in the two circular buffers.
3. `inner_dim` and `w` are our major indices, and thus are conveniently incremented in the loop body, while the `offset` variables are updated according to the size of the matrix dimension they are indexing into. This provides a good example of the style of indexing used in TT.

Following this computation, one output-subblock's worth of partial results is held within the DST register. We won't need these particular partial results until we NoC the next set of input blocks, so where do we keep them? In an intermediate buffer!

```C++
// Move partial result to interm buffer
cb_reserve_back(tt::CBIndex::c_24, out_subblock_num_tiles);
for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
    pack_tile(i, tt::CBIndex::c_24);
}
cb_push_back(tt::CBIndex::c_24, out_subblock_num_tiles);
```
#### Side note about Circular Buffers.
 This intermediate buffer is single-buffered and *shares memory with the output buffer*. The compute kernel is mostly agnostic to this fact and treats the two buffers as logically distinct. This logical separation in the code allows the writer kernel to wait on the output buffer even as the memory associated with the output buffer is filling up with intermediate results. Below is the host code which assigns a single region in SRAM two different circular buffer indices using the `set_page_size` function:
```C++
uint32_t output_cb_index = tt::CBIndex::c_16;
uint32_t interm0_cb_index = 24;
std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec{
    {output_cb_index, cb_data_format}, {interm0_cb_index, cb_data_format}};
CircularBufferConfig cb_output_config = CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
                                            .set_page_size(output_cb_index, single_tile_size)
                                            .set_page_size(interm0_cb_index, single_tile_size);
auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);
```

This ends the side note on Circular Buffers.

The intermediate buffer fills with partial results until all the subblocks of the two input blocks have been read and matmul-ed together. The compute kernel then pops the CBs containing the input blocks and waits on the reader kernel to NoC new blocks. Once the next input blocks are NoC'ed, the compute kernel can begin accumulating into the partial results from the earlier blocks. Before resuming the computation of an output subblock, the kernel must read the partial results of that subblock from the intermediate buffer to the DST register.
```C++
if (enable_reload) {
    copy_tile_to_dst_init_short(tt::CBIndex::c_24);
    cb_wait_front(tt::CBIndex::c_24, out_subblock_num_tiles);
    for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
        copy_tile(tt::CBIndex::c_24, i, i);
    }
    cb_pop_front(tt::CBIndex::c_24, out_subblock_num_tiles);
    mm_init_short();
}
```
Naturally, `enable_reload` is initialized to false and is set to true once the first round of partial results have been sent to the intermediate buffer for every output subblock. From here, the computation proceeds as before. Note that the `matmul_tiles` API call accumulates into the DST register.


## Avenues for Block (BSR) SpMM as adapted from this example.

Let's let "**Block**" refer to a nonzero block in BSR format,
and       "**block**" refer to the unit of tiles which are NoC'ed into SRAM on a core at one time.

What has to change in the compute kernel? Maybe nothing. It receives blocks, it computes output subblocks, it pushes to buffers. It's the reader that has to change to send the reader the correct blocks, and it's the writer that has to change to write the output subblocks to the correct locations in DRAM, and the host which has to configure the circular buffers correctly. Ah. Hold on. We only get 32 circular buffer indices. If we wanted each *set of rows of equal length* to share a CB config (or even simpler, assuming Block sizes are a multiple of block sizes, let each row of nonzero Blocks share a CB config), then we run out of CB indices very quickly. The dense implementation carries a huge advantage by only needing two... wait. Brotherman the whole point of buffers is to be flexible in the number of elements which pass through them. None of this is necessary.

It's the runtime arguments, as described in the Google Doc, which need to be set exactly for each core. For now, hold fast to your assumption that any input BSR matrix will have fewer than 241 nonzero Blocks in a single row. Later we will devise a more flexible algorithm.

### Reader kernel runtime args
Instead of iterating over all the blocks in a dense row, we iterate over all the Blocks in a sparse row and all of the blocks in a dense row of a Block.

We could make some synthetic datasets with low Block sizes (close to tile or subblock sizes) and programmatically check. If Block size is small, we should be able to treat Blocks as blocks.

From either point, the reader kernel will need the col indices for the entire row of interest of the BSR matrix.

### Writer kernel runtime args
The writer kernel simply iterates over the number subblocks in each dimension it expects, then over the shape of a subblock, and NoCs those tiles to DRAM. Nothing changes with this first-shot Block SpMM.

The host has to tell the writer which block it is writing to.
