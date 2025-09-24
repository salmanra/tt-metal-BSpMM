


/*

Match order of RK
for num_iters_x:
    for num_iters_y:
        business as usual
        out_tensor_start_tile_id += in0_stride_h
    out_tensor_start_tile_id += in1_block_w

*/