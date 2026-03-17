#include <stdint.h>
#include <cstdint>
#include "dataflow_api.h"
#include "hostdevcommon/kernel_structs.h"
#include "debug/dprint.h"
#include <tools/profiler/kernel_profiler.hpp>
#include "tt_metal/programming_examples/rahmy/block_spmm/kernels/common/spmm_reader_common.hpp"
#include "tt_metal/programming_examples/rahmy/block_spmm/kernels/common/spmm_tile_ops.hpp"
#include "tt_metal/programming_examples/rahmy/block_spmm/kernels/common/spmm_indexing.hpp"
#include "tt_metal/programming_examples/rahmy/block_spmm/kernels/common/spmm_profiling.hpp"

// Compile-time profiling zone toggle (override to 0 via CreateKernel defines)
#ifndef PROFILE_READ_IN1
#define PROFILE_READ_IN1 1
#endif
#ifndef PROFILE_WRITE_OUT
#define PROFILE_WRITE_OUT 1
#endif

// Ablation skip flags (set to 1 via CreateKernel defines to skip that phase)
#ifndef SKIP_IN1_DRAM_READ
#define SKIP_IN1_DRAM_READ 0
#endif
#ifndef SKIP_DRAM_WRITE
#define SKIP_DRAM_WRITE 0
#endif

// CDA action codes
constexpr uint32_t CDA_SOLO      = 0;
constexpr uint32_t CDA_DRAM_READ = 1;
constexpr uint32_t CDA_RECEIVE   = 2;

void kernel_main(){
    ///////////////////////////////////////////////////////////////////////
    /// COMPILETIME ARGS //////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    constexpr bool in0_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool in1_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr bool col_indices_is_dram = get_compile_time_arg_val(2) == 1;
    constexpr bool indptr_is_dram = get_compile_time_arg_val(3) == 1;

    constexpr uint32_t in0_tensor_addr = get_compile_time_arg_val(4);
    constexpr uint32_t in0_tensor_stride_w = get_compile_time_arg_val(5);
    constexpr uint32_t in0_tensor_stride_h = get_compile_time_arg_val(6);

    constexpr uint32_t in0_block_w = get_compile_time_arg_val(7);
    constexpr uint32_t in0_block_h = get_compile_time_arg_val(8);
    constexpr uint32_t in0_block_num_tiles = get_compile_time_arg_val(9);

    constexpr uint32_t in1_tensor_addr = get_compile_time_arg_val(10);
    constexpr uint32_t in1_tensor_stride_w = get_compile_time_arg_val(11);
    constexpr uint32_t in1_tensor_stride_h = get_compile_time_arg_val(12);

    constexpr uint32_t in1_block_w = get_compile_time_arg_val(13);
    constexpr uint32_t in1_block_h = get_compile_time_arg_val(14);
    constexpr uint32_t in1_block_num_tiles = get_compile_time_arg_val(15);

    constexpr uint32_t col_indices_addr = get_compile_time_arg_val(16);
    constexpr uint32_t indptr_addr = get_compile_time_arg_val(17);

    constexpr uint32_t col_indices_num_tiles = get_compile_time_arg_val(18);
    constexpr uint32_t indptr_num_tiles = get_compile_time_arg_val(19);

    constexpr uint32_t is_output_writer = get_compile_time_arg_val(20);

    // writer args (only used when is_output_writer == 1)
    constexpr uint32_t out_tensor_addr = get_compile_time_arg_val(21);
    constexpr uint32_t RtNt = get_compile_time_arg_val(22);
    constexpr uint32_t Nt = get_compile_time_arg_val(23);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(24);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(25);

    // CDA semaphores
    uint32_t in1_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(26));
    uint32_t in1_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(27));

    ///////////////////////////////////////////////////////////////////////
    /// END COMPILETIME ARGS //////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////
    /// RUNTIME ARGS //////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    uint32_t arg_index = 0;
    const uint32_t num_iters_x = get_arg_val<uint32_t>(arg_index++);
    const uint32_t num_iters_y = get_arg_val<uint32_t>(arg_index++);
    const uint32_t output_idx_x_start = get_arg_val<uint32_t>(arg_index++);
    const uint32_t my_core_idx_y = get_arg_val<uint32_t>(arg_index++);
    const uint32_t num_cores_in_column = get_arg_val<uint32_t>(arg_index++);
    const uint32_t noc_x_for_column = get_arg_val<uint32_t>(arg_index++);

    // VLAs sized to actual values (RISC-V stack is small — no oversized static arrays)
    uint32_t noc_y_table[num_cores_in_column];
    for (uint32_t r = 0; r < num_cores_in_column; r++) {
        noc_y_table[r] = get_arg_val<uint32_t>(arg_index++);
    }

    // This core's y_coords and folded_y_coords
    uint32_t y_coords[num_iters_y];
    uint32_t folded_y_coords[num_iters_y];
    for (uint32_t i = 0; i < num_iters_y; i++) {
        y_coords[i] = get_arg_val<uint32_t>(arg_index++);
        if constexpr (is_output_writer) {
            folded_y_coords[i] = get_arg_val<uint32_t>(arg_index++);
        }
    }
    uint32_t out_tensor_start_tile_id = 0;
    if constexpr (is_output_writer) {
        out_tensor_start_tile_id = get_arg_val<uint32_t>(arg_index++);
    }

    // Column-wide schedule
    uint32_t all_num_iters_y[num_cores_in_column];
    for (uint32_t r = 0; r < num_cores_in_column; r++) {
        all_num_iters_y[r] = get_arg_val<uint32_t>(arg_index++);
    }

    // Compute offsets into the flattened all_y_coords array
    uint32_t all_y_coords_offsets[num_cores_in_column];
    uint32_t total_all_y_coords = 0;
    for (uint32_t r = 0; r < num_cores_in_column; r++) {
        all_y_coords_offsets[r] = total_all_y_coords;
        total_all_y_coords += all_num_iters_y[r];
    }

    // Read all_y_coords flattened (VLA — actual size, not worst-case)
    uint32_t all_y_coords_flat[total_all_y_coords];
    for (uint32_t i = 0; i < total_all_y_coords; i++) {
        all_y_coords_flat[i] = get_arg_val<uint32_t>(arg_index++);
    }

    ///////////////////////////////////////////////////////////////////////
    /// END RUNTIME ARGS //////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    DPRINT_DATA0(DPRINT << "CDA in1: core_idx_y=" << my_core_idx_y << " num_cores_col=" << num_cores_in_column << ENDL());

    const auto ti = spmm::get_tile_info();
    const uint32_t output_tile_size = get_tile_size(spmm::cb_id_out);
    const DataFormat output_format = get_dataformat(spmm::cb_id_out);

    const InterleavedAddrGenFast<in1_is_dram> s1 = {
        .bank_base_address = in1_tensor_addr, .page_size = ti.in1_tile_size, .data_format = ti.in1_format};
    const InterleavedAddrGenFast<true> out_s = {
        .bank_base_address = out_tensor_addr, .page_size = output_tile_size, .data_format = output_format};

    // CDA semaphore setup — MUST happen before indexing load so that
    // semaphores are initialized before any core can signal us.
    // (A fast core may finish wait_for_indexing, enter the main loop,
    //  and noc_semaphore_inc our sender_sem before we reach this point.)
    volatile tt_l1_ptr uint32_t* in1_sender_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_sender_semaphore_addr);
    *(in1_sender_sem_ptr) = 0;
    volatile tt_l1_ptr uint32_t* in1_receiver_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_receiver_semaphore_addr);
    *(in1_receiver_sem_ptr) = 0;

    // Load or wait for sparse indexing data
    uint32_t* col_indices;
    uint32_t* indptr;
    if constexpr (is_output_writer){
        col_indices = spmm::load_indexing_tiled<col_indices_is_dram>(
            spmm::cb_id_col_indices, col_indices_addr,
            ti.col_indices_tile_size, ti.col_indices_format, col_indices_num_tiles);
        indptr = spmm::load_indexing_tiled<indptr_is_dram>(
            spmm::cb_id_indptr, indptr_addr,
            ti.indptr_tile_size, ti.indptr_format, indptr_num_tiles);
    }
    else {
        indptr = spmm::wait_for_indexing(spmm::cb_id_indptr, indptr_num_tiles);
        col_indices = spmm::wait_for_indexing(spmm::cb_id_col_indices, col_indices_num_tiles);
    }

    // Writer setup (computed unconditionally; values are only used when is_output_writer == 1)
    uint32_t out_num_subblocks_w = in1_block_w / out_subblock_w;
    uint32_t out_num_subblocks_h = in0_block_h / out_subblock_h;
    uint32_t out_tensor_next_subblock_stride_w = out_subblock_w;
    uint32_t out_tensor_next_subblock_stride_h = out_subblock_h * Nt;
    uint32_t out_tensor_stride_w = 1;
    uint32_t out_tensor_stride_h = Nt;

    ///////////////////////////////////////////////////////////////////////
    /// PROGRAM BODY //////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    uint32_t out_tensor_x_coord_offset = 0;
    uint32_t output_idx_y, output_idx_x;
    for (uint32_t iter_y = 0; iter_y < num_iters_y; iter_y++){
        // Get y_coord for this iter
        output_idx_y = y_coords[iter_y];
        uint32_t my_block_start = indptr[output_idx_y];
        uint32_t my_block_end = indptr[output_idx_y + 1];

        // Compute max_blocks across all cores in column for this iter_y
        uint32_t max_blocks = my_block_end - my_block_start;
        for (uint32_t r = 0; r < num_cores_in_column; r++) {
            if (iter_y >= all_num_iters_y[r]) continue;
            uint32_t their_row = all_y_coords_flat[all_y_coords_offsets[r] + iter_y];
            uint32_t their_blocks = indptr[their_row + 1] - indptr[their_row];
            if (their_blocks > max_blocks) max_blocks = their_blocks;
        }

        // Block bytes for forwarding
        uint32_t current_block_bytes = ti.in1_tile_size * in1_block_num_tiles;

        for (uint32_t iter_x = 0; iter_x < num_iters_x; iter_x++){
            output_idx_x = output_idx_x_start + iter_x;
            uint32_t in1_tensor_start_tile_id = in1_block_w * output_idx_x;
            uint32_t in1_block_stride = in1_block_h * in1_tensor_stride_h;

            for (uint32_t vstep = 0; vstep < max_blocks; vstep++){
                // NOP check: if this core has no block at this vstep, skip
                if (my_block_start + vstep >= my_block_end) continue;

                uint32_t my_col = col_indices[my_block_start + vstep];

                // ── Share set computation ──
                // Scan all cores in column. Find:
                // - share_set_size: how many cores need this same col index at this vstep
                // - injector_idx: max core_idx_y in share set (reads from DRAM)
                // - my_sender_idx: smallest r > my_core_idx_y in share set (sends data to us)
                // - my_downstream_idx: largest r < my_core_idx_y in share set (we forward to them)
                uint32_t share_set_size = 0;
                uint32_t injector_idx = my_core_idx_y;  // guaranteed in set
                bool found_sender = false;
                uint32_t my_sender_idx = 0;
                bool found_downstream = false;
                uint32_t my_downstream_idx = 0;

                for (uint32_t r = 0; r < num_cores_in_column; r++) {
                    if (iter_y >= all_num_iters_y[r]) continue;
                    uint32_t their_row = all_y_coords_flat[all_y_coords_offsets[r] + iter_y];
                    uint32_t their_start = indptr[their_row];
                    uint32_t their_end = indptr[their_row + 1];
                    if (their_start + vstep >= their_end) continue;

                    uint32_t their_col = col_indices[their_start + vstep];
                    if (their_col == my_col) {
                        share_set_size++;
                        // injector = max core_idx_y in share set
                        if (r > injector_idx) {
                            injector_idx = r;
                        }
                        // my_sender_idx = smallest r > my_core_idx_y in share set
                        if (r > my_core_idx_y) {
                            if (!found_sender || r < my_sender_idx) {
                                my_sender_idx = r;
                                found_sender = true;
                            }
                        }
                        // my_downstream_idx = largest r < my_core_idx_y in share set
                        if (r < my_core_idx_y) {
                            if (!found_downstream || r > my_downstream_idx) {
                                my_downstream_idx = r;
                                found_downstream = true;
                            }
                        }
                    }
                }

                // ── Determine action ──
                uint32_t action;
                if (share_set_size <= 1) {
                    action = CDA_SOLO;
                } else if (my_core_idx_y == injector_idx) {
                    action = CDA_DRAM_READ;
                } else {
                    action = CDA_RECEIVE;
                }

                // ── Execute action ──
                cb_reserve_back(spmm::cb_id_in1, in1_block_num_tiles);
                uint32_t l1_write_addr_in1 = get_write_ptr(spmm::cb_id_in1);
                uint32_t l1_write_addr_in1_start = l1_write_addr_in1;  // Save before read_block_by_tile mutates it

                if (action == CDA_SOLO || action == CDA_DRAM_READ) {
#if SKIP_IN1_DRAM_READ == 0
                    // DRAM read
#if PROFILE_READ_IN1 == 1
                    DeviceZoneScopedN("SpMM Zone: CDA Reading dense block of in1 from DRAM");
#endif
                    DPRINT_DATA0(DPRINT << "in1 DRAM Read: " << action << ENDL());

                    spmm::read_block_by_tile(
                        in1_tensor_start_tile_id + my_col * in1_block_stride,
                        s1, l1_write_addr_in1,
                        ti.in1_tile_size, in1_block_h, in1_block_w,
                        in1_tensor_stride_h, in1_tensor_stride_w);
                    noc_async_read_barrier();
#endif
                } else {
                    // RECEIVE — wait for data from sender
                    DPRINT_DATA0(DPRINT << "in1 receiving from x: " << noc_x_for_column << ", y: " << noc_y_table[my_sender_idx] << ENDL());

                    noc_semaphore_set(in1_receiver_sem_ptr, 0);
                    DPRINT_DATA0(DPRINT << "Receiving set local semaphore"<< ENDL());
                    uint64_t sender_sem_noc = get_noc_addr(noc_x_for_column, noc_y_table[my_sender_idx], in1_sender_semaphore_addr);
                    noc_semaphore_inc(sender_sem_noc, 1);
                    DPRINT_DATA0(DPRINT << "Receiving set NoC sender semaphore"<< ENDL());
                    noc_semaphore_wait(in1_receiver_sem_ptr, 1);
                    DPRINT_DATA0(DPRINT << "Receiving got past commit"<< ENDL());

                }

                cb_push_back(spmm::cb_id_in1, in1_block_num_tiles);

                // Forward to downstream if applicable
                if (found_downstream && action != CDA_SOLO) {
                    DPRINT_DATA0(DPRINT << "in1 sharing to x:" << noc_x_for_column << ", y: " << noc_y_table[my_downstream_idx] << ENDL());

                    noc_semaphore_wait(in1_sender_sem_ptr, 1);
                    DPRINT_DATA0(DPRINT << "Sharing got past waiting semaphore"<< ENDL());
                    noc_semaphore_set(in1_sender_sem_ptr, 0);
                    DPRINT_DATA0(DPRINT << "Sharing got past setting semaphore"<< ENDL());


                    uint64_t dest_data_addr = get_noc_addr(noc_x_for_column, noc_y_table[my_downstream_idx], l1_write_addr_in1_start);
                    noc_async_write(l1_write_addr_in1_start, dest_data_addr, current_block_bytes);
                    noc_async_write_barrier();
                    DPRINT_DATA0(DPRINT << "Sharing got past write barrier"<< ENDL());

                    uint64_t dest_recv_sem = get_noc_addr(noc_x_for_column, noc_y_table[my_downstream_idx], in1_receiver_semaphore_addr);
                    noc_semaphore_inc(dest_recv_sem, 1);
                    noc_async_atomic_barrier();
                    DPRINT_DATA0(DPRINT << "Sharing got past commit"<< ENDL());

                }
            }

            // Output write — IDENTICAL to SNF in1 reader
            if constexpr (is_output_writer) {
                uint32_t out_tensor_y_coord_offset = RtNt * folded_y_coords[iter_y];
                uint32_t out_block_num_tiles = in0_block_h * in1_block_w;
                uint32_t out_tensor_sbh_start_tile_id = out_tensor_start_tile_id + out_tensor_y_coord_offset + out_tensor_x_coord_offset;

                cb_wait_front(spmm::cb_id_out, out_block_num_tiles);

                uint32_t l1_read_addr = get_read_ptr(spmm::cb_id_out);
#if SKIP_DRAM_WRITE == 0
#if PROFILE_WRITE_OUT == 1
                DeviceZoneScopedN("SpMM Zone: Writing Block back to DRAM");
#endif
                for (uint32_t sbh = 0; sbh < out_num_subblocks_h; sbh++) {
                    uint32_t out_tensor_sbw_start_tile_id = out_tensor_sbh_start_tile_id;
                    for (uint32_t sbw = 0; sbw < out_num_subblocks_w; sbw++) {
                        spmm::write_subblock_by_tile(
                            out_tensor_sbw_start_tile_id,
                            out_s, l1_read_addr,
                            output_tile_size, out_subblock_h, out_subblock_w,
                            out_tensor_stride_h, out_tensor_stride_w);
                        out_tensor_sbw_start_tile_id += out_tensor_next_subblock_stride_w;
                    }
                    out_tensor_sbh_start_tile_id += out_tensor_next_subblock_stride_h;
                }
                noc_async_write_barrier();
#endif
                cb_pop_front(spmm::cb_id_out, out_block_num_tiles);
                out_tensor_x_coord_offset += out_num_subblocks_w * out_tensor_next_subblock_stride_w;
            }
        }
        out_tensor_x_coord_offset = 0;
    }

    if constexpr (is_output_writer) {
        cb_pop_front(spmm::cb_id_col_indices, col_indices_num_tiles);
        cb_pop_front(spmm::cb_id_indptr, indptr_num_tiles);
    }
    DPRINT_DATA0(DPRINT << "CDA in1 kernel complete" << ENDL());

}
