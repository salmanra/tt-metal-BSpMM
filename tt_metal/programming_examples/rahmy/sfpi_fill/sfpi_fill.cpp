

/*

SFPI program for initializing a DRAM buffer with zeros

- Host creates DRAM buffer
- Host calls on as many cores as convenient (all?) to fill the 
    buffer with zeros
    - probably each core writes a single tile to a CB, then 
        NoCs as many times as necessary to fill its part of DRAM
- profit

*/

// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/test_tiles.hpp>
#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/work_split.hpp>
#include <../matmul_common/bmm_op.hpp>
#include <tt-metalium/tilize_untilize.hpp>
#include <tt-metalium/device.hpp>

using namespace tt;
using namespace tt::tt_metal;

// void init_dram_multicore(vector<bfloat16>& output, IDevice* device, uint32_t M, uint32_t N) {

//     auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
//     uint32_t num_cores_x = compute_with_storage_grid_size.x;
//     uint32_t num_cores_y = compute_with_storage_grid_size.y;


//     uint32_t dram_buffer_size = M * N * 2;  // Size in bytes (bfloat16)
//     uint32_t dram_buffer_size_in_tiles = dram_buffer_size / detail::TileSize(tt::DataFormat::Float16_b);
    
//     tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
//     uint32_t single_tile_size = detail::TileSize(cb_data_format);
        
//     // From tt_metal/common/constants.hpp
//     auto num_output_tiles_total = (M * N) / TILE_HW;

//     tt_metal::InterleavedBufferConfig dram_buffer_config = {
//         .data_format = cb_data_format,
//         .size = dram_buffer_size,
//         .single_tile_size = single_tile_size,
//         .buffer_type = tt_metal::BufferType::DRAM
//     };

//     auto dst_dram_buffer = CreateBuffer(dram_buffer_config);
//     uint32_t dst_addr = dst_dram_buffer->address(); 

//     /*
//      * Use a helper function to deduce the splits needed to co-operatively do
//      * this matmul.
//      */
//     auto
//         [num_cores,
//          all_cores,
//          core_group_1,
//          core_group_2,
//          num_output_tiles_per_core_group_1,
//          num_output_tiles_per_core_group_2] =
//             split_work_to_cores(compute_with_storage_grid_size, num_output_tiles_total);

//     // Configure and Create Void Kernel
//     // Wait the compute kernels all do the same work. It's the writer kernel that will write some variable number of tiles to the DRAM buffer.

//     std::vector<uint32_t> compute_kernel_args = {};
//     KernelHandle compute_kernel_id = CreateKernel(
//         program,
//         "tt_metal/programming_examples/rahmy/sfpi_fill/compute/fill.cpp",
//         all_cores,
//         ComputeConfig{
//             .math_fidelity = MathFidelity::HiFi4,
//             .fp32_dest_acc_en = false,
//             .math_approx_mode = false,
//             .compile_args = compute_kernel_args});
    


//     std::vector<uint32_t> writer_kernel_args_group_1 = {num_output_tiles_per_core_group_1}
//     KernelHandle writer_kernel_id_group_1 = CreateKernel(
//         program,
//         "tt_metal/programming_examples/rahmy/sfpi_fill/dataflow/write_fill.cpp",
//         core_group_1,
//         ComputeConfig{
//             .math_fidelity = MathFidelity::HiFi4,
//             .fp32_dest_acc_en = false,
//             .math_approx_mode = false,
//             .compile_args = writer_kernel_args_group_1});

//     if (!core_group_2.ranges().empty()) {
//         // Configure and Create Reader Kernel
//         std::vector<uint32_t> writer_kernel_args_group_2 = {num_output_tiles_per_core_group_2};

//         KernelHandle writer_kernel_id_group_2 = CreateKernel(
//             program,
//             "tt_metal/programming_examples/rahmy/sfpi_fill/dataflow/write_fill.cpp",
//             core_group_2,
//             ComputeConfig{
//                 .math_fidelity = MathFidelity::HiFi4,
//                 .fp32_dest_acc_en = false,
//                 .math_approx_mode = false,
//                 .compile_args = writer_kernel_args_group_2});

//         //SetRuntimeArgs(program, reader_kernel_id, core_group_2, {});
//     }
//     // Configure Program and Start Program Execution on Device

//     SetRuntimeArgs(program, void_compute_kernel_id, all_cores, {});
//     EnqueueProgram(cq, program, true);
//     EnqueueReadBuffer(cq, )

//     Finish(cq);
//     CloseDevice(device);
// }


void init_dram_single_tensix(std::vector<bfloat16>& output, IDevice* device, uint32_t M, uint32_t N) {
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();

    tt::DataFormat data_format = tt::DataFormat::Float16_b;
    uint32_t single_tile_size = detail::TileSize(data_format);

    uint32_t num_output_tiles = M * N / TILE_HW;

    // Create DRAM buffer
    uint32_t dram_buffer_size = output.size() * sizeof(bfloat16);
    tt_metal::InterleavedBufferConfig dram_config{
        .device = device,
        .size = dram_buffer_size,
        .page_size = detail::TileSize(data_format),
        .buffer_type = tt_metal::BufferType::DRAM};

    auto dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_addr = dram_buffer->address();

    // config CB for filling. Just needs to be a single tile of bfloat16
    uint32_t fill_cb_index = tt::CBIndex::c_24; // interm index bc idk
    uint32_t fill_cb_size = single_tile_size; // one tile!
    CircularBufferConfig fill_cb_config = CircularBufferConfig(fill_cb_size, {{fill_cb_index, data_format}})
                                              .set_page_size(fill_cb_index, single_tile_size);
    auto cb_fill = tt_metal::CreateCircularBuffer(program, CoreCoord{0, 0}, fill_cb_config);

    // Fill the DRAM buffer with zeros using SFPU
    
    auto fill_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/rahmy/sfpi_fill/compute/fill.cpp",
        CoreRangeSet{CoreCoord{0, 0}},
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = {}});

    // comptime args: num tiles 
    // runtime args: start addr
    std::vector<uint32_t> writer_kernel_args = {num_output_tiles};
    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/rahmy/sfpi_fill/dataflow/write_fill.cpp",
        CoreCoord{0,0},
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_kernel_args});

    SetRuntimeArgs(program, writer_kernel_id, CoreRangeSet{CoreCoord{0, 0}}, {dram_addr});

    EnqueueProgram(cq, program, true);
    EnqueueReadBuffer(cq, dram_buffer, output, true);
    Finish(cq);

}

int main(int argc, char** argv) {
    // Initialize Program and Device

    bool pass = true;
    int device_id = 0;
    IDevice* device = CreateDevice(device_id);


    // DRAM buffer is some dense matrix, so we can 
    uint32_t M = 2048;  // Rows
    uint32_t N = 2048;  // Columns

    // output vec, init with zeros. If DRAM buffer is not initialized, then this vec will be overwritten with garbage values. 
    std::vector<bfloat16> output(M * N, bfloat16(0.0f)); 
    std::vector<bfloat16> golden(M * N, bfloat16(0.0f)); 

    init_dram_single_tensix(output, device, M, N);

    uint32_t total_bad = 0;
    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++) {
            if (std::abs(output[i*N + j].to_float() - golden[i*N + j].to_float()) > 1e-5)
                total_bad++;
        }
    }

    std::cout << "here are some output elts: " << output[0].to_float() << ' ' << output[1000].to_float() << std::endl;

    // let's write the output vectors to a file
    std::string local_path = "/home/user/tt-metal/tt_metal/programming_examples/rahmy/sfpi_fill";
    std::string output_file = local_path + "/output.txt";
    std::ofstream out(output_file);
    if (!out.is_open()) {
        TT_THROW("Failed to open output file: {}", output_file);
    }
    for (size_t i = 0; i < output.size(); i++) {
        out << output[i].to_float() << "\n";
    }
    out.close();

    log_info(tt::LogVerif, "Output written to {}", output_file);
    // let's write the golden vector to a file
    std::string golden_file = local_path + "/golden.txt";
    std::ofstream golden_out(golden_file);
    if (!golden_out.is_open()) {
        TT_THROW("Failed to open golden file: {}", golden_file);
    }
    for (size_t i = 0; i < golden.size(); i++) {
        golden_out << golden[i].to_float() << "\n";
    }
    golden_out.close(); 

    TT_FATAL(total_bad == 0, "Not all close! Total bad={} ", total_bad);


    // float pearson = check_bfloat16_vector_pcc(golden, output);
    // log_info(tt::LogVerif, "Metalium vs Golden -- PCC = {}", pearson);
    // TT_FATAL(pearson > 0.99, "PCC not high enough. Result PCC: {}, Expected PCC: 0.99", pearson);

    pass &= CloseDevice(device);

    if (pass) {
        tt::log_info(tt::LogTest, "Test Passed");
    }
    else {
        TT_THROW("Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
