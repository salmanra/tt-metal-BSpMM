

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
#include <matmul_common/bmm_op.hpp>
#include <tt-metalium/tilize_untilize.hpp>
#include <tt-metalium/device.hpp>

using namespace tt;
using namespace tt::tt_metal;

int main(int argc, char** argv) {
    // Initialize Program and Device

    int device_id = 0;
    IDevice* device = CreateDevice(device_id);
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    // DRAM buffer is some dense matrix, so we can 
    uint32_t M = 1024;  // Rows
    uint32_t N = 1024;  // Columns

    uint32_t dram_buffer_size = M * N * 2;  // Size in bytes (bfloat16)
    uint32_t dram_buffer_size_in_tiles = dram_buffer_size / detail::TileSize(tt::DataFormat::Float16_b);
    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    uint32_t single_tile_size = detail::TileSize(cb_data_format);

        // From tt_metal/common/constants.hpp
    auto num_output_tiles_total = (M * N) / TILE_HW;

    /*
     * Use a helper function to deduce the splits needed to co-operatively do
     * this matmul.
     */
    auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_output_tiles_per_core_group_1,
         num_output_tiles_per_core_group_2] =
            split_work_to_cores(compute_with_storage_grid_size, num_output_tiles_total);

    // TODO: keep going

    // Configure and Create Void Kernel

    std::vector<uint32_t> compute_kernel_args = {};
    KernelHandle void_compute_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/rahmy/hello_world_compute_kernel_remix/kernels/void_compute_kernel.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args});

    // Configure Program and Start Program Execution on Device

    SetRuntimeArgs(program, void_compute_kernel_id, core, {});
    EnqueueProgram(cq, program, false);
    printf("Hello, Core {0, 0} on Device 0, I am sending you a compute kernel. Standby awaiting communication.\n");

    // Wait Until Program Finishes, Print "Hello World!", and Close Device

    Finish(cq);
    printf("Thank you, Core {0, 0} on Device 0, for the completed task.\n");
    CloseDevice(device);

    return 0;
}
