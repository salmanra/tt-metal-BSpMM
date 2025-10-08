// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

int main() {
    using namespace tt;
    using namespace tt::tt_metal;

    char* env_var = std::getenv("TT_METAL_DPRINT_CORES");
    if (env_var == nullptr) {
        std::cerr << "WARNING: Please set the environment variable TT_METAL_DPRINT_CORES to 0,0 to see the output of "
                     "the Data Movement kernels."
                  << std::endl;
        std::cerr << "WARNING: For example, export TT_METAL_DPRINT_CORES=0,0" << std::endl;
    }

    // Initialize mesh device (1x1), command queue, workload, device range, and program.
    // We are going to use the first device (0) and the first core (0, 0) on the device.
    constexpr CoreCoord core = {0, 0};
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    // Command queue lets us submit work (execute programs and read/write buffers) to the device.
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    // Prepare a workload and a device coordinate range that spans the mesh.
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();

    // Configure and create Data Movement kernels
    // There are 2 Data Movement cores per Tensix. In applications one is usually used for reading data from DRAM and
    // the other for writing data. However for demonstration purposes, we will create 2 Data Movement kernels that
    // simply prints a message to show them running at the same time.
    KernelHandle data_movement_kernel_0 = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "hello_world_datamovement_kernel/kernels/dataflow/void_dataflow_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    KernelHandle data_movement_kernel_1 = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "hello_world_datamovement_kernel/kernels/dataflow/void_dataflow_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

<<<<<<< HEAD
    // Configure Program and Start Program Execution on Device

    // a single core is given two kernels to run.
    // the first kernel is specified to run on the 0th data movement RISCV processor,
    // and the second kernel on the 1st, so these can run in parallel?
    // RESEARCH: what happens when you specify multiple kernels onto a single RISCV processor?
    //           are they run sequentially? concurrently?
    //           is this an ill-defined program?
    //           is this a key to writing optimal programs?
    //              ie, a core exhibits fine-grained multithreading to hide latency
    SetRuntimeArgs(program, void_dataflow_kernel_noc0_id, core, {});
    SetRuntimeArgs(program, void_dataflow_kernel_noc1_id, core, {});
    EnqueueProgram(cq, program, false);
    printf("Hello, Core {0, 0} on Device 0, I am sending you some data. Standby awaiting communication.\n");

    // Wait Until Program Finishes, Print "Hello World!", and Close Device
=======
    // Set Runtime Arguments for the Data Movement Kernels (none in this case and execute the program)
    SetRuntimeArgs(program, data_movement_kernel_0, core, {});
    SetRuntimeArgs(program, data_movement_kernel_1, core, {});
    std::cout << "Hello, Core {0, 0} on Device 0, Please start execution. I will standby for your communication."
              << std::endl;
>>>>>>> main

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);
    // Wait Until MeshWorkload Finishes. The program should print the following (NC and BR is Data movement core 1 and 0
    // respectively):
    //
    // 0:(x=0,y=0):NC: My logical coordinates are 0,0
    // 0:(x=0,y=0):NC: Hello, host, I am running a void data movement kernel on Data Movement core 1.
    // 0:(x=0,y=0):BR: My logical coordinates are 0,0
    // 0:(x=0,y=0):BR: Hello, host, I am running a void data movement kernel on Data Movement core 0.
    //
    // Deconstructing the output:
    // 0: - Device ID
    // (x=0,y=0): - Tensix core coordinates (so both Data Movement cores are on the same Tensix core
    // NC: - Data Movement core 1
    // BR: - Data Movement core 0

    std::cout << "Thank you, Core {0, 0} on Device 0, for the completed task." << std::endl;
    mesh_device->close();
    return 0;
}
