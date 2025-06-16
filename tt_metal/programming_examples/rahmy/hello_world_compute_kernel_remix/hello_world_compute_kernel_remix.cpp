#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>

using namespace tt;
using namespace tt::tt_metal;

int main(int argc, char** argv) {
    // Initialize Program and Device

    constexpr CoreCoord core = {0, 0};
    int device_id = 0;
    IDevice* device = CreateDevice(device_id);
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();
    Program program2 = CreateProgram();

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
    // what if we create a second compute kernel on the same core?
    KernelHandle void_compute_kernel_id_2 = CreateKernel(
        program2,
        "tt_metal/programming_examples/rahmy/hello_world_compute_kernel_remix/kernels/void2.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args});

    SetRuntimeArgs(program, void_compute_kernel_id, core, {});
    SetRuntimeArgs(program2, void_compute_kernel_id_2, core, {});

    printf("Hello, Core {0, 0} on Device 0 on Program 0, take this compute kernel.\n");
    EnqueueProgram(cq, program, false);

    printf("Hello, Core {0, 0} on Device 0 on Program 1, take this compute kernel.\n");
    EnqueueProgram(cq, program2, false);

    Finish(cq);
    printf("Thanks, Core {0, 0} on Device 0.\n");

    CloseDevice(device);

    return 0;
}
