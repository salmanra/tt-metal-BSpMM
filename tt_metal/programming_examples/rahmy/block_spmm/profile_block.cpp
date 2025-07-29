#include <common/TracyColor.hpp>
#include <string>
#include "include_me.hpp"
#include "test_suite.hpp"
#include "host_code.hpp"

#include <tracy/Tracy.hpp>
#include "hostdevcommon/profiler_common.h"

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;

using namespace bsr_test_suite;
using namespace bsr_host_code;

void profile_test(
    HostCodeFunctionPtr host_func,
    bsr_matrix<bfloat16>& a,
    dense_matrix<bfloat16>& b,
    std::string& test_name,
    int num_iters = 10);

int main(int argc, char** argv) {

    const int big_test_id = 30;
    const int host_code_id = 0;

    // uhhh pick a host function pick a test and run it ten times.
    int host_code_num = argc > 1 ? std::stoi(argv[1]) : host_code_id;
    int test_num = argc > 2 ? std::stoi(argv[2]) : big_test_id;

    HostCodeFunctionPtr host_function = HostCodeRegistry[host_code_num].first;
    std::string host_function_name = HostCodeRegistry[host_code_num].second;
    auto [a, b, test_name] = TestRegistry[test_num]();

    std::cout << "---------------------------------------------------------------------------------" << std::endl;
    std::cout << "--- ⚠️⚠️⚠️ PROFILING RESULTS WILL ONLY BE WRITTEN IF THIS PROGRAM IS BUILT WITH PROFILING ENABLED AND IS RUN WITH TRACY LISTENING VIA THE './capture-release' COMMAND" << std::endl;
    std::cout << "---------------------------------------------------------------------------------" << std::endl;
    std::cout << "--- Host code function: " << host_function_name << std::endl;
    std::cout << "---------------------------------------------------------------------------------" << std::endl;
    std::cout << "--- Test case: " << test_name << std::endl;
    std::cout << "---------------------------------------------------------------------------------" << std::endl;

    profile_test(host_function, a, b, test_name);

    std::cout << "--- Profiling complete ----------------------------------------------------------" << std::endl;
    std::cout << "--- ⚠️⚠️⚠️ PROFILING RESULTS WILL ONLY BE WRITTEN IF THIS PROGRAM IS BUILT WITH PROFILING ENABLED AND IS RUN WITH TRACY LISTENING VIA THE './capture-release' COMMAND" << std::endl;
    return 0;
}

void profile_test(
        HostCodeFunctionPtr host_func,
        bsr_matrix<bfloat16>& a,
        dense_matrix<bfloat16>& b,
        std::string& test_name,
        int num_iters) {
    // device setup
    constexpr int device_id = 0;
    IDevice* device = CreateDevice(device_id);
    {
        // matmul params setup
        uint32_t M = a.H;
        uint32_t N = b.W;
        uint32_t K = a.W;
        // block params setup
        uint32_t R = a.R;
        uint32_t C = a.C;
        uint32_t nblocks = a.nblocks;
        uint32_t block_matrix_height = M / R;

        uint32_t Rt = R / TILE_HEIGHT;
        uint32_t Ct = C / TILE_WIDTH;


        // initialize output_data
        dense_matrix<float> tmp(M, N, 0.0f);
        dense_matrix<bfloat16> output = tmp.bfloat16_cast();

        // run sequential spmm
        dense_matrix<bfloat16> golden = a.spmm_bfloat16(b);

        // tilize input data
        tilize(a.data, R, C);
        tilize(b.data, K, N);

        // warm up
        host_func(a, b, output, false, nblocks, M, N, K, R, C, 1, device, false);
        {
            ZoneScopedNC("Program Loop", tracy::Color::Aquamarine);
            for (int count = 0; count < num_iters; count++){
                host_func(a, b, output, false, nblocks, M, N, K, R, C, 1, device, false);
            }
        }

        untilize(output.data, M, N);
    }

    tt_metal::detail::DumpDeviceProfileResults(device);
    CloseDevice(device);
}