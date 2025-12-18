#include <common/TracyColor.hpp>
#include <cstdio>
#include <string>
#include "../inc/include_me.hpp"
#include "../inc/test_suite.hpp"
#include "../inc/host_code.hpp"

#include <system_error>
#include <tracy/Tracy.hpp>
#include "hostdevcommon/profiler_common.h"


#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
#include "hostdevcommon/profiler_common.h"

#include <cstdlib> // required to start ./capture-release listening

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;

using namespace bsr_host_code;
using namespace profiling_suite;

void profile_test(
    HostCodeFunctionPtr host_func,
    bsr_matrix<bfloat16>& a,
    dense_matrix<bfloat16>& b,
    std::string& test_name,
    int num_iters = 10);

int main(int argc, char** argv) {

    const int num_host_programs = sizeof(HostCodeRegistry) / sizeof(HostCodeRegistry[0]);

    const int test_id = 0;
    const int host_code_id = 0;

    // let's make the test registry and test index required arguments
    // and the host code index
    // then let num_iters be optional
    int test_num = argc > 1 ? std::stoi(argv[1]) : test_id;
    int host_code_num = argc > 2 ? std::stoi(argv[2]) : host_code_id;
    int registry_number = argc > 3 ? std::stoi(argv[3]) : 0;
    int num_iters = argc > 4 ? std::stoi(argv[3]) : 10;

    ProfileCaseFunctionPtr *Registry = nullptr;
    std::string registry_name;
    switch (registry_number) {
        case 0:
            Registry = ProfileCaseRegistry;
            registry_name = "ProfileSuiteSparseVersioning";
            break;
        case 1:
            Registry = ProfileDenseAblationRegistry;
            registry_name = "DenseAblationKProfileSuite";
            break;
    }


    // get the host code and test case
    HostCodeFunctionPtr host_function = HostCodeRegistry[host_code_num].first;
    std::string host_function_name = HostCodeRegistry[host_code_num].second;
    auto [a, b, test_name] = Registry[test_num]();


    // set up command strings to direct and capture the trace (and its csv file)
    char buf[1000];
    size_t n = sprintf(buf, "/home/user/tt-metal/profiles/bsr/%s/%s/", registry_name.c_str(), host_function_name.c_str());
    std::string trace_directory(buf, n);
    std::string trace_file_location = trace_directory + test_name + ".tracy";

    n = sprintf(buf, "mkdir -p %s", trace_directory.c_str());
    std::string mkdir_command(buf, n);
    
    n = sprintf(buf, "./capture-release -f -o %s &", trace_file_location.c_str());
    std::string capture_trace_command(buf, n);

    n = sprintf(buf, "/home/user/tt-metal/profiles/csvs/%s/%s/", registry_name.c_str(), host_function_name.c_str());
    std::string csv_directory(buf);
    std::string csv_file_location = csv_directory + test_name + ".csv";

    n = sprintf(buf, "mkdir -p %s", csv_directory.c_str());
    std::string csv_mkdir_command(buf, n);

    n = sprintf(buf, "./csvexport-release %s > %s", trace_file_location.c_str(), csv_file_location.c_str());
    std::string csvexport_command(buf);

    // print header
    std::cout << "---------------------------------------------------------------------------------" << std::endl;
    std::cout << "--- Host code function: " << host_function_name << std::endl;
    std::cout << "---------------------------------------------------------------------------------" << std::endl;
    std::cout << "--- Test case: " << test_name << std::endl;
    std::cout << "---------------------------------------------------------------------------------" << std::endl;
    std::cout << "--- Num iters: " << num_iters << std::endl;
    std::cout << "---------------------------------------------------------------------------------" << std::endl;
    std::cout << "--- Output file: " << trace_file_location << std::endl;
    std::cout << "---------------------------------------------------------------------------------" << std::endl;

    // run ./capture-release to allow the profiler to listen for the program 
    std::system(mkdir_command.c_str());
    std::system(capture_trace_command.c_str());

    // // // run the program
    profile_test(host_function, a, b, test_name, num_iters);

    // print footer
    std::cout << "---------------------------------------------------------------------------------" << std::endl;
    std::cout << "--- Host code function: " << host_function_name << std::endl;
    std::cout << "---------------------------------------------------------------------------------" << std::endl;
    std::cout << "--- Test case: " << test_name << std::endl;
    std::cout << "---------------------------------------------------------------------------------" << std::endl;
    std::cout << "--- Num iters: " << num_iters << std::endl;
    std::cout << "---------------------------------------------------------------------------------" << std::endl;
    std::cout << "--- Output file: " << trace_file_location << std::endl;
    std::cout << "---------------------------------------------------------------------------------" << std::endl;

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
        ZoneScopedNC("Post-device setup", tracy::Color::DarkOliveGreen);
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
        tilize_nfaces(a.data, R, C);
        tilize_nfaces(b.data, K, N);

        // warm up
        host_func(a, b, output, false, nblocks, M, N, K, R, C, 1, device, false);
        {
            ZoneScopedNC("Program Loop", tracy::Color::Aquamarine);
            for (int count = 0; count < num_iters; count++){
                host_func(a, b, output, false, nblocks, M, N, K, R, C, 1, device, false);
            }
        }

        untilize_nfaces(output.data, M, N);
    }

    // tt_metal::detail::DumpDeviceProfileResults(device);
    tt_metal::detail::ReadDeviceProfilerResults(device);
    CloseDevice(device);
}