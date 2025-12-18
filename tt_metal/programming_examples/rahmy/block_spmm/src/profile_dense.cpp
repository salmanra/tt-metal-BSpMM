// // SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
// //
// // SPDX-License-Identifier: Apache-2.0

// #include <common/TracyColor.hpp>
// #include <tracy/Tracy.hpp>
// #include "block_spmm/inc/bsr_matrix.hpp"
// #include "hostdevcommon/profiler_common.h"

// #include "../inc/include_me.hpp"
// #include "../inc/test_suite.hpp"
// #include "../inc/host_code.hpp"
// #include "tt-metalium/bfloat16.hpp"

// #include <cstdlib> // required to start ./capture-release listening


// using namespace tt::constants;
// using namespace std;
// using namespace tt;
// using namespace tt::tt_metal;

// using namespace dense_host_code;
// using namespace profiling_suite;

// #define NUM_ITERS 10

// void golden_matmul(
//     std::vector<bfloat16>& a,
//     std::vector<bfloat16>& b,
//     std::vector<bfloat16>& output,
//     uint32_t M,
//     uint32_t N,
//     uint32_t K,
//     uint32_t B) {
//     std::uint32_t idx_c = 0;
//     std::uint32_t idx_a = 0;
//     std::uint32_t idx_b = 0;

//     float c_f;
//     float float_tmp;
//     std::vector<bfloat16> c_bf(M * N, 0);

//     for (int i = 0; i < M; i++) {
//         for (int j = 0; j < N; j++) {
//             idx_c = j + (i * N);
//             idx_a = i * K;
//             idx_b = j;
//             c_f = 0;
//             for (int k_m = 0; k_m < K; k_m++) {
//                 float_tmp = a[idx_a].to_float() * b[idx_b].to_float();
//                 c_f += float_tmp;
//                 idx_a += 1;
//                 idx_b += N;
//             }
//             output.at(idx_c) = bfloat16(c_f);
//         }
//     }
// }

// ///////////////////////////////////////

// void profile_dense_test(
//     DenseHostCodeFunctionPtr host_func,
//     dense_matrix<bfloat16>& a,
//     dense_matrix<bfloat16>& b,
//     std::string& test_name,
//     int num_iters = 10);

// int main(int argc, char** argv) {

//     const int num_host_programs = sizeof(DenseHostCodeRegistry) / sizeof(DenseHostCodeRegistry[0]);

//     const int big_test_id = 0;
//     const int host_code_id = 0;

//     // uhhh pick a host function pick a test and run it ten times.
//     int test_num = argc > 1 ? std::stoi(argv[1]) : big_test_id;
//     int host_code_num = argc > 2 ? std::stoi(argv[2]) : host_code_id;
//     int registry_number = argc > 3 ? std::stoi(argv[3]) : 0;
//     int num_iters = argc > 4 ? std::stoi(argv[3]) : 10;

//     ProfileCaseFunctionPtr *Registry;
//     std::string registry_name;
//     switch (registry_number) {
//         case 0:
//             Registry = ProfileCaseRegistry;
//             registry_name = "ProfileSuite";
//             break;
//         case 1:
//             Registry = ProfileDenseAblationRegistry;
//             registry_name = "DenseAblationKProfileSuite";
//             break;
//     }
//     DenseHostCodeFunctionPtr host_function = DenseHostCodeRegistry[host_code_num].first;
//     std::string host_function_name = DenseHostCodeRegistry[host_code_num].second;
//     auto [tmp, b, test_name] = Registry[test_num]();
//     dense_matrix<bfloat16> a = tmp.to_dense();

//     // set up command strings to direct and capture the trace
//     char buf[1000];
//     size_t n = sprintf(buf, "/home/user/tt-metal/profiles/dense/%s/%s/", registry_name.c_str(), host_function_name.c_str());
//     std::string trace_directory(buf, n);
//     std::string trace_file_location = trace_directory + test_name + ".tracy";

//     n = sprintf(buf, "mkdir -p %s", trace_directory.c_str());
//     std::string mkdir_command(buf, n);

//     n = sprintf(buf, "./capture-release -f -o %s &", trace_file_location.c_str());
//     std::string capture_trace_command(buf, n);

//     n = sprintf(buf, "/home/user/tt-metal/profiles/csvs/%s/%s/", registry_name.c_str(), host_function_name.c_str());
//     std::string csv_directory(buf);
//     std::string csv_file_location = csv_directory + test_name + ".csv";

//     n = sprintf(buf, "./csvexport-release %s > %s", trace_file_location.c_str(), csv_file_location.c_str());
//     std::string csvexport_command(buf);

//     // print header
//     std::cout << "---------------------------------------------------------------------------------" << std::endl;
//     std::cout << "--- Host code function: " << host_function_name << std::endl;
//     std::cout << "---------------------------------------------------------------------------------" << std::endl;
//     std::cout << "--- Test case: " << test_name << std::endl;
//     std::cout << "---------------------------------------------------------------------------------" << std::endl;
//     std::cout << "--- Num iters: " << num_iters << std::endl;
//     std::cout << "---------------------------------------------------------------------------------" << std::endl;
//     std::cout << "--- Output file: " << trace_file_location << std::endl;
//     std::cout << "---------------------------------------------------------------------------------" << std::endl;

//     // run ./capture-release to allow the profiler to listen for the program
//     std::system(mkdir_command.c_str());
//     std::system(capture_trace_command.c_str());

//     profile_dense_test(host_function, a, b, test_name);

//     // print footer
//     std::cout << "---------------------------------------------------------------------------------" << std::endl;
//     std::cout << "--- Host code function: " << host_function_name << std::endl;
//     std::cout << "---------------------------------------------------------------------------------" << std::endl;
//     std::cout << "--- Test case: " << test_name << std::endl;
//     std::cout << "---------------------------------------------------------------------------------" << std::endl;
//     std::cout << "--- Num iters: " << num_iters << std::endl;
//     std::cout << "---------------------------------------------------------------------------------" << std::endl;
//     std::cout << "--- Output file: " << trace_file_location << std::endl;
//     std::cout << "---------------------------------------------------------------------------------" << std::endl;

//     // std::system(csvexport_command.c_str());

//     return 0;
// }

// void profile_dense_test(
//         DenseHostCodeFunctionPtr host_func,
//         dense_matrix<bfloat16> & a,
//         dense_matrix<bfloat16> & b,
//         std::string & test_name,
//         int num_iters) {
//     constexpr int device_id = 0;
//     IDevice* device = CreateDevice(device_id);
//     {
//         ZoneScopedNC("Post-device setup", tracy::Color::DarkOliveGreen);
//         // matmul params setup
//         uint32_t M = a.H;
//         uint32_t N = b.W;
//         uint32_t K = a.W;

//         // initialize output_data
//         dense_matrix<float> tmp(M, N, 0.0f);
//         dense_matrix<bfloat16> output = tmp.bfloat16_cast();

//         // run sequential spmm
//         dense_matrix<bfloat16> golden = a.gemm_bfloat16(b);

//         // tilize input data
//         tilize(a.data, M, K);
//         tilize(b.data, K, N);

//         // warm up
//         host_func(a.data, b.data, output.data, false, M, N, K, 1, device);
//         {
//             ZoneScopedNC("Program Loop", tracy::Color::Aquamarine);
//             for (int count = 0; count < num_iters; count++){
//                 host_func(a.data, b.data, output.data, false, M, N, K, 1, device);
//             }
//         }

//         untilize(output.data, M, N);
//     }

//     tt_metal::detail::DumpDeviceProfileResults(device);
//     CloseDevice(device);
// }
