#include <common/TracyColor.hpp>
#include <cstdio>
#include <string>
#include "../inc/include_me.hpp"
#include "../inc/test_suite.hpp"
#include "../inc/host_code.hpp"

#include <system_error>
#include <tracy/Tracy.hpp>
#include "hostdevcommon/profiler_common.h"

#include <cstdlib> // required to start ./capture-release listening

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;

using namespace bsr_host_code;
using namespace profiling_suite;

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

    ProfileCaseFunctionPtr *Registry;
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

    std::system(csv_mkdir_command.c_str());
    std::system(csvexport_command.c_str());

    return 0;
}
