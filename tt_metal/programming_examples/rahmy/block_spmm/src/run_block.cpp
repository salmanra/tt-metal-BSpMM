
#include <cstdio>
#include <string>
#include "../inc/include_me.hpp"
#include "../inc/test_suite.hpp"
#include "../inc/profiling_suite.hpp"
#include "../inc/host_code.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;

using namespace bsr_test_suite;
using namespace bsr_host_code;
using namespace profiling_suite;

#define ESC "\033["
#define BLACK_BKG "106"
#define GREEN_TXT "118"
#define RED_TXT "196"
#define RESET "\033[m"

// Print to the *original* console, regardless of where stdout is redirected
void console_printf(const char* fmt, ...) {
    static int console_fd = -1;
    if (console_fd == -1) {
        // If not initialized, we fail-safe to /dev/tty (works when a TTY is present),
        // but you could also inject the saved fd via a setter if you prefer.
        console_fd = ::open("/dev/tty", O_WRONLY | O_CLOEXEC);
        // If /dev/tty isn't available (e.g. no controlling terminal), this will be -1.
    }
    if (console_fd == -1) return; // quietly drop if no console is available

    va_list ap;
    va_start(ap, fmt);
    ::vdprintf(console_fd, fmt, ap);
    va_end(ap);
}


void run_test(
    HostCodeFunctionPtr host_func,
    bsr_matrix<bfloat16>& a,
    dense_matrix<bfloat16>& b,
    std::string& test_name) {

    /*
    Requires: a, b to be initialized on CPU
    Modifies: can modifiy output files and log data
    Effects:

    Returns the PCC between the sequential matmul of a and b and the multicore matmul of a and b.
    */

    // device setup
    console_printf("Setting up the device!\n");

    constexpr int device_id = 0;
    IDevice* device = CreateDevice(device_id);


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

    console_printf("Initalizing output data!\n");

    // initialize output_data
    dense_matrix<float> tmp(M, N, 0.0f);
    dense_matrix<bfloat16> output = tmp.bfloat16_cast();


    a.pretty_print();


    // tilize input data
    console_printf("Tilizing!\n");

    a.data = tilize_nfaces(a.data, R, C);
    b.data = tilize_nfaces(b.data, K, N);

    // run bsr_spmm_multicore_reuse
    console_printf("Entering host code\n");
    
    host_func(a, b, output, false, nblocks, M, N, K, R, C, 1, device);
    console_printf("exiting host code\n");

    // untile output data
    output.data = untilize_nfaces(output.data, M, N);

    // this is useless when matrices are not tiny with tiny elements. I get it now.
    // PCC is faulty and gives false positives for say, equality up to scaling, but
    // all_close is simply not suitable for bfloat16.
    // surely there is a version of all_close which bases its tolerance on the norm of the input matrices?
    CloseDevice(device);
}

void run_full_test(int host_code_num, int test_num, TestFunctionPtr* registry){
    auto [a, b, test_name] = registry[test_num]();
    run_test(HostCodeRegistryVerbose[host_code_num].first, a, b, test_name);

    console_printf("--------------------------------------------------------\n");
    console_printf("--- Single Test results --------------------------------\n");
    console_printf("--------------------------------------------------------\n");
    console_printf("--- Host Code function: ");
    console_printf(HostCodeRegistryVerbose[host_code_num].second.c_str());
    console_printf("\n");
    console_printf("--------------------------------------------------------\n");

    console_printf("--- Test #");
    console_printf(std::to_string(test_num).c_str());
    console_printf(", ");
    console_printf(test_name.c_str());
    console_printf(" ---\n");
    console_printf("--------------------------------------------------------\n");
    console_printf("--- COMPLETE!!! ----------------------------------------\n");
    console_printf("--------------------------------------------------------\n");
}

int main(int argc, char** argv) {
    int test_num = 0;
    int host_code_index = 0;
    if (argc > 1) {
        test_num = std::stoi(argv[1]);
    }
    if (argc > 2) {
        host_code_index = std::stoi(argv[2]);
    }

    // Registry selection (mirrors profile_block.cpp)
    int registry_number = argc > 3 ? std::stoi(argv[3]) : -1;
    std::string registry_name = "";
    TestFunctionPtr *Registry = nullptr;
    switch (registry_number) {
        case 0:
            Registry = ProfileCaseRegistry;
            registry_name = "ProfileSuiteSparseVersioning";
            break;
        case 1:
            Registry = ProfileDenseAblationRegistry;
            registry_name = "DenseAblationKProfileSuite";
            break;
        case 2:
            Registry = ProfileLargeSparseRegistry;
            registry_name = "ProfileSuiteLargeSparseVersioning";
            break;
        case 3:
            Registry = ProfileLargeSparseLargeBlocksRegistry;
            registry_name = "ProfileSuiteLargeSparseLargeBlocksVersioning";
            break;
        case 4:
            Registry = ProfileSweepNRegistry;
            registry_name = "ProfileSweepN";
            break;
        case 5:
            Registry = ProfileSweepDensityRegistry;
            registry_name = "ProfileSweepDensity";
            break;
        case 6:
            Registry = ProfileSweepKRegistry;
            registry_name = "ProfileSweepK";
            break;
        case 7:
            Registry = ProfileSweepBlockSizeRegistry;
            registry_name = "ProfileSweepBlockSize";
            break;
        case 8:
            Registry = ProfileSweepSparsityPatternRegistry;
            registry_name = "ProfileSweepSparsityPattern";
            break;
        case 9:
            Registry = ProfileSweepSparsityPatternRegistryD10;
            registry_name = "ProfileSweepSparsityPatternD10";
            break;
        case 10:
            Registry = ProfileSweepSparsityPatternRegistryD5;
            registry_name = "ProfileSweepSparsityPatternD5";
            break;
        case 11:
            Registry = ProfileSweepSparsityPatternRegistryD50;
            registry_name = "ProfileSweepSparsityPatternD50";
            break;
        default:
            Registry = TestRegistry;
            break;
    }

    test_num = argc > 1 ? std::stoi(argv[1]) : -1;
    if (test_num == -1) {
        console_printf("No test specified. Returning.\n");
        return 0;
    }
    run_full_test(host_code_index, test_num, Registry);
    console_printf("Leaving the test program\n");
}
