
#include <cstdio>
#include <string>
#include "../inc/include_me.hpp"
#include "../inc/profiling_suite.hpp"
#include "../inc/host_code.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;

using namespace bsr_host_code;
using namespace profiling_suite;

#define ESC "\033["
#define BLACK_BKG "106"
#define GREEN_TXT "118"
#define RED_TXT "196"
#define RESET "\033[m"

// Verbose registry: 3 SC26 algorithms with verbose=true, profiling=false
static std::pair<HostCodeFunctionPtr, std::string> HostCodeRegistryVerbose[] = {
    {bsr_spmm_multicore_load_balanced_new_DM<true, false>, "bsr_spmm_multicore_load_balanced_new_DM"},
    {bsr_spmm_multicore_snf<true, false>, "bsr_spmm_multicore_snf"},
    {bsr_spmm_multicore_snfin0_cdain1<true, false>, "bsr_spmm_multicore_snfin0_cdain1"},
};

// Print to the *original* console, regardless of where stdout is redirected
void console_printf(const char* fmt, ...) {
    static int console_fd = -1;
    if (console_fd == -1) {
        console_fd = ::open("/dev/tty", O_WRONLY | O_CLOEXEC);
    }
    if (console_fd == -1) return;

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

    console_printf("Setting up the device!\n");

    constexpr int device_id = 0;
    IDevice* device = CreateDevice(device_id);

    uint32_t M = a.H;
    uint32_t N = b.W;
    uint32_t K = a.W;
    uint32_t R = a.R;
    uint32_t C = a.C;
    uint32_t nblocks = a.nblocks;

    console_printf("Initalizing output data!\n");

    dense_matrix<float> tmp(M, N, 0.0f);
    dense_matrix<bfloat16> output = tmp.bfloat16_cast();

    a.pretty_print();

    console_printf("Tilizing!\n");

    a.data = tilize_nfaces(a.data, R, C);
    b.data = tilize_nfaces(b.data, K, N);

    console_printf("Entering host code\n");

    host_func(a, b, output, false, nblocks, M, N, K, R, C, 1, device);
    console_printf("exiting host code\n");

    output.data = untilize_nfaces(output.data, M, N);

    CloseDevice(device);
}

void run_full_test(int host_code_num, int test_num, ProfileCaseFunctionPtr* registry){
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
    bool run_all = true;

    int test_num = 0;
    int host_code_index = 0;
    if (argc > 1) {
        run_all = std::string(argv[1]) == "all";
    }
    if (argc > 2) {
        host_code_index = std::stoi(argv[2]);
    }

    int registry_number = argc > 3 ? std::stoi(argv[3]) : 0;

    if (registry_number < 0 || registry_number >= NUM_REGISTRIES) {
        fprintf(stderr, "Invalid registry number %d (valid: 0-%d)\n", registry_number, NUM_REGISTRIES - 1);
        return 1;
    }

    ProfileCaseFunctionPtr *Registry = Registries[registry_number];
    std::string registry_name = RegistryNames[registry_number];
    int num_tests = RegistrySizes[registry_number];

    if (run_all) {
        int saved_stdout = ::dup(STDOUT_FILENO);
        if (saved_stdout == -1) {
            std::perror("dup");
            return 1;
        }
        int log_fd = ::open("std.out.log", O_CREAT | O_WRONLY | O_TRUNC, 0644);
        if (log_fd == -1) {
            std::perror("open");
            return 1;
        }
        if (::dup2(log_fd, STDOUT_FILENO) == -1) {
            std::perror("dup2");
            return 1;
        }
        ::close(log_fd);

        for (int i = 0; i < num_tests; i++) {
            run_full_test(host_code_index, i, Registry);
        }
    } else {
        test_num = argc > 1 ? std::stoi(argv[1]) : -1;
        if (test_num == -1) {
            console_printf("No test specified. Returning.\n");
            return 0;
        }
        run_full_test(host_code_index, test_num, Registry);
        console_printf("Leaving the test program\n");
    }
}
