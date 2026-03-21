
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

using namespace bsr_sddmm_test_suite;
using namespace bsr_sddmm_host_code;
using namespace sddmm_profiling_suite;

#define ESC "\033["
#define GREEN_TXT "118"
#define RED_TXT "196"
#define RESET "\033[m"

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
    SDDMMHostCodeFunctionPtr host_func,
    bsr_matrix<bfloat16>& mask,
    dense_matrix<bfloat16>& c,
    dense_matrix<bfloat16>& d,
    std::string& test_name) {

    // Device setup
    console_printf("Setting up the device!\n");

    constexpr int device_id = 0;
    IDevice* device = CreateDevice(device_id);

    // Matrix params
    uint32_t M = mask.H;
    uint32_t N = mask.W;
    uint32_t K = c.W;
    uint32_t R = mask.R;
    uint32_t C_block = mask.C;

    console_printf("Running SDDMM: mask(%zux%zu, %zu blocks) ⊙ (C(%zux%zu) × D(%zux%zu))\n",
                   mask.H, mask.W, mask.nblocks, c.H, c.W, d.H, d.W);

    // Run SDDMM via host code
    bsr_matrix<bfloat16> output;
    host_func(mask, c, d, output, M, N, K, R, C_block, 1, device);

    console_printf("SDDMM complete. Output: %zux%zu with %zu blocks\n",
                   output.H, output.W, output.nblocks);

    // Verify against CPU reference
    bsr_matrix<bfloat16> expected = mask.sddmm(c, d);

    // Convert both to dense for comparison
    dense_matrix<bfloat16> output_dense = output.to_dense();
    dense_matrix<bfloat16> expected_dense = expected.to_dense();

    float pcc = check_bfloat16_vector_pcc(output_dense.data, expected_dense.data);
    console_printf("PCC against CPU reference: %f\n", pcc);

    if (pcc > 0.99f || (output_dense.data.size() == 0 && expected_dense.data.size() == 0)) {
        console_printf(ESC "38;5;" GREEN_TXT "m" "PASS" RESET "\n");
    } else {
        console_printf(ESC "38;5;" RED_TXT "m" "FAIL (PCC = %f)" RESET "\n", pcc);
    }

    CloseDevice(device);
}

void run_full_test(int host_code_num, int test_num, TestFunctionPtr* registry) {
    auto [mask, c, d, test_name] = registry[test_num]();
    run_test(HostCodeRegistryVerbose[host_code_num].first, mask, c, d, test_name);

    console_printf("--------------------------------------------------------\n");
    console_printf("--- SDDMM Test results ---------------------------------\n");
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
    size_t num_tests = 0;
    int registry_number = argc > 3 ? std::stoi(argv[3]) : -1;
    std::string registry_name = "";
    TestFunctionPtr *Registry = nullptr;
    switch (registry_number) {
        case 0:
            Registry = reinterpret_cast<TestFunctionPtr*>(ProfileCaseRegistry);
            registry_name = "SDDMMProfileSuite";
            num_tests = sizeof(ProfileCaseRegistry) / sizeof(ProfileCaseRegistry[0]);
            break;
        default:
            Registry = TestRegistry;
            num_tests = sizeof(TestRegistry) / sizeof(TestRegistry[0]);
            break;
    }

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

        for (size_t i = 0; i < num_tests; i++) {
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
