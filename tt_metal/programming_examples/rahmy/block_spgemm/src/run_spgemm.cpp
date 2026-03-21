
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

using namespace bsr_spgemm_test_suite;
using namespace bsr_spgemm_host_code;
using namespace spgemm_profiling_suite;

#define ESC "\033["
#define GREEN_TXT "118"
#define RED_TXT "196"
#define RESET "\033[m"

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
    SpGEMMHostCodeFunctionPtr host_func,
    bsr_matrix<bfloat16>& a,
    bsr_matrix<bfloat16>& b,
    std::string& test_name) {

    // Device setup
    console_printf("Setting up the device!\n");

    constexpr int device_id = 0;
    IDevice* device = CreateDevice(device_id);

    // Matrix params
    uint32_t M = a.H;
    uint32_t K = a.W;
    uint32_t R = a.R;
    uint32_t C = a.C;

    console_printf("Running SpGEMM: A(%zux%zu, %zu blocks) x B(%zux%zu, %zu blocks)\n",
                   a.H, a.W, a.nblocks, b.H, b.W, b.nblocks);

    // Run SpGEMM via host code
    bsr_matrix<bfloat16> output;
    host_func(a, b, output, M, K, R, C, 1, device);

    console_printf("SpGEMM complete. Output: %zux%zu with %zu blocks\n",
                   output.H, output.W, output.nblocks);

    // Verify against CPU reference
    bsr_matrix<bfloat16> expected = a.spgemm(b);

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
    auto [a, b, test_name] = registry[test_num]();
    run_test(HostCodeRegistryVerbose[host_code_num].first, a, b, test_name);

    console_printf("--------------------------------------------------------\n");
    console_printf("--- SpGEMM Test results --------------------------------\n");
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
    // Registry selection
    int registry_number = argc > 3 ? std::stoi(argv[3]) : -1;
    std::string registry_name = "";
    TestFunctionPtr *Registry = nullptr;
    switch (registry_number) {
        case 0:
            Registry = reinterpret_cast<TestFunctionPtr*>(ProfileCaseRegistry);
            registry_name = "SpGEMMProfileSuite";
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
