
#include <cstdio>
#include <string>
#include "../inc/include_me.hpp"
#include "../inc/test_suite.hpp"
#include "../inc/host_code.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;

using namespace bsr_sddmm_test_suite;
using namespace bsr_sddmm_host_code;

// CPU-only test harness for SDDMM.
// Runs all test cases through the CPU sddmm() and verifies correctness.
int main(int argc, char** argv) {
    size_t num_tests = sizeof(TestRegistry) / sizeof(TestRegistry[0]);
    int num_passed = 0;
    int num_failed = 0;

    for (size_t i = 0; i < num_tests; i++) {
        auto [mask, c, d, test_name] = TestRegistry[i]();
        printf("Test %zu: %s ... ", i, test_name.c_str());

        // Run CPU SDDMM
        bsr_matrix<bfloat16> result = mask.sddmm(c, d);

        // Sanity checks
        bool pass = true;

        // Output must have same dimensions as mask
        if (result.H != mask.H) {
            printf("FAIL (output H=%zu, expected %zu)\n", result.H, mask.H);
            pass = false;
        }
        if (result.W != mask.W) {
            printf("FAIL (output W=%zu, expected %zu)\n", result.W, mask.W);
            pass = false;
        }
        if (result.R != mask.R) {
            printf("FAIL (output R=%zu, expected %zu)\n", result.R, mask.R);
            pass = false;
        }
        if (result.C != mask.C) {
            printf("FAIL (output C=%zu, expected %zu)\n", result.C, mask.C);
            pass = false;
        }

        // Output must have same sparsity pattern as mask
        if (result.nblocks != mask.nblocks) {
            printf("FAIL (output nblocks=%zu, expected %zu)\n", result.nblocks, mask.nblocks);
            pass = false;
        }

        // indptr must match exactly
        if (result.indptr != mask.indptr) {
            printf("FAIL (indptr doesn't match mask)\n");
            pass = false;
        }

        // indices must match exactly
        if (result.indices != mask.indices) {
            printf("FAIL (indices doesn't match mask)\n");
            pass = false;
        }

        // Verify data size
        if (result.data.size() != mask.nblocks * mask.R * mask.C) {
            printf("FAIL (data size=%zu, expected %zu)\n",
                   result.data.size(), mask.nblocks * mask.R * mask.C);
            pass = false;
        }

        if (pass) {
            printf("PASS (%zu output blocks, sparsity pattern matches)\n", result.nblocks);
            num_passed++;
        } else {
            num_failed++;
        }
    }

    printf("\n--- SDDMM CPU Test Summary ---\n");
    printf("Passed: %d / %zu\n", num_passed, num_tests);
    if (num_failed > 0) {
        printf("Failed: %d\n", num_failed);
    }
    return num_failed > 0 ? 1 : 0;
}
