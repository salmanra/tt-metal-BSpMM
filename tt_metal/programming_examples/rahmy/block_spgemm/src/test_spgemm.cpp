
#include <cstdio>
#include <string>
#include "../inc/include_me.hpp"
#include "../inc/test_suite.hpp"
#include "../inc/host_code.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;

using namespace bsr_spgemm_test_suite;
using namespace bsr_spgemm_host_code;

// Quick CPU-only test harness for SpGEMM.
// Runs all test cases through the CPU stub and verifies correctness.
int main(int argc, char** argv) {
    size_t num_tests = sizeof(TestRegistry) / sizeof(TestRegistry[0]);
    int num_passed = 0;
    int num_failed = 0;

    for (size_t i = 0; i < num_tests; i++) {
        auto [a, b, test_name] = TestRegistry[i]();
        printf("Test %zu: %s ... ", i, test_name.c_str());

        // Run CPU SpGEMM
        bsr_matrix<bfloat16> result = a.spgemm(b);

        // Basic sanity checks
        bool pass = true;
        if (result.H != a.H) {
            printf("FAIL (output H=%zu, expected %zu)\n", result.H, a.H);
            pass = false;
        }
        if (result.W != b.W) {
            printf("FAIL (output W=%zu, expected %zu)\n", result.W, b.W);
            pass = false;
        }
        if (result.R != a.R) {
            printf("FAIL (output R=%zu, expected %zu)\n", result.R, a.R);
            pass = false;
        }
        if (result.C != b.C) {
            printf("FAIL (output C=%zu, expected %zu)\n", result.C, b.C);
            pass = false;
        }

        // Verify indptr is monotonically non-decreasing
        for (size_t j = 1; j < result.indptr.size(); j++) {
            if (result.indptr[j] < result.indptr[j-1]) {
                printf("FAIL (indptr not monotonic at index %zu)\n", j);
                pass = false;
                break;
            }
        }

        // Verify nblocks matches indptr
        if (result.indptr.size() > 0 && result.indptr.back() != (int)result.nblocks) {
            printf("FAIL (indptr.back()=%d != nblocks=%zu)\n", result.indptr.back(), result.nblocks);
            pass = false;
        }

        if (pass) {
            printf("PASS (%zu output blocks)\n", result.nblocks);
            num_passed++;
        } else {
            num_failed++;
        }
    }

    printf("\n--- SpGEMM CPU Test Summary ---\n");
    printf("Passed: %d / %zu\n", num_passed, num_tests);
    if (num_failed > 0) {
        printf("Failed: %d\n", num_failed);
    }
    return num_failed > 0 ? 1 : 0;
}
