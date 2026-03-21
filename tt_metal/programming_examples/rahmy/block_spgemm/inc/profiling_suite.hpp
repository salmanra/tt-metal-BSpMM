#pragma once

#include <cstdint>
#include <cmath>
#include <random>
#include "include_me.hpp"
#include "sparse_common/bsr_matrix.hpp"
#include "tt-metalium/bfloat16.hpp"

using namespace tt;

namespace spgemm_profiling_suite {

    using ProfileCaseReturnType = std::tuple<bsr_matrix<bfloat16>, bsr_matrix<bfloat16>, std::string>;

    // Fully parametric sparse cases for sweeps.
    // Template params: M, K, R, C, DensityPercentA, DensityPercentB
    template <uint32_t M, uint32_t K, uint32_t R, uint32_t C, uint32_t DensityA, uint32_t DensityB>
    ProfileCaseReturnType profile_case_parametric_random() {
        uint32_t blocked_h_a = M / R;
        uint32_t blocked_w_a = K / C;
        uint32_t nblocks_a = std::max(1u, (uint32_t)std::round(blocked_h_a * blocked_w_a * DensityA / 100.0));

        uint32_t blocked_h_b = K / R;
        uint32_t blocked_w_b = M / C;  // square output for simplicity
        uint32_t nblocks_b = std::max(1u, (uint32_t)std::round(blocked_h_b * blocked_w_b * DensityB / 100.0));

        bsr_matrix<bfloat16> a(M, K, R, C, nblocks_a, RAND);
        bsr_matrix<bfloat16> b(K, M, R, C, nblocks_b, RAND);

        std::string name = "parametric_M" + std::to_string(M) + "_K" + std::to_string(K) +
                           "_R" + std::to_string(R) + "_C" + std::to_string(C) +
                           "_dA" + std::to_string(DensityA) + "_dB" + std::to_string(DensityB);
        return {a, b, name};
    }

    using ProfileCaseFunctionPtr = ProfileCaseReturnType (*)();

    // Minimal initial registry
    static ProfileCaseFunctionPtr ProfileCaseRegistry[] = {
        profile_case_parametric_random<1024, 1024, 32, 32, 25, 25>,
        profile_case_parametric_random<2048, 2048, 64, 64, 25, 25>,
        profile_case_parametric_random<1024, 1024, 32, 32, 10, 10>,
        profile_case_parametric_random<1024, 1024, 32, 32, 50, 50>,
    };

} // namespace spgemm_profiling_suite
