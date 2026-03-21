#pragma once

#include <cstdint>
#include <cmath>
#include <random>
#include "include_me.hpp"
#include "sparse_common/bsr_matrix.hpp"
#include "tt-metalium/bfloat16.hpp"

using namespace tt;

namespace sddmm_profiling_suite {

    using ProfileCaseReturnType = std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, dense_matrix<bfloat16>, std::string>;

    // Fully parametric SDDMM cases for sweeps.
    // Template params: M, N, K, R, C, DensityPercent (for the mask)
    // C and D are always dense.
    template <uint32_t M, uint32_t N, uint32_t K, uint32_t R, uint32_t Cb, uint32_t Density>
    ProfileCaseReturnType profile_case_parametric_random() {
        uint32_t blocked_h = M / R;
        uint32_t blocked_w = N / Cb;
        uint32_t nblocks = std::max(1u, (uint32_t)std::round(blocked_h * blocked_w * Density / 100.0));

        bsr_matrix<bfloat16> mask(M, N, R, Cb, nblocks, RAND);
        dense_matrix<bfloat16> c(M, K, RAND);
        dense_matrix<bfloat16> d(K, N, RAND);

        std::string name = "parametric_M" + std::to_string(M) + "_N" + std::to_string(N) +
                           "_K" + std::to_string(K) + "_R" + std::to_string(R) +
                           "_C" + std::to_string(Cb) + "_d" + std::to_string(Density);
        return {mask, c, d, name};
    }

    using ProfileCaseFunctionPtr = ProfileCaseReturnType (*)();

    // Minimal initial registry
    static ProfileCaseFunctionPtr ProfileCaseRegistry[] = {
        profile_case_parametric_random<1024, 1024, 512, 32, 32, 25>,
        profile_case_parametric_random<2048, 2048, 512, 64, 64, 25>,
        profile_case_parametric_random<1024, 1024, 256, 32, 32, 10>,
        profile_case_parametric_random<1024, 1024, 256, 32, 32, 50>,
    };

} // namespace sddmm_profiling_suite
