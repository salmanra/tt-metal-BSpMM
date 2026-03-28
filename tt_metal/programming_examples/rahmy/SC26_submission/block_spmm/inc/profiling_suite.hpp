#pragma once

#include <cstdint>
#include <cmath>
#include <random>
#include "include_me.hpp"
#include "bsr_matrix.hpp"
#include "tt-metalium/bfloat16.hpp"
#include "tt-metalium/bfloat4.hpp"

using namespace tt;

using CoreSpec = std::variant<CoreCoord, CoreRange, CoreRangeSet>;

namespace profiling_suite {

    using ProfileCaseReturnType = std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string>;

    // ── Parametric case templates ──
    template <uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>
    ProfileCaseReturnType profile_case_parametric_random();
    template <uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>
    ProfileCaseReturnType profile_case_parametric_row();
    template <uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>
    ProfileCaseReturnType profile_case_parametric_col();
    template <uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>
    ProfileCaseReturnType profile_case_parametric_multi_diag();

    // ── Registry arrays ──
    using ProfileCaseFunctionPtr = ProfileCaseReturnType (*)();

    // Registry 0: Microbench d=25% — sweep block size, random pattern
    static ProfileCaseFunctionPtr MicrobenchD25Registry[] = {
        profile_case_parametric_random<8192, 8192, 8192,  32,  32, 25>,
        profile_case_parametric_random<8192, 8192, 8192,  64,  64, 25>,
        profile_case_parametric_random<8192, 8192, 8192, 128, 128, 25>,
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 25>,
    };

    // Registry 1: Microbench d=5% — sweep block size, random pattern
    static ProfileCaseFunctionPtr MicrobenchD5Registry[] = {
        profile_case_parametric_random<8192, 8192, 8192,  32,  32,  5>,
        profile_case_parametric_random<8192, 8192, 8192,  64,  64,  5>,
        profile_case_parametric_random<8192, 8192, 8192, 128, 128,  5>,
        profile_case_parametric_random<8192, 8192, 8192, 256, 256,  5>,
    };

    // Registry 2: Sparsity pattern sweep d=5% — R=C=256
    static ProfileCaseFunctionPtr PatternD5Registry[] = {
        profile_case_parametric_row       <8192, 8192, 8192, 256, 256,  5>,
        profile_case_parametric_col       <8192, 8192, 8192, 256, 256,  5>,
        profile_case_parametric_multi_diag<8192, 8192, 8192, 256, 256,  5>,
        profile_case_parametric_random    <8192, 8192, 8192, 256, 256,  5>,
    };

    // Registry 3: Sparsity pattern sweep d=10% — R=C=256
    static ProfileCaseFunctionPtr PatternD10Registry[] = {
        profile_case_parametric_row       <8192, 8192, 8192, 256, 256, 10>,
        profile_case_parametric_col       <8192, 8192, 8192, 256, 256, 10>,
        profile_case_parametric_multi_diag<8192, 8192, 8192, 256, 256, 10>,
        profile_case_parametric_random    <8192, 8192, 8192, 256, 256, 10>,
    };

    // Registry 4: Sparsity pattern sweep d=25% — R=C=256
    static ProfileCaseFunctionPtr PatternD25Registry[] = {
        profile_case_parametric_row       <8192, 8192, 8192, 256, 256, 25>,
        profile_case_parametric_col       <8192, 8192, 8192, 256, 256, 25>,
        profile_case_parametric_multi_diag<8192, 8192, 8192, 256, 256, 25>,
        profile_case_parametric_random    <8192, 8192, 8192, 256, 256, 25>,
    };

    // Registry 5: Sparsity pattern sweep d=50% — R=C=256
    static ProfileCaseFunctionPtr PatternD50Registry[] = {
        profile_case_parametric_row       <8192, 8192, 8192, 256, 256, 50>,
        profile_case_parametric_col       <8192, 8192, 8192, 256, 256, 50>,
        profile_case_parametric_multi_diag<8192, 8192, 8192, 256, 256, 50>,
        profile_case_parametric_random    <8192, 8192, 8192, 256, 256, 50>,
    };

    // Registry 6: Sweep N — M=K=8192, R=C=256, d=10%
    static ProfileCaseFunctionPtr SweepNRegistry[] = {
        profile_case_parametric_random<8192,  512, 8192, 256, 256, 10>,
        profile_case_parametric_random<8192, 1024, 8192, 256, 256, 10>,
        profile_case_parametric_random<8192, 2048, 8192, 256, 256, 10>,
        profile_case_parametric_random<8192, 4096, 8192, 256, 256, 10>,
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 10>,
    };

    // Registry 7: Sweep K — M=N=8192, R=C=256, d=10%
    static ProfileCaseFunctionPtr SweepKRegistry[] = {
        profile_case_parametric_random<8192, 8192,  512, 256, 256, 10>,
        profile_case_parametric_random<8192, 8192, 1024, 256, 256, 10>,
        profile_case_parametric_random<8192, 8192, 2048, 256, 256, 10>,
        profile_case_parametric_random<8192, 8192, 4096, 256, 256, 10>,
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 10>,
    };

    // Registry 8: Sweep block size — M=N=K=8192, d=10%
    static ProfileCaseFunctionPtr SweepBlockSizeRegistry[] = {
        profile_case_parametric_random<8192, 8192, 8192,  32,  32, 10>,
        profile_case_parametric_random<8192, 8192, 8192,  64,  64, 10>,
        profile_case_parametric_random<8192, 8192, 8192, 128, 128, 10>,
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 10>,
    };

    // Registry 9: Sweep density — M=N=K=8192, R=C=256
    static ProfileCaseFunctionPtr SweepDensityRegistry[] = {
        profile_case_parametric_random<8192, 8192, 8192, 256, 256,  5>,
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 10>,
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 25>,
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 50>,
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 75>,
    };

    // ── Registry metadata (sizes and names) ──
    static const int NUM_REGISTRIES = 10;
    static const int RegistrySizes[] = {4, 4, 4, 4, 4, 4, 5, 5, 4, 5};
    static const char* RegistryNames[] = {
        "MicrobenchD25", "MicrobenchD5",
        "PatternD5", "PatternD10", "PatternD25", "PatternD50",
        "SweepN", "SweepK", "SweepBlockSize", "SweepDensity",
    };
    static ProfileCaseFunctionPtr* Registries[] = {
        MicrobenchD25Registry, MicrobenchD5Registry,
        PatternD5Registry, PatternD10Registry, PatternD25Registry, PatternD50Registry,
        SweepNRegistry, SweepKRegistry, SweepBlockSizeRegistry, SweepDensityRegistry,
    };

    // ── Parametric case definitions ──

    template <uint32_t M = 8192, uint32_t N = 8192, uint32_t K = 8192,
              uint32_t R = 64, uint32_t C = 64, uint32_t DensityPercent = 25>
    inline ProfileCaseReturnType profile_case_parametric_random() {
        uint32_t block_matrix_height = M / R;
        uint32_t block_matrix_width  = K / C;
        constexpr float density = DensityPercent / 100.0f;
        uint32_t divisor = uint32_t(std::round(1.0 / density));
        uint32_t nblocks = (block_matrix_height * block_matrix_width) / divisor;

        bsr_matrix<float> bsr(M, K, R, C, nblocks, RAND);
        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16   = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        char buf[100];
        size_t n = sprintf(buf, "parametric_M%u_N%u_K%u_R%u_C%u_d%u", M, N, K, R, C, DensityPercent);
        std::string test_name(buf, n);
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, test_name);
    }

    template <uint32_t M = 8192, uint32_t N = 8192, uint32_t K = 8192,
              uint32_t R = 64, uint32_t C = 64, uint32_t DensityPercent = 25>
    inline ProfileCaseReturnType profile_case_parametric_row() {
        uint32_t block_matrix_height = M / R;
        uint32_t block_matrix_width  = K / C;
        constexpr float density = DensityPercent / 100.0f;
        uint32_t divisor = uint32_t(std::round(1.0 / density));
        uint32_t nblocks = (block_matrix_height * block_matrix_width) / divisor;

        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_ROW, RAND);
        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16   = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        char buf[100];
        size_t n = sprintf(buf, "parametric_row_M%u_N%u_K%u_R%u_C%u_d%u", M, N, K, R, C, DensityPercent);
        std::string test_name(buf, n);
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, test_name);
    }

    template <uint32_t M = 8192, uint32_t N = 8192, uint32_t K = 8192,
              uint32_t R = 64, uint32_t C = 64, uint32_t DensityPercent = 25>
    inline ProfileCaseReturnType profile_case_parametric_col() {
        uint32_t block_matrix_height = M / R;
        uint32_t block_matrix_width  = K / C;
        constexpr float density = DensityPercent / 100.0f;
        uint32_t divisor = uint32_t(std::round(1.0 / density));
        uint32_t nblocks = (block_matrix_height * block_matrix_width) / divisor;

        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_COL, RAND);
        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16   = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        char buf[100];
        size_t n = sprintf(buf, "parametric_col_M%u_N%u_K%u_R%u_C%u_d%u", M, N, K, R, C, DensityPercent);
        std::string test_name(buf, n);
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, test_name);
    }

    template <uint32_t M = 8192, uint32_t N = 8192, uint32_t K = 8192,
              uint32_t R = 64, uint32_t C = 64, uint32_t DensityPercent = 25>
    inline ProfileCaseReturnType profile_case_parametric_multi_diag() {
        uint32_t block_matrix_height = M / R;
        uint32_t block_matrix_width  = K / C;
        constexpr float density = DensityPercent / 100.0f;
        uint32_t divisor = uint32_t(std::round(1.0 / density));
        uint32_t nblocks = (block_matrix_height * block_matrix_width) / divisor;

        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_MULTI_DIAG, RAND);
        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16   = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        char buf[100];
        size_t n = sprintf(buf, "parametric_multi_diag_M%u_N%u_K%u_R%u_C%u_d%u", M, N, K, R, C, DensityPercent);
        std::string test_name(buf, n);
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, test_name);
    }

} // namespace profiling_suite
