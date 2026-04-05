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

    // ── Parametric case templates (last param is DensityPPM: parts per million) ──
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
        profile_case_parametric_random<8192, 8192, 8192,  32,  32, 250000>,
        profile_case_parametric_random<8192, 8192, 8192,  64,  64, 250000>,
        profile_case_parametric_random<8192, 8192, 8192, 128, 128, 250000>,
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 250000>,
    };

    // Registry 1: Microbench d=5% — sweep block size, random pattern
    static ProfileCaseFunctionPtr MicrobenchD5Registry[] = {
        profile_case_parametric_random<8192, 8192, 8192,  32,  32,  50000>,
        profile_case_parametric_random<8192, 8192, 8192,  64,  64,  50000>,
        profile_case_parametric_random<8192, 8192, 8192, 128, 128,  50000>,
        profile_case_parametric_random<8192, 8192, 8192, 256, 256,  50000>,
    };

    // Registry 2: Sparsity pattern sweep d=5% — R=C=256
    static ProfileCaseFunctionPtr PatternD5Registry[] = {
        profile_case_parametric_row       <8192, 8192, 8192, 256, 256,  50000>,
        profile_case_parametric_col       <8192, 8192, 8192, 256, 256,  50000>,
        profile_case_parametric_multi_diag<8192, 8192, 8192, 256, 256,  50000>,
        profile_case_parametric_random    <8192, 8192, 8192, 256, 256,  50000>,
    };

    // Registry 3: Sparsity pattern sweep d=10% — R=C=256
    static ProfileCaseFunctionPtr PatternD10Registry[] = {
        profile_case_parametric_row       <8192, 8192, 8192, 256, 256, 100000>,
        profile_case_parametric_col       <8192, 8192, 8192, 256, 256, 100000>,
        profile_case_parametric_multi_diag<8192, 8192, 8192, 256, 256, 100000>,
        profile_case_parametric_random    <8192, 8192, 8192, 256, 256, 100000>,
    };

    // Registry 4: Sparsity pattern sweep d=25% — R=C=256
    static ProfileCaseFunctionPtr PatternD25Registry[] = {
        profile_case_parametric_row       <8192, 8192, 8192, 256, 256, 250000>,
        profile_case_parametric_col       <8192, 8192, 8192, 256, 256, 250000>,
        profile_case_parametric_multi_diag<8192, 8192, 8192, 256, 256, 250000>,
        profile_case_parametric_random    <8192, 8192, 8192, 256, 256, 250000>,
    };

    // Registry 5: Sparsity pattern sweep d=50% — R=C=256
    static ProfileCaseFunctionPtr PatternD50Registry[] = {
        profile_case_parametric_row       <8192, 8192, 8192, 256, 256, 500000>,
        profile_case_parametric_col       <8192, 8192, 8192, 256, 256, 500000>,
        profile_case_parametric_multi_diag<8192, 8192, 8192, 256, 256, 500000>,
        profile_case_parametric_random    <8192, 8192, 8192, 256, 256, 500000>,
    };

    // Registry 6: Sweep N — M=K=8192, R=C=256, d=10%
    static ProfileCaseFunctionPtr SweepNRegistry[] = {
        profile_case_parametric_random<8192,  512, 8192, 256, 256, 100000>,
        profile_case_parametric_random<8192, 1024, 8192, 256, 256, 100000>,
        profile_case_parametric_random<8192, 2048, 8192, 256, 256, 100000>,
        profile_case_parametric_random<8192, 4096, 8192, 256, 256, 100000>,
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 100000>,
    };

    // Registry 7: Sweep K — M=N=8192, R=C=256, d=10%
    static ProfileCaseFunctionPtr SweepKRegistry[] = {
        profile_case_parametric_random<8192, 8192,  512, 256, 256, 100000>,
        profile_case_parametric_random<8192, 8192, 1024, 256, 256, 100000>,
        profile_case_parametric_random<8192, 8192, 2048, 256, 256, 100000>,
        profile_case_parametric_random<8192, 8192, 4096, 256, 256, 100000>,
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 100000>,
    };

    // Registry 8: Sweep block size — M=N=K=8192, d=10%
    static ProfileCaseFunctionPtr SweepBlockSizeRegistry[] = {
        profile_case_parametric_random<8192, 8192, 8192,  32,  32, 100000>,
        profile_case_parametric_random<8192, 8192, 8192,  64,  64, 100000>,
        profile_case_parametric_random<8192, 8192, 8192, 128, 128, 100000>,
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 100000>,
    };

    // Registry 9: Sweep density — M=N=K=8192, R=C=256
    static ProfileCaseFunctionPtr SweepDensityRegistry[] = {
        profile_case_parametric_random<8192, 8192, 8192, 256, 256,  50000>,  //  5%
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 100000>,  // 10%
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 250000>,  // 25%
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 500000>,  // 50%
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 750000>,  // 75%
    };

    // Registry 10: Ultra-low density — M=N=K=8192, R=C=32, 30–10000 PPM
    static ProfileCaseFunctionPtr UltraLowDensity32Registry[] = {
        profile_case_parametric_random<8192, 8192, 8192, 32, 32,    30>,  //  0.003%  (~2 blocks / 65536)
        profile_case_parametric_random<8192, 8192, 8192, 32, 32,   100>,  //  0.01%   (~7 blocks)
        profile_case_parametric_random<8192, 8192, 8192, 32, 32,   300>,  //  0.03%   (~20 blocks)
        profile_case_parametric_random<8192, 8192, 8192, 32, 32,  1000>,  //  0.1%    (~66 blocks)
        profile_case_parametric_random<8192, 8192, 8192, 32, 32,  3000>,  //  0.3%    (~197 blocks)
        profile_case_parametric_random<8192, 8192, 8192, 32, 32, 10000>,  //  1.0%    (~655 blocks)
    };

    // Registry 11: Ultra-low density — M=N=K=8192, R=C=64, 60–10000 PPM
    static ProfileCaseFunctionPtr UltraLowDensity64Registry[] = {
        profile_case_parametric_random<8192, 8192, 8192, 64, 64,    60>,  //  0.006%  (~1 block / 16384)
        profile_case_parametric_random<8192, 8192, 8192, 64, 64,   200>,  //  0.02%   (~3 blocks)
        profile_case_parametric_random<8192, 8192, 8192, 64, 64,   600>,  //  0.06%   (~10 blocks)
        profile_case_parametric_random<8192, 8192, 8192, 64, 64,  2000>,  //  0.2%    (~33 blocks)
        profile_case_parametric_random<8192, 8192, 8192, 64, 64,  6000>,  //  0.6%    (~98 blocks)
        profile_case_parametric_random<8192, 8192, 8192, 64, 64, 10000>,  //  1.0%    (~164 blocks)
    };

    // ── Registry metadata (sizes and names) ──
    static const int NUM_REGISTRIES = 12;
    static const int RegistrySizes[] = {4, 4, 4, 4, 4, 4, 5, 5, 4, 5, 6, 6};
    static const char* RegistryNames[] = {
        "MicrobenchD25", "MicrobenchD5",
        "PatternD5", "PatternD10", "PatternD25", "PatternD50",
        "SweepN", "SweepK", "SweepBlockSize", "SweepDensity",
        "UltraLowDensity32", "UltraLowDensity64",
    };
    static ProfileCaseFunctionPtr* Registries[] = {
        MicrobenchD25Registry, MicrobenchD5Registry,
        PatternD5Registry, PatternD10Registry, PatternD25Registry, PatternD50Registry,
        SweepNRegistry, SweepKRegistry, SweepBlockSizeRegistry, SweepDensityRegistry,
        UltraLowDensity32Registry, UltraLowDensity64Registry,
    };

    // ── Parametric case definitions ──

    template <uint32_t M = 8192, uint32_t N = 8192, uint32_t K = 8192,
              uint32_t R = 64, uint32_t C = 64, uint32_t DensityPPM = 250000>
    inline ProfileCaseReturnType profile_case_parametric_random() {
        uint32_t block_matrix_height = M / R;
        uint32_t block_matrix_width  = K / C;
        constexpr float density = DensityPPM / 1000000.0f;
        uint32_t nblocks = std::max(1u, uint32_t(std::round(block_matrix_height * block_matrix_width * density)));

        bsr_matrix<float> bsr(M, K, R, C, nblocks, RAND);
        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16   = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        char buf[128];
        size_t n = sprintf(buf, "parametric_M%u_N%u_K%u_R%u_C%u_dppm%u", M, N, K, R, C, DensityPPM);
        std::string test_name(buf, n);
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, test_name);
    }

    template <uint32_t M = 8192, uint32_t N = 8192, uint32_t K = 8192,
              uint32_t R = 64, uint32_t C = 64, uint32_t DensityPPM = 250000>
    inline ProfileCaseReturnType profile_case_parametric_row() {
        uint32_t block_matrix_height = M / R;
        uint32_t block_matrix_width  = K / C;
        constexpr float density = DensityPPM / 1000000.0f;
        uint32_t nblocks = std::max(1u, uint32_t(std::round(block_matrix_height * block_matrix_width * density)));

        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_ROW, RAND);
        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16   = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        char buf[128];
        size_t n = sprintf(buf, "parametric_row_M%u_N%u_K%u_R%u_C%u_dppm%u", M, N, K, R, C, DensityPPM);
        std::string test_name(buf, n);
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, test_name);
    }

    template <uint32_t M = 8192, uint32_t N = 8192, uint32_t K = 8192,
              uint32_t R = 64, uint32_t C = 64, uint32_t DensityPPM = 250000>
    inline ProfileCaseReturnType profile_case_parametric_col() {
        uint32_t block_matrix_height = M / R;
        uint32_t block_matrix_width  = K / C;
        constexpr float density = DensityPPM / 1000000.0f;
        uint32_t nblocks = std::max(1u, uint32_t(std::round(block_matrix_height * block_matrix_width * density)));

        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_COL, RAND);
        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16   = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        char buf[128];
        size_t n = sprintf(buf, "parametric_col_M%u_N%u_K%u_R%u_C%u_dppm%u", M, N, K, R, C, DensityPPM);
        std::string test_name(buf, n);
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, test_name);
    }

    template <uint32_t M = 8192, uint32_t N = 8192, uint32_t K = 8192,
              uint32_t R = 64, uint32_t C = 64, uint32_t DensityPPM = 250000>
    inline ProfileCaseReturnType profile_case_parametric_multi_diag() {
        uint32_t block_matrix_height = M / R;
        uint32_t block_matrix_width  = K / C;
        constexpr float density = DensityPPM / 1000000.0f;
        uint32_t nblocks = std::max(1u, uint32_t(std::round(block_matrix_height * block_matrix_width * density)));

        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_MULTI_DIAG, RAND);
        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16   = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        char buf[128];
        size_t n = sprintf(buf, "parametric_multi_diag_M%u_N%u_K%u_R%u_C%u_dppm%u", M, N, K, R, C, DensityPPM);
        std::string test_name(buf, n);
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, test_name);
    }

} // namespace profiling_suite
