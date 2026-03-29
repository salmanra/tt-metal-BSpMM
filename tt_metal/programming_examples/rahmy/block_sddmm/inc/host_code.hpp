#pragma once

#include <sys/types.h>
#include <algorithm>
#include <cstdint>

#include "include_me.hpp"
#include "sparse_common/host_code_utils.hpp"
#include "tt-metalium/assert.hpp"
#include "tt-metalium/bfloat16.hpp"
#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/host_api.hpp"

// profiler includes
#include <system_error>
#include <tracy/Tracy.hpp>
#include <common/TracyColor.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
#include "hostdevcommon/profiler_common.h"

using namespace tt;
using namespace tt::tt_metal;

using CoreSpec = std::variant<CoreCoord, CoreRange, CoreRangeSet>;

namespace bsr_sddmm_host_code {

using sparse_common::MakeBuffer;
using sparse_common::MakeCircularBuffer;

// SDDMM host code function declarations
// Signature: A = B ⊙ (C × D)
//   B: sparse BSR sampling mask (M×N)
//   C: dense (M×K)
//   D: dense (K×N)
//   A: sparse BSR output (M×N, same sparsity pattern as B)
//
// Key properties:
//   - Output sparsity pattern is known a priori (same as B)
//   - Only compute entries of C×D where B is nonzero
//   - Has both N and K params (C is M×K, D is K×N)

template<bool verbose = false, bool is_profiling = false>
void bsr_sddmm_multicore_naive(
    bsr_matrix<bfloat16>& sampling_mask,
    dense_matrix<bfloat16>& c,
    dense_matrix<bfloat16>& d,
    bsr_matrix<bfloat16>& output,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t R,
    uint32_t C_block,
    uint32_t B,
    IDevice* device);

template<bool verbose = false, bool is_profiling = false>
void bsr_sddmm_multicore_CDA(
    bsr_matrix<bfloat16>& sampling_mask,
    dense_matrix<bfloat16>& c,
    dense_matrix<bfloat16>& d,
    bsr_matrix<bfloat16>& output,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t R,
    uint32_t C_block,
    uint32_t B,
    IDevice* device);

// Function pointer type for SDDMM host code
using SDDMMHostCodeFunctionPtr = void (*)(
    bsr_matrix<bfloat16>& sampling_mask,
    dense_matrix<bfloat16>& c,
    dense_matrix<bfloat16>& d,
    bsr_matrix<bfloat16>& output,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t R,
    uint32_t C_block,
    uint32_t B,
    IDevice* device);

static std::pair<SDDMMHostCodeFunctionPtr, std::string> HostCodeRegistry[] = {
    {bsr_sddmm_multicore_naive<false, false>, "bsr_sddmm_multicore_naive"},
    {bsr_sddmm_multicore_CDA<false, false>, "bsr_sddmm_multicore_CDA"},
};

static std::pair<SDDMMHostCodeFunctionPtr, std::string> HostCodeRegistryVerbose[] = {
    {bsr_sddmm_multicore_naive<true, false>, "bsr_sddmm_multicore_naive"},
    {bsr_sddmm_multicore_CDA<true, false>, "bsr_sddmm_multicore_CDA"},
};

static std::pair<SDDMMHostCodeFunctionPtr, std::string> HostCodeRegistryProfiling[] = {
    {bsr_sddmm_multicore_naive<false, true>, "bsr_sddmm_multicore_naive"},
    {bsr_sddmm_multicore_CDA<false, true>, "bsr_sddmm_multicore_CDA"},
};

} // namespace bsr_sddmm_host_code
