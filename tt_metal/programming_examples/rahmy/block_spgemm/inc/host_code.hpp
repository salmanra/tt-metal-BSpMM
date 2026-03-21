#pragma once

#include <sys/types.h>
#include <algorithm>
#include <cstdint>

#include "include_me.hpp"
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

namespace bsr_spgemm_host_code {

// SpGEMM host code function declarations
// Signature: BSR x BSR -> BSR
// Key differences from SpMM:
//   - b is bsr_matrix (not dense_matrix)
//   - output is bsr_matrix (not dense_matrix)
//   - No N parameter (output width = b.W)
//   - No bcast_batch (not applicable to BSR x BSR)

template<bool verbose = false, bool is_profiling = false>
void bsr_spgemm_multicore_naive(
    bsr_matrix<bfloat16>& a,
    bsr_matrix<bfloat16>& b,
    bsr_matrix<bfloat16>& output,
    uint32_t M,
    uint32_t K,
    uint32_t R,
    uint32_t C,
    uint32_t B,
    IDevice* device);

// Function pointer type for SpGEMM host code
using SpGEMMHostCodeFunctionPtr = void (*)(
    bsr_matrix<bfloat16>& a,
    bsr_matrix<bfloat16>& b,
    bsr_matrix<bfloat16>& output,
    uint32_t M,
    uint32_t K,
    uint32_t R,
    uint32_t C,
    uint32_t B,
    IDevice* device);

// Registry: only one algorithm (naive) for now
static std::pair<SpGEMMHostCodeFunctionPtr, std::string> HostCodeRegistry[] = {
    {bsr_spgemm_multicore_naive<false, false>, "bsr_spgemm_multicore_naive"},
};

static std::pair<SpGEMMHostCodeFunctionPtr, std::string> HostCodeRegistryVerbose[] = {
    {bsr_spgemm_multicore_naive<true, false>, "bsr_spgemm_multicore_naive"},
};

static std::pair<SpGEMMHostCodeFunctionPtr, std::string> HostCodeRegistryProfiling[] = {
    {bsr_spgemm_multicore_naive<false, true>, "bsr_spgemm_multicore_naive"},
};

} // namespace bsr_spgemm_host_code
