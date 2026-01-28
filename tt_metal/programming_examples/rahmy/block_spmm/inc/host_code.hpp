#pragma once

#include <sys/types.h>
#include <algorithm>
#include <cstdint>

#include "include_me.hpp"
#include "tt-metalium/assert.hpp"
#include "tt-metalium/bfloat16.hpp"
// #include "tt-metalium/buffer_constants.hpp"
// #include "tt-metalium/circular_buffer_types.hpp"
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

namespace bsr_host_code {

// list of host code function declarations
template<bool verbose = false, bool is_profiling = false>
void bsr_spmm_multicore_reuse(
    bsr_matrix<bfloat16>& a,
    dense_matrix<bfloat16>& b,
    dense_matrix<bfloat16>& output,
    bool bcast_batch,
    uint32_t nnz_blocks,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t R,
    uint32_t C,
    uint32_t B,
    IDevice* device);

template<bool verbose = false, bool is_profiling = false>
void bsr_spmm_multicore_reuse_naive(
    bsr_matrix<bfloat16>& a,
    dense_matrix<bfloat16>& b,
    dense_matrix<bfloat16>& output,
    bool bcast_batch,
    uint32_t nnz_blocks,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t R,
    uint32_t C,
    uint32_t B,
    IDevice* device);

template<bool verbose = false, bool is_profiling = false>
void bsr_spmm_multicore_reuse_many_blocks_per_core(
    bsr_matrix<bfloat16>& a,
    dense_matrix<bfloat16>& b,
    dense_matrix<bfloat16>& output,
    bool bcast_batch,
    uint32_t nnz_blocks,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t R,
    uint32_t C,
    uint32_t B,
    IDevice* device);

template<bool verbose = false, bool is_profiling = false>
void bsr_spmm_multicore_reuse_iteration(
    bsr_matrix<bfloat16>& a,
    dense_matrix<bfloat16>& b,
    dense_matrix<bfloat16>& output,
    bool bcast_batch,
    uint32_t nnz_blocks,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t R,
    uint32_t C,
    uint32_t B,
    IDevice* device);

template<bool verbose = false, bool is_profiling = false>
void bsr_spmm_multicore_load_balanced(
    bsr_matrix<bfloat16>& a,
    dense_matrix<bfloat16>& b,
    dense_matrix<bfloat16>& output,
    bool bcast_batch,
    uint32_t nnz_blocks,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t R,
    uint32_t C,
    uint32_t B,
    IDevice* device);

// TEST: reuse host code with iter device code (iters set to 1)
template<bool verbose = false, bool is_profiling = false>
void bsr_spmm_multicore_host_reuse_device_iter(
    bsr_matrix<bfloat16>& a,
    dense_matrix<bfloat16>& b,
    dense_matrix<bfloat16>& output,
    bool bcast_batch,
    uint32_t nnz_blocks,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t R,
    uint32_t C,
    uint32_t B,
    IDevice* device);

template<bool verbose = false, bool is_profiling = false>
void bsr_spmm_multicore_sparse_mcast(
    bsr_matrix<bfloat16>& a,
    dense_matrix<bfloat16>& b,
    dense_matrix<bfloat16>& output,
    bool bcast_batch,
    uint32_t nnz_blocks,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t R,
    uint32_t C,
    uint32_t B,
    IDevice* device);

template<bool verbose = false, bool is_profiling = false>
void bsr_spmm_multicore_snf(
    bsr_matrix<bfloat16>& a,
    dense_matrix<bfloat16>& b,
    dense_matrix<bfloat16>& output,
    bool bcast_batch,
    uint32_t nnz_blocks,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t R,
    uint32_t C,
    uint32_t B,
    IDevice* device);


using HostCodeFunctionPtr = void (*)(
    bsr_matrix<bfloat16>& a,
    dense_matrix<bfloat16>& b,
    dense_matrix<bfloat16>& output,
    bool bcast_batch,
    uint32_t nnz_blocks,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t R,
    uint32_t C,
    uint32_t B,
    IDevice* device);


static std::pair<HostCodeFunctionPtr, std::string> HostCodeRegistry[] = {
    {bsr_spmm_multicore_snf<false, false>, "bsr_spmm_multicore_snf"},
    // {bsr_spmm_multicore_sparse_mcast<false, false>, "bsr_spmm_multicore_sparse_mcast"},
    {bsr_spmm_multicore_load_balanced<false, false>, "bsr_spmm_multicore_load_balanced"},
    {bsr_spmm_multicore_reuse_iteration<false, false>, "bsr_spmm_multicore_reuse_iteration"},
    // {bsr_spmm_multicore_reuse_many_blocks_per_core<false, false>, "bsr_spmm_multicore_reuse_many_blocks_per_core"}, // Defunct!
    {bsr_spmm_multicore_reuse<false, false>, "bsr_spmm_multicore_reuse"},
    {bsr_spmm_multicore_reuse_naive<false, false>, "bsr_spmm_multicore_reuse_naive"},
    // {bsr_spmm_multicore_host_reuse_device_iter<false, false>, "bsr_spmm_multicore_host_reuse_device_iter"} // TEST
};

static std::pair<HostCodeFunctionPtr, std::string> HostCodeRegistryVerbose[] = {
    {bsr_spmm_multicore_snf<true, false>, "bsr_spmm_multicore_snf"},
    // {bsr_spmm_multicore_sparse_mcast<true, false>, "bsr_spmm_multicore_sparse_mcast"},
    {bsr_spmm_multicore_load_balanced<true, false>, "bsr_spmm_multicore_load_balanced"},
    {bsr_spmm_multicore_reuse_iteration<true, false>, "bsr_spmm_multicore_reuse_iteration"},
    // {bsr_spmm_multicore_reuse_many_blocks_per_core<true, false>, "bsr_spmm_multicore_reuse_many_blocks_per_core"}, // Defunct!
    {bsr_spmm_multicore_reuse<true, false>, "bsr_spmm_multicore_reuse"},
    {bsr_spmm_multicore_reuse_naive<true, false>, "bsr_spmm_multicore_reuse_naive"},
    // {bsr_spmm_multicore_host_reuse_device_iter<true, false>, "bsr_spmm_multicore_host_reuse_device_iter"} // TEST
};

static std::pair<HostCodeFunctionPtr, std::string> HostCodeRegistryProfiling[] = {
    {bsr_spmm_multicore_snf<false, true>, "bsr_spmm_multicore_snf"},
    // {bsr_spmm_multicore_sparse_mcast<false, true>, "bsr_spmm_multicore_sparse_mcast"},
    {bsr_spmm_multicore_load_balanced<false, true>, "bsr_spmm_multicore_load_balanced"},
    {bsr_spmm_multicore_reuse_iteration<false, true>, "bsr_spmm_multicore_reuse_iteration"},
    // {bsr_spmm_multicore_reuse_many_blocks_per_core<false, true>, "bsr_spmm_multicore_reuse_many_blocks_per_core"}, // Defunct!
    {bsr_spmm_multicore_reuse<false, true>, "bsr_spmm_multicore_reuse"},
    {bsr_spmm_multicore_reuse_naive<false, true>, "bsr_spmm_multicore_reuse_naive"},
    // {bsr_spmm_multicore_host_reuse_device_iter<false, true>, "bsr_spmm_multicore_host_reuse_device_iter"} // TEST
};

CoreCoord clamped_prev(const std::vector<CoreCoord>& order, uint32_t index);

CoreCoord clamped_next(const std::vector<CoreCoord>& order, uint32_t index);

std::shared_ptr<Buffer> MakeBuffer(IDevice* device, uint32_t size, uint32_t page_size, bool sram = false);

std::shared_ptr<Buffer> MakeBuffer(IDevice* device, uint32_t n_tiles, size_t element_size, bool sram = false);

CBHandle MakeCircularBuffer(
    Program& program, const CoreSpec& core, tt::CBIndex cb, uint32_t size, uint32_t page_size, tt::DataFormat format);

CBHandle MakeCircularBufferFP32(Program& program, const CoreSpec& core, tt::CBIndex cb, uint32_t n_tiles);


uint32_t _get_maximum_block_dim_with_NoC_args(int32_t block_dim, int32_t in0_block_w, int32_t num_tiles_in_NoC_args);

uint32_t get_Npc_from_BSR_block_size(uint32_t Nt, uint32_t Mpc, uint32_t in0_block_w, uint32_t num_cores_x, uint32_t num_tiles_for_indexing);

template<class Vals>
void sortingPermutation(const Vals& values, std::vector<int>& v){
    int size = values.size();
    v.clear(); v.reserve(size);
    for(int i=0; i < size; ++i)
        v.push_back(i);

    std::sort(v.begin(), v.end(), [&values](int a, int b) -> bool {
        return values[a] > values[b];
    });
}
}

namespace dense_host_code {

    void matmul_multicore_reuse(
        std::vector<bfloat16>& a,
        std::vector<bfloat16>& b,
        std::vector<bfloat16>& output,
        bool bcast_batch, uint32_t M, uint32_t N, uint32_t K, uint32_t B, IDevice* device);

    void matmul_multicore_reuse_mcast(
        std::vector<bfloat16>& a,
        std::vector<bfloat16>& b,
        std::vector<bfloat16>& output,
        bool bcast_batch, uint32_t M, uint32_t N, uint32_t K, uint32_t B, IDevice* device);

    using DenseHostCodeFunctionPtr = void (*)(
        std::vector<bfloat16>& a,
        std::vector<bfloat16>& b,
        std::vector<bfloat16>& output,
        bool bcast_batch,
        uint32_t M,
        uint32_t N,
        uint32_t K,
        uint32_t B,
        IDevice* device);

}
