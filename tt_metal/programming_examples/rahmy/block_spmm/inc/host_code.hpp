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

template<bool verbose = false, bool is_profiling = false, bool use_optimal_noc = true>
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

template<bool verbose = false, bool is_profiling = false, bool use_optimal_noc = true>
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

template<bool verbose = false, bool is_profiling = false, bool use_optimal_noc = true>
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

template<bool verbose = false, bool is_profiling = false, bool use_optimal_noc = true,
         bool in0_left_to_right = true, bool in1_bottom_to_top = true>
void bsr_spmm_multicore_snfin0_cdain1(
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

template<bool verbose = false, bool is_profiling = false, bool use_optimal_noc = true>
void bsr_spmm_multicore_naive_new_DM(
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

template<bool verbose = false, bool is_profiling = false, bool use_optimal_noc = true>
void bsr_spmm_multicore_load_balanced_new_DM(
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

// Ablation skip wrapper declarations (no_a_read, no_b_read, no_compute, no_write)
#define DECLARE_ABLATION_WRAPPERS(func_name) \
template<bool verbose = false, bool is_profiling = false, bool use_optimal_noc = true> \
void func_name##_no_a_read( \
    bsr_matrix<bfloat16>& a, dense_matrix<bfloat16>& b, dense_matrix<bfloat16>& output, \
    bool bcast_batch, uint32_t nnz_blocks, uint32_t M, uint32_t N, uint32_t K, \
    uint32_t R, uint32_t C, uint32_t B, IDevice* device); \
template<bool verbose = false, bool is_profiling = false, bool use_optimal_noc = true> \
void func_name##_no_b_read( \
    bsr_matrix<bfloat16>& a, dense_matrix<bfloat16>& b, dense_matrix<bfloat16>& output, \
    bool bcast_batch, uint32_t nnz_blocks, uint32_t M, uint32_t N, uint32_t K, \
    uint32_t R, uint32_t C, uint32_t B, IDevice* device); \
template<bool verbose = false, bool is_profiling = false, bool use_optimal_noc = true> \
void func_name##_no_compute( \
    bsr_matrix<bfloat16>& a, dense_matrix<bfloat16>& b, dense_matrix<bfloat16>& output, \
    bool bcast_batch, uint32_t nnz_blocks, uint32_t M, uint32_t N, uint32_t K, \
    uint32_t R, uint32_t C, uint32_t B, IDevice* device); \
template<bool verbose = false, bool is_profiling = false, bool use_optimal_noc = true> \
void func_name##_no_write( \
    bsr_matrix<bfloat16>& a, dense_matrix<bfloat16>& b, dense_matrix<bfloat16>& output, \
    bool bcast_batch, uint32_t nnz_blocks, uint32_t M, uint32_t N, uint32_t K, \
    uint32_t R, uint32_t C, uint32_t B, IDevice* device);

DECLARE_ABLATION_WRAPPERS(bsr_spmm_multicore_snf)
DECLARE_ABLATION_WRAPPERS(bsr_spmm_multicore_load_balanced)
DECLARE_ABLATION_WRAPPERS(bsr_spmm_multicore_reuse_iteration)
DECLARE_ABLATION_WRAPPERS(bsr_spmm_multicore_naive_new_DM)
DECLARE_ABLATION_WRAPPERS(bsr_spmm_multicore_load_balanced_new_DM)
#undef DECLARE_ABLATION_WRAPPERS

// 5-param version for snfin0_cdain1 (adds in0_left_to_right, in1_bottom_to_top)
#define DECLARE_ABLATION_WRAPPERS_5(func_name) \
template<bool verbose = false, bool is_profiling = false, bool use_optimal_noc = true, \
         bool in0_left_to_right = true, bool in1_bottom_to_top = true> \
void func_name##_no_a_read( \
    bsr_matrix<bfloat16>& a, dense_matrix<bfloat16>& b, dense_matrix<bfloat16>& output, \
    bool bcast_batch, uint32_t nnz_blocks, uint32_t M, uint32_t N, uint32_t K, \
    uint32_t R, uint32_t C, uint32_t B, IDevice* device); \
template<bool verbose = false, bool is_profiling = false, bool use_optimal_noc = true, \
         bool in0_left_to_right = true, bool in1_bottom_to_top = true> \
void func_name##_no_b_read( \
    bsr_matrix<bfloat16>& a, dense_matrix<bfloat16>& b, dense_matrix<bfloat16>& output, \
    bool bcast_batch, uint32_t nnz_blocks, uint32_t M, uint32_t N, uint32_t K, \
    uint32_t R, uint32_t C, uint32_t B, IDevice* device); \
template<bool verbose = false, bool is_profiling = false, bool use_optimal_noc = true, \
         bool in0_left_to_right = true, bool in1_bottom_to_top = true> \
void func_name##_no_compute( \
    bsr_matrix<bfloat16>& a, dense_matrix<bfloat16>& b, dense_matrix<bfloat16>& output, \
    bool bcast_batch, uint32_t nnz_blocks, uint32_t M, uint32_t N, uint32_t K, \
    uint32_t R, uint32_t C, uint32_t B, IDevice* device); \
template<bool verbose = false, bool is_profiling = false, bool use_optimal_noc = true, \
         bool in0_left_to_right = true, bool in1_bottom_to_top = true> \
void func_name##_no_write( \
    bsr_matrix<bfloat16>& a, dense_matrix<bfloat16>& b, dense_matrix<bfloat16>& output, \
    bool bcast_batch, uint32_t nnz_blocks, uint32_t M, uint32_t N, uint32_t K, \
    uint32_t R, uint32_t C, uint32_t B, IDevice* device);

DECLARE_ABLATION_WRAPPERS_5(bsr_spmm_multicore_snfin0_cdain1)

#undef DECLARE_ABLATION_WRAPPERS_5


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
    {bsr_spmm_multicore_snfin0_cdain1<false, false>, "bsr_spmm_multicore_snfin0_cdain1"},
    {bsr_spmm_multicore_load_balanced<false, false>, "bsr_spmm_multicore_load_balanced"},
    {bsr_spmm_multicore_reuse_iteration<false, false>, "bsr_spmm_multicore_reuse_iteration"},
    {bsr_spmm_multicore_naive_new_DM<false, false>, "bsr_spmm_multicore_naive_new_DM"},
    {bsr_spmm_multicore_load_balanced_new_DM<false, false>, "bsr_spmm_multicore_load_balanced_new_DM"},
};

static std::pair<HostCodeFunctionPtr, std::string> HostCodeRegistryVerbose[] = {
    {bsr_spmm_multicore_snf<true, false>, "bsr_spmm_multicore_snf"},
    {bsr_spmm_multicore_snfin0_cdain1<true, false>, "bsr_spmm_multicore_snfin0_cdain1"},
    {bsr_spmm_multicore_load_balanced<true, false>, "bsr_spmm_multicore_load_balanced"},
    {bsr_spmm_multicore_reuse_iteration<true, false>, "bsr_spmm_multicore_reuse_iteration"},
    {bsr_spmm_multicore_naive_new_DM<true, false>, "bsr_spmm_multicore_naive_new_DM"},
    {bsr_spmm_multicore_load_balanced_new_DM<true, false>, "bsr_spmm_multicore_load_balanced_new_DM"},
};

static std::pair<HostCodeFunctionPtr, std::string> HostCodeRegistryProfiling[] = {
    // [0-5] Full algorithms
    {bsr_spmm_multicore_snf<false, true>, "bsr_spmm_multicore_snf"},
    {bsr_spmm_multicore_load_balanced<false, true>, "bsr_spmm_multicore_load_balanced"},
    {bsr_spmm_multicore_reuse_iteration<false, true>, "bsr_spmm_multicore_reuse_iteration"},
    {bsr_spmm_multicore_naive_new_DM<false, true>, "bsr_spmm_multicore_naive_new_DM"},
    {bsr_spmm_multicore_load_balanced_new_DM<false, true>, "bsr_spmm_multicore_load_balanced_new_DM"},
    {bsr_spmm_multicore_snfin0_cdain1<false, true>, "bsr_spmm_multicore_snfin0_cdain1"},
    // [6-11] no_a_read ablations (SKIP_IN0_DRAM_READ=1)
    {bsr_spmm_multicore_snf_no_a_read<false, true>, "bsr_spmm_multicore_snf_no_a_read"},
    {bsr_spmm_multicore_load_balanced_no_a_read<false, true>, "bsr_spmm_multicore_load_balanced_no_a_read"},
    {bsr_spmm_multicore_reuse_iteration_no_a_read<false, true>, "bsr_spmm_multicore_reuse_iteration_no_a_read"},
    {bsr_spmm_multicore_naive_new_DM_no_a_read<false, true>, "bsr_spmm_multicore_naive_new_DM_no_a_read"},
    {bsr_spmm_multicore_load_balanced_new_DM_no_a_read<false, true>, "bsr_spmm_multicore_load_balanced_new_DM_no_a_read"},
    {bsr_spmm_multicore_snfin0_cdain1_no_a_read<false, true>, "bsr_spmm_multicore_snfin0_cdain1_no_a_read"},
    // [12-17] no_b_read ablations (SKIP_IN1_DRAM_READ=1)
    {bsr_spmm_multicore_snf_no_b_read<false, true>, "bsr_spmm_multicore_snf_no_b_read"},
    {bsr_spmm_multicore_load_balanced_no_b_read<false, true>, "bsr_spmm_multicore_load_balanced_no_b_read"},
    {bsr_spmm_multicore_reuse_iteration_no_b_read<false, true>, "bsr_spmm_multicore_reuse_iteration_no_b_read"},
    {bsr_spmm_multicore_naive_new_DM_no_b_read<false, true>, "bsr_spmm_multicore_naive_new_DM_no_b_read"},
    {bsr_spmm_multicore_load_balanced_new_DM_no_b_read<false, true>, "bsr_spmm_multicore_load_balanced_new_DM_no_b_read"},
    {bsr_spmm_multicore_snfin0_cdain1_no_b_read<false, true>, "bsr_spmm_multicore_snfin0_cdain1_no_b_read"},
    // [18-23] no_compute ablations (SKIP_COMPUTE=1)
    {bsr_spmm_multicore_snf_no_compute<false, true>, "bsr_spmm_multicore_snf_no_compute"},
    {bsr_spmm_multicore_load_balanced_no_compute<false, true>, "bsr_spmm_multicore_load_balanced_no_compute"},
    {bsr_spmm_multicore_reuse_iteration_no_compute<false, true>, "bsr_spmm_multicore_reuse_iteration_no_compute"},
    {bsr_spmm_multicore_naive_new_DM_no_compute<false, true>, "bsr_spmm_multicore_naive_new_DM_no_compute"},
    {bsr_spmm_multicore_load_balanced_new_DM_no_compute<false, true>, "bsr_spmm_multicore_load_balanced_new_DM_no_compute"},
    {bsr_spmm_multicore_snfin0_cdain1_no_compute<false, true>, "bsr_spmm_multicore_snfin0_cdain1_no_compute"},
    // [24-29] no_write ablations (SKIP_DRAM_WRITE=1)
    {bsr_spmm_multicore_snf_no_write<false, true>, "bsr_spmm_multicore_snf_no_write"},
    {bsr_spmm_multicore_load_balanced_no_write<false, true>, "bsr_spmm_multicore_load_balanced_no_write"},
    {bsr_spmm_multicore_reuse_iteration_no_write<false, true>, "bsr_spmm_multicore_reuse_iteration_no_write"},
    {bsr_spmm_multicore_naive_new_DM_no_write<false, true>, "bsr_spmm_multicore_naive_new_DM_no_write"},
    {bsr_spmm_multicore_load_balanced_new_DM_no_write<false, true>, "bsr_spmm_multicore_load_balanced_new_DM_no_write"},
    {bsr_spmm_multicore_snfin0_cdain1_no_write<false, true>, "bsr_spmm_multicore_snfin0_cdain1_no_write"},
    // [30-35] flip_noc full algorithms
    {bsr_spmm_multicore_snf<false, true, false>, "bsr_spmm_multicore_snf_flip_noc"},
    {bsr_spmm_multicore_load_balanced<false, true, false>, "bsr_spmm_multicore_load_balanced_flip_noc"},
    {bsr_spmm_multicore_reuse_iteration<false, true, false>, "bsr_spmm_multicore_reuse_iteration_flip_noc"},
    {bsr_spmm_multicore_naive_new_DM<false, true, false>, "bsr_spmm_multicore_naive_new_DM_flip_noc"},
    {bsr_spmm_multicore_load_balanced_new_DM<false, true, false>, "bsr_spmm_multicore_load_balanced_new_DM_flip_noc"},
    {bsr_spmm_multicore_snfin0_cdain1<false, true, false>, "bsr_spmm_multicore_snfin0_cdain1_flip_noc"},
    // [36-41] flip_noc no_a_read
    {bsr_spmm_multicore_snf_no_a_read<false, true, false>, "bsr_spmm_multicore_snf_no_a_read_flip_noc"},
    {bsr_spmm_multicore_load_balanced_no_a_read<false, true, false>, "bsr_spmm_multicore_load_balanced_no_a_read_flip_noc"},
    {bsr_spmm_multicore_reuse_iteration_no_a_read<false, true, false>, "bsr_spmm_multicore_reuse_iteration_no_a_read_flip_noc"},
    {bsr_spmm_multicore_naive_new_DM_no_a_read<false, true, false>, "bsr_spmm_multicore_naive_new_DM_no_a_read_flip_noc"},
    {bsr_spmm_multicore_load_balanced_new_DM_no_a_read<false, true, false>, "bsr_spmm_multicore_load_balanced_new_DM_no_a_read_flip_noc"},
    {bsr_spmm_multicore_snfin0_cdain1_no_a_read<false, true, false>, "bsr_spmm_multicore_snfin0_cdain1_no_a_read_flip_noc"},
    // [42-47] flip_noc no_b_read
    {bsr_spmm_multicore_snf_no_b_read<false, true, false>, "bsr_spmm_multicore_snf_no_b_read_flip_noc"},
    {bsr_spmm_multicore_load_balanced_no_b_read<false, true, false>, "bsr_spmm_multicore_load_balanced_no_b_read_flip_noc"},
    {bsr_spmm_multicore_reuse_iteration_no_b_read<false, true, false>, "bsr_spmm_multicore_reuse_iteration_no_b_read_flip_noc"},
    {bsr_spmm_multicore_naive_new_DM_no_b_read<false, true, false>, "bsr_spmm_multicore_naive_new_DM_no_b_read_flip_noc"},
    {bsr_spmm_multicore_load_balanced_new_DM_no_b_read<false, true, false>, "bsr_spmm_multicore_load_balanced_new_DM_no_b_read_flip_noc"},
    {bsr_spmm_multicore_snfin0_cdain1_no_b_read<false, true, false>, "bsr_spmm_multicore_snfin0_cdain1_no_b_read_flip_noc"},
    // [48-53] flip_noc no_compute
    {bsr_spmm_multicore_snf_no_compute<false, true, false>, "bsr_spmm_multicore_snf_no_compute_flip_noc"},
    {bsr_spmm_multicore_load_balanced_no_compute<false, true, false>, "bsr_spmm_multicore_load_balanced_no_compute_flip_noc"},
    {bsr_spmm_multicore_reuse_iteration_no_compute<false, true, false>, "bsr_spmm_multicore_reuse_iteration_no_compute_flip_noc"},
    {bsr_spmm_multicore_naive_new_DM_no_compute<false, true, false>, "bsr_spmm_multicore_naive_new_DM_no_compute_flip_noc"},
    {bsr_spmm_multicore_load_balanced_new_DM_no_compute<false, true, false>, "bsr_spmm_multicore_load_balanced_new_DM_no_compute_flip_noc"},
    {bsr_spmm_multicore_snfin0_cdain1_no_compute<false, true, false>, "bsr_spmm_multicore_snfin0_cdain1_no_compute_flip_noc"},
    // [54-59] flip_noc no_write
    {bsr_spmm_multicore_snf_no_write<false, true, false>, "bsr_spmm_multicore_snf_no_write_flip_noc"},
    {bsr_spmm_multicore_load_balanced_no_write<false, true, false>, "bsr_spmm_multicore_load_balanced_no_write_flip_noc"},
    {bsr_spmm_multicore_reuse_iteration_no_write<false, true, false>, "bsr_spmm_multicore_reuse_iteration_no_write_flip_noc"},
    {bsr_spmm_multicore_naive_new_DM_no_write<false, true, false>, "bsr_spmm_multicore_naive_new_DM_no_write_flip_noc"},
    {bsr_spmm_multicore_load_balanced_new_DM_no_write<false, true, false>, "bsr_spmm_multicore_load_balanced_new_DM_no_write_flip_noc"},
    {bsr_spmm_multicore_snfin0_cdain1_no_write<false, true, false>, "bsr_spmm_multicore_snfin0_cdain1_no_write_flip_noc"},
};

// Direction sweep registry: all 4 in0×in1 direction combos × 2 NoC configs
//                                                                         verbose prof opt_noc L2R   B2T
static std::pair<HostCodeFunctionPtr, std::string> HostCodeRegistryDirectionSweep[] = {
    // optimal NoC
    {bsr_spmm_multicore_snfin0_cdain1<false, true, true, true, true>,   "snfin0_cdain1_L2R_B2T"},
    {bsr_spmm_multicore_snfin0_cdain1<false, true, true, true, false>,  "snfin0_cdain1_L2R_T2B"},
    {bsr_spmm_multicore_snfin0_cdain1<false, true, true, false, true>,  "snfin0_cdain1_R2L_B2T"},
    {bsr_spmm_multicore_snfin0_cdain1<false, true, true, false, false>, "snfin0_cdain1_R2L_T2B"},
    // flip NoC
    {bsr_spmm_multicore_snfin0_cdain1<false, true, false, true, true>,   "snfin0_cdain1_L2R_B2T_flip_noc"},
    {bsr_spmm_multicore_snfin0_cdain1<false, true, false, true, false>,  "snfin0_cdain1_L2R_T2B_flip_noc"},
    {bsr_spmm_multicore_snfin0_cdain1<false, true, false, false, true>,  "snfin0_cdain1_R2L_B2T_flip_noc"},
    {bsr_spmm_multicore_snfin0_cdain1<false, true, false, false, false>, "snfin0_cdain1_R2L_T2B_flip_noc"},
};

// Direction sweep registry (verbose, non-profiling): all 4 in0×in1 direction combos
//                                                                            verbose prof opt_noc L2R   B2T
static std::pair<HostCodeFunctionPtr, std::string> HostCodeRegistryDirectionSweepVerbose[] = {
    {bsr_spmm_multicore_snfin0_cdain1<true, false, true, true, true>,   "snfin0_cdain1_L2R_B2T"}, // 0
    {bsr_spmm_multicore_snfin0_cdain1<true, false, true, true, false>,  "snfin0_cdain1_L2R_T2B"}, // 1
    {bsr_spmm_multicore_snfin0_cdain1<true, false, true, false, true>,  "snfin0_cdain1_R2L_B2T"}, // 2
    {bsr_spmm_multicore_snfin0_cdain1<true, false, true, false, false>, "snfin0_cdain1_R2L_T2B"}, // 3
    // flip NoC
    {bsr_spmm_multicore_snfin0_cdain1<true, false, false, true, true>,   "snfin0_cdain1_L2R_B2T_flip_noc"}, // 4
    {bsr_spmm_multicore_snfin0_cdain1<true, false, false, true, false>,  "snfin0_cdain1_L2R_T2B_flip_noc"}, // 5
    {bsr_spmm_multicore_snfin0_cdain1<true, false, false, false, true>,  "snfin0_cdain1_R2L_B2T_flip_noc"}, // 6
    {bsr_spmm_multicore_snfin0_cdain1<true, false, false, false, false>, "snfin0_cdain1_R2L_T2B_flip_noc"}, // 7
};

CoreCoord clamped_prev(const std::vector<CoreCoord>& order, uint32_t index);

CoreCoord clamped_next(const std::vector<CoreCoord>& order, uint32_t index);

std::shared_ptr<Buffer> MakeBuffer(IDevice* device, uint32_t size, uint32_t page_size, bool sram = false);

std::shared_ptr<Buffer> MakeBuffer(IDevice* device, uint32_t n_tiles, size_t element_size, bool sram = false);

CBHandle MakeCircularBuffer(
    Program& program, const CoreSpec& core, tt::CBIndex cb, uint32_t size, uint32_t page_size, tt::DataFormat format);

CBHandle MakeCircularBufferFP32(Program& program, const CoreSpec& core, tt::CBIndex cb, uint32_t n_tiles);


uint32_t _get_maximum_block_dim_with_NoC_args(int32_t block_dim, int32_t in0_block_w, int32_t num_tiles_in_NoC_args);

uint32_t get_Npc_from_BSR_block_size(uint32_t Nt, uint32_t Mpc, uint32_t in0_block_w, uint32_t num_cores_x, uint32_t num_cores_y, uint32_t num_tiles_for_indexing, uint32_t nnz_rows);

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
