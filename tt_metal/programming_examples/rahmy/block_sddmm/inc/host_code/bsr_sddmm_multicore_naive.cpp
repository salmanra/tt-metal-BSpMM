#include "../host_code.hpp"
#include "sddmm_zone_config.hpp"

namespace bsr_sddmm_host_code {

template<bool verbose, bool is_profiling>
void bsr_sddmm_multicore_naive_impl(
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
    IDevice* device,
    const std::map<std::string, std::string>& extra_defines = {}) {

    // STUB: For now, just run CPU SDDMM as the "device" implementation.
    //
    // The real implementation will need to:
    // 1. Transfer sampling mask structure (indptr, indices) + block data to device
    // 2. Transfer dense C and D to device
    // 3. For each nonzero block (i,j) in mask:
    //    - Compute dense block of C×D at position (i,j)
    //    - Element-wise multiply with mask block
    //    - Write result block to output
    //
    // Key advantage over SpGEMM: output sparsity pattern is known a priori,
    // so output buffers can be pre-allocated with exact sizes.

    if constexpr (verbose) {
        log_info(tt::LogVerif, "bsr_sddmm_multicore_naive: STUB - using CPU sddmm");
        log_info(tt::LogVerif, "Mask: {}x{} with {} nonzero blocks ({}x{})",
                 sampling_mask.H, sampling_mask.W, sampling_mask.nblocks,
                 sampling_mask.R, sampling_mask.C);
        log_info(tt::LogVerif, "C: {}x{} (dense)", c.H, c.W);
        log_info(tt::LogVerif, "D: {}x{} (dense)", d.H, d.W);
    }

    output = sampling_mask.sddmm(c, d);

    if constexpr (verbose) {
        log_info(tt::LogVerif, "Output: {}x{} with {} nonzero blocks",
                 output.H, output.W, output.nblocks);
    }
}

// Public wrapper
template<bool verbose, bool is_profiling>
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
    IDevice* device) {
    bsr_sddmm_multicore_naive_impl<verbose, is_profiling>(
        sampling_mask, c, d, output, M, N, K, R, C_block, B, device, {});
}

// Explicit template instantiations
template void bsr_sddmm_multicore_naive<false, false>(
    bsr_matrix<bfloat16>&, dense_matrix<bfloat16>&, dense_matrix<bfloat16>&,
    bsr_matrix<bfloat16>&,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, IDevice*);
template void bsr_sddmm_multicore_naive<true, false>(
    bsr_matrix<bfloat16>&, dense_matrix<bfloat16>&, dense_matrix<bfloat16>&,
    bsr_matrix<bfloat16>&,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, IDevice*);
template void bsr_sddmm_multicore_naive<false, true>(
    bsr_matrix<bfloat16>&, dense_matrix<bfloat16>&, dense_matrix<bfloat16>&,
    bsr_matrix<bfloat16>&,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, IDevice*);
template void bsr_sddmm_multicore_naive<true, true>(
    bsr_matrix<bfloat16>&, dense_matrix<bfloat16>&, dense_matrix<bfloat16>&,
    bsr_matrix<bfloat16>&,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, IDevice*);

} // namespace bsr_sddmm_host_code
