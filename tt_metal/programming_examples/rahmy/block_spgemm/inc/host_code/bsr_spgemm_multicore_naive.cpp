#include "../host_code.hpp"
#include "spgemm_zone_config.hpp"

namespace bsr_spgemm_host_code {

template<bool verbose, bool is_profiling>
void bsr_spgemm_multicore_naive_impl(
    bsr_matrix<bfloat16>& a,
    bsr_matrix<bfloat16>& b,
    bsr_matrix<bfloat16>& output,
    uint32_t M,
    uint32_t K,
    uint32_t R,
    uint32_t C,
    uint32_t B,
    IDevice* device,
    const std::map<std::string, std::string>& extra_defines = {}) {

    // STUB: For now, just run CPU SpGEMM as the "device" implementation.
    //
    // The real implementation will need to:
    // 1. Symbolic phase: compute output sparsity structure (which output blocks are nonzero)
    // 2. Numeric phase: compute the actual block values via device kernels
    // 3. Handle dynamic output allocation (output nnz unknown at launch)
    //
    // Possible approaches:
    //   a) Host-side symbolic pass + single device numeric pass
    //   b) Two device passes (symbolic + numeric)
    //   c) Over-allocate output buffer, compact on host after readback
    //   d) Row-by-row processing with dynamic L1 accumulation

    if constexpr (verbose) {
        log_info(tt::LogVerif, "bsr_spgemm_multicore_naive: STUB - using CPU spgemm");
        log_info(tt::LogVerif, "A: {}x{} with {} nonzero blocks ({}x{})", a.H, a.W, a.nblocks, a.R, a.C);
        log_info(tt::LogVerif, "B: {}x{} with {} nonzero blocks ({}x{})", b.H, b.W, b.nblocks, b.R, b.C);
    }

    output = a.spgemm(b);

    if constexpr (verbose) {
        log_info(tt::LogVerif, "Output: {}x{} with {} nonzero blocks", output.H, output.W, output.nblocks);
    }
}

// Public wrapper
template<bool verbose, bool is_profiling>
void bsr_spgemm_multicore_naive(
    bsr_matrix<bfloat16>& a,
    bsr_matrix<bfloat16>& b,
    bsr_matrix<bfloat16>& output,
    uint32_t M,
    uint32_t K,
    uint32_t R,
    uint32_t C,
    uint32_t B,
    IDevice* device) {
    bsr_spgemm_multicore_naive_impl<verbose, is_profiling>(a, b, output, M, K, R, C, B, device, {});
}

// Explicit template instantiations
template void bsr_spgemm_multicore_naive<false, false>(
    bsr_matrix<bfloat16>&, bsr_matrix<bfloat16>&, bsr_matrix<bfloat16>&,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, IDevice*);
template void bsr_spgemm_multicore_naive<true, false>(
    bsr_matrix<bfloat16>&, bsr_matrix<bfloat16>&, bsr_matrix<bfloat16>&,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, IDevice*);
template void bsr_spgemm_multicore_naive<false, true>(
    bsr_matrix<bfloat16>&, bsr_matrix<bfloat16>&, bsr_matrix<bfloat16>&,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, IDevice*);
template void bsr_spgemm_multicore_naive<true, true>(
    bsr_matrix<bfloat16>&, bsr_matrix<bfloat16>&, bsr_matrix<bfloat16>&,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, IDevice*);

} // namespace bsr_spgemm_host_code
