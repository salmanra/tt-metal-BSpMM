// STUB: Compute kernel for SpGEMM block multiply.
//
// Performs: output[i,j] += A[i,k] * B[k,j] at the tile level.
//
// The compute pattern is the same as SpMM's bmm_iter at the individual
// block-multiply level: for each pair of matching blocks, do a tiled
// matmul and accumulate into the output block's DST register.
//
// The key difference is the outer iteration: instead of iterating over
// a dense column of B, we iterate over the sparse intersection of
// A's column indices and B's row indices.

#include <cstdint>
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"

namespace NAMESPACE {
void MAIN {
    // STUB: To be implemented when device-side SpGEMM algorithm is designed.
}
} // namespace NAMESPACE
