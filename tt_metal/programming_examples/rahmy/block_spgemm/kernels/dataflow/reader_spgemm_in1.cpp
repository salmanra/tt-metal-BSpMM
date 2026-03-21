// STUB: Reader kernel for sparse matrix B in SpGEMM.
//
// Will read BSR matrix B blocks from DRAM, including:
//   - data blocks (tile data for each nonzero block)
//   - indptr array (block row pointers)
//   - indices array (block column indices)
//
// Key difference from SpMM: B is also sparse, so we need to look up
// which blocks of B exist for a given block row (determined by the
// column index of the current A block).

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    // STUB: To be implemented when device-side SpGEMM algorithm is designed.
}
