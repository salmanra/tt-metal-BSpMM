// STUB: Reader kernel for sparse matrix A in SpGEMM.
//
// Will read BSR matrix A blocks from DRAM, including:
//   - data blocks (tile data for each nonzero block)
//   - indptr array (block row pointers)
//   - indices array (block column indices)
//
// The reader must coordinate with the compute kernel to stream
// blocks in the correct order for the row-by-row SpGEMM pattern.

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    // STUB: To be implemented when device-side SpGEMM algorithm is designed.
}
