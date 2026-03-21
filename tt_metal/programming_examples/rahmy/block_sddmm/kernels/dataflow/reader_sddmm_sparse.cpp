// STUB: Reader kernel for the sparse sampling mask B in SDDMM.
//
// Will read BSR matrix B structure from DRAM, including:
//   - data blocks (tile data for each nonzero block — the mask values)
//   - indptr array (block row pointers)
//   - indices array (block column indices)
//
// The mask structure tells the compute kernel which blocks of C×D
// to compute. Since the output sparsity is known a priori, the reader
// can pre-determine the exact work schedule.

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    // STUB: To be implemented when device-side SDDMM algorithm is designed.
}
