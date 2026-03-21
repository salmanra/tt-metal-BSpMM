// STUB: Reader kernel for dense matrix D in SDDMM.
//
// Will read columns of D from DRAM. For each nonzero block (i,j) in
// the sampling mask, we need columns [j*C, (j+1)*C) of D. Multiple
// blocks in the same block column share the same D columns.

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    // STUB: To be implemented when device-side SDDMM algorithm is designed.
}
