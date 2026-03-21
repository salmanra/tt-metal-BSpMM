// STUB: Writer kernel for SDDMM output.
//
// Will write output BSR blocks to DRAM. Unlike SpGEMM, the output
// sparsity pattern is known a priori (same as the sampling mask B),
// so output buffers can be pre-allocated with exact sizes. This
// simplifies the writer significantly compared to SpGEMM.

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    // STUB: To be implemented when device-side SDDMM algorithm is designed.
}
