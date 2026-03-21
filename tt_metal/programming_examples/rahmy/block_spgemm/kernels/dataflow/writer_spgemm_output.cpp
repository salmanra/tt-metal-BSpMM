// STUB: Writer kernel for SpGEMM output.
//
// Will write output BSR blocks to DRAM.
//
// Key challenge: the output sparsity pattern is not known at kernel launch time.
// Options to explore:
//   a) Pre-allocated dense output buffer, compact to BSR on host after readback
//   b) Over-sized BSR output buffer with a count written by the kernel
//   c) Two-pass: symbolic pass determines structure, numeric pass fills values

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    // STUB: To be implemented when device-side SpGEMM algorithm is designed.
}
