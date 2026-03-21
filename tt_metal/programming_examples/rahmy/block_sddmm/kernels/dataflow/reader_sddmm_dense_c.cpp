// STUB: Reader kernel for dense matrix C in SDDMM.
//
// Will read rows of C from DRAM. For each nonzero block (i,j) in the
// sampling mask, we need rows [i*R, (i+1)*R) of C. Since multiple
// blocks in the same block row share the same C rows, there is an
// opportunity for data reuse (read C rows once per block row).

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    // STUB: To be implemented when device-side SDDMM algorithm is designed.
}
