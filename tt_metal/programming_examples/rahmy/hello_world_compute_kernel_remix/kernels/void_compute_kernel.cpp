

#include <unistd.h>

#include "debug/dprint.h"
#include "compute_kernel_api.h"

namespace NAMESPACE {

void MAIN {
    DPRINT_MATH(DPRINT << "Hello Host, I am runnning a void compute kernel and I am waiting for 5 seconds." << ENDL());

    // sleep(5);
}
}  // namespace NAMESPACE
