# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

TT-Metal (Metalium) is Tenstorrent's low-level programming framework for AI accelerator hardware (Grayskull, Wormhole, Blackhole). This repository is primarily used for **scientific kernel development using the TT-Metal layer**, programming Tensix cores directly with custom data movement and compute kernels.

**Primary Focus**: Writing high-performance scientific kernels at the TT-Metal level (C++ kernels running on RISC-V cores), not using the high-level TTNN library.

The repository contains:
- **TT-Metalium**: Low-level kernel development framework for programming Tensix cores directly
- **TT-NN (ttnn)**: High-level neural network operations library (less relevant for this work)
- **Model Demos**: Reference implementations primarily for benchmarking

## Essential Environment Setup

Before building or running code, you must set these environment variables:

```bash
export ARCH_NAME=<grayskull|wormhole_b0|blackhole>  # Match your hardware
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
```

**Critical**: Always activate the Python environment after building:
```bash
source python_env/bin/activate
```

## Build Commands

### Standard Build
```bash
./build_metal.sh                    # Release build with default options
./build_metal.sh --clean            # Clean all build artifacts
```

### Common Build Options for Kernel Development
```bash
./build_metal.sh --build-metal-tests          # Build TT-Metal C++ tests
./build_metal.sh --build-programming-examples # Build programming examples
./build_metal.sh --debug                      # Debug build (recommended for kernel dev)
./build_metal.sh --enable-profiler            # Enable Tracy profiler
./build_metal.sh -e                           # Export compile_commands.json
```

Build artifacts go to `build_<BuildType>/` (e.g., `build_Release/`). The `build/` symlink points to the active build directory.

### CMake Direct Usage
```bash
cmake -B build_Release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build_Release --target install
```

## Running Tests

### C++ Tests (Primary for Kernel Development)
```bash
# Build tests first
cmake --build build --target tests

# Run individual test binary
./build/test/tt_metal/test_matmul_single_core
./build/test/tt_metal/test_add_two_ints

# Google Test (gtest) binaries are in build/test/
./build/test/tt_metal/<test_binary>
```

### Python Tests (pytest)
Used mainly for higher-level integration testing:
```bash
# Single test file
pytest tests/tt_metal/test_custom_kernel.py -v

# Run with specific markers
pytest -m post_commit tests/
```

### Environment Variables for Kernel Development/Debugging
```bash
# Enable debug logging
export TT_METAL_LOGGER_LEVEL=Debug
export TT_METAL_LOGGER_TYPES=Op

# Watcher for hang detection during kernel development
export TT_METAL_WATCHER=1
export TT_METAL_WATCHER_INTERVAL=5000  # Check every 5s

# Device ID for this system
export TT_METAL_DEVICE_ID=5

# Set CPU performance governor (recommended for performance tests)
sudo cpupower frequency-set -g performance
```

## Architecture Overview for Kernel Development

### Directory Structure (Focus on TT-Metal)
```
tt-metal/
├── tt_metal/              # TT-Metalium: Core kernel framework
│   ├── hw/               # Hardware abstraction
│   │   ├── firmware/     # RISC-V firmware for Tensix cores
│   │   ├── ckernels/     # Compute kernel building blocks
│   │   └── inc/          # Hardware headers
│   ├── impl/             # Core implementation (buffers, programs, kernels)
│   ├── llrt/             # Low-level runtime
│   ├── jit_build/        # JIT compilation for kernels
│   ├── include/          # Kernel API headers
│   │   ├── compute_kernel_api/     # Compute kernel APIs
│   │   ├── dataflow_kernel_api/    # Data movement kernel APIs
│   │   └── debug/                  # Debug utilities (DPRINT, etc.)
│   ├── kernels/          # Example and utility kernels
│   └── programming_examples/  # Educational kernel examples
├── tests/tt_metal/       # TT-Metal kernel tests
├── ttnn/                 # High-level library (not primary focus)
├── models/demos/         # Reference models (for benchmarking)
├── tt_fabric/            # Multi-chip communication
└── infra/                # CI/CD infrastructure
```

### Tensix Core Architecture (Hardware You're Programming)

Each Tensix core contains:
- **5 RISC-V processors ("Baby RISCVs")**:
  - 1x Brisc: Data movement from DRAM
  - 1x Ncrisc: Data movement on NoC
  - 3x Trisc: Compute (Unpack, Math, Pack)
- **1 MB SRAM (L1)**: Local scratchpad memory
- **Matrix Engine (FPU)**: 32x32 tile matrix operations
- **Vector Engine (SFPU)**: Special functions (exp, sqrt, GELU, etc.)
- **Data Movement Engine**: NoC read/write, connected to 2 NoCs

**Programming Model**: You write bare-metal C/C++ kernels that run directly on these RISC-V cores. No OS, no threads - direct core-to-core mapping (1 kernel = 1 core).

### Kernel Types in TT-Metal

1. **Data Movement Kernels** (`dataflow_kernel_api`):
   - Run on Brisc/Ncrisc RISCVs
   - Move data between DRAM ↔ L1 ↔ other cores via NoC
   - Use `cb_push_back()`, `cb_reserve_back()`, `noc_async_read()`, `noc_async_write()`
   - File extension: typically `.cpp` in a `dataflow/` directory

2. **Compute Kernels** (`compute_kernel_api`):
   - Run on Trisc RISCVs (automatically split into Unpack, Math, Pack)
   - Operate on tiles in Circular Buffers
   - Use `matmul_tiles()`, `add_tiles()`, `pack_tile()`, `cb_wait_front()`, etc.
   - File extension: typically `.cpp` in a `compute/` directory

3. **Ethernet Kernels** (for multi-chip):
   - Data movement between chips via Ethernet cores
   - Less commonly used unless doing multi-chip scientific computing

**Key Insight**: Data movement and compute are **decoupled and explicit**. You manually orchestrate:
- When to fetch data from DRAM to L1
- When compute can start (via Circular Buffer synchronization)
- When to write results back

### Memory Hierarchy for Kernel Development

- **DRAM**: Off-chip GDDR6 memory
  - Accessed via `noc_async_read_dram()` / `noc_async_write_dram()`
  - Organized into banks; can be interleaved or contiguous

- **L1 (SRAM)**: 1 MB per Tensix core
  - Your primary workspace
  - Contains Circular Buffers (CBs) for data staging
  - Accessed directly by compute and data movement engines

- **Circular Buffers (CBs)**:
  - Fixed-size FIFO queues in L1
  - Connect data movement ↔ compute kernels
  - Indexed: `tt::CBIndex::c_0`, `c_1`, ..., `c_16` (output typically)
  - Producer-consumer synchronization via `cb_push_back()` / `cb_wait_front()`

### Tile-Based Compute

**Critical**: All operations work on **tiles** (32x32 matrices), not individual scalars.

- Matrix Engine operates on tiles of shape [32, 32]
- Data in L1 Circular Buffers organized as tiles
- For non-32-aligned data, use row-major layout or manual padding
- Tile size varies by datatype: bfloat16 tile = 2KB (32×32×2 bytes)

### Kernel Compilation Flow

1. **Write kernel**: C++ file using TT-Metal APIs
2. **JIT compilation**: TT-Metal compiles kernels at runtime for target arch
3. **RISC-V cross-compile**: Uses toolchain in `tt_metal/third_party/sfpi/compiler/`
4. **Artifacts**: Cached in `built/` directory (keyed by kernel hash)

To inspect compiled kernels:
```bash
# Find compiled artifacts
ls built/<device_id>/kernels/<kernel_name>/<hash>/

# View preprocessed compute kernel
# Use -DUCK_CHLKC_UNPACK/-MATH/-PACK to see individual RISC-V kernels
```

## Common Kernel Development Patterns

### Structure of a TT-Metal Program

```cpp
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"

// 1. Create device and program
Device *device = CreateDevice(0);
Program program = CreateProgram();

// 2. Define Circular Buffers (on device L1)
CircularBufferConfig cb_src0_config = CircularBufferConfig(num_tiles * tile_size, {{CB::c_in0, data_format}})
    .set_page_size(CB::c_in0, tile_size);
CreateCircularBuffer(program, core, cb_src0_config);

// 3. Create data movement kernel
KernelHandle reader_kernel = CreateKernel(
    program,
    "path/to/reader_kernel.cpp",
    core,
    DataMovementConfig{.processor = DataMovementProcessor::RISCV_0}
);

// 4. Create compute kernel
KernelHandle compute_kernel = CreateKernel(
    program,
    "path/to/compute_kernel.cpp",
    core,
    ComputeConfig{}
);

// 5. Set runtime arguments
SetRuntimeArgs(program, reader_kernel, core, {dram_addr, num_tiles});

// 6. Execute
EnqueueProgram(device->command_queue(), program, false);
Finish(device->command_queue());
```

### Typical Data Movement Kernel

```cpp
#include "dataflow_api.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_id = tt::CB::c_in0;

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_reserve_back(cb_id, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_id);

        noc_async_read_tile(i, src_addr, l1_write_addr);
        noc_async_read_barrier();

        cb_push_back(cb_id, 1);
    }
}
```

### Typical Compute Kernel

```cpp
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_in0 = tt::CB::c_in0;
    constexpr uint32_t cb_in1 = tt::CB::c_in1;
    constexpr uint32_t cb_out = tt::CB::c_16;

    mm_init();

    for (uint32_t i = 0; i < num_tiles; i++) {
        acquire_dst();

        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);

        matmul_tiles(cb_in0, cb_in1, 0, 0, 0, false);

        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);

        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);

        release_dst();
    }
}
}
```

### Debugging Kernels

**DPRINT**: Print from device kernels to host console
```cpp
#include "debug/dprint.h"

void kernel_main() {
    DPRINT << "Hello from core " << NOC_X << "," << NOC_Y << ENDL();
    DPRINT << "Value: " << HEX() << my_value << ENDL();
}
```

Enable with:
```bash
export TT_METAL_DPRINT_CORES=0,0  # Print from core (0,0)
export TT_METAL_DPRINT_CHIPS=5    # Print from chip 5 (our device ID)
```

**Watcher**: Detect hangs and show kernel state
```bash
export TT_METAL_WATCHER=1
export TT_METAL_WATCHER_INTERVAL=5000
```

**Device Reset**: If device hangs, reset it with:
```bash
tt-smi -tr 5  # Reset device ID 5
```

### Multi-Core Kernel Dispatch

Distribute work across multiple Tensix cores:

```cpp
// Define core range (e.g., 8x8 grid)
CoreRange core_range({0, 0}, {7, 7});

// Create kernel for all cores
KernelHandle kernel = CreateKernel(program, "kernel.cpp", core_range, config);

// Set per-core runtime args
for (uint32_t x = 0; x < 8; x++) {
    for (uint32_t y = 0; y < 8; y++) {
        CoreCoord core(x, y);
        SetRuntimeArgs(program, kernel, core, {core_specific_data});
    }
}
```

## Programming Examples Reference

Study these examples in `tt_metal/programming_examples/`:
- `hello_world_compute_kernel`: Basic compute kernel
- `hello_world_datamovement_kernel`: Basic data movement
- `matmul_single_core`: Single-core matmul
- `matmul_multi_core`: Multi-core matmul with work distribution
- `eltwise_binary`: Element-wise operations
- `dram_loopback`: DRAM ↔ L1 data movement

Build and run:
```bash
./build_metal.sh --build-programming-examples
./build/programming_examples/matmul_single_core
```

## Important API Headers

- `tt_metal/host_api.hpp`: Host-side API (CreateDevice, CreateProgram, etc.)
- `dataflow_api.h`: Data movement kernel APIs
- `compute_kernel_api/*.h`: Compute kernel APIs (matmul, eltwise, etc.)
- `debug/dprint.h`: Debug printing from kernels
- `tt_metal/common/constants.hpp`: Core coordinates, buffer IDs

## Common Issues in Kernel Development

**Kernel hang**: Check CB synchronization. Ensure producers push exactly what consumers wait for.

**NoC timeout**: Data movement taking too long. Check source addresses and use `noc_async_read_barrier()`.

**Tile alignment**: Ensure data is properly aligned. Use `#pragma pack` or intrinsics if needed.

**Wrong results**: Check compute kernel logic. Use DPRINT to inspect intermediate values.

**Old kernel cache**: Run `cmake --build build --target clean-built` to clear `built/` directory.

## Important Files

- `build_metal.sh`: Main build script
- `CMakeLists.txt`: Top-level build configuration
- `METALIUM_GUIDE.md`: Detailed architecture and programming guide
- `tt_metal/programming_examples/`: Educational examples
- `tech_reports/`: Technical deep-dives on architecture
- `CONTRIBUTING.md`: Development workflow

## Hardware-Specific Notes

**ARCH_NAME values**:
- `grayskull`: e150 cards
- `wormhole_b0`: n150, n300 cards, T3000 systems
- `blackhole`: Next-gen architecture

**Core Grid Sizes** (approximate, check device spec):
- Grayskull: 12x10 worker cores
- Wormhole: 8x8 worker cores (varies by config)

**NoC Configuration**: 2 NoCs per chip (NoC-0, NoC-1). Use for parallel data movement.
