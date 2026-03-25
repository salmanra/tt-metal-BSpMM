# DRAM Bank Mapping: Which DRAM Tile Services a `noc_async_read_tile` Call?

**Yes, it's fully deterministic.** Given a tile ID, you can compute exactly which DRAM tile on the NOC grid will fetch the data. Here's the complete picture.

## Wormhole Physical Layout

| Level | Count | What it is |
|-------|-------|-----------|
| GDDR6 chips | 6 | Physical chips (2 on left edge, 4 on right edge of NOC grid) |
| NOC tiles per chip | 3 | Each chip has 3 bridge tiles on the NOC grid (**18 total**) |
| DRAM views (= banks) | 12 | Each chip split into 2 x 1GB halves; this is what software sees as `NUM_DRAM_BANKS` |

Each GDDR6 chip provides 2GB. Three NOC tiles share access to the same chip, but software
routes worker-initiated reads through a **single preferred tile per view**, chosen by the
SoC descriptor.

## The 18 DRAM Tiles on the NOC Grid

From `wormhole_b0_80_arch.yaml:11-19` and `umd/.../wormhole_implementation.hpp:130-141`:

| Channel (chip) | NOC Tile 0 | NOC Tile 1 | NOC Tile 2 |
|----------------|-----------|-----------|-----------|
| 0 | (0, 0) | (0, 1) | (0, 11) |
| 1 | (0, 5) | (0, 6) | (0, 7) |
| 2 | (5, 0) | (5, 1) | (5, 11) |
| 3 | (5, 2) | (5, 9) | (5, 10) |
| 4 | (5, 3) | (5, 4) | (5, 8) |
| 5 | (5, 5) | (5, 6) | (5, 7) |

All DRAM tiles sit on the left edge (x=0) and right edge (x=5) of the 10x12 grid.
Within each chip, the 3 tiles are physically adjacent (or near-adjacent) on the same edge.

## The 12 DRAM Views (Banks)

From `wormhole_b0_80_arch.yaml:25-99`. Each view selects one chip half (0 or 1GB offset)
and one preferred NOC tile per chip:

| Bank | Channel | Address Offset | Worker Endpoint [noc0, noc1] | Resolved NOC Tile (noc0) |
|------|---------|---------------|------------------------------|--------------------------|
| 0 | 0 | 0 | [2, 2] | **(0, 11)** |
| 1 | 0 | 1 GB | [1, 1] | **(0, 1)** |
| 2 | 1 | 0 | [0, 0] | **(0, 5)** |
| 3 | 1 | 1 GB | [2, 2] | **(0, 7)** |
| 4 | 2 | 0 | [1, 1] | **(5, 1)** |
| 5 | 2 | 1 GB | [2, 2] | **(5, 11)** |
| 6 | 3 | 0 | [0, 0] | **(5, 2)** |
| 7 | 3 | 1 GB | [1, 1] | **(5, 9)** |
| 8 | 4 | 0 | [2, 2] | **(5, 8)** |
| 9 | 4 | 1 GB | [0, 0] | **(5, 3)** |
| 10 | 5 | 0 | [0, 0] | **(5, 5)** |
| 11 | 5 | 1 GB | [2, 2] | **(5, 7)** |

The "Worker Endpoint" is a subchannel index into the chip's 3-tile array. For example,
bank 0 uses endpoint [2, 2], which indexes into channel 0's tiles `[(0,0), (0,1), (0,11)]`
at position 2, giving **(0, 11)**.

Of the 18 physical DRAM tiles, **12 are used** as preferred worker endpoints (one per view).
The remaining tiles may serve ethernet traffic or provide alternative NOC routing.

## The Mapping: tile_id -> DRAM Tile

From `dataflow_api_addrgen.h:16-51`:

```
bank_index        = tile_id % 12          // round-robin across 12 views
bank_offset_index = tile_id / 12          // which slot within that bank
noc_xy            = dram_bank_to_noc_xy[noc][bank_index]   // lookup table
addr              = tile_size * bank_offset_index
                  + bank_base_address
                  + bank_to_dram_offset[bank_index]         // 0 or 1GB
noc_addr          = (noc_xy << 32) | addr
```

**Examples**:
- `tile_id = 0`  -> `0 % 12 = 0`  -> bank 0 -> channel 0, 0GB half -> NOC tile **(0, 11)**
- `tile_id = 25` -> `25 % 12 = 1` -> bank 1 -> channel 0, 1GB half -> NOC tile **(0, 1)**
- `tile_id = 14` -> `14 % 12 = 2` -> bank 2 -> channel 1, 0GB half -> NOC tile **(0, 5)**

## What You Need to Determine It

To determine which DRAM tile services any read, you need just two things:

1. **The tile ID** (first argument to `noc_async_read_tile`)
2. **`tile_id % 12`** indexes into the bank table above

That's it -- no runtime state, no dynamic routing. The mapping is baked into the
`dram_bank_to_noc_xy` lookup table that gets written to every Tensix core's L1 at
device init (`metal_context.cpp:717-789`).

## How the Lookup Tables Get to the Device

1. **Host computes tables** in `MetalContext::generate_device_bank_to_noc_tables()`
   (`metal_context.cpp:717-765`). For each bank, it calls
   `soc_d.get_preferred_worker_core_for_dram_view(channel, noc)` to resolve the
   physical NOC coordinates, then packs them into `uint16_t` entries.

2. **Host writes tables to L1** in `initialize_device_bank_to_noc_tables()`
   (`metal_context.cpp:780-789`), storing them at a scratch address
   (`MEM_BANK_TO_NOC_SCRATCH`) on each Tensix core.

3. **Firmware copies to local memory** via `noc_bank_table_init()` at boot
   (`firmware_common.h:42-52`), populating the `dram_bank_to_noc_xy` and
   `bank_to_dram_offset` arrays that `InterleavedAddrGenFast::get_noc_addr()` reads.

## File References

| What | File |
|------|------|
| Round-robin bank math | `tt_metal/hw/inc/dataflow_api_addrgen.h:16-51` |
| `get_noc_addr()` full path | `tt_metal/hw/inc/dataflow_api_addrgen.h:350-358` |
| Lookup table declarations | `tt_metal/hw/inc/dataflow_api_common.h:17-18` |
| Host populates tables from SoC descriptor | `tt_metal/impl/context/metal_context.cpp:717-765` |
| Host writes tables to device L1 | `tt_metal/impl/context/metal_context.cpp:780-789` |
| SoC descriptor resolves view -> NOC tile | `tt_metal/common/metal_soc_descriptor.cpp:15-20` |
| `NUM_DRAM_BANKS` JIT-compiled as `get_num_dram_views()` | `tt_metal/jit_build/build_env_manager.cpp:60,79` |
| DRAM tile coordinates + 12 view definitions | `tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml:11-99` |
| UMD constants (6 banks x 3 ports) | `umd/.../wormhole_implementation.hpp:130-141` |
| Physical layout context | [corsix.org/content/tt-wh-part1](https://www.corsix.org/content/tt-wh-part1) |
