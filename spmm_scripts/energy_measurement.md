# Energy / Power Measurement

## What Are You Actually Measuring?

Different methods measure power at different points in the system. It's critical
to know what's included and excluded when reporting energy numbers.

```
                                    What each method captures
                                    ─────────────────────────
  Wall outlet
  │
  ├─ Wall meter (Kill-A-Watt)      Everything: PSU losses, fans, disks, CPU,
  │                                 DRAM, accelerator, motherboard, NICs, ...
  │
  └─ PSU
     │
     ├─ BMC / IPMI                  Total server input power (post-wall, pre-PSU
     │                              efficiency loss is already gone)
     │
     ├─ 12V rails → motherboard
     │   │
     │   ├─ CPU package             RAPL (Intel/AMD): package, cores, DRAM,
     │   │                          uncore — but NOT fans, disks, NICs
     │   │
     │   └─ Host DRAM               RAPL DRAM domain (Intel only, some SKUs)
     │
     └─ PCIe slot + aux power
         │
         └─ Accelerator card
             │
             ├─ On-board VRMs
             │   │
             │   ├─ ASIC core       ← tt-smi "power" (TDP register)
             │   │                    Wormhole: 1W resolution, core only
             │   │
             │   ├─ ASIC I/O
             │   │
             │   └─ On-card DRAM    (GDDR6, HBM, etc.) — may or may not be
             │                       included in the reported "board power"
             │
             └─ Fans, misc          NOT included in tt-smi
```

### What tt-smi measures

tt-smi reads the TDP register on the Wormhole ASIC. This is the **ASIC core
power only** — it does NOT include:
- PCIe bus power overhead
- On-card DRAM power (the MVDDQ_POWER register exists but is not parsed)
- On-card fans or VRM conversion losses
- Host CPU/DRAM consumed by the host-side driver and data movement
- PSU efficiency losses

For apples-to-apples comparison with other accelerators (e.g., nvidia-smi
reports **board power** including GPU + VRAM), keep this scope difference in
mind.

## Measurement Methods Overview

### 1. On-chip telemetry (software, no extra hardware)

| Tool | Platform | What it measures | Resolution | Sample rate |
|------|----------|-----------------|------------|-------------|
| **tt-smi / pyluwen** | Tenstorrent | ASIC core power | 1 W | ~10 kHz via pyluwen |
| **nvidia-smi** | NVIDIA GPU | Board power (GPU+VRAM) | ~1 W | ~20 Hz |
| **rocm-smi** | AMD GPU | Board power | ~1 W | varies |
| **RAPL** (`perf`, `likwid`, `powertop`) | Intel/AMD CPU | Package, cores, DRAM (separate domains) | ~1 mJ (energy counter) | µs-level counters |

These are the easiest to use but measure only a specific component, not the
whole system. RAPL deserves special mention: it provides **energy counters**
(not power), so you read the counter before and after your workload and subtract.
No sampling/integration needed.

### 2. BMC / IPMI (software, server-class machines)

```bash
ipmitool dcmi power reading     # instantaneous server power
ipmitool sdr list               # all sensor readings including PSU input power
```

Measures total server input power. Available on most server-class machines with
a BMC (e.g., Dell iDRAC, HPE iLO, Supermicro). Resolution is typically 1W,
sample rate ~1 Hz. Includes everything the server draws except the PDU and
upstream infrastructure.

### 3. Wall-plug power meters (external hardware)

| Device | What it measures | Resolution | Sample rate | Cost |
|--------|-----------------|------------|-------------|------|
| Kill-A-Watt | Entire system at wall | 0.1 W | ~1 Hz | ~$30 |
| Watts Up Pro | Entire system at wall | 0.1 W | ~1 Hz, USB logging | ~$100 |
| Yokogawa WT series | Entire system at wall | 0.01% | up to 100 kHz | $5k+ |

These measure **everything**: PSU efficiency losses (typically 80–95%), fans,
disks, NICs, host CPU/DRAM, and the accelerator. Best for total system energy
claims. The gap between wall power and component-level telemetry tells you
the overhead of the rest of the system.

### 4. PCB-level current sensing (hardware modification)

Shunt resistors or INA226/INA3221 current-sense ICs on specific power rails.
Can target individual voltage domains (core, I/O, DRAM) with mA-level
resolution at kHz sample rates. Requires board access and possibly soldering.
This is what board designers use during bring-up.

### 5. DAQ systems (lab-grade)

National Instruments DAQ, Keysight, etc. Measure voltage/current on specific
wires at up to MHz sample rates with µW resolution. Gold standard for
research-grade power characterization, but expensive ($1k–$50k+) and requires
physical instrumentation of the power delivery path.

### 6. PDU-level monitoring (data center)

Smart PDUs (e.g., Raritan, ServerTech) provide per-outlet power monitoring via
SNMP or web API. Measures entire server(s) at the rack level. Resolution ~1W,
sample rate ~1 Hz. Useful for fleet-level energy accounting, not per-kernel
measurement.

## Practical Guidance

**For comparing SpMM kernel variants on Tenstorrent:**
- tt-smi (pyluwen) is sufficient — you're comparing the *same chip* across
  algorithms, so the scope (core-only) is consistent
- 1W resolution is adequate when workloads draw 13–100W
- Always report idle power as a baseline so readers can compute dynamic energy

**For comparing Tenstorrent vs. other accelerators:**
- Use the same measurement point (ideally wall-plug or at least board-level)
- tt-smi core-only vs. nvidia-smi board-level is NOT a fair comparison
- If wall-plug isn't available, clearly state what's included/excluded

**For publication-quality results:**
- Report: measurement method, what's included, sample rate, idle baseline,
  ambient temperature, and whether the workload is steady-state or transient
- Consider multiple runs and report variance

---

## tt-smi Details

### tt-smi Power Telemetry

tt-smi exposes per-device: **power** (W), **voltage** (V), **current** (A), **temperature** (°C), plus TDP/TDC limits.
Wormhole n150 L reads ~13W at idle with a 100W TDP limit.

Key CLI commands:
```bash
tt-smi -s                    # JSON snapshot to stdout
tt-smi -s --snapshot_no_tty  # JSON snapshot without color codes
tt-smi -f output.json        # Save snapshot to file
```

## No Built-in Continuous Logging

tt-smi has no `--watch` or `--interval` mode. Each `tt-smi -s` invocation takes ~600ms (chip detection overhead), limiting to ~1.6 samples/sec.

## Approaches

### A: Shell loop with `tt-smi -s` (simple, slow ~1.6 Hz)

Good enough for long-running workloads but low resolution due to per-invocation overhead.

### B: Direct `pyluwen` Python script (recommended, fast)

The underlying `pyluwen` library reads telemetry in <0.05ms — you just pay the chip detection cost once (~600ms). 10 Hz sampling is plenty given the sensor's 1W resolution.

Steps:
1. Detect chip once via `tt_tools_common.utils_common.tools_utils.detect_chips_with_callback()`
2. Call `chip.as_wh().get_telemetry()` in a background thread at a fixed interval
3. Extract power as `telemetry.tdp & 0xFFFF` (integer watts)
4. Run workload as a subprocess
5. Trapezoidal integration of power samples over time → total energy in Joules

```
Energy (J) = Σ [(P[i] + P[i-1]) / 2 × Δt]
```

### Limitations

- 1W resolution from the firmware register — fine for comparing kernels at 13–100W, not micro-watt level
- Temperature is `(ASIC_TEMPERATURE & 0xFFFF) / 16` in °C for Wormhole

## Key Source Files

| File | Purpose |
|------|---------|
| `tt_smi/tt_smi_backend.py` | Telemetry parsing: `get_wh_chip_telemetry()`, `update_telem()` |
| `tt_smi/constants.py` | Field definitions: `TELEM_LIST`, `LIMITS`, `GUI_INTERVAL_TIME = 0.1` |
| `tt_smi/tt_smi.py` | GUI + CLI entry point |
