# Energy Measurement with tt-smi

## tt-smi Power Telemetry

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
