#!/usr/bin/bash
# run_profiling.sh - Orchestrate the SC26 SpMM profiling suite
#
# Usage:
#   ./run_profiling.sh [--phase <microbench|throughput|scaling|lowdensity|all>]
#                      [--no-zones]
#                      [--no-build]
#                      [--dry-run]
#                      [--export-only]
#                      [--profile-only]
#                      [--list]
#
# Phases:
#   microbench  - All 15 host codes (3 algos × 5 variants) against MicrobenchD25/D5.
#                 Registries 0-1, host codes 0-14.  (120 runs)
#   throughput  - 3 base algorithms against 4 pattern registries.
#                 Registries 2-5, host codes 0-2.   (48 runs)
#   scaling     - DDA only against 4 sweep registries.
#                 Registries 6-9, host code 2.       (19 runs)
#   lowdensity  - 3 base algorithms against ultra-low density registries.
#                 Registries 10-11, host codes 0-2.  (36 runs)
#   all         - All four phases.                   (223 runs total)
#
# Options:
#   --with-zones    Enable device profiling zones (PROFILE_* env vars = 1; off by default)
#   --no-build      Skip the build step
#   --dry-run       Print commands without running them
#   --export-only   Only run export_to_csv (skip profiling)
#   --profile-only  Only run profile_block (skip export)
#   --list          List all registries and host codes, then exit
#
# Host code index map (HostCodeRegistryProfiling, 15 entries):
#   [0]  bsr_spmm_multicore_naive                  (full)
#   [1]  bsr_spmm_multicore_snf_in0_naive_in1      (full)
#   [2]  bsr_spmm_multicore_snf_in0_dda_in1        (full)
#   [3-5]   no_a_read ablations
#   [6-8]   no_b_read ablations
#   [9-11]  no_compute ablations
#   [12-14] no_write ablations
#
# Registry index map (profiling_suite.hpp, 12 registries):
#   0:  MicrobenchD25 (4)     6:  SweepN (5)
#   1:  MicrobenchD5 (4)      7:  SweepK (5)
#   2:  PatternD5 (4)         8:  SweepBlockSize (4)
#   3:  PatternD10 (4)        9:  SweepDensity (5)
#   4:  PatternD25 (4)        10: UltraLowDensity32 (6)
#   5:  PatternD50 (4)        11: UltraLowDensity64 (6)

set -euo pipefail

TT_METAL_DIR="/home/user/tt-metal"
HOST_CODE_HPP="$TT_METAL_DIR/tt_metal/programming_examples/rahmy/block_spmm/inc/host_code.hpp"
PROFILING_SUITE_HPP="$TT_METAL_DIR/tt_metal/programming_examples/rahmy/block_spmm/inc/profiling_suite.hpp"

PROFILE_BIN="$TT_METAL_DIR/build/programming_examples/rahmy/profile_block"
EXPORT_BIN="$TT_METAL_DIR/build/programming_examples/rahmy/export_to_csv"

# Registry display names — indices match profiling_suite.hpp
REGISTRY_NAMES=(
    "MicrobenchD25" "MicrobenchD5"
    "PatternD5" "PatternD10" "PatternD25" "PatternD50"
    "SweepN" "SweepK" "SweepBlockSize" "SweepDensity"
    "UltraLowDensity32" "UltraLowDensity64"
)
REGISTRY_SIZES=(4 4 4 4 4 4 5 5 4 5 6 6)
NUM_REGISTRIES=12

###############################################################################
# Registry parsing
###############################################################################

function parse_registry {
    local file="$1"
    local array_name="$2"
    local in_array=0

    while IFS= read -r line; do
        if [[ $in_array -eq 0 ]]; then
            [[ "$line" =~ ^[[:space:]]*//.* ]] && continue
            if [[ "$line" == *"${array_name}[]"* ]]; then
                in_array=1
            fi
            continue
        fi
        [[ "$line" == *"};" ]] && break
        [[ "$line" =~ ^[[:space:]]*//.* ]] && continue
        [[ -z "${line// /}" ]] && continue
        if [[ "$line" =~ \"([^\"]+)\" ]]; then
            echo "${BASH_REMATCH[1]}"
        fi
    done < "$file"
}

function read_host_codes {
    local -n arr_ref=$1
    arr_ref=()
    while IFS= read -r entry; do
        arr_ref+=("$entry")
    done < <(parse_registry "$HOST_CODE_HPP" "HostCodeRegistryProfiling")
}

###############################################################################
# Display
###############################################################################

function list_plan {
    echo "=== HostCodeRegistryProfiling ==="
    local hc_entries=()
    read_host_codes hc_entries
    local i=0
    for entry in "${hc_entries[@]}"; do
        local group=""
        if   (( i >= 0  && i <= 2  )); then group="[full]"
        elif (( i >= 3  && i <= 5  )); then group="[no_a_read]"
        elif (( i >= 6  && i <= 8  )); then group="[no_b_read]"
        elif (( i >= 9  && i <= 11 )); then group="[no_compute]"
        elif (( i >= 12 && i <= 14 )); then group="[no_write]"
        fi
        printf "  [%2d] %-12s %s\n" "$i" "$group" "$entry"
        i=$(( i + 1 ))
    done
    echo ""

    for (( reg=0; reg<NUM_REGISTRIES; reg++ )); do
        echo "=== Registry $reg: ${REGISTRY_NAMES[$reg]} (${REGISTRY_SIZES[$reg]} cases) ==="
    done
    echo ""

    echo "=== Phase breakdown ==="
    echo "  microbench:  regs 0-1,   hc 0-14  (${REGISTRY_SIZES[0]}+${REGISTRY_SIZES[1]})×15 = $(( (REGISTRY_SIZES[0]+REGISTRY_SIZES[1]) * 15 )) runs"
    echo "  throughput:  regs 2-5,   hc 0-2   (${REGISTRY_SIZES[2]}+${REGISTRY_SIZES[3]}+${REGISTRY_SIZES[4]}+${REGISTRY_SIZES[5]})×3 = $(( (REGISTRY_SIZES[2]+REGISTRY_SIZES[3]+REGISTRY_SIZES[4]+REGISTRY_SIZES[5]) * 3 )) runs"
    echo "  scaling:     regs 6-9,   hc 2     ${REGISTRY_SIZES[6]}+${REGISTRY_SIZES[7]}+${REGISTRY_SIZES[8]}+${REGISTRY_SIZES[9]} = $(( REGISTRY_SIZES[6]+REGISTRY_SIZES[7]+REGISTRY_SIZES[8]+REGISTRY_SIZES[9] )) runs"
    echo "  lowdensity:  regs 10-11, hc 0-2   (${REGISTRY_SIZES[10]}+${REGISTRY_SIZES[11]})×3 = $(( (REGISTRY_SIZES[10]+REGISTRY_SIZES[11]) * 3 )) runs"
    local total=$(( (REGISTRY_SIZES[0]+REGISTRY_SIZES[1])*15 + (REGISTRY_SIZES[2]+REGISTRY_SIZES[3]+REGISTRY_SIZES[4]+REGISTRY_SIZES[5])*3 + REGISTRY_SIZES[6]+REGISTRY_SIZES[7]+REGISTRY_SIZES[8]+REGISTRY_SIZES[9] + (REGISTRY_SIZES[10]+REGISTRY_SIZES[11])*3 ))
    echo "  TOTAL: $total runs"
}

###############################################################################
# Build
###############################################################################

function rebuild {
    if [[ "$OPT_NO_BUILD" == "1" ]]; then
        return
    fi
    echo "[build] Reset board with tt-smi -r"
    tt-smi -r > /dev/null
    pushd "$TT_METAL_DIR" > /dev/null
    echo "[build] Building with profiling enabled..."
    ./build_metal.sh --enable-profiler --build-programming-examples > build.log 2> build_err.log
    local rc=$?
    popd > /dev/null
    if [[ $rc -ne 0 ]]; then
        echo "[build] FAILED — see $TT_METAL_DIR/build_err.log"
        exit 1
    fi
    echo "[build] Succeeded."
}

###############################################################################
# Run one (profile_block + export_to_csv)
###############################################################################

RUN_COUNT=0

function run_one {
    local profile_case="$1"
    local host_code="$2"
    local registry="$3"
    local hc_name="$4"

    local reg_name="${REGISTRY_NAMES[$registry]}"
    RUN_COUNT=$(( RUN_COUNT + 1 ))

    echo "  [$RUN_COUNT] registry=$registry ($reg_name)  host_code=$host_code ($hc_name)  case=$profile_case"

    if [[ "$OPT_DRY_RUN" == "1" ]]; then
        if [[ "$OPT_EXPORT_ONLY" != "1" ]]; then
            echo "    [dry-run] TT_METAL_DEVICE_PROFILER=1 profile_block $profile_case $host_code $registry"
        fi
        if [[ "$OPT_PROFILE_ONLY" != "1" ]]; then
            echo "    [dry-run] export_to_csv $profile_case $host_code $registry"
        fi
        return
    fi

    # Clear stale zone source location logs
    rm -f "$TT_METAL_DIR/generated/profiler/.logs/zone_src_locations.log"
    rm -f "$TT_METAL_DIR/generated/profiler/.logs/new_zone_src_locations.log"

    rebuild

    if [[ "$OPT_EXPORT_ONLY" != "1" ]]; then
        TT_METAL_DEVICE_PROFILER=1 "$PROFILE_BIN" "$profile_case" "$host_code" "$registry"
    fi

    if [[ "$OPT_PROFILE_ONLY" != "1" ]]; then
        "$EXPORT_BIN" "$profile_case" "$host_code" "$registry"
    fi
}

###############################################################################
# Phase runners
###############################################################################

function run_phase {
    local phase_name="$1"
    local reg_start="$2"
    local reg_end="$3"
    local hc_start="$4"
    local hc_end="$5"

    local hc_entries=()
    read_host_codes hc_entries

    echo ""
    echo "###################################################################"
    echo "### $phase_name"
    echo "### registries $reg_start-$reg_end, host codes $hc_start-$hc_end"
    echo "###################################################################"

    for (( reg=reg_start; reg<=reg_end; reg++ )); do
        local num_cases="${REGISTRY_SIZES[$reg]}"
        local reg_name="${REGISTRY_NAMES[$reg]}"
        echo ""
        echo "=== Registry $reg: $reg_name ($num_cases cases) ==="

        for (( pc=0; pc<num_cases; pc++ )); do
            for (( hc=hc_start; hc<=hc_end; hc++ )); do
                run_one "$pc" "$hc" "$reg" "${hc_entries[$hc]}"
            done
        done
    done
}

function run_microbench {
    run_phase "MICROBENCH (ablation)" 0 1 0 14
}

function run_throughput {
    run_phase "THROUGHPUT (3 algos × 4 pattern registries)" 2 5 0 2
}

function run_scaling {
    run_phase "SCALING (DDA only × 4 sweep registries)" 6 9 2 2
}

function run_lowdensity {
    run_phase "LOW DENSITY (3 algos × 2 ultra-low registries)" 10 11 0 2
}

###############################################################################
# Main
###############################################################################

function main {
    OPT_PHASE="all"
    OPT_NO_BUILD=0
    OPT_WITH_ZONES=0
    OPT_DRY_RUN=0
    OPT_EXPORT_ONLY=0
    OPT_PROFILE_ONLY=0

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --list|-l)
                list_plan
                exit 0
                ;;
            --phase)
                OPT_PHASE="$2"; shift 2 ;;
            --with-zones)
                OPT_WITH_ZONES=1; shift ;;
            --no-build)
                OPT_NO_BUILD=1; shift ;;
            --dry-run)
                OPT_DRY_RUN=1; shift ;;
            --export-only)
                OPT_EXPORT_ONLY=1; shift ;;
            --profile-only)
                OPT_PROFILE_ONLY=1; shift ;;
            *)
                echo "Unknown option: $1"
                echo "Usage: $0 [--phase microbench|throughput|scaling|lowdensity|all]"
                echo "          [--with-zones] [--no-build] [--dry-run]"
                echo "          [--export-only] [--profile-only] [--list]"
                exit 1
                ;;
        esac
    done

    case "$OPT_PHASE" in
        microbench|throughput|scaling|lowdensity|all) ;;
        *)
            echo "Error: --phase must be microbench, throughput, scaling, lowdensity, or all"
            exit 1
            ;;
    esac

    if [[ "$OPT_WITH_ZONES" == "1" ]]; then
        echo "[config] Enabling device profiling zones"
        export PROFILE_READ_IN0=1
        export PROFILE_WAIT_IN0=1
        export PROFILE_WRITE_OUT=1
        export PROFILE_READ_IN1=1
        export PROFILE_COMPUTE=1
    fi

    case "$OPT_PHASE" in
        microbench)  run_microbench ;;
        throughput)  run_throughput ;;
        scaling)     run_scaling ;;
        lowdensity)  run_lowdensity ;;
        all)
            run_microbench
            run_throughput
            run_scaling
            run_lowdensity
            ;;
    esac

    echo ""
    echo "=== run_profiling.sh complete ($RUN_COUNT runs) ==="
}

main "$@"
