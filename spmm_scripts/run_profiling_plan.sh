#!/usr/bin/bash
# run_profiling_plan.sh - Orchestrate the full SpMM profiling plan
#
# Usage:
#   ./run_profiling_plan.sh [--phase <ablation|sweep|all>]
#                           [--host-code <index|all>]
#                           [--registry <index|all>]
#                           [--ablation-registry <index>]
#                           [--no-build]
#                           [--dry-run]
#                           [--list]
#
# Phases:
#   ablation  - Run 4 skip-ablation variants (no_a_read, no_b_read, no_compute,
#               no_write) for each algorithm against a chosen reference registry.
#               Host codes 5-24 in HostCodeRegistryProfiling, registry default=2.
#   sweep     - Run the 5 base algorithms against the 4 parametric sweep registries
#               (N sweep, density sweep, K sweep, block-size sweep).
#               Host codes 0-4 in HostCodeRegistryProfiling, registries 4-7.
#   all       - Run both phases (default).
#
# Options:
#   --host-code <i|all>       Override host-code index (0-4 for base, 5-24 for ablation)
#   --registry <i|all>        Override profile registry for sweep phase (4-7, or all 4-7)
#   --ablation-registry <i>   Registry to use for ablation phase (default: 2)
#   --no-build                Skip the build step
#   --dry-run                 Print commands without running them
#   --list                    List all registries and host codes, then exit
#
# Registry index map (matches profile_block.cpp switch statement):
#   0  ProfileCaseRegistry              (small sparse cases)
#   1  ProfileDenseAblationRegistry     (dense K-sweep)
#   2  ProfileLargeSparseRegistry       (large sparse, R/C 32/64/128)
#   3  ProfileLargeSparseLargeBlocksRegistry (large sparse, R/C 512)
#   4  ProfileSweepN                    (parametric, sweep N)
#   5  ProfileSweepDensity              (parametric, sweep density)
#   6  ProfileSweepK                    (parametric, sweep K)
#   7  ProfileSweepBlockSize            (parametric, sweep block size)
#
# Host code index map in HostCodeRegistryProfiling:
#   [0-4]   Full algorithms: snf, load_balanced, reuse_iteration, naive_new_DM, lb_new_DM
#   [5-9]   no_a_read  variants (SKIP_IN0_DRAM_READ=1)
#   [10-14] no_b_read  variants (SKIP_IN1_DRAM_READ=1)
#   [15-19] no_compute variants (SKIP_COMPUTE=1)
#   [20-24] no_write   variants (SKIP_DRAM_WRITE=1)

set -euo pipefail

TT_METAL_DIR="/home/user/tt-metal"
HOST_CODE_HPP="$TT_METAL_DIR/tt_metal/programming_examples/rahmy/block_spmm/inc/host_code.hpp"
PROFILING_SUITE_HPP="$TT_METAL_DIR/tt_metal/programming_examples/rahmy/block_spmm/inc/profiling_suite.hpp"

# Registry array names and display names — indices must match profile_block.cpp
PROFILE_REGISTRY_ARRAY_NAMES=(
    "ProfileCaseRegistry"
    "ProfileDenseAblationRegistry"
    "ProfileLargeSparseRegistry"
    "ProfileLargeSparseLargeBlocksRegistry"
    "ProfileSweepNRegistry"
    "ProfileSweepDensityRegistry"
    "ProfileSweepKRegistry"
    "ProfileSweepBlockSizeRegistry"
)
PROFILE_REGISTRY_DISPLAY_NAMES=(
    "ProfileSuiteSparseVersioning"
    "DenseAblationKProfileSuite"
    "ProfileSuiteLargeSparseVersioning"
    "ProfileSuiteLargeSparseLargeBlocksVersioning"
    "ProfileSweepN"
    "ProfileSweepDensity"
    "ProfileSweepK"
    "ProfileSweepBlockSize"
)

###############################################################################
# Registry parsing (mirrors get_trace.sh)
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

        # Skip commented-out entries and blank lines
        [[ "$line" =~ ^[[:space:]]*//.* ]] && continue
        [[ -z "${line// /}" ]] && continue

        # Extract quoted string (HostCodeRegistryProfiling style)
        # or function_name<args> (ProfileCaseRegistry style)
        if [[ "$line" =~ \"([^\"]+)\" ]]; then
            echo "${BASH_REMATCH[1]}"
        elif [[ "$line" =~ ([a-zA-Z_]+)\<([^\>]+)\> ]]; then
            echo "${BASH_REMATCH[1]}<${BASH_REMATCH[2]}>"
        fi
    done < "$file"
}

function registry_size {
    parse_registry "$1" "$2" | wc -l
}

function read_registry_into {
    local -n arr_ref=$1
    arr_ref=()
    while IFS= read -r entry; do
        arr_ref+=("$entry")
    done < <(parse_registry "$2" "$3")
}

###############################################################################
# Display
###############################################################################

function list_plan {
    echo "=== HostCodeRegistryProfiling ==="
    local hc_entries=()
    read_registry_into hc_entries "$HOST_CODE_HPP" "HostCodeRegistryProfiling"
    local i=0
    for entry in "${hc_entries[@]}"; do
        local group=""
        if   (( i >= 0  && i <= 4  )); then group="[full]"
        elif (( i >= 5  && i <= 9  )); then group="[no_a_read]"
        elif (( i >= 10 && i <= 14 )); then group="[no_b_read]"
        elif (( i >= 15 && i <= 19 )); then group="[no_compute]"
        elif (( i >= 20 && i <= 24 )); then group="[no_write]"
        fi
        printf "  [%2d] %-12s %s\n" "$i" "$group" "$entry"
        i=$(( i + 1 ))
    done
    echo ""

    local num_regs=${#PROFILE_REGISTRY_ARRAY_NAMES[@]}
    for (( reg=0; reg<num_regs; reg++ )); do
        local arr_name="${PROFILE_REGISTRY_ARRAY_NAMES[$reg]}"
        local disp_name="${PROFILE_REGISTRY_DISPLAY_NAMES[$reg]}"
        echo "=== Profile Registry $reg: $disp_name ==="
        local j=0
        while IFS= read -r entry; do
            printf "  [%d] %s\n" "$j" "$entry"
            j=$(( j + 1 ))
        done < <(parse_registry "$PROFILING_SUITE_HPP" "$arr_name")
        echo ""
    done
}

###############################################################################
# Build
###############################################################################

_BUILT=0

function build_if_needed {
    if [[ "$OPT_NO_BUILD" == "1" || "$_BUILT" == "1" ]]; then
        return
    fi
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
    _BUILT=1
}

function just_build {
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
    _BUILT=1
}

###############################################################################
# Run one (profile_block + export_to_csv)
###############################################################################

function run_one {
    local profile_case="$1"
    local host_code="$2"
    local registry="$3"
    local hc_name="${4:-?}"

    local reg_name="${PROFILE_REGISTRY_DISPLAY_NAMES[$registry]:-registry$registry}"

    echo "  → registry=$registry ($reg_name)  host_code=$host_code ($hc_name)  profile_case=$profile_case"

    if [[ "$OPT_DRY_RUN" == "1" ]]; then
        echo "    [dry-run] TT_METAL_DEVICE_PROFILER=1 profile_block $profile_case $host_code $registry"
        echo "    [dry-run] export_to_csv $profile_case $host_code $registry"
        return
    fi

    # Clear stale zone source location logs
    rm -f "$TT_METAL_DIR/generated/profiler/.logs/zone_src_locations.log"
    rm -f "$TT_METAL_DIR/generated/profiler/.logs/new_zone_src_locations.log"

    # reset and rebuild because there are clearly profiling artifacts on the device which sometimes cause errors
    # when running the entire suite
    just_build

    TT_METAL_DEVICE_PROFILER=1 \
        "$TT_METAL_DIR/build/programming_examples/rahmy/profile_block" \
        "$profile_case" "$host_code" "$registry"

    "$TT_METAL_DIR/build/programming_examples/rahmy/export_to_csv" \
        "$profile_case" "$host_code" "$registry"
}

###############################################################################
# Run registry
###############################################################################

# Run all profile cases in a registry against a list of host code indices.
# Arguments: registry_idx, hc_start, hc_end, hc_entries_array_name
function run_registry {
    local registry="$1"
    local hc_start="$2"
    local hc_end="$3"
    local -n hc_entries_ref=$4

    local arr_name="${PROFILE_REGISTRY_ARRAY_NAMES[$registry]}"
    local disp_name="${PROFILE_REGISTRY_DISPLAY_NAMES[$registry]}"
    local num_profiles
    num_profiles=$(registry_size "$PROFILING_SUITE_HPP" "$arr_name")

    echo "========================================================================"
    echo "  Registry $registry: $disp_name  ($num_profiles cases)"
    echo "  Host codes: $hc_start .. $hc_end"
    echo "========================================================================"

    for (( pc=0; pc<num_profiles; pc++ )); do
        for (( hc=hc_start; hc<=hc_end; hc++ )); do
            run_one "$pc" "$hc" "$registry" "${hc_entries_ref[$hc]:-?}"
        done
    done
}

###############################################################################
# Ablation phase
###############################################################################

# Run the 4 ablation groups (no_a_read, no_b_read, no_compute, no_write)
# for the 5 algorithms against a single reference registry.
#
# Host code index layout in HostCodeRegistryProfiling:
#   group 0 (no_a_read):  host codes 5-9   (5 algorithms)
#   group 1 (no_b_read):  host codes 10-14
#   group 2 (no_compute): host codes 15-19
#   group 3 (no_write):   host codes 20-24
function run_ablation_phase {
    local ablation_registry="$1"   # reference registry (default 2)
    local hc_override="${2:-all}"  # "all" or a single algorithm index 0-4

    local hc_entries=()
    read_registry_into hc_entries "$HOST_CODE_HPP" "HostCodeRegistryProfiling"

    local ABLATION_GROUPS=(
        "no_a_read:5:9"
        "no_b_read:10:14"
        "no_compute:15:19"
        "no_write:20:24"
    )

    echo ""
    echo "###################################################################"
    echo "### ABLATION PHASE — registry=$ablation_registry               ###"
    echo "###################################################################"

    local arr_name="${PROFILE_REGISTRY_ARRAY_NAMES[$ablation_registry]}"
    local num_profiles
    num_profiles=$(registry_size "$PROFILING_SUITE_HPP" "$arr_name")

    for group_spec in "${ABLATION_GROUPS[@]}"; do
        local group_name="${group_spec%%:*}"
        local rest="${group_spec#*:}"
        local group_hc_start="${rest%%:*}"
        local group_hc_end="${rest#*:}"

        # If user passed a specific algorithm index (0-4), map it into this group
        local hc_start hc_end
        if [[ "$hc_override" == "all" ]]; then
            hc_start="$group_hc_start"
            hc_end="$group_hc_end"
        else
            hc_start=$(( group_hc_start + hc_override ))
            hc_end="$hc_start"
        fi

        echo ""
        echo "--- Ablation group: $group_name (host codes $hc_start..$hc_end) ---"

        for (( pc=0; pc<num_profiles; pc++ )); do
            for (( hc=hc_start; hc<=hc_end; hc++ )); do
                run_one "$pc" "$hc" "$ablation_registry" "${hc_entries[$hc]:-?}"
            done
        done
    done
}

###############################################################################
# Sweep phase
###############################################################################

# Run the 5 base algorithms (host codes 0-4) against all 4 sweep registries (4-7).
function run_sweep_phase {
    local registry_override="${1:-all}"  # "all" or a single registry index 4-7
    local hc_override="${2:-all}"        # "all" or a single algorithm index 0-4

    local hc_entries=()
    read_registry_into hc_entries "$HOST_CODE_HPP" "HostCodeRegistryProfiling"

    # Base algorithm host codes: 0-4
    local hc_start hc_end
    if [[ "$hc_override" == "all" ]]; then
        hc_start=0
        hc_end=4
    else
        hc_start="$hc_override"
        hc_end="$hc_override"
    fi

    # Sweep registries: 4-7
    local reg_start reg_end
    if [[ "$registry_override" == "all" ]]; then
        reg_start=4
        reg_end=7
    else
        reg_start="$registry_override"
        reg_end="$registry_override"
    fi

    echo ""
    echo "###################################################################"
    echo "### SWEEP PHASE — registries $reg_start..$reg_end             ###"
    echo "###################################################################"

    for (( reg=reg_start; reg<=reg_end; reg++ )); do
        run_registry "$reg" "$hc_start" "$hc_end" hc_entries
    done
}

###############################################################################
# Main
###############################################################################
function main {
    OPT_PHASE="all"
    OPT_HOST_CODE="all"
    OPT_REGISTRY="all"
    OPT_ABLATION_REGISTRY=2
    OPT_NO_BUILD=0
    OPT_DRY_RUN=0

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --list|-l)
                list_plan
                exit 0
                ;;
            --phase)
                OPT_PHASE="$2"; shift 2 ;;
            --host-code)
                OPT_HOST_CODE="$2"; shift 2 ;;
            --registry)
                OPT_REGISTRY="$2"; shift 2 ;;
            --ablation-registry)
                OPT_ABLATION_REGISTRY="$2"; shift 2 ;;
            --no-build)
                OPT_NO_BUILD=1; shift ;;
            --dry-run)
                OPT_DRY_RUN=1; shift ;;
            *)
                echo "Unknown option: $1"
                echo "Usage: $0 [--phase ablation|sweep|all] [--host-code <i|all>]"
                echo "          [--registry <i|all>] [--ablation-registry <i>]"
                echo "          [--no-build] [--dry-run] [--list]"
                exit 1
                ;;
        esac
    done

    # Validate phase
    case "$OPT_PHASE" in
        ablation|sweep|all) ;;
        *)
            echo "Error: --phase must be 'ablation', 'sweep', or 'all'"
            exit 1
            ;;
    esac

    tt-smi -r > /dev/null
    build_if_needed

    case "$OPT_PHASE" in
        ablation)
            run_ablation_phase "$OPT_ABLATION_REGISTRY" "$OPT_HOST_CODE"
            ;;
        sweep)
            run_sweep_phase "$OPT_REGISTRY" "$OPT_HOST_CODE"
            ;;
        all)
            run_ablation_phase "$OPT_ABLATION_REGISTRY" "$OPT_HOST_CODE"
            run_sweep_phase "$OPT_REGISTRY" "$OPT_HOST_CODE"
            ;;
    esac

    echo ""
    echo "=== run_profiling_plan.sh complete ==="

}

main "$@"
