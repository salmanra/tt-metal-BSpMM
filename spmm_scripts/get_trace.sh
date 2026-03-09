#!/usr/bin/bash
# get_trace.sh - Build, profile, and export BSR SpMM traces
#
# Usage:
#   ./get_trace.sh <profile_case> <host_code> [registry]
#   ./get_trace.sh --list
#
# Arguments:
#   profile_case  - index number, or "all"
#   host_code     - index number, or "all"
#   registry      - index number (0-3), or "all" (default: 2)
#

set -euo pipefail

TT_METAL_DIR="/home/user/tt-metal"
HOST_CODE_HPP="$TT_METAL_DIR/tt_metal/programming_examples/rahmy/block_spmm/inc/host_code.hpp"
PROFILING_SUITE_HPP="$TT_METAL_DIR/tt_metal/programming_examples/rahmy/block_spmm/inc/profiling_suite.hpp"

# Profile registry number -> (array name, display name)
# Must match the switch statement in profile_block.cpp / export_to_csv.cpp
PROFILE_REGISTRY_ARRAY_NAMES=("ProfileCaseRegistry" "ProfileDenseAblationRegistry" "ProfileLargeSparseRegistry" "ProfileLargeSparseLargeBlocksRegistry")
PROFILE_REGISTRY_DISPLAY_NAMES=("ProfileSuiteSparseVersioning" "DenseAblationKProfileSuite" "ProfileSuiteLargeSparseVersioning" "ProfileSuiteLargeSparseLargeBlocksVersioning")
NUM_PROFILE_REGISTRIES=${#PROFILE_REGISTRY_ARRAY_NAMES[@]}

###############################################################################
# Registry parsing
###############################################################################

# Parse uncommented entries from a C++ static array declaration in a header.
# For HostCodeRegistry (pairs): extracts the quoted string name.
# For ProfileCaseRegistry (bare function ptrs): extracts function_name<template_args>.
# Outputs one entry per line.
#
# Usage: parse_registry <header_file> <array_variable_name>
function parse_registry {
    local file="$1"
    local array_name="$2"
    local in_array=0

    while IFS= read -r line; do
        if [[ $in_array -eq 0 ]]; then
            # Skip comment lines
            [[ "$line" =~ ^[[:space:]]*//.* ]] && continue
            # Look for the array declaration
            if [[ "$line" == *"${array_name}[]"* ]]; then
                in_array=1
            fi
            continue
        fi

        # End of array
        [[ "$line" == *"};" ]] && break

        # Skip commented-out entries and blank lines
        [[ "$line" =~ ^[[:space:]]*//.* ]] && continue
        [[ -z "${line// /}" ]] && continue

        # Extract: prefer a quoted string (HostCodeRegistry style),
        # otherwise grab function_name<args> (ProfileCaseRegistry style)
        if [[ "$line" =~ \"([^\"]+)\" ]]; then
            echo "${BASH_REMATCH[1]}"
        elif [[ "$line" =~ ([a-zA-Z_]+)\<([^\>]+)\> ]]; then
            echo "${BASH_REMATCH[1]}<${BASH_REMATCH[2]}>"
        fi
    done < "$file"
}

# Get the number of entries in a parsed registry.
function registry_size {
    parse_registry "$1" "$2" | wc -l
}

# Read parsed entries into a bash array variable.
# Usage: read_registry_into <varname> <header_file> <array_name>
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

function list_registries {
    echo "=== Host Code Registry (HostCodeRegistry) ==="
    local i=0
    while IFS= read -r entry; do
        printf "  [%d] %s\n" "$i" "$entry"
        i=$((i + 1))
    done < <(parse_registry "$HOST_CODE_HPP" "HostCodeRegistry")
    echo ""

    for reg_idx in "${!PROFILE_REGISTRY_ARRAY_NAMES[@]}"; do
        local arr_name="${PROFILE_REGISTRY_ARRAY_NAMES[$reg_idx]}"
        local disp_name="${PROFILE_REGISTRY_DISPLAY_NAMES[$reg_idx]}"
        echo "=== Profile Registry $reg_idx: $disp_name ($arr_name) ==="
        local j=0
        while IFS= read -r entry; do
            printf "  [%d] %s\n" "$j" "$entry"
            j=$((j + 1))
        done < <(parse_registry "$PROFILING_SUITE_HPP" "$arr_name")
        echo ""
    done
}

###############################################################################
# Build & run
###############################################################################

function build_with_profiling_enabled {
    pushd "$TT_METAL_DIR" > /dev/null
    tt-smi -r > /dev/null
    echo "Building with profiling enabled..."
    ./build_metal.sh --enable-profiler --build-programming-examples > build.log 2> build_err.log
    local rc=$?
    popd > /dev/null
    if [[ $rc -ne 0 ]]; then
        echo "Build failed"
        exit 1
    fi
    echo "Build Succeeded!"
}

function get_trace {
    local profile_case="$1"
    local host_code="$2"
    local registry="$3"

    build_with_profiling_enabled
    pkill capture-release 2>/dev/null || true
    # Clear stale profiler zone mappings (they accumulate across JIT compilations)
    rm -f "$TT_METAL_DIR/generated/profiler/.logs/zone_src_locations.log"
    rm -f "$TT_METAL_DIR/generated/profiler/.logs/new_zone_src_locations.log"
    # Zone env vars (e.g. PROFILE_WRITE_OUT=0) are inherited from the parent shell
    TT_METAL_DEVICE_PROFILER=1 "$TT_METAL_DIR/build/programming_examples/rahmy/profile_block" \
        "$profile_case" "$host_code" "$registry"
    "$TT_METAL_DIR/build/programming_examples/rahmy/export_to_csv" \
        "$profile_case" "$host_code" "$registry"
}

function just_export_to_csv {
    local profile_case="$1"
    local host_code="$2"
    local registry="$3"

    "$TT_METAL_DIR/build/programming_examples/rahmy/export_to_csv" \
        "$profile_case" "$host_code" "$registry"
}

# Parse --disable-zones flag and export env vars.
# Usage: parse_disable_zones "read_in0,wait_in0,read_in1,write_out,compute"
function parse_disable_zones {
    IFS=',' read -ra zones <<< "$1"
    for z in "${zones[@]}"; do
        local upper_z
        upper_z="PROFILE_$(echo "$z" | tr '[:lower:]' '[:upper:]')"
        export "${upper_z}=0"
        echo "  Zone disabled: ${upper_z}=0"
    done
}

###############################################################################
# Main
###############################################################################

function main {
    local profile_arg="${1:?Usage: $0 <profile_case|all|--list> <host_code|all> [registry|all] [--disable-zones zone1,zone2,...]}"
    local host_code_arg="${2:-all}"
    local registry_arg="${3:-2}"

    # --list: show available registries and exit
    if [[ "$profile_arg" == "--list" || "$profile_arg" == "-l" ]]; then
        list_registries
        return
    fi

    # Parse optional --disable-zones flag (can appear as 4th positional arg)
    # What I want now is for the set of disabled zones to define their own output trace
    if [[ "${4:-}" == "--disable-zones" && -n "${5:-}" ]]; then
        parse_disable_zones "$5"
    fi

    # Clear stale profiler zone source location logs (they accumulate across JIT
    # compilations and can cause hash mismatches when kernel files are edited)
    rm -f "$TT_METAL_DIR/generated/profiler/.logs/zone_src_locations.log"
    rm -f "$TT_METAL_DIR/generated/profiler/.logs/new_zone_src_locations.log"

    tt-smi -r > /dev/null
    build_with_profiling_enabled


    # Determine which profile registries to iterate over
    local reg_start reg_end
    if [[ "$registry_arg" == "all" ]]; then
        reg_start=0
        reg_end=$(( NUM_PROFILE_REGISTRIES - 1 ))
    else
        reg_start="$registry_arg"
        reg_end="$registry_arg"
    fi

    # Read host code entries to know the valid range
    local host_code_entries=()
    read_registry_into host_code_entries "$HOST_CODE_HPP" "HostCodeRegistry"
    local num_host_codes=${#host_code_entries[@]}

    for (( reg=reg_start; reg<=reg_end; reg++ )); do
        local arr_name="${PROFILE_REGISTRY_ARRAY_NAMES[$reg]}"
        local disp_name="${PROFILE_REGISTRY_DISPLAY_NAMES[$reg]}"
        local num_profiles
        num_profiles=$(registry_size "$PROFILING_SUITE_HPP" "$arr_name")

        echo "========================================================================="
        echo "  Registry $reg: $disp_name ($num_profiles profile cases)"
        echo "========================================================================="

        # When both profile_arg and host_code_arg are not "all", just pass
        # them straight to the C++ programs (they handle single indices directly).
        if [[ "$profile_arg" != "all" && "$host_code_arg" != "all" ]]; then
            echo "--- profile_case=$profile_arg  host_code=$host_code_arg (${host_code_entries[$host_code_arg]}) ---"
            get_trace "$profile_arg" "$host_code_arg" "$reg"
            continue
        fi

        # When "all" is used, iterate in bash so we can print progress.
        local pc_start pc_end
        if [[ "$profile_arg" == "all" ]]; then
            pc_start=0
            pc_end=$(( num_profiles - 1 ))
        else
            pc_start="$profile_arg"
            pc_end="$profile_arg"
        fi

        local hc_start hc_end
        if [[ "$host_code_arg" == "all" ]]; then
            hc_start=0
            hc_end=$(( num_host_codes - 1 ))
        else
            hc_start="$host_code_arg"
            hc_end="$host_code_arg"
        fi

        for (( pc=pc_start; pc<=pc_end; pc++ )); do
            for (( hc=hc_start; hc<=hc_end; hc++ )); do
                echo "--- profile_case=$pc  host_code=$hc (${host_code_entries[$hc]})  registry=$reg ---"
                get_trace "$pc" "$hc" "$reg"
            done
        done
    done
}

main "$@"
