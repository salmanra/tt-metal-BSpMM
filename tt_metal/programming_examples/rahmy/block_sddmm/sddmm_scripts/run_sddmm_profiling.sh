#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/user/tt-metal"
BUILD_DIR="${REPO_ROOT}/build/programming_examples/rahmy"

# Defaults
DO_BUILD=true
DO_PROFILE=true
DO_EXPORT=true
DRY_RUN=false
REGISTRIES=(1 2 3 4)

# Registry sizes (number of test cases per registry)
declare -A REGISTRY_SIZES=(
    [0]=4   # ProfileCaseRegistry
    [1]=5   # ProfileSweepNRegistry
    [2]=5   # ProfileSweepDensityRegistry
    [3]=5   # ProfileSweepKRegistry
    [4]=4   # ProfileSweepBlockSizeRegistry
)

declare -A REGISTRY_NAMES=(
    [0]="SDDMMProfileSuite"
    [1]="SDDMMSweepN"
    [2]="SDDMMSweepDensity"
    [3]="SDDMMSweepK"
    [4]="SDDMMSweepBlockSize"
)

NUM_HOST_CODES=2
HOST_CODES=()  # empty = run all

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Run SDDMM profiling across sweep registries.
Each test case is profiled in a separate process invocation.

Registries:
  0  ProfileCaseRegistry          (4 small cases)
  1  ProfileSweepNRegistry        (5 cases, sweep dense output width N)
  2  ProfileSweepDensityRegistry  (5 cases, sweep sparsity density)
  3  ProfileSweepKRegistry        (5 cases, sweep reduction dimension K)
  4  ProfileSweepBlockSizeRegistry (4 cases, sweep block size R=C)

Options:
  --no-build       Skip cmake build step
  --profile-only   Run profiling only (skip CSV export)
  --export-only    Run CSV export only (skip profiling)
  --registry N     Only run registry N (0-4); can be repeated
  --host-code N    Only run host code N (0-$((NUM_HOST_CODES-1))); can be repeated
  --dry-run        Print commands without executing
  -h, --help       Show this help
EOF
    exit 0
}

# Parse arguments
CUSTOM_REGISTRIES=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-build)     DO_BUILD=false; shift ;;
        --profile-only) DO_EXPORT=false; shift ;;
        --export-only)  DO_PROFILE=false; shift ;;
        --registry)     CUSTOM_REGISTRIES+=("$2"); shift 2 ;;
        --host-code)    HOST_CODES+=("$2"); shift 2 ;;
        --dry-run)      DRY_RUN=true; shift ;;
        -h|--help)      usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

if [[ ${#CUSTOM_REGISTRIES[@]} -gt 0 ]]; then
    REGISTRIES=("${CUSTOM_REGISTRIES[@]}")
fi

# Default host codes: all of them
if [[ ${#HOST_CODES[@]} -eq 0 ]]; then
    for (( i=0; i<NUM_HOST_CODES; i++ )); do
        HOST_CODES+=("$i")
    done
fi

run_cmd() {
    echo "+ $*"
    if [[ $DRY_RUN == false ]]; then
        "$@"
    fi
}

just_build() {
    echo "[build] Reset board with tt-smi -r"
    tt-smi -r > /dev/null
    pushd "$REPO_ROOT" > /dev/null
    echo "[build] Building with profiling enabled..."
    ./build_metal.sh --enable-profiler --build-programming-examples > build.log 2> build_err.log
    local rc=$?
    popd > /dev/null
    if [[ $rc -ne 0 ]]; then
        echo "[build] FAILED — see $REPO_ROOT/build_err.log"
        exit 1
    fi
    echo "[build] Succeeded."
}

# Build
if [[ "$DO_BUILD" == true ]]; then
    run_cmd just_build
fi

PROFILE_BIN="${BUILD_DIR}/profile_sddmm"
EXPORT_BIN="${BUILD_DIR}/export_sddmm"

# Profile + Export (combined so device profiler log is read before next run overwrites it)
for reg in "${REGISTRIES[@]}"; do
    num_cases=${REGISTRY_SIZES[$reg]}
    echo "=== Registry ${reg} (${REGISTRY_NAMES[$reg]}, ${num_cases} cases) ==="
    for (( tc=0; tc<num_cases; tc++ )); do
        for hc in "${HOST_CODES[@]}"; do
            run_cmd just_build
            if [[ "$DO_PROFILE" == true ]]; then
                run_cmd env TT_METAL_DEVICE_PROFILER=1 "$PROFILE_BIN" "$tc" "$hc" "$reg"
            fi
            if [[ "$DO_EXPORT" == true ]]; then
                run_cmd "$EXPORT_BIN" "$tc" "$hc" "$reg"
            fi
        done
    done
    echo ""
done

echo "=== Done ==="
