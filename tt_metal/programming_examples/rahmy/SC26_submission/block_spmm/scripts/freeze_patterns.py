#!/usr/bin/env python3
"""
freeze_patterns.py — Parse profiles_sc26 sparse.log files and generate
sc26_profiling_suite.hpp with frozen (hardcoded) sparsity patterns.

The generated header mirrors the template-based style of profiling_suite.hpp:
  - frozen_parametric_random<M,N,K,R,C,DensityPPM> — explicit specializations
  - frozen_parametric_{row,col,multi_diag}<...>     — generic templates (deterministic)

Usage:
    python3 freeze_patterns.py [--log-dir DIR] [--output FILE]
"""

import argparse
import os
import re
import sys
from pathlib import Path
from collections import OrderedDict


# ── Density label ↔ PPM mapping ─────────────────────────────────────────────

DENSITY_LABEL_TO_PPM = {
    "d5":    50000,
    "d10":   100000,
    "d25":   250000,
    "d50":   500000,
    "d75":   750000,
}

def density_label_to_ppm(label: str) -> int:
    """Convert a density label like 'd25' or 'dppm300' to PPM integer."""
    if label in DENSITY_LABEL_TO_PPM:
        return DENSITY_LABEL_TO_PPM[label]
    m = re.match(r"dppm(\d+)", label)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot parse density label: {label}")


# ── Log parsing ──────────────────────────────────────────────────────────────

def parse_sparse_log(path: str) -> dict:
    """Extract BSR metadata, indptr, and indices from a _sparse.log file."""
    with open(path) as f:
        lines = f.readlines()

    result = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Size (H x W):"):
            m = re.match(r"Size \(H x W\):\s*(\d+)\s*x\s*(\d+)", line)
            result["H"] = int(m.group(1))
            result["W"] = int(m.group(2))
        elif line.startswith("Block Size (R x C):"):
            m = re.match(r"Block Size \(R x C\):\s*(\d+)\s*x\s*(\d+)", line)
            result["R"] = int(m.group(1))
            result["C"] = int(m.group(2))
        elif line.startswith("Number of blocks:"):
            result["nblocks"] = int(line.split(":")[1].strip())
        elif line == "Indptr:":
            i += 1
            result["indptr"] = list(map(int, lines[i].strip().split()))
        elif line == "Indices:":
            i += 1
            result["indices"] = list(map(int, lines[i].strip().split()))
        i += 1

    assert "indptr" in result, f"No Indptr found in {path}"
    assert "indices" in result, f"No Indices found in {path}"
    assert result["indptr"][-1] == result["nblocks"], \
        f"indptr[-1]={result['indptr'][-1]} != nblocks={result['nblocks']} in {path}"
    assert len(result["indices"]) == result["nblocks"], \
        f"len(indices)={len(result['indices'])} != nblocks={result['nblocks']} in {path}"

    return result


def extract_params_from_basename(basename: str) -> dict:
    """Extract M, N, K, R, C, DensityPPM from a test basename."""
    params = {}
    for key in ["M", "N", "K", "R", "C"]:
        m = re.search(rf"_{key}(\d+)", basename)
        if m:
            params[key] = int(m.group(1))

    # Density: either _d25 or _dppm10000 (at end of string)
    m = re.search(r"_(dppm\d+|d\d+)$", basename)
    if m:
        params["density_label"] = m.group(1)
        params["DensityPPM"] = density_label_to_ppm(m.group(1))

    return params


def is_deterministic_pattern(basename: str) -> bool:
    """True if the test uses a deterministic fill pattern (row, col, multi_diag)."""
    return any(basename.startswith(f"parametric_{p}_") for p in ["row", "col", "multi_diag"])


# ── Log discovery ────────────────────────────────────────────────────────────

PREFERRED_ALGO = "bsr_spmm_multicore_snfin0_cdain1"


def discover_canonical_logs(log_dir: str) -> dict:
    """
    Find one canonical sparse.log per unique test basename.
    Returns: {basename: {"path": str, "group": str, "algo": str}}
    """
    log_dir = Path(log_dir)
    available = {}  # basename -> [(group, algo, path)]
    for group_dir in sorted(log_dir.iterdir()):
        if not group_dir.is_dir():
            continue
        group = group_dir.name
        for algo_dir in sorted(group_dir.iterdir()):
            if not algo_dir.is_dir():
                continue
            algo = algo_dir.name
            for log_file in algo_dir.iterdir():
                if log_file.name.endswith("_sparse.log"):
                    basename = log_file.name.replace("_sparse.log", "")
                    if basename not in available:
                        available[basename] = []
                    available[basename].append((group, algo, str(log_file)))

    canonical = {}
    for basename, entries in sorted(available.items()):
        preferred = [e for e in entries if e[1] == PREFERRED_ALGO]
        if preferred:
            preferred.sort(key=lambda e: e[0])
            chosen = preferred[0]
        else:
            entries.sort(key=lambda e: (e[0], e[1]))
            chosen = entries[0]
        canonical[basename] = {
            "path": chosen[2],
            "group": chosen[0],
            "algo": chosen[1],
        }

    return canonical


# ── Registry specification ───────────────────────────────────────────────────
# Maps group name → list of (basename, is_random) tuples.
# Order within each registry matches the existing profiling_suite.hpp.

REGISTRY_SPEC = OrderedDict([
    ("MicrobenchD25", [
        "parametric_M8192_N8192_K8192_R32_C32_d25",
        "parametric_M8192_N8192_K8192_R64_C64_d25",
        "parametric_M8192_N8192_K8192_R128_C128_d25",
        "parametric_M8192_N8192_K8192_R256_C256_d25",
    ]),
    ("MicrobenchD5", [
        "parametric_M8192_N8192_K8192_R32_C32_d5",
        "parametric_M8192_N8192_K8192_R64_C64_d5",
        "parametric_M8192_N8192_K8192_R128_C128_d5",
        "parametric_M8192_N8192_K8192_R256_C256_d5",
    ]),
    ("PatternD5", [
        "parametric_row_M8192_N8192_K8192_R256_C256_d5",
        "parametric_col_M8192_N8192_K8192_R256_C256_d5",
        "parametric_multi_diag_M8192_N8192_K8192_R256_C256_d5",
        "parametric_M8192_N8192_K8192_R256_C256_d5",
    ]),
    ("PatternD10", [
        "parametric_row_M8192_N8192_K8192_R256_C256_d10",
        "parametric_col_M8192_N8192_K8192_R256_C256_d10",
        "parametric_multi_diag_M8192_N8192_K8192_R256_C256_d10",
        "parametric_M8192_N8192_K8192_R256_C256_d10",
    ]),
    ("PatternD25", [
        "parametric_row_M8192_N8192_K8192_R256_C256_d25",
        "parametric_col_M8192_N8192_K8192_R256_C256_d25",
        "parametric_multi_diag_M8192_N8192_K8192_R256_C256_d25",
        "parametric_M8192_N8192_K8192_R256_C256_d25",
    ]),
    ("PatternD50", [
        "parametric_row_M8192_N8192_K8192_R256_C256_d50",
        "parametric_col_M8192_N8192_K8192_R256_C256_d50",
        "parametric_multi_diag_M8192_N8192_K8192_R256_C256_d50",
        "parametric_M8192_N8192_K8192_R256_C256_d50",
    ]),
    ("SweepN", [
        "parametric_M8192_N512_K8192_R256_C256_d10",
        "parametric_M8192_N1024_K8192_R256_C256_d10",
        "parametric_M8192_N2048_K8192_R256_C256_d10",
        "parametric_M8192_N4096_K8192_R256_C256_d10",
        "parametric_M8192_N8192_K8192_R256_C256_d10",
    ]),
    ("SweepK", [
        "parametric_M8192_N8192_K512_R256_C256_d10",
        "parametric_M8192_N8192_K1024_R256_C256_d10",
        "parametric_M8192_N8192_K2048_R256_C256_d10",
        "parametric_M8192_N8192_K4096_R256_C256_d10",
        "parametric_M8192_N8192_K8192_R256_C256_d10",
    ]),
    ("SweepBlockSize", [
        "parametric_M8192_N8192_K8192_R32_C32_d10",
        "parametric_M8192_N8192_K8192_R64_C64_d10",
        "parametric_M8192_N8192_K8192_R128_C128_d10",
        "parametric_M8192_N8192_K8192_R256_C256_d10",
    ]),
    ("SweepDensity", [
        "parametric_M8192_N8192_K8192_R256_C256_d5",
        "parametric_M8192_N8192_K8192_R256_C256_d10",
        "parametric_M8192_N8192_K8192_R256_C256_d25",
        "parametric_M8192_N8192_K8192_R256_C256_d50",
        "parametric_M8192_N8192_K8192_R256_C256_d75",
    ]),
    ("UltraLowDensity32", [
        "parametric_M8192_N8192_K8192_R32_C32_dppm30",
        "parametric_M8192_N8192_K8192_R32_C32_dppm100",
        "parametric_M8192_N8192_K8192_R32_C32_dppm300",
        "parametric_M8192_N8192_K8192_R32_C32_dppm1000",
        "parametric_M8192_N8192_K8192_R32_C32_dppm3000",
        "parametric_M8192_N8192_K8192_R32_C32_dppm10000",
    ]),
    ("UltraLowDensity64", [
        "parametric_M8192_N8192_K8192_R64_C64_dppm60",
        "parametric_M8192_N8192_K8192_R64_C64_dppm200",
        "parametric_M8192_N8192_K8192_R64_C64_dppm600",
        "parametric_M8192_N8192_K8192_R64_C64_dppm2000",
        "parametric_M8192_N8192_K8192_R64_C64_dppm6000",
        "parametric_M8192_N8192_K8192_R64_C64_dppm10000",
    ]),
])


# ── C++ code generation ──────────────────────────────────────────────────────

def format_int_array(values: list, items_per_line: int = 16) -> str:
    """Format a list of ints as C++ initializer elements with line wrapping."""
    if len(values) == 0:
        return ""
    lines = []
    for i in range(0, len(values), items_per_line):
        chunk = values[i:i + items_per_line]
        lines.append("            " + ", ".join(str(v) for v in chunk))
    return ",\n".join(lines)


def template_args(params: dict) -> str:
    """Return C++ template argument list like '<8192, 8192, 8192, 256, 256, 250000>'."""
    return "<{M}, {N}, {K}, {R}, {C}, {DensityPPM}>".format(**params)


def frozen_specialization(basename: str, parsed: dict) -> str:
    """Generate an explicit template specialization for a frozen random pattern."""
    params = extract_params_from_basename(basename)
    targs = template_args(params)
    M = parsed["H"]
    K = parsed["W"]
    N = params["N"]
    R = parsed["R"]
    C = parsed["C"]
    nblocks = parsed["nblocks"]

    indptr_str = format_int_array(parsed["indptr"])
    indices_str = format_int_array(parsed["indices"])

    return f"""\
    template <>
    inline ProfileCaseReturnType frozen_parametric_random{targs}() {{
        std::vector<int> indptr = {{
{indptr_str}
        }};
        std::vector<int> indices = {{
{indices_str}
        }};
        return make_frozen_case(indptr, indices, {nblocks},
            {M}, {N}, {K}, {R}, {C},
            "parametric_M{M}_N{N}_K{K}_R{R}_C{C}_dppm{params['DensityPPM']}");
    }}
"""


def registry_entry(basename: str) -> str:
    """Return the C++ expression to reference this test case in a registry."""
    params = extract_params_from_basename(basename)
    targs = template_args(params)
    if is_deterministic_pattern(basename):
        if basename.startswith("parametric_row_"):
            return f"frozen_parametric_row{targs}"
        elif basename.startswith("parametric_col_"):
            return f"frozen_parametric_col{targs}"
        elif basename.startswith("parametric_multi_diag_"):
            return f"frozen_parametric_multi_diag{targs}"
    return f"frozen_parametric_random{targs}"


def generate_header(canonical: dict) -> str:
    """Generate the complete sc26_profiling_suite.hpp."""

    # Collect unique random-pattern specializations needed
    # Key by (M,N,K,R,C,DensityPPM) to deduplicate
    random_basenames = sorted(b for b in canonical if not is_deterministic_pattern(b))
    seen_params = set()
    unique_random = []
    for basename in random_basenames:
        params = extract_params_from_basename(basename)
        key = (params["M"], params["N"], params["K"], params["R"], params["C"], params["DensityPPM"])
        if key not in seen_params:
            seen_params.add(key)
            unique_random.append(basename)

    # Parse logs for unique random patterns
    parsed_logs = {}
    for basename in unique_random:
        parsed_logs[basename] = parse_sparse_log(canonical[basename]["path"])

    det_basenames = sorted(set(b for b in canonical if is_deterministic_pattern(b)))

    # Generate specializations
    specializations = []
    for basename in unique_random:
        specializations.append(frozen_specialization(basename, parsed_logs[basename]))

    # Generate registries
    registry_blocks = []
    for group_name, entries in REGISTRY_SPEC.items():
        lines = []
        for entry in entries:
            lines.append(f"        {registry_entry(entry)},")
        body = "\n".join(lines)
        registry_blocks.append(
            f"    // {group_name}\n"
            f"    static ProfileCaseFunctionPtr {group_name}Registry[] = {{\n"
            f"{body}\n"
            f"    }};\n"
        )

    # Registry metadata
    reg_names = list(REGISTRY_SPEC.keys())
    reg_sizes = [len(v) for v in REGISTRY_SPEC.values()]

    header = f"""\
#pragma once
// Auto-generated by freeze_patterns.py — do not edit manually.
// Source: profiles_sc26/csvs/*_sparse.log
// {len(unique_random)} frozen random-pattern specializations
// {len(det_basenames)} deterministic-pattern templates (row, col, multi_diag)

#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <tuple>
#include <string>
#include "include_me.hpp"
#include "bsr_matrix.hpp"
#include "tt-metalium/bfloat16.hpp"

using namespace tt;

using CoreSpec = std::variant<CoreCoord, CoreRange, CoreRangeSet>;

namespace sc26_profiling_suite {{

    using ProfileCaseReturnType = std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string>;
    using ProfileCaseFunctionPtr = ProfileCaseReturnType (*)();

    // ── Forward declarations ──
    template <uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>
    ProfileCaseReturnType frozen_parametric_random();
    template <uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>
    ProfileCaseReturnType frozen_parametric_row();
    template <uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>
    ProfileCaseReturnType frozen_parametric_col();
    template <uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>
    ProfileCaseReturnType frozen_parametric_multi_diag();

    // ── Registry arrays ──

{chr(10).join(registry_blocks)}
    // ── Registry metadata ──
    static const int NUM_REGISTRIES = {len(reg_names)};
    static const int RegistrySizes[] = {{{", ".join(str(s) for s in reg_sizes)}}};
    static const char* RegistryNames[] = {{
        {", ".join(f'"{n}"' for n in reg_names)},
    }};
    static ProfileCaseFunctionPtr* Registries[] = {{
        {", ".join(f'{n}Registry' for n in reg_names)},
    }};

    // ═══════════════════════════════════════════════════════════════════════
    // Helper: construct frozen BSR from hardcoded indptr/indices
    // ═══════════════════════════════════════════════════════════════════════

    inline ProfileCaseReturnType make_frozen_case(
        const std::vector<int>& indptr,
        const std::vector<int>& indices,
        int nblocks, int M, int N, int K, int R, int C,
        const std::string& name)
    {{
        // UNIFORM block data (all 1.0f) — values don't affect profiling timing
        std::vector<float> block_data((size_t)nblocks * R * C, 1.0f);
        bsr_matrix<float> bsr(
            std::move(block_data),
            std::vector<int>(indptr),
            std::vector<int>(indices),
            M, K, R, C, nblocks);
        bsr_matrix<bfloat16> bsr_bf16 = bsr.bfloat16_cast();

        srand(42);  // fixed seed for reproducible dense data
        dense_matrix<float> dense(K, N, RAND);
        dense_matrix<bfloat16> dense_bf16 = dense.bfloat16_cast();

        return {{bsr_bf16, dense_bf16, name}};
    }}

    // ═══════════════════════════════════════════════════════════════════════
    // Frozen random-pattern explicit specializations ({len(unique_random)})
    // ═══════════════════════════════════════════════════════════════════════

{chr(10).join(specializations)}
    // ═══════════════════════════════════════════════════════════════════════
    // Deterministic-pattern template definitions
    // ═══════════════════════════════════════════════════════════════════════

    template <uint32_t M = 8192, uint32_t N = 8192, uint32_t K = 8192,
              uint32_t R = 64, uint32_t C = 64, uint32_t DensityPPM = 250000>
    inline ProfileCaseReturnType frozen_parametric_row() {{
        uint32_t block_matrix_height = M / R;
        uint32_t block_matrix_width  = K / C;
        constexpr float density = DensityPPM / 1000000.0f;
        uint32_t nblocks = std::max(1u, uint32_t(std::round(block_matrix_height * block_matrix_width * density)));

        srand(42);
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_ROW, RAND);
        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16   = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        char buf[128];
        size_t n = sprintf(buf, "parametric_row_M%u_N%u_K%u_R%u_C%u_dppm%u", M, N, K, R, C, DensityPPM);
        std::string test_name(buf, n);
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, test_name);
    }}

    template <uint32_t M = 8192, uint32_t N = 8192, uint32_t K = 8192,
              uint32_t R = 64, uint32_t C = 64, uint32_t DensityPPM = 250000>
    inline ProfileCaseReturnType frozen_parametric_col() {{
        uint32_t block_matrix_height = M / R;
        uint32_t block_matrix_width  = K / C;
        constexpr float density = DensityPPM / 1000000.0f;
        uint32_t nblocks = std::max(1u, uint32_t(std::round(block_matrix_height * block_matrix_width * density)));

        srand(42);
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_COL, RAND);
        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16   = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        char buf[128];
        size_t n = sprintf(buf, "parametric_col_M%u_N%u_K%u_R%u_C%u_dppm%u", M, N, K, R, C, DensityPPM);
        std::string test_name(buf, n);
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, test_name);
    }}

    template <uint32_t M = 8192, uint32_t N = 8192, uint32_t K = 8192,
              uint32_t R = 64, uint32_t C = 64, uint32_t DensityPPM = 250000>
    inline ProfileCaseReturnType frozen_parametric_multi_diag() {{
        uint32_t block_matrix_height = M / R;
        uint32_t block_matrix_width  = K / C;
        constexpr float density = DensityPPM / 1000000.0f;
        uint32_t nblocks = std::max(1u, uint32_t(std::round(block_matrix_height * block_matrix_width * density)));

        srand(42);
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_MULTI_DIAG, RAND);
        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16   = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        char buf[128];
        size_t n = sprintf(buf, "parametric_multi_diag_M%u_N%u_K%u_R%u_C%u_dppm%u", M, N, K, R, C, DensityPPM);
        std::string test_name(buf, n);
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, test_name);
    }}

}} // namespace sc26_profiling_suite
"""
    return header


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate frozen SC26 profiling suite")
    script_dir = Path(__file__).parent
    parser.add_argument("--log-dir",
        default=str(script_dir / "../../../../../../profiles_sc26/csvs"),
        help="Path to profiles_sc26/csvs directory")
    parser.add_argument("--output",
        default=str(script_dir / "../inc/sc26_profiling_suite.hpp"),
        help="Output header file path")
    args = parser.parse_args()

    log_dir = os.path.realpath(args.log_dir)
    output = os.path.realpath(args.output)

    print(f"Scanning logs in: {log_dir}")
    canonical = discover_canonical_logs(log_dir)
    print(f"Found {len(canonical)} unique test basenames")

    random_count = sum(1 for b in canonical if not is_deterministic_pattern(b))
    det_count = sum(1 for b in canonical if is_deterministic_pattern(b))
    print(f"  {random_count} random patterns (will be frozen as template specializations)")
    print(f"  {det_count} deterministic patterns (generic templates)")

    # Verify all registry entries have a canonical log
    missing = []
    for group_name, entries in REGISTRY_SPEC.items():
        for entry in entries:
            if entry not in canonical:
                missing.append(f"  {group_name}: {entry}")
    if missing:
        print("WARNING: Registry entries without canonical logs:")
        for m in missing:
            print(m)
        sys.exit(1)

    print(f"Generating: {output}")
    header = generate_header(canonical)

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        f.write(header)

    size_kb = os.path.getsize(output) / 1024
    print(f"Generated {output} ({size_kb:.1f} KB)")
    print("Done.")


if __name__ == "__main__":
    main()
