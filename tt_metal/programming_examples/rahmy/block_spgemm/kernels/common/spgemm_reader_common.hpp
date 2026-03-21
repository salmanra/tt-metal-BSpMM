#pragma once
#include <stdint.h>

// Circular buffer assignments for SpGEMM.
// SpGEMM needs CB slots for TWO sparse inputs (A and B) plus output.
// Unlike SpMM where B is dense, here both inputs carry indptr/indices metadata.

namespace spgemm {

constexpr uint32_t cb_id_in0         = 0;   // A data blocks
constexpr uint32_t cb_id_in1         = 1;   // B data blocks
constexpr uint32_t cb_id_a_indices   = 2;   // A col_indices
constexpr uint32_t cb_id_a_indptr    = 3;   // A indptr
constexpr uint32_t cb_id_b_indices   = 4;   // B col_indices
constexpr uint32_t cb_id_b_indptr    = 5;   // B indptr
constexpr uint32_t cb_id_out         = 16;  // Output data blocks
constexpr uint32_t cb_id_intermed    = 24;  // Intermediate accumulation

} // namespace spgemm
