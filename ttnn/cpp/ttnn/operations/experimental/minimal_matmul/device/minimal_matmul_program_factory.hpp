// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"

namespace ttnn::experimental::ccl {
// Stub for fused-op signaler used by the minimal_matmul factory helper.
// The standalone minimal_matmul_factory never populates the optional, so these
// methods are never called; they exist only to satisfy the compiler.
struct MinimalMatmulFusedOpSignaler {
    bool read_local_slice_from_input = false;
    std::optional<tt::tt_metal::Tensor> ag_input = std::nullopt;

    void init_fused_op(
        tt::tt_metal::Program& /*program*/,
        const tt::tt_metal::IDevice* /*device*/,
        const CoreRange& /*cores*/) {}

    void push_matmul_fused_op_rt_args(
        std::vector<uint32_t>& /*out_rt_args*/,
        uint32_t /*k_blocks*/,
        uint32_t /*k_block_tiles*/) {}
};
}  // namespace ttnn::experimental::ccl

namespace ttnn::operations::experimental::minimal_matmul {

namespace helpers {
void override_program_parameters(
    const ttnn::operations::experimental::minimal_matmul::minimal_matmul_override_variables_t& override_variables,
    const void* operation,
    tt::tt_metal::Program& program,
    const std::vector<tt::tt_metal::Tensor>& input_tensors,
    const std::vector<std::optional<const tt::tt_metal::Tensor>>& optional_input_tensors,
    const std::vector<tt::tt_metal::Tensor>& output_tensors);
}

namespace detail {
ttnn::operations::experimental::minimal_matmul::minimal_matmul_override_variables_t minimal_matmul_factory_helper(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const std::optional<const Tensor>& bias_tensor,
    const std::optional<unary::UnaryWithParam>& fused_activation,
    const std::optional<const MinimalMatmulConfig>& config,
    const Tensor& output_tensor,
    const DeviceComputeKernelConfig& compute_kernel_config,
    std::optional<ttnn::experimental::ccl::MinimalMatmulFusedOpSignaler>& fused_op_signaler);

tt::tt_metal::operation::ProgramWithCallbacks minimal_matmul_factory(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const std::optional<const Tensor>& bias_tensor,
    const std::optional<unary::UnaryWithParam>& fused_activation,
    const std::optional<const MinimalMatmulConfig>& config,
    const Tensor& output_tensor,
    const DeviceComputeKernelConfig& compute_kernel_config);

}  // namespace detail
}  // namespace ttnn::operations::experimental::minimal_matmul
