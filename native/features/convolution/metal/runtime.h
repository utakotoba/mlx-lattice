#pragma once

#include <vector>

#include "features/convolution/contract.h"
#include "mlx/stream.h"

namespace mlx_lattice::backend::metal::conv {

void eval_quantized(
    QuantizedSparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
);

void eval(
    SparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
);

void eval_sorted_implicit_gemm(
    SparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
);

void eval_sorted_direct_reference(
    SparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
);

void eval_input_grad(
    SparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
);

void eval_weight_grad(
    SparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
);

} // namespace mlx_lattice::backend::metal::conv
