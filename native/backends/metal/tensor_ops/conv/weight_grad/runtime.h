#pragma once

#include <vector>

#include "mlx/stream.h"
#include "ops/exec/types.h"

namespace mlx_lattice::backend::metal::tensor_ops::conv::weight_grad {

bool supports(SparseConvShape shape);
bool is_preferred(SparseConvShape shape, const mx::Stream& stream);

void encode(
    SparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    mx::array& out
);

} // namespace mlx_lattice::backend::metal::tensor_ops::conv::weight_grad
