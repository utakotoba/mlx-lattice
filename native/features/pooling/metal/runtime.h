#pragma once

#include <vector>

#include "features/pooling/contract.h"
#include "mlx/stream.h"

namespace mlx_lattice::backend::metal::pool {

void eval(
    PoolReduceOp reduce,
    SparsePoolShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
);

void eval_grad(
    PoolReduceOp reduce,
    SparsePoolShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
);

void eval_jvp(
    PoolReduceOp reduce,
    SparsePoolShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
);

} // namespace mlx_lattice::backend::metal::pool
