#pragma once

#include <vector>

#include "mlx/stream.h"
#include "ops/exec/types.h"

namespace mlx_lattice::exec::metal {

bool supports(
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids
);

bool supports_pool(
    const mx::array& feats,
    const mx::array& in_rows,
    const mx::array& out_rows
);

void eval_spmm_edges(
    SpmmEdgesShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
);

void eval_pool_edges(
    PoolReduceOp op,
    PoolEdgesShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
);

} // namespace mlx_lattice::exec::metal
