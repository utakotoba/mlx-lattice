#pragma once

#include "ops/exec.h"
#include "ops/exec/types.h"

namespace mlx_lattice {

mx::array dispatch_spmm_edges(
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    int n_out_rows
);

mx::array dispatch_pool_edges(
    PoolReduceOp op,
    const mx::array& feats,
    const mx::array& in_rows,
    const mx::array& out_rows,
    int n_out_rows
);

} // namespace mlx_lattice
