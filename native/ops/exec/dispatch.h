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
    const mx::array& edge_count,
    int n_out_rows
);

mx::array dispatch_pool_edges(
    PoolReduceOp op,
    const mx::array& feats,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& edge_count,
    int n_out_rows
);

NativeSparseTensorOutput dispatch_sparse_conv(
    SparseMapOp op,
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& offsets,
    Triple stride,
    Triple padding
);

NativeSparseTensorOutput dispatch_sparse_pool(
    PoolReduceOp reduce,
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& feats,
    const mx::array& offsets,
    Triple stride,
    Triple padding
);

} // namespace mlx_lattice
