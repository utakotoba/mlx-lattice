#pragma once

#include "mlx/array.h"
#include "ops/exec/types.h"

namespace mlx_lattice {

namespace mx = mlx::core;

mx::array spmm_edges(
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    const mx::array& edge_count,
    int n_out_rows
);

mx::array pool_sum_edges(
    const mx::array& feats,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& edge_count,
    int n_out_rows
);

mx::array pool_max_edges(
    const mx::array& feats,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& edge_count,
    int n_out_rows
);

NativeSparseTensorOutput sparse_conv(
    SparseMapOp op,
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& feats,
    const mx::array& weights,
    Triple kernel_size,
    Triple stride,
    Triple padding,
    Triple dilation
);

NativeSparseTensorOutput sparse_pool(
    PoolReduceOp op,
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& feats,
    Triple kernel_size,
    Triple stride,
    Triple padding,
    Triple dilation
);

} // namespace mlx_lattice
