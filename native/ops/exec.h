#pragma once

#include "ops/exec/types.h"

namespace mlx_lattice {

namespace mx = mlx::core;

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

mx::array sparse_conv_features(
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    const mx::array& counts,
    int out_capacity,
    int n_kernels
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
