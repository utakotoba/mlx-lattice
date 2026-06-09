#pragma once

#include "ops/exec.h"

namespace mlx_lattice {

mx::array make_sparse_conv_features(
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    const mx::array& counts,
    int out_capacity,
    int n_kernels
);

NativeSparseTensorOutput make_sparse_pool(
    PoolReduceOp reduce,
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& feats,
    const mx::array& offsets,
    Triple stride,
    Triple padding
);

} // namespace mlx_lattice
