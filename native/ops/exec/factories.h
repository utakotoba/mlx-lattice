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
    const mx::array& row_offsets,
    int out_capacity,
    int n_kernels
);

mx::array make_sparse_pool_features(
    PoolReduceOp reduce,
    const mx::array& feats,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    const mx::array& row_offsets,
    const mx::array& counts,
    int out_capacity,
    int n_kernels,
    PoolInputLayout input_layout
);

} // namespace mlx_lattice
