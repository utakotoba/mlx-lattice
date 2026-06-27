#pragma once

#include "features/pooling/contract.h"

namespace mlx_lattice {

mx::array sparse_pool_features(
    PoolReduceOp op,
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
