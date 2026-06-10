#pragma once

#include "ops/exec/types.h"

namespace mlx_lattice {

mx::Stream sparse_conv_features_stream(
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    const mx::array& counts,
    const mx::array& row_offsets
);

mx::Stream sparse_pool_features_stream(
    const mx::array& feats,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    const mx::array& row_offsets,
    const mx::array& counts
);

} // namespace mlx_lattice
