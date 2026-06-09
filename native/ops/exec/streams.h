#pragma once

#include "ops/exec/types.h"

namespace mlx_lattice {

mx::Stream sparse_conv_stream(
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& offsets
);

mx::Stream sparse_conv_features_stream(
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    const mx::array& counts
);

mx::Stream sparse_pool_stream(
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& feats,
    const mx::array& offsets
);

} // namespace mlx_lattice
