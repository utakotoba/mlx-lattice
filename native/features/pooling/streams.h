#pragma once

#include "features/pooling/contract.h"

namespace mlx_lattice {

mx::Stream sparse_pool_features_stream(
    const mx::array& feats,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    const mx::array& row_offsets,
    const mx::array& counts
);

} // namespace mlx_lattice
