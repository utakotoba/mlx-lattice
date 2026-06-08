#pragma once

#include "ops/exec.h"

namespace mlx_lattice {

void validate_spmm_edges(
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    const mx::array& edge_count,
    int n_out_rows
);

void validate_pool_edges(
    const mx::array& feats,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& edge_count,
    int n_out_rows
);

void validate_sparse_conv(
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& feats,
    const mx::array& weights
);

void validate_sparse_pool(
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& feats
);

} // namespace mlx_lattice
