#pragma once

#include "ops/exec.h"

namespace mlx_lattice {

void validate_spmm_edges(
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    int n_out_rows
);

void validate_pool_edges(
    const mx::array& feats,
    const mx::array& in_rows,
    const mx::array& out_rows,
    int n_out_rows
);

} // namespace mlx_lattice
