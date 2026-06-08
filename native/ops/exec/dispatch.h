#pragma once

#include "ops/exec.h"

namespace mlx_lattice {

mx::array dispatch_spmm_edges(
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    int n_out_rows
);

} // namespace mlx_lattice
