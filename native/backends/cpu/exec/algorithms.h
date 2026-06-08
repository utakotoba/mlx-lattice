#pragma once

#include <vector>

#include "ops/exec/types.h"

namespace mlx_lattice::exec::cpu {

void eval_spmm_edges(
    SpmmEdgesShape shape,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
);

} // namespace mlx_lattice::exec::cpu
