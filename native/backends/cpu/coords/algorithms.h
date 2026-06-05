#pragma once

#include <vector>

#include "ops/coords/types.h"

namespace mlx_lattice::coords::cpu {

void eval_set_coords(
    CoordSetOp op,
    Triple stride,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
);

void eval_lookup_coords(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
);

void eval_generic_kernel_map(
    CoordMapOp op,
    Triple stride,
    Triple padding,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
);

void eval_generative_kernel_map(
    Triple stride,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
);

} // namespace mlx_lattice::coords::cpu
