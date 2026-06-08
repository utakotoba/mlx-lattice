#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "ops/exec/types.h"

namespace mlx_lattice::exec::cpu {

using Coord = std::array<int64_t, 4>;
using Edge = std::array<int32_t, 3>;

struct Plan {
    std::vector<Coord> out_coords;
    std::vector<Edge> edges;
};

Plan build_plan(
    SparseMapOp op,
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& offsets,
    Triple stride,
    Triple padding
);

void write_coords(mx::array& out, const std::vector<Coord>& coords);
void write_counts(mx::array& out, const Plan& plan);

std::vector<int32_t> pool_degrees(const Plan& plan, int n_out_rows);

} // namespace mlx_lattice::exec::cpu
