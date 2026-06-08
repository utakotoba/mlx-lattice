#pragma once

#include "mlx/array.h"

namespace mlx_lattice {

namespace mx = mlx::core;

struct SpmmEdgesShape {
    int edge_count;
    int in_channels;
    int out_channels;
    int n_out_rows;
};

} // namespace mlx_lattice
