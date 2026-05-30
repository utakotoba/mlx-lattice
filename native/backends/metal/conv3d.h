#pragma once

#include <vector>

#include "mlx/array.h"
#include "mlx/stream.h"

namespace mlx_lattice::metal {

namespace mx = mlx::core;

void eval_conv3d_feats(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs,
    mx::Stream stream,
    int rows,
    int in_channels,
    int out_channels
);

} // namespace mlx_lattice::metal
