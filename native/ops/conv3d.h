#pragma once

#include "mlx/array.h"
#include "mlx/ops.h"
#include "mlx/utils.h"

namespace mlx_lattice {

namespace mx = mlx::core;

mx::array conv3d_feats(
    const mx::array& feats,
    const mx::array& weight,
    const mx::array& maps,
    const mx::array& kernels,
    int out_rows,
    mx::StreamOrDevice stream = {}
);

mx::array conv3d_subm_feats(
    const mx::array& feats,
    const mx::array& weight,
    const mx::array& maps,
    const mx::array& kernels,
    int center_kernel,
    mx::StreamOrDevice stream = {}
);

mx::array conv3d_residual_feats(
    const mx::array& base,
    const mx::array& feats,
    const mx::array& weight,
    const mx::array& maps,
    const mx::array& kernels,
    const mx::array& offsets,
    mx::StreamOrDevice stream = {}
);

} // namespace mlx_lattice
