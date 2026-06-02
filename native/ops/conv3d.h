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

mx::array pool3d_feats(
    const mx::array& feats,
    const mx::array& maps,
    const mx::array& kernels,
    const mx::array& offsets,
    int out_rows,
    mx::StreamOrDevice stream = {}
);

mx::array max_pool3d_feats(
    const mx::array& feats,
    const mx::array& maps,
    const mx::array& kernels,
    const mx::array& offsets,
    int out_rows,
    mx::StreamOrDevice stream = {}
);

mx::array pool3d_feats_grad(
    const mx::array& grad,
    const mx::array& maps,
    const mx::array& kernels,
    int rows,
    mx::StreamOrDevice stream = {}
);

mx::array conv3d_feats_grad(
    const mx::array& grad,
    const mx::array& weight,
    const mx::array& maps,
    const mx::array& kernels,
    int rows,
    mx::StreamOrDevice stream = {}
);

mx::array conv3d_weight_grad(
    const mx::array& feats,
    const mx::array& grad,
    const mx::array& maps,
    const mx::array& kernels,
    int kernel_count,
    mx::StreamOrDevice stream = {}
);

} // namespace mlx_lattice
