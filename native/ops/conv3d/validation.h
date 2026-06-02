#pragma once

#include "mlx/array.h"

namespace mlx_lattice {

namespace mx = mlx::core;

void validate_conv3d_feats(
    const mx::array& feats,
    const mx::array& weight,
    const mx::array& maps,
    const mx::array& kernels,
    int out_rows
);

void validate_conv3d_residual_feats(
    const mx::array& base,
    const mx::array& feats,
    const mx::array& weight,
    const mx::array& maps,
    const mx::array& kernels,
    const mx::array& offsets
);

void validate_pool3d_feats(
    const mx::array& feats,
    const mx::array& maps,
    const mx::array& kernels,
    const mx::array& offsets,
    int out_rows
);

void validate_max_pool3d_feats(
    const mx::array& feats,
    const mx::array& maps,
    const mx::array& kernels,
    const mx::array& offsets,
    int out_rows
);

void validate_pool3d_feats_grad(
    const mx::array& grad,
    const mx::array& maps,
    const mx::array& kernels,
    int rows
);

void validate_conv3d_feats_grad(
    const mx::array& grad,
    const mx::array& weight,
    const mx::array& maps,
    const mx::array& kernels,
    int rows
);

void validate_conv3d_weight_grad(
    const mx::array& feats,
    const mx::array& grad,
    const mx::array& maps,
    const mx::array& kernels,
    int kernel_count
);

} // namespace mlx_lattice
