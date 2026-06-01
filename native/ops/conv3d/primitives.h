#pragma once

#include <memory>

#include "mlx/primitives.h"
#include "mlx/stream.h"

namespace mlx_lattice {

namespace mx = mlx::core;

std::shared_ptr<mx::Primitive> make_conv3d_feats_primitive(
    mx::Stream stream,
    int rows,
    int in_channels,
    int out_channels
);

std::shared_ptr<mx::Primitive> make_conv3d_subm_feats_primitive(
    mx::Stream stream,
    int rows,
    int in_channels,
    int out_channels,
    int center_kernel
);

std::shared_ptr<mx::Primitive> make_conv3d_residual_feats_primitive(
    mx::Stream stream,
    int rows,
    int in_channels,
    int out_channels
);

std::shared_ptr<mx::Primitive>
make_pool3d_feats_primitive(mx::Stream stream, int rows, int channels);

std::shared_ptr<mx::Primitive>
make_pool3d_feats_grad_primitive(mx::Stream stream, int rows, int channels);

std::shared_ptr<mx::Primitive> make_conv3d_feats_grad_primitive(
    mx::Stream stream,
    int rows,
    int in_channels,
    int out_channels
);

std::shared_ptr<mx::Primitive> make_conv3d_weight_grad_primitive(
    mx::Stream stream,
    int kernels,
    int in_channels,
    int out_channels
);

} // namespace mlx_lattice
