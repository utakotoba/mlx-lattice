#pragma once

#include "features/convolution/contract.h"

namespace mlx_lattice {

mx::Stream sparse_quantized_conv_stream(
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& scales,
    const mx::array& biases,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    const mx::array& counts,
    const mx::array& row_offsets
);

mx::Stream sparse_conv_features_stream(
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    const mx::array& counts,
    const mx::array& row_offsets
);

mx::Stream sparse_conv_implicit_gemm_stream(
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& out_in_map
);

mx::Stream sparse_conv_grad_stream(
    const mx::array& lhs,
    const mx::array& rhs,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    const mx::array& counts,
    const mx::array& row_offsets,
    const mx::array& grouped_row_offsets,
    const mx::array& grouped_edge_ids
);

} // namespace mlx_lattice
