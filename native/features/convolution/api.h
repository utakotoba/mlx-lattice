#pragma once

#include "features/convolution/contract.h"

namespace mlx_lattice {

mx::array sparse_quantized_conv_features(
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& scales,
    const mx::array& biases,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    const mx::array& counts,
    const mx::array& row_offsets,
    int out_capacity,
    int n_kernels,
    int in_channels,
    int out_channels,
    int storage_in_channels,
    int group_size,
    int bits,
    const QuantizedSparseConvPlan& plan
);

mx::array sparse_conv_features(
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    const mx::array& counts,
    const mx::array& row_offsets,
    const mx::array& in_row_offsets,
    const mx::array& in_edge_ids,
    const mx::array& kernel_row_offsets,
    const mx::array& kernel_edge_ids,
    int out_capacity,
    int n_kernels
);

mx::array sparse_conv_features_sorted_implicit_gemm(
    const mx::array& feats,
    const mx::array& weights,
    const SparseConvSortedImplicitGemmView& sorted_view,
    const SparseRelationEdges& edges,
    const SparseRelationContract& contract,
    const SparseRelationExecutionViews& execution_views,
    bool store_sorted = false
);

mx::array sparse_conv_features_sorted_direct_reference(
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& sorted_out_in_map,
    const mx::array& reorder_rows,
    const mx::array& tile_masks,
    int out_capacity,
    int n_kernels,
    bool store_sorted = false
);

} // namespace mlx_lattice
