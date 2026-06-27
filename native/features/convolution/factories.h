#pragma once

#include "features/convolution/api.h"

namespace mlx_lattice {

mx::array make_sparse_conv_features(
    const mx::array& feats,
    const mx::array& weights,
    const SparseRelationEdges& edges,
    const SparseRelationContract& contract,
    const SparseRelationExecutionViews& views
);

mx::array make_sparse_conv_features_sorted_implicit_gemm(
    const mx::array& feats,
    const mx::array& weights,
    const SparseConvSortedImplicitGemmView& sorted_view,
    const SparseRelationEdges& edges,
    const SparseRelationContract& contract,
    const SparseRelationExecutionViews& views,
    bool store_sorted
);

mx::array make_sparse_conv_features_sorted_direct_reference(
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& sorted_out_in_map,
    const mx::array& reorder_rows,
    const mx::array& tile_masks,
    int out_capacity,
    int n_kernels,
    bool store_sorted
);

} // namespace mlx_lattice
