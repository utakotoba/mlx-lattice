#pragma once

#include "ops/exec.h"

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
    const mx::array& sorted_out_in_map,
    const mx::array& reorder_rows,
    const mx::array& tile_masks,
    int out_capacity,
    int n_kernels
);

mx::array make_sparse_pool_features(
    PoolReduceOp reduce,
    const mx::array& feats,
    const SparseRelationEdges& edges,
    const SparseRelationContract& contract,
    const SparseRelationCSRView& output_csr,
    PoolInputLayout input_layout
);

} // namespace mlx_lattice
