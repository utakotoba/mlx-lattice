#pragma once

#include "features/pooling/api.h"

namespace mlx_lattice {

mx::array make_sparse_pool_features(
    PoolReduceOp reduce,
    const mx::array& feats,
    const SparseRelationEdges& edges,
    const SparseRelationContract& contract,
    const SparseRelationCSRView& output_csr,
    PoolInputLayout input_layout
);

} // namespace mlx_lattice
