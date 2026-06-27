#include "features/pooling/api.h"

#include "features/pooling/factories.h"
#include "features/pooling/validation.h"

namespace mlx_lattice {

mx::array sparse_pool_features(
    PoolReduceOp op,
    const mx::array& feats,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    const mx::array& row_offsets,
    const mx::array& counts,
    int out_capacity,
    int n_kernels,
    PoolInputLayout input_layout
) {
    validate_sparse_pool_features(
        feats,
        in_rows,
        out_rows,
        kernel_ids,
        row_offsets,
        counts,
        out_capacity,
        n_kernels
    );
    return make_sparse_pool_features(
        op,
        feats,
        SparseRelationEdges{in_rows, out_rows, kernel_ids},
        SparseRelationContract{counts, out_capacity, n_kernels},
        SparseRelationCSRView{row_offsets, row_offsets},
        input_layout
    );
}

} // namespace mlx_lattice
