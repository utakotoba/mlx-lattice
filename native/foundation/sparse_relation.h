#pragma once

#include "mlx/array.h"

namespace mlx_lattice {

namespace mx = mlx::core;

struct SparseRelationEdges {
    mx::array in_rows;
    mx::array out_rows;
    mx::array kernel_ids;
};

struct SparseRelationContract {
    mx::array counts;
    int out_capacity;
    int n_kernels;
};

struct SparseRelationCSRView {
    mx::array row_offsets;
    mx::array edge_ids;
};

struct SparseRelationExecutionViews {
    SparseRelationCSRView output_csr;
    SparseRelationCSRView input_csr;
    SparseRelationCSRView kernel_csr;
};

} // namespace mlx_lattice
