#pragma once

#include "ops/coords/types.h"

namespace mlx_lattice {

namespace mx = mlx::core;

enum class PoolReduceOp {
    Sum,
    Max,
    Avg,
};

enum class PoolInputLayout {
    Overlap,
    Exclusive,
};

enum class SparseMapOp {
    Forward,
    Transposed,
    Generative,
};

enum SparseOutputSlot : std::size_t {
    SparseOutFeats = 0,
    SparseOutCoords,
    SparseCounts,
    SparseOutputCount,
};

struct NativeSparseTensorOutput {
    mx::array coords;
    mx::array feats;
    mx::array counts;
};

struct SparseConvShape {
    int in_capacity;
    int out_capacity;
    int n_kernels;
    int in_channels;
    int out_channels;
    int weight_layout;
    int kernel_x;
    int kernel_y;
    int kernel_z;
    int store_sorted = 0;
};

inline bool operator==(SparseConvShape lhs, SparseConvShape rhs) {
    return lhs.in_capacity == rhs.in_capacity &&
           lhs.out_capacity == rhs.out_capacity &&
           lhs.n_kernels == rhs.n_kernels &&
           lhs.in_channels == rhs.in_channels &&
           lhs.out_channels == rhs.out_channels &&
           lhs.weight_layout == rhs.weight_layout &&
           lhs.kernel_x == rhs.kernel_x && lhs.kernel_y == rhs.kernel_y &&
           lhs.kernel_z == rhs.kernel_z && lhs.store_sorted == rhs.store_sorted;
}

inline bool operator!=(SparseConvShape lhs, SparseConvShape rhs) {
    return !(lhs == rhs);
}

struct SparseConvPlan {
    mx::array in_row_offsets;
    mx::array in_edge_ids;
    mx::array kernel_row_offsets;
    mx::array kernel_edge_ids;
};

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

struct SparsePoolShape {
    int in_capacity;
    int out_capacity;
    int n_kernels;
    int channels;
    bool input_exclusive;
};

inline bool operator==(SparsePoolShape lhs, SparsePoolShape rhs) {
    return lhs.in_capacity == rhs.in_capacity &&
           lhs.out_capacity == rhs.out_capacity &&
           lhs.n_kernels == rhs.n_kernels && lhs.channels == rhs.channels &&
           lhs.input_exclusive == rhs.input_exclusive;
}

inline bool operator!=(SparsePoolShape lhs, SparsePoolShape rhs) {
    return !(lhs == rhs);
}

} // namespace mlx_lattice
