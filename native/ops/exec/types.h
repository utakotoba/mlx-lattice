#pragma once

#include "mlx/array.h"
#include "ops/coords/types.h"

namespace mlx_lattice {

namespace mx = mlx::core;

struct SpmmEdgesShape {
    int edge_count;
    int in_channels;
    int out_channels;
    int n_in_rows;
    int n_out_rows;
    int n_kernels;
};

enum class PoolReduceOp {
    Sum,
    Max,
    Avg,
};

struct PoolEdgesShape {
    int edge_count;
    int channels;
    int n_in_rows;
    int n_out_rows;
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
    int n_in_rows;
    int n_out_rows;
    int n_kernels;
    int in_channels;
    int out_channels;
    int weight_layout;
    int kernel_x;
    int kernel_y;
    int kernel_z;
};

struct SparsePoolShape {
    int n_in_rows;
    int n_out_rows;
    int n_kernels;
    int channels;
};

} // namespace mlx_lattice
