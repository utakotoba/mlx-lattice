#pragma once

#include "features/coordinates/types.h"
#include "foundation/sparse_relation.h"

namespace mlx_lattice {

struct SparseConvProblem {
    int in_capacity;
    int out_capacity;
    int kernel_count;
    int in_channels;
    int out_channels;
    int kernel_x;
    int kernel_y;
    int kernel_z;
};

enum class ConvWeightLayout : std::uint8_t {
    KernelMajor = 0,
    Dense5d = 1,
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

struct QuantizedSparseConvShape {
    int in_capacity;
    int out_capacity;
    int n_kernels;
    int in_channels;
    int out_channels;
    int storage_in_channels;
    int group_size;
    int bits;
};

inline bool
operator==(QuantizedSparseConvShape lhs, QuantizedSparseConvShape rhs) {
    return lhs.in_capacity == rhs.in_capacity &&
           lhs.out_capacity == rhs.out_capacity &&
           lhs.n_kernels == rhs.n_kernels &&
           lhs.in_channels == rhs.in_channels &&
           lhs.out_channels == rhs.out_channels &&
           lhs.storage_in_channels == rhs.storage_in_channels &&
           lhs.group_size == rhs.group_size && lhs.bits == rhs.bits;
}

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

struct SparseConvSortedImplicitGemmView {
    mx::array sorted_out_in_map;
    mx::array sorted_kv_out_in_map;
    mx::array reorder_rows;
    mx::array tile_masks;
};

} // namespace mlx_lattice
