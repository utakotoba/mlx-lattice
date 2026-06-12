#pragma once

#include <vector>

#include "ops/coords/types.h"

namespace mlx_lattice {

std::vector<Triple> kernel_offsets(Triple kernel_size);
std::vector<Triple> kernel_offsets(Triple kernel_size, Triple dilation);

NativeCoordSet downsample_coords(const mx::array& coords, Triple stride);
NativeCoordSet union_coords(const mx::array& lhs, const mx::array& rhs);
NativeCoordSet intersection_coords(const mx::array& lhs, const mx::array& rhs);
mx::array lookup_coords(const mx::array& coords, const mx::array& queries);
mx::array morton_codes(const mx::array& coords);
NativeSparseOccupancy
occupancy_downsample(const mx::array& coords, const mx::array& active_rows);
NativeOccupancyExpansion occupancy_expand(
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& occupancy
);
mx::array child_coords_from_indices(
    const mx::array& parent_coords,
    const mx::array& child_indices
);

NativeSparseQuantization sparse_quantize(
    const mx::array& points,
    const mx::array& batch_indices,
    const mx::array& active_rows,
    QuantizationSpec spec
);

mx::array voxelize_features(
    const mx::array& feats,
    const mx::array& inverse_rows,
    const mx::array& voxel_counts,
    const mx::array& active_rows,
    VoxelReduceOp reduce
);

NativeKernelRelation build_kernel_relation(
    const mx::array& coords,
    const mx::array& active_rows,
    Triple kernel_size,
    Triple stride,
    Triple padding,
    Triple dilation
);

NativeKernelRelation build_generative_relation(
    const mx::array& coords,
    const mx::array& active_rows,
    Triple kernel_size,
    Triple stride
);

NativeKernelRelation build_transposed_kernel_relation(
    const mx::array& coords,
    const mx::array& active_rows,
    Triple kernel_size,
    Triple stride,
    Triple padding,
    Triple dilation
);

NativeKernelRelation build_target_kernel_relation(
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& target_coords,
    const mx::array& target_active_rows,
    Triple kernel_size,
    Triple stride,
    Triple padding,
    Triple dilation
);

NativeNeighborRelation build_knn_relation(
    const mx::array& source_coords,
    const mx::array& source_active_rows,
    const mx::array& query_coords,
    const mx::array& query_active_rows,
    int k
);

NativeNeighborRelation build_radius_relation(
    const mx::array& source_coords,
    const mx::array& source_active_rows,
    const mx::array& query_coords,
    const mx::array& query_active_rows,
    double radius,
    int max_neighbors
);

} // namespace mlx_lattice
