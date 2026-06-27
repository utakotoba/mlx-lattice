#pragma once

#include "features/coordinates/api.h"

namespace mlx_lattice {

NativeCoordSet make_downsample_coords(const mx::array& coords, Triple stride);
NativeCoordSet make_union_coords(const mx::array& lhs, const mx::array& rhs);
NativeCoordSet
make_intersection_coords(const mx::array& lhs, const mx::array& rhs);
mx::array make_lookup_coords(const mx::array& coords, const mx::array& queries);
mx::array make_morton_codes(const mx::array& coords);
NativeSparseOccupancy make_occupancy_downsample(
    const mx::array& coords,
    const mx::array& active_rows
);
NativeOccupancyExpansion make_occupancy_expand(
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& occupancy
);
mx::array make_child_coords_from_indices(
    const mx::array& parent_coords,
    const mx::array& child_indices
);

NativeSparseQuantization make_sparse_quantize(
    const mx::array& points,
    const mx::array& batch_indices,
    const mx::array& active_rows,
    QuantizationSpec spec
);

mx::array make_voxelize_features(
    const mx::array& feats,
    const mx::array& inverse_rows,
    const mx::array& voxel_counts,
    const mx::array& active_rows,
    VoxelReduceOp reduce
);

NativeKernelRelation make_kernel_relation(
    const mx::array& coords,
    const mx::array& active_rows,
    Triple kernel_size,
    Triple stride,
    Triple padding,
    Triple dilation
);

NativeKernelRelation make_generative_relation(
    const mx::array& coords,
    const mx::array& active_rows,
    Triple kernel_size,
    Triple stride
);

NativeKernelRelation make_transposed_kernel_relation(
    const mx::array& coords,
    const mx::array& active_rows,
    Triple kernel_size,
    Triple stride,
    Triple padding,
    Triple dilation
);

NativeKernelRelation make_target_kernel_relation(
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& target_coords,
    const mx::array& target_active_rows,
    Triple kernel_size,
    Triple stride,
    Triple padding,
    Triple dilation
);

NativeKernelRelationViews make_kernel_relation_views(
    const mx::array& in_rows,
    const mx::array& kernel_ids,
    const mx::array& counts,
    int in_capacity,
    int kernel_count
);

NativeRelationGroupedView make_relation_grouped_view(
    const mx::array& group_ids,
    const mx::array& counts,
    int group_count
);

NativeRelationDirectView make_relation_direct_view(
    const mx::array& group_ids,
    const mx::array& counts,
    int group_count
);

NativeRelationImplicitGemmView make_relation_implicit_gemm_view(
    const mx::array& source_coords,
    const mx::array& source_active_rows,
    const mx::array& output_coords,
    const mx::array& output_active_rows,
    const mx::array& offsets,
    CoordRelationOp op,
    Triple stride,
    Triple padding
);

NativeNeighborRelation make_knn_relation(
    const mx::array& source_coords,
    const mx::array& source_active_rows,
    const mx::array& query_coords,
    const mx::array& query_active_rows,
    int k
);

NativeNeighborRelation make_radius_relation(
    const mx::array& source_coords,
    const mx::array& source_active_rows,
    const mx::array& query_coords,
    const mx::array& query_active_rows,
    double radius,
    int max_neighbors
);

} // namespace mlx_lattice
