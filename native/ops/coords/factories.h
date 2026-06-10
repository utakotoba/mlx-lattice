#pragma once

#include "ops/coords.h"

namespace mlx_lattice {

NativeCoordSet make_downsample_coords(const mx::array& coords, Triple stride);
NativeCoordSet make_union_coords(const mx::array& lhs, const mx::array& rhs);
NativeCoordSet
make_intersection_coords(const mx::array& lhs, const mx::array& rhs);
mx::array make_lookup_coords(const mx::array& coords, const mx::array& queries);

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
