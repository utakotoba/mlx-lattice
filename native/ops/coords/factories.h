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
