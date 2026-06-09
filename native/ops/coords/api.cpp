#include "ops/coords.h"

#include <stdexcept>

#include "ops/coords/factories.h"
#include "ops/coords/validation.h"

namespace mlx_lattice {

// MARK: - set ops

NativeCoordSet downsample_coords(const mx::array& coords, Triple stride) {
    validate_coords(coords);
    validate_positive(stride, "stride");
    return make_downsample_coords(coords, stride);
}

NativeCoordSet union_coords(const mx::array& lhs, const mx::array& rhs) {
    validate_coord_pair(lhs, rhs);
    return make_union_coords(lhs, rhs);
}

NativeCoordSet intersection_coords(const mx::array& lhs, const mx::array& rhs) {
    validate_coord_pair(lhs, rhs);
    return make_intersection_coords(lhs, rhs);
}

mx::array lookup_coords(const mx::array& coords, const mx::array& queries) {
    validate_coord_pair(coords, queries);
    return make_lookup_coords(coords, queries);
}

// MARK: - quantization

NativeSparseQuantization sparse_quantize(
    const mx::array& points,
    const mx::array& batch_indices,
    const mx::array& active_rows,
    QuantizationSpec spec
) {
    validate_points(points);
    validate_batch_indices(batch_indices, points.shape(0));
    validate_active_rows(active_rows);
    validate_positive(spec.voxel_size, "voxel_size");
    return make_sparse_quantize(points, batch_indices, active_rows, spec);
}

mx::array voxelize_features(
    const mx::array& feats,
    const mx::array& inverse_rows,
    const mx::array& voxel_counts,
    const mx::array& active_rows,
    VoxelReduceOp reduce
) {
    validate_feature_matrix(feats);
    validate_inverse_rows(inverse_rows, feats.shape(0));
    validate_voxel_counts(voxel_counts, feats.shape(0));
    validate_active_rows(active_rows);
    return make_voxelize_features(
        feats, inverse_rows, voxel_counts, active_rows, reduce
    );
}

// MARK: - relations

NativeKernelRelation build_kernel_relation(
    const mx::array& coords,
    const mx::array& active_rows,
    Triple kernel_size,
    Triple stride,
    Triple padding,
    Triple dilation
) {
    validate_coords(coords);
    validate_active_rows(active_rows);
    validate_positive(kernel_size, "kernel_size");
    validate_positive(stride, "stride");
    validate_nonnegative(padding, "padding");
    validate_positive(dilation, "dilation");
    return make_kernel_relation(
        coords, active_rows, kernel_size, stride, padding, dilation
    );
}

NativeKernelRelation build_generative_relation(
    const mx::array& coords,
    const mx::array& active_rows,
    Triple kernel_size,
    Triple stride
) {
    validate_coords(coords);
    validate_active_rows(active_rows);
    validate_positive(kernel_size, "kernel_size");
    validate_positive(stride, "stride");
    return make_generative_relation(coords, active_rows, kernel_size, stride);
}

NativeKernelRelation build_transposed_kernel_relation(
    const mx::array& coords,
    const mx::array& active_rows,
    Triple kernel_size,
    Triple stride,
    Triple padding,
    Triple dilation
) {
    validate_coords(coords);
    validate_active_rows(active_rows);
    validate_positive(kernel_size, "kernel_size");
    validate_positive(stride, "stride");
    validate_nonnegative(padding, "padding");
    validate_positive(dilation, "dilation");
    return make_transposed_kernel_relation(
        coords, active_rows, kernel_size, stride, padding, dilation
    );
}

NativeNeighborRelation build_knn_relation(
    const mx::array& source_coords,
    const mx::array& source_active_rows,
    const mx::array& query_coords,
    const mx::array& query_active_rows,
    int k
) {
    validate_coord_pair(source_coords, query_coords);
    validate_active_rows(source_active_rows);
    validate_active_rows(query_active_rows);
    if (k <= 0) {
        throw std::invalid_argument("k must be positive.");
    }
    return make_knn_relation(
        source_coords, source_active_rows, query_coords, query_active_rows, k
    );
}

NativeNeighborRelation build_radius_relation(
    const mx::array& source_coords,
    const mx::array& source_active_rows,
    const mx::array& query_coords,
    const mx::array& query_active_rows,
    double radius,
    int max_neighbors
) {
    validate_coord_pair(source_coords, query_coords);
    validate_active_rows(source_active_rows);
    validate_active_rows(query_active_rows);
    if (radius < 0.0) {
        throw std::invalid_argument("radius must be non-negative.");
    }
    if (max_neighbors < 0) {
        throw std::invalid_argument("max_neighbors must be non-negative.");
    }
    return make_radius_relation(
        source_coords,
        source_active_rows,
        query_coords,
        query_active_rows,
        radius,
        max_neighbors
    );
}

} // namespace mlx_lattice
