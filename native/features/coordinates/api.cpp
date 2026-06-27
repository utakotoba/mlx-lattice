#include "features/coordinates/api.h"

#include <stdexcept>

#include "features/coordinates/factories.h"
#include "features/coordinates/validation.h"

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

NativeSparseAlignment build_sparse_alignment(
    const mx::array& lhs_coords,
    const mx::array& lhs_active_rows,
    const mx::array& rhs_coords,
    const mx::array& rhs_active_rows,
    SparseJoinOp join
) {
    validate_coord_pair(lhs_coords, rhs_coords);
    validate_active_rows(lhs_active_rows);
    validate_active_rows(rhs_active_rows);
    return make_sparse_alignment(
        lhs_coords, lhs_active_rows, rhs_coords, rhs_active_rows, join
    );
}

mx::array morton_codes(const mx::array& coords) {
    validate_coords(coords);
    return make_morton_codes(coords);
}

NativeSparseOccupancy
occupancy_downsample(const mx::array& coords, const mx::array& active_rows) {
    validate_coords(coords);
    validate_active_rows(active_rows);
    return make_occupancy_downsample(coords, active_rows);
}

NativeOccupancyExpansion occupancy_expand(
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& occupancy
) {
    validate_coords(coords);
    validate_active_rows(active_rows);
    validate_row_indices(occupancy, coords.shape(0), "occupancy");
    return make_occupancy_expand(coords, active_rows, occupancy);
}

mx::array child_coords_from_indices(
    const mx::array& parent_coords,
    const mx::array& child_indices
) {
    validate_coords(parent_coords);
    validate_row_indices(
        child_indices, parent_coords.shape(0), "child_indices"
    );
    return make_child_coords_from_indices(parent_coords, child_indices);
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

NativePointVoxelMap build_point_voxel_map(
    const mx::array& points,
    const mx::array& batch_indices,
    const mx::array& point_active_rows,
    const mx::array& voxel_coords,
    const mx::array& voxel_active_rows,
    QuantizationSpec spec,
    PointVoxelInterpolationOp interpolation
) {
    validate_points(points);
    validate_batch_indices(batch_indices, points.shape(0));
    validate_active_rows(point_active_rows);
    validate_coords(voxel_coords);
    validate_active_rows(voxel_active_rows);
    validate_positive(spec.voxel_size, "voxel_size");
    return make_point_voxel_map(
        points,
        batch_indices,
        point_active_rows,
        voxel_coords,
        voxel_active_rows,
        spec,
        interpolation
    );
}

mx::array interpolate_point_features(
    const mx::array& voxel_feats,
    const mx::array& rows,
    const mx::array& weights
) {
    validate_feature_matrix(voxel_feats);
    validate_interpolation_rows(rows, rows.shape(0));
    validate_interpolation_weights(weights, rows.shape(0));
    return make_interpolate_point_features(voxel_feats, rows, weights);
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

NativeKernelRelation build_target_kernel_relation(
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& target_coords,
    const mx::array& target_active_rows,
    Triple kernel_size,
    Triple stride,
    Triple padding,
    Triple dilation
) {
    validate_coord_pair(coords, target_coords);
    validate_active_rows(active_rows);
    validate_active_rows(target_active_rows);
    validate_positive(kernel_size, "kernel_size");
    validate_positive(stride, "stride");
    validate_nonnegative(padding, "padding");
    validate_positive(dilation, "dilation");
    return make_target_kernel_relation(
        coords,
        active_rows,
        target_coords,
        target_active_rows,
        kernel_size,
        stride,
        padding,
        dilation
    );
}

NativeRelationImplicitGemmView build_relation_implicit_gemm_view(
    const mx::array& source_coords,
    const mx::array& source_active_rows,
    const mx::array& output_coords,
    const mx::array& output_active_rows,
    const mx::array& offsets,
    CoordRelationOp op,
    Triple stride,
    Triple padding
) {
    validate_coord_pair(source_coords, output_coords);
    validate_active_rows(source_active_rows);
    validate_active_rows(output_active_rows);
    validate_positive(stride, "stride");
    validate_nonnegative(padding, "padding");
    if (offsets.ndim() != 2 || offsets.shape(1) != 3 ||
        offsets.dtype() != mx::int32) {
        throw std::invalid_argument(
            "offsets must have shape (K, 3) and int32 dtype."
        );
    }
    if (op != CoordRelationOp::Forward) {
        throw std::invalid_argument(
            "implicit GEMM view currently supports forward-style relations."
        );
    }
    return make_relation_implicit_gemm_view(
        source_coords,
        source_active_rows,
        output_coords,
        output_active_rows,
        offsets,
        op,
        stride,
        padding
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
