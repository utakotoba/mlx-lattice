#include "ops/exec.h"

#include <vector>

#include "ops/exec/factories.h"
#include "ops/exec/validation.h"

namespace mlx_lattice {

mx::array sparse_conv_features(
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    const mx::array& counts,
    const mx::array& row_offsets,
    const mx::array& in_row_offsets,
    const mx::array& in_edge_ids,
    const mx::array& kernel_row_offsets,
    const mx::array& kernel_edge_ids,
    int out_capacity,
    int n_kernels
) {
    if (feats.ndim() != 2) {
        throw std::invalid_argument("feats must have shape (N, C_in).");
    }
    if (weights.ndim() != 3 && weights.ndim() != 5) {
        throw std::invalid_argument(
            "weights must have shape (K, C_in, C_out) or "
            "(C_out, Kx, Ky, Kz, C_in)."
        );
    }
    if (feats.dtype() != mx::float32 && feats.dtype() != mx::float16) {
        throw std::invalid_argument(
            "sparse_conv_features supports float32 and float16 feats."
        );
    }
    if (weights.dtype() != feats.dtype()) {
        throw std::invalid_argument(
            "sparse_conv_features weights must match feats dtype."
        );
    }
    if (in_rows.ndim() != 1 || out_rows.ndim() != 1 || kernel_ids.ndim() != 1) {
        throw std::invalid_argument(
            "relation rows and kernel_ids must be one-dimensional."
        );
    }
    if (in_rows.dtype() != mx::int32 || out_rows.dtype() != mx::int32 ||
        kernel_ids.dtype() != mx::int32) {
        throw std::invalid_argument(
            "relation rows and kernel_ids must use int32 dtype."
        );
    }
    if (in_rows.shape(0) != out_rows.shape(0) ||
        in_rows.shape(0) != kernel_ids.shape(0)) {
        throw std::invalid_argument(
            "relation row arrays and kernel_ids must have equal capacity."
        );
    }
    if (counts.shape() != mx::Shape{2} || counts.dtype() != mx::int32) {
        throw std::invalid_argument(
            "counts must have shape (2,) and int32 dtype."
        );
    }
    if (row_offsets.ndim() != 1 || row_offsets.dtype() != mx::int32) {
        throw std::invalid_argument(
            "row_offsets must be a one-dimensional int32 array."
        );
    }
    if (in_row_offsets.ndim() != 1 || in_row_offsets.dtype() != mx::int32 ||
        kernel_row_offsets.ndim() != 1 ||
        kernel_row_offsets.dtype() != mx::int32) {
        throw std::invalid_argument(
            "plan row_offsets must be one-dimensional int32 arrays."
        );
    }
    if (in_edge_ids.ndim() != 1 || kernel_edge_ids.ndim() != 1 ||
        in_edge_ids.dtype() != mx::int32 ||
        kernel_edge_ids.dtype() != mx::int32) {
        throw std::invalid_argument(
            "plan edge_ids must be one-dimensional int32 arrays."
        );
    }
    if (row_offsets.shape(0) != out_capacity + 1) {
        throw std::invalid_argument(
            "row_offsets length must match out_capacity + 1."
        );
    }
    auto weight_in_channels =
        weights.ndim() == 3 ? weights.shape(1) : weights.shape(4);
    if (feats.shape(1) != weight_in_channels) {
        throw std::invalid_argument(
            "feats channels must match weights input channels."
        );
    }
    if (out_capacity < 0 || n_kernels <= 0) {
        throw std::invalid_argument(
            "out_capacity must be nonnegative and n_kernels must be positive."
        );
    }
    if (in_row_offsets.shape(0) != feats.shape(0) + 1) {
        throw std::invalid_argument(
            "in_row_offsets length must match input capacity + 1."
        );
    }
    if (kernel_row_offsets.shape(0) != n_kernels + 1) {
        throw std::invalid_argument(
            "kernel_row_offsets length must match n_kernels + 1."
        );
    }
    if (in_edge_ids.shape(0) != in_rows.shape(0) ||
        kernel_edge_ids.shape(0) != in_rows.shape(0)) {
        throw std::invalid_argument(
            "plan edge_ids must match relation edge capacity."
        );
    }
    if (weights.ndim() == 3 && weights.shape(0) != n_kernels) {
        throw std::invalid_argument(
            "weights kernel rows must match n_kernels."
        );
    }
    if (weights.ndim() == 5 &&
        weights.shape(1) * weights.shape(2) * weights.shape(3) != n_kernels) {
        throw std::invalid_argument(
            "weights kernel rows must match n_kernels."
        );
    }
    auto edges = SparseRelationEdges{in_rows, out_rows, kernel_ids};
    auto contract = SparseRelationContract{counts, out_capacity, n_kernels};
    auto views = SparseRelationExecutionViews{
        SparseRelationCSRView{row_offsets, row_offsets},
        SparseRelationCSRView{in_row_offsets, in_edge_ids},
        SparseRelationCSRView{kernel_row_offsets, kernel_edge_ids},
    };
    return make_sparse_conv_features(feats, weights, edges, contract, views);
}

mx::array sparse_conv_features_sorted_implicit_gemm(
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& sorted_out_in_map,
    const mx::array& reorder_rows,
    const mx::array& tile_masks,
    int out_capacity,
    int n_kernels
) {
    if (sorted_out_in_map.ndim() != 2 ||
        sorted_out_in_map.dtype() != mx::int32) {
        throw std::invalid_argument(
            "sorted_out_in_map must have shape (N_out, K) and int32 dtype."
        );
    }
    if (reorder_rows.ndim() != 1 || reorder_rows.dtype() != mx::int32) {
        throw std::invalid_argument(
            "reorder_rows must have shape (N_out,) and int32 dtype."
        );
    }
    if (tile_masks.ndim() != 1 || tile_masks.dtype() != mx::int32) {
        throw std::invalid_argument(
            "tile_masks must have shape (ceil(N_out / 16),) and int32 dtype."
        );
    }
    if (sorted_out_in_map.shape(0) != out_capacity ||
        sorted_out_in_map.shape(1) != n_kernels ||
        reorder_rows.shape(0) != out_capacity) {
        throw std::invalid_argument(
            "sorted implicit GEMM view shape must match out_capacity and "
            "n_kernels."
        );
    }
    auto expected_tiles = (out_capacity + 15) / 16;
    if (tile_masks.shape(0) != expected_tiles) {
        throw std::invalid_argument(
            "tile_masks length must match ceil(out_capacity / 16)."
        );
    }
    if (feats.ndim() != 2) {
        throw std::invalid_argument("feats must have shape (N, C_in).");
    }
    if (weights.ndim() != 3 && weights.ndim() != 5) {
        throw std::invalid_argument(
            "weights must have shape (K, C_in, C_out) or "
            "(C_out, Kx, Ky, Kz, C_in)."
        );
    }
    if (feats.dtype() != mx::float32 && feats.dtype() != mx::float16) {
        throw std::invalid_argument(
            "sparse_conv_features_sorted_implicit_gemm supports float32 and "
            "float16 feats."
        );
    }
    if (weights.dtype() != feats.dtype()) {
        throw std::invalid_argument(
            "sparse_conv_features_sorted_implicit_gemm weights must match "
            "feats dtype."
        );
    }
    auto weight_in_channels =
        weights.ndim() == 3 ? weights.shape(1) : weights.shape(4);
    if (feats.shape(1) != weight_in_channels) {
        throw std::invalid_argument(
            "feats channels must match weights input channels."
        );
    }
    if (out_capacity < 0 || n_kernels <= 0) {
        throw std::invalid_argument(
            "out_capacity must be nonnegative and n_kernels must be positive."
        );
    }
    if (weights.ndim() == 3 && weights.shape(0) != n_kernels) {
        throw std::invalid_argument(
            "weights kernel rows must match n_kernels."
        );
    }
    if (weights.ndim() == 5 &&
        weights.shape(1) * weights.shape(2) * weights.shape(3) != n_kernels) {
        throw std::invalid_argument(
            "weights kernel rows must match n_kernels."
        );
    }
    return make_sparse_conv_features_sorted_implicit_gemm(
        feats,
        weights,
        sorted_out_in_map,
        reorder_rows,
        tile_masks,
        out_capacity,
        n_kernels
    );
}

mx::array sparse_pool_features(
    PoolReduceOp op,
    const mx::array& feats,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    const mx::array& row_offsets,
    const mx::array& counts,
    int out_capacity,
    int n_kernels,
    PoolInputLayout input_layout
) {
    validate_sparse_pool_features(
        feats,
        in_rows,
        out_rows,
        kernel_ids,
        row_offsets,
        counts,
        out_capacity,
        n_kernels
    );
    return make_sparse_pool_features(
        op,
        feats,
        SparseRelationEdges{in_rows, out_rows, kernel_ids},
        SparseRelationContract{counts, out_capacity, n_kernels},
        SparseRelationCSRView{row_offsets, row_offsets},
        input_layout
    );
}

} // namespace mlx_lattice
