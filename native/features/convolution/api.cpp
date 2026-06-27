#include "features/convolution/api.h"

#include <string>
#include <vector>

#include "features/convolution/factories.h"

namespace mlx_lattice {

namespace {

void validate_sorted_conv_common(
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& sorted_out_in_map,
    const mx::array& reorder_rows,
    const mx::array& tile_masks,
    int out_capacity,
    int n_kernels,
    const char* op_name
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
            "tile_masks must have shape (ceil(N_out / 64) * 4,) and int32 "
            "dtype."
        );
    }
    if (sorted_out_in_map.shape(0) != out_capacity ||
        sorted_out_in_map.shape(1) != n_kernels ||
        reorder_rows.shape(0) != out_capacity) {
        throw std::invalid_argument(
            "sorted convolution view shape must match out_capacity and "
            "n_kernels."
        );
    }
    auto expected_tile_words = ((out_capacity + 63) / 64) * 4;
    if (tile_masks.shape(0) != expected_tile_words) {
        throw std::invalid_argument(
            "tile_masks length must match ceil(out_capacity / 64) * 4."
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
            std::string(op_name) + " supports float32 and float16 feats."
        );
    }
    if (weights.dtype() != feats.dtype()) {
        throw std::invalid_argument(
            std::string(op_name) + " weights must match feats dtype."
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
}

void validate_sorted_kv_map(
    const mx::array& sorted_kv_out_in_map,
    int out_capacity,
    int n_kernels
) {
    if (sorted_kv_out_in_map.ndim() != 2 ||
        sorted_kv_out_in_map.dtype() != mx::int32) {
        throw std::invalid_argument(
            "sorted_kv_out_in_map must have shape (K, N_out) and int32 dtype."
        );
    }
    if (sorted_kv_out_in_map.shape(0) != n_kernels ||
        sorted_kv_out_in_map.shape(1) != out_capacity) {
        throw std::invalid_argument(
            "sorted_kv_out_in_map shape must match out_capacity and "
            "n_kernels."
        );
    }
}

} // namespace

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
    const SparseConvSortedImplicitGemmView& sorted_view,
    const SparseRelationEdges& edges,
    const SparseRelationContract& contract,
    const SparseRelationExecutionViews& execution_views,
    bool store_sorted
) {
    validate_sorted_conv_common(
        feats,
        weights,
        sorted_view.sorted_out_in_map,
        sorted_view.reorder_rows,
        sorted_view.tile_masks,
        contract.out_capacity,
        contract.n_kernels,
        "sparse_conv_features_sorted_implicit_gemm"
    );
    validate_sorted_kv_map(
        sorted_view.sorted_kv_out_in_map,
        contract.out_capacity,
        contract.n_kernels
    );
    return make_sparse_conv_features_sorted_implicit_gemm(
        feats,
        weights,
        sorted_view,
        edges,
        contract,
        execution_views,
        store_sorted
    );
}

mx::array sparse_conv_features_sorted_direct_reference(
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& sorted_out_in_map,
    const mx::array& reorder_rows,
    const mx::array& tile_masks,
    int out_capacity,
    int n_kernels,
    bool store_sorted
) {
    validate_sorted_conv_common(
        feats,
        weights,
        sorted_out_in_map,
        reorder_rows,
        tile_masks,
        out_capacity,
        n_kernels,
        "sparse_conv_features_sorted_direct_reference"
    );
    return make_sparse_conv_features_sorted_direct_reference(
        feats,
        weights,
        sorted_out_in_map,
        reorder_rows,
        tile_masks,
        out_capacity,
        n_kernels,
        store_sorted
    );
}

} // namespace mlx_lattice
