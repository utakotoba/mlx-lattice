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
    if (feats.dtype() != mx::float32 || weights.dtype() != mx::float32) {
        throw std::invalid_argument(
            "sparse_conv_features currently supports float32 feats and weights."
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
    return make_sparse_conv_features(
        feats,
        weights,
        in_rows,
        out_rows,
        kernel_ids,
        counts,
        row_offsets,
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
        in_rows,
        out_rows,
        kernel_ids,
        row_offsets,
        counts,
        out_capacity,
        n_kernels,
        input_layout
    );
}

} // namespace mlx_lattice
