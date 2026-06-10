#include "ops/exec/validation.h"

#include <stdexcept>

namespace mlx_lattice {

void validate_sparse_pool_features(
    const mx::array& feats,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    const mx::array& row_offsets,
    const mx::array& counts,
    int out_capacity,
    int n_kernels
) {
    if (feats.ndim() != 2) {
        throw std::invalid_argument("feats must have shape (N, C).");
    }
    if (feats.dtype() != mx::float32) {
        throw std::invalid_argument(
            "sparse_pool_features currently supports float32 feats."
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
    if (row_offsets.ndim() != 1 || row_offsets.dtype() != mx::int32) {
        throw std::invalid_argument(
            "row_offsets must have shape (N_out + 1,) and int32 dtype."
        );
    }
    if (counts.shape() != mx::Shape{2} || counts.dtype() != mx::int32) {
        throw std::invalid_argument(
            "counts must have shape (2,) and int32 dtype."
        );
    }
    if (out_capacity < 0 || n_kernels <= 0) {
        throw std::invalid_argument(
            "out_capacity must be nonnegative and n_kernels must be positive."
        );
    }
    if (row_offsets.shape(0) != out_capacity + 1) {
        throw std::invalid_argument(
            "row_offsets length must match out_capacity + 1."
        );
    }
}

} // namespace mlx_lattice
