#include "ops/exec/validation.h"

#include <stdexcept>
#include <string>

namespace mlx_lattice {

namespace {

void validate_row_index(const mx::array& rows, const char* name) {
    if (rows.ndim() != 1) {
        throw std::invalid_argument(
            std::string(name) + " must have shape (E,)."
        );
    }
    if (rows.dtype() != mx::int32) {
        throw std::invalid_argument(
            std::string(name) + " must be int32 for spmm_edges."
        );
    }
}

} // namespace

void validate_spmm_edges(
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids, // NOLINT(bugprone-easily-swappable-parameters)
    const mx::array& edge_count,
    int n_out_rows
) {
    if (feats.ndim() != 2) {
        throw std::invalid_argument("feats must have shape (N_in, C_in).");
    }
    if (weights.ndim() != 3 && weights.ndim() != 5) {
        throw std::invalid_argument(
            "weights must have shape (K, C_in, C_out) or "
            "(C_out, Kx, Ky, Kz, C_in)."
        );
    }
    if (feats.dtype() != mx::float32 || weights.dtype() != mx::float32) {
        throw std::invalid_argument(
            "spmm_edges currently supports float32 feats and weights."
        );
    }
    auto weight_in_channels =
        weights.ndim() == 3 ? weights.shape(1) : weights.shape(4);
    if (feats.shape(1) != weight_in_channels) {
        throw std::invalid_argument(
            "feats channels must match weights input channels."
        );
    }
    validate_row_index(in_rows, "in_rows");
    validate_row_index(out_rows, "out_rows");
    validate_row_index(kernel_ids, "kernel_ids");
    if (edge_count.shape() != mx::Shape{1} || edge_count.dtype() != mx::int32) {
        throw std::invalid_argument(
            "edge_count must have shape (1,) and int32 dtype."
        );
    }
    if (in_rows.shape(0) != out_rows.shape(0) ||
        in_rows.shape(0) != kernel_ids.shape(0)) {
        throw std::invalid_argument(
            "in_rows, out_rows, and kernel_ids must have the same length."
        );
    }
    if (n_out_rows < 0) {
        throw std::invalid_argument("n_out_rows must be non-negative.");
    }
}

void validate_pool_edges(
    const mx::array& feats, // NOLINT(bugprone-easily-swappable-parameters)
    const mx::array& in_rows,
    const mx::array& out_rows, // NOLINT(bugprone-easily-swappable-parameters)
    const mx::array& edge_count,
    int n_out_rows
) {
    if (feats.ndim() != 2) {
        throw std::invalid_argument("feats must have shape (N_in, C).");
    }
    if (feats.dtype() != mx::float32) {
        throw std::invalid_argument(
            "pool edge reductions currently support float32 feats."
        );
    }
    validate_row_index(in_rows, "in_rows");
    validate_row_index(out_rows, "out_rows");
    if (edge_count.shape() != mx::Shape{1} || edge_count.dtype() != mx::int32) {
        throw std::invalid_argument(
            "edge_count must have shape (1,) and int32 dtype."
        );
    }
    if (in_rows.shape(0) != out_rows.shape(0)) {
        throw std::invalid_argument(
            "in_rows and out_rows must have the same length."
        );
    }
    if (n_out_rows < 0) {
        throw std::invalid_argument("n_out_rows must be non-negative.");
    }
}

void validate_sparse_conv(
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& feats,
    const mx::array& weights
) {
    if (coords.ndim() != 2 || coords.shape(1) != 4) {
        throw std::invalid_argument("coords must have shape (N, 4).");
    }
    if (coords.dtype() != mx::int32 && coords.dtype() != mx::int64) {
        throw std::invalid_argument("coords must be int32 or int64.");
    }
    if (active_rows.shape() != mx::Shape{1} ||
        active_rows.dtype() != mx::int32) {
        throw std::invalid_argument(
            "active_rows must have shape (1,) and int32 dtype."
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
    if (coords.shape(0) != feats.shape(0)) {
        throw std::invalid_argument(
            "coords and feats must have the same row count."
        );
    }
    if (feats.dtype() != mx::float32 || weights.dtype() != mx::float32) {
        throw std::invalid_argument(
            "sparse_conv currently supports float32 feats and weights."
        );
    }
    auto weight_in_channels =
        weights.ndim() == 3 ? weights.shape(1) : weights.shape(4);
    if (feats.shape(1) != weight_in_channels) {
        throw std::invalid_argument(
            "feats channels must match weights input channels."
        );
    }
}

void validate_sparse_pool(
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& feats
) {
    if (coords.ndim() != 2 || coords.shape(1) != 4) {
        throw std::invalid_argument("coords must have shape (N, 4).");
    }
    if (coords.dtype() != mx::int32 && coords.dtype() != mx::int64) {
        throw std::invalid_argument("coords must be int32 or int64.");
    }
    if (active_rows.shape() != mx::Shape{1} ||
        active_rows.dtype() != mx::int32) {
        throw std::invalid_argument(
            "active_rows must have shape (1,) and int32 dtype."
        );
    }
    if (feats.ndim() != 2) {
        throw std::invalid_argument("feats must have shape (N, C).");
    }
    if (coords.shape(0) != feats.shape(0)) {
        throw std::invalid_argument(
            "coords and feats must have the same row count."
        );
    }
    if (feats.dtype() != mx::float32) {
        throw std::invalid_argument(
            "sparse_pool currently supports float32 feats."
        );
    }
}

} // namespace mlx_lattice
