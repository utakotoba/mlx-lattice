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
    const mx::array& kernel_ids,
    int n_out_rows
) {
    if (feats.ndim() != 2) {
        throw std::invalid_argument("feats must have shape (N_in, C_in).");
    }
    if (weights.ndim() != 3) {
        throw std::invalid_argument(
            "weights must have shape (K, C_in, C_out)."
        );
    }
    if (feats.dtype() != mx::float32 || weights.dtype() != mx::float32) {
        throw std::invalid_argument(
            "spmm_edges currently supports float32 feats and weights."
        );
    }
    if (feats.shape(1) != weights.shape(1)) {
        throw std::invalid_argument(
            "feats channels must match weights input channels."
        );
    }
    validate_row_index(in_rows, "in_rows");
    validate_row_index(out_rows, "out_rows");
    validate_row_index(kernel_ids, "kernel_ids");
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
    const mx::array& out_rows,
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
    if (in_rows.shape(0) != out_rows.shape(0)) {
        throw std::invalid_argument(
            "in_rows and out_rows must have the same length."
        );
    }
    if (n_out_rows < 0) {
        throw std::invalid_argument("n_out_rows must be non-negative.");
    }
}

} // namespace mlx_lattice
