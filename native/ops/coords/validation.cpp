#include "ops/coords/validation.h"

#include <stdexcept>
#include <string>

namespace mlx_lattice {

void validate_coords(const mx::array& coords) {
    if (coords.ndim() != 2 || coords.shape(1) != 4) {
        throw std::invalid_argument("coords must have shape (N, 4).");
    }
    if (coords.dtype() != mx::int32 && coords.dtype() != mx::int64) {
        throw std::invalid_argument("coords must be int32 or int64.");
    }
}

void validate_coord_pair(const mx::array& lhs, const mx::array& rhs) {
    validate_coords(lhs);
    validate_coords(rhs);
    if (lhs.dtype() != rhs.dtype()) {
        throw std::invalid_argument(
            "coordinate arrays must have matching dtype."
        );
    }
}

void validate_points(const mx::array& points) {
    if (points.ndim() != 2 || points.shape(1) != 3) {
        throw std::invalid_argument("points must have shape (N, 3).");
    }
    if (points.dtype() != mx::float32) {
        throw std::invalid_argument("points must be float32.");
    }
}

void validate_active_rows(const mx::array& active_rows) {
    if (active_rows.shape() != mx::Shape{1} ||
        active_rows.dtype() != mx::int32) {
        throw std::invalid_argument(
            "active_rows must have shape (1,) and int32 dtype."
        );
    }
}

void validate_batch_indices(const mx::array& batch_indices, int rows) {
    if (batch_indices.shape() != mx::Shape{rows} ||
        batch_indices.dtype() != mx::int32) {
        throw std::invalid_argument(
            "batch_indices must have shape (N,) and int32 dtype."
        );
    }
}

void validate_inverse_rows(const mx::array& inverse_rows, int rows) {
    if (inverse_rows.shape() != mx::Shape{rows} ||
        inverse_rows.dtype() != mx::int32) {
        throw std::invalid_argument(
            "inverse_rows must have shape (N,) and int32 dtype."
        );
    }
}

void validate_voxel_counts(const mx::array& voxel_counts, int rows) {
    if (voxel_counts.shape() != mx::Shape{rows} ||
        voxel_counts.dtype() != mx::int32) {
        throw std::invalid_argument(
            "voxel_counts must have shape (N,) and int32 dtype."
        );
    }
}

void validate_feature_matrix(const mx::array& feats) {
    if (feats.ndim() != 2) {
        throw std::invalid_argument("feats must have shape (N, C).");
    }
    if (feats.dtype() != mx::float32) {
        throw std::invalid_argument("feats must be float32.");
    }
}

void validate_positive(Triple values, const char* name) {
    for (auto value : values) {
        if (value <= 0) {
            throw std::invalid_argument(
                std::string(name) + " values must be positive."
            );
        }
    }
}

void validate_positive(FloatTriple values, const char* name) {
    for (auto value : values) {
        if (value <= 0.0F) {
            throw std::invalid_argument(
                std::string(name) + " values must be positive."
            );
        }
    }
}

void validate_nonnegative(Triple values, const char* name) {
    for (auto value : values) {
        if (value < 0) {
            throw std::invalid_argument(
                std::string(name) + " values must be non-negative."
            );
        }
    }
}

} // namespace mlx_lattice
