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

void validate_active_rows(const mx::array& active_rows) {
    if (active_rows.shape() != mx::Shape{1} ||
        active_rows.dtype() != mx::int32) {
        throw std::invalid_argument(
            "active_rows must have shape (1,) and int32 dtype."
        );
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
