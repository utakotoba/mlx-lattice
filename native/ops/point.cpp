#include "ops/point.h"

#include <stdexcept>

#include "backends/cpu/point.h"

namespace mlx_lattice {

namespace {

// MARK: - validation

void validate_coords(const mx::array& coords) {
    if (coords.ndim() != 2 || coords.shape(1) != 4) {
        throw std::invalid_argument("coords must have shape (N, 4).");
    }
    if (coords.dtype() != mx::int32 && coords.dtype() != mx::int64) {
        throw std::invalid_argument("coords must be int32 or int64.");
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

} // namespace

// MARK: - helpers

std::vector<Triple> kernel_offsets(Triple kernel_size) {
    validate_positive(kernel_size, "kernel_size");

    std::array<std::vector<int>, 3> axes;
    for (int axis = 0; axis < 3; ++axis) {
        int size = kernel_size[axis];
        if (size % 2 == 1) {
            int radius = size / 2;
            for (int value = -radius; value <= radius; ++value) {
                axes[axis].push_back(value);
            }
        } else {
            for (int value = 0; value < size; ++value) {
                axes[axis].push_back(value);
            }
        }
    }

    std::vector<Triple> offsets;
    offsets.reserve(axes[0].size() * axes[1].size() * axes[2].size());
    for (auto x : axes[0]) {
        for (auto y : axes[1]) {
            for (auto z : axes[2]) {
                offsets.push_back({x, y, z});
            }
        }
    }
    return offsets;
}

// MARK: - api

mx::array downsample_coords(const mx::array& coords, Triple stride) {
    validate_coords(coords);
    validate_positive(stride, "stride");
    return cpu::downsample_coords(coords, stride);
}

KernelMapData
build_kernel_map(const mx::array& coords, Triple kernel_size, Triple stride) {
    validate_coords(coords);
    validate_positive(stride, "stride");
    return cpu::build_kernel_map(coords, kernel_size, stride);
}

} // namespace mlx_lattice
