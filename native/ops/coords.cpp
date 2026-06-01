#include "ops/coords.h"

#include <stdexcept>

#include "backends/cpu/coords.h"
#include "ops/coords/dispatch.h"

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

void validate_nonnegative(Triple values, const char* name) {
    for (auto value : values) {
        if (value < 0) {
            throw std::invalid_argument(
                std::string(name) + " values must be non-negative."
            );
        }
    }
}

bool has_center_offset(Triple kernel_size) {
    for (auto size : kernel_size) {
        if (size % 2 == 0) {
            return false;
        }
    }
    return true;
}

} // namespace

// MARK: - helpers

std::vector<Triple> kernel_offsets(Triple kernel_size) {
    return kernel_offsets(kernel_size, {1, 1, 1});
}

std::vector<Triple> kernel_offsets(Triple kernel_size, Triple dilation) {
    validate_positive(kernel_size, "kernel_size");
    validate_positive(dilation, "dilation");

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
                offsets.push_back(
                    {x * dilation[0], y * dilation[1], z * dilation[2]}
                );
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

KernelMapData build_kernel_map(
    const mx::array& coords,
    Triple kernel_size,
    Triple stride,
    Triple padding,
    Triple dilation
) {
    validate_coords(coords);
    validate_positive(stride, "stride");
    validate_nonnegative(padding, "padding");
    validate_positive(dilation, "dilation");
    if (stride == Triple{1, 1, 1} && padding == Triple{0, 0, 0} &&
        dilation == Triple{1, 1, 1} && has_center_offset(kernel_size) &&
        coords.dtype() == mx::int32 && has_gpu_coordinate_backend()) {
        return build_gpu_subm_kernel_map(coords, kernel_size);
    }
    return cpu::build_kernel_map(
        coords, kernel_size, stride, padding, dilation
    );
}

KernelMapData build_generative_map(
    const mx::array& coords,
    Triple kernel_size,
    Triple stride
) {
    validate_coords(coords);
    validate_positive(kernel_size, "kernel_size");
    validate_positive(stride, "stride");
    if (kernel_size == stride && coords.dtype() == mx::int32 &&
        has_gpu_coordinate_backend()) {
        return build_gpu_generative_map(coords, kernel_size, stride);
    }
    return cpu::build_generative_map(coords, kernel_size, stride);
}

} // namespace mlx_lattice
