#include "features/coordinates/api.h"

#include "features/coordinates/validation.h"

namespace mlx_lattice {

std::vector<Triple> kernel_offsets(Triple kernel_size) {
    return kernel_offsets(kernel_size, {1, 1, 1});
}

std::vector<Triple> kernel_offsets(
    Triple kernel_size, // NOLINT(bugprone-easily-swappable-parameters)
    Triple dilation
) {
    validate_positive(kernel_size, "kernel_size");
    validate_positive(dilation, "dilation");

    std::array<std::vector<int>, 3> axes;
    for (int axis = 0; axis < 3; ++axis) {
        auto size = kernel_size[axis];
        if (size % 2 == 1) {
            auto radius = size / 2;
            for (int value = -radius; value <= radius; ++value) {
                axes[axis].push_back(value);
            }
        } else {
            for (int value = 0; value < size; ++value) {
                axes[axis].push_back(value);
            }
        }
    }

    std::vector<Triple> out;
    out.reserve(axes[0].size() * axes[1].size() * axes[2].size());
    for (auto x : axes[0]) {
        for (auto y : axes[1]) {
            for (auto z : axes[2]) {
                out.push_back(
                    {x * dilation[0], y * dilation[1], z * dilation[2]}
                );
            }
        }
    }
    return out;
}

} // namespace mlx_lattice
