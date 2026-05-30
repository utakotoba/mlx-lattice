#pragma once

#include <array>
#include <vector>

#include "mlx/array.h"

namespace mlx_lattice {

namespace mx = mlx::core;

using Triple = std::array<int, 3>;

struct KernelMapData {
    mx::array maps;
    mx::array sizes;
    mx::array kernels;
    mx::array out_coords;
    mx::array offsets;
};

std::vector<Triple> kernel_offsets(Triple kernel_size);

mx::array downsample_coords(const mx::array& coords, Triple stride);

KernelMapData
build_kernel_map(const mx::array& coords, Triple kernel_size, Triple stride);

} // namespace mlx_lattice
