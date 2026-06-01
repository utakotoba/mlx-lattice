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
    mx::array residual_maps;
    mx::array residual_kernels;
    mx::array residual_offsets;
    mx::array out_coords;
    mx::array offsets;
};

std::vector<Triple> kernel_offsets(Triple kernel_size);
std::vector<Triple> kernel_offsets(Triple kernel_size, Triple dilation);

mx::array downsample_coords(const mx::array& coords, Triple stride);

mx::array union_coords(const mx::array& lhs, const mx::array& rhs);

mx::array intersection_coords(const mx::array& lhs, const mx::array& rhs);

mx::array lookup_coords(const mx::array& coords, const mx::array& queries);

KernelMapData build_kernel_map(
    const mx::array& coords,
    Triple kernel_size,
    Triple stride,
    Triple padding,
    Triple dilation
);

KernelMapData build_generative_map(
    const mx::array& coords,
    Triple kernel_size,
    Triple stride
);

KernelMapData build_transposed_kernel_map(
    const mx::array& coords,
    Triple kernel_size,
    Triple stride,
    Triple padding,
    Triple dilation
);

} // namespace mlx_lattice
