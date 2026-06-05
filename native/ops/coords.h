#pragma once

#include <vector>

#include "ops/coords/types.h"

namespace mlx_lattice {

std::vector<Triple> kernel_offsets(Triple kernel_size);
std::vector<Triple> kernel_offsets(Triple kernel_size, Triple dilation);

mx::array downsample_coords(const mx::array& coords, Triple stride);
mx::array union_coords(const mx::array& lhs, const mx::array& rhs);
mx::array intersection_coords(const mx::array& lhs, const mx::array& rhs);
mx::array lookup_coords(const mx::array& coords, const mx::array& queries);

NativeKernelMap build_kernel_map(
    const mx::array& coords,
    Triple kernel_size,
    Triple stride,
    Triple padding,
    Triple dilation
);

NativeKernelMap build_generative_map(
    const mx::array& coords,
    Triple kernel_size,
    Triple stride
);

NativeKernelMap build_transposed_kernel_map(
    const mx::array& coords,
    Triple kernel_size,
    Triple stride,
    Triple padding,
    Triple dilation
);

} // namespace mlx_lattice
