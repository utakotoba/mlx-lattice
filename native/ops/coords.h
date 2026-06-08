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

NativeKernelRelation build_kernel_relation(
    const mx::array& coords,
    const mx::array& active_rows,
    Triple kernel_size,
    Triple stride,
    Triple padding,
    Triple dilation
);

NativeKernelRelation build_generative_relation(
    const mx::array& coords,
    const mx::array& active_rows,
    Triple kernel_size,
    Triple stride
);

NativeKernelRelation build_transposed_kernel_relation(
    const mx::array& coords,
    const mx::array& active_rows,
    Triple kernel_size,
    Triple stride,
    Triple padding,
    Triple dilation
);

} // namespace mlx_lattice
