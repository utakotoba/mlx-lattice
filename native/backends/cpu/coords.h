#pragma once

#include "ops/coords.h"

namespace mlx_lattice::cpu {

mx::array downsample_coords(const mx::array& coords, Triple stride);

KernelMapData build_kernel_map(
    const mx::array& coords,
    Triple kernel_size,
    Triple stride,
    Triple padding
);

KernelMapData build_generative_map(
    const mx::array& coords,
    Triple kernel_size,
    Triple stride
);

} // namespace mlx_lattice::cpu
