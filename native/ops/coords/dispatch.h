#pragma once

#include "ops/coords.h"

namespace mlx_lattice {

bool has_gpu_coordinate_backend();

KernelMapData
build_gpu_subm_kernel_map(const mx::array& coords, Triple kernel_size);

KernelMapData build_gpu_generative_map(
    const mx::array& coords,
    Triple kernel_size,
    Triple stride
);

} // namespace mlx_lattice
