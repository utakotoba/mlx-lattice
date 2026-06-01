#include "ops/coords/dispatch.h"

#include <stdexcept>

#if MLX_LATTICE_HAS_CUDA
#include "backends/cuda/coords.h"
#include "mlx/backend/cuda/cuda.h"
#endif
#include "backends/metal/coords.h"
#include "mlx/device.h"

namespace mlx_lattice {

bool has_gpu_coordinate_backend() {
#if MLX_LATTICE_HAS_CUDA
    if (mx::cu::is_available()) {
        return true;
    }
#endif
#if MLX_LATTICE_HAS_METAL
    return mx::is_available(mx::Device::gpu);
#else
    return false;
#endif
}

KernelMapData
build_gpu_subm_kernel_map(const mx::array& coords, Triple kernel_size) {
#if MLX_LATTICE_HAS_CUDA
    if (mx::cu::is_available()) {
        return cuda::build_subm_kernel_map(coords, kernel_size);
    }
#endif
#if MLX_LATTICE_HAS_METAL
    return metal::build_subm_kernel_map(coords, kernel_size);
#else
    throw std::runtime_error("No GPU coordinate backend is available.");
#endif
}

KernelMapData build_gpu_generative_map(
    const mx::array& coords,
    Triple kernel_size,
    Triple stride
) {
#if MLX_LATTICE_HAS_CUDA
    if (mx::cu::is_available()) {
        return cuda::build_generative_map(coords, kernel_size, stride);
    }
#endif
#if MLX_LATTICE_HAS_METAL
    return metal::build_generative_map(coords, kernel_size, stride);
#else
    throw std::runtime_error("No GPU coordinate backend is available.");
#endif
}

} // namespace mlx_lattice
