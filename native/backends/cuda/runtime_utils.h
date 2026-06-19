#pragma once

#include <cstddef>

#if __has_include(<cuda_runtime_api.h>)
#include <cuda_runtime_api.h>
#else
struct dim3 {
    unsigned int x;
    unsigned int y;
    unsigned int z;

    constexpr explicit dim3(
        unsigned int x_value = 1,
        unsigned int y_value = 1,
        unsigned int z_value = 1
    )
        : x(x_value), y(y_value), z(z_value) {}
};
#endif

#include "mlx/array.h"
#include "mlx/stream.h"

namespace mlx_lattice::backend::cuda {

namespace mx = mlx::core;

struct Launch1D {
    dim3 grid;
    dim3 block;
};

Launch1D launch_1d(std::size_t elements, int block_size = 256);

mx::array make_int32_temp(int elements);
mx::array make_float32_temp(int elements);

} // namespace mlx_lattice::backend::cuda
