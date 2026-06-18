#pragma once

#include <cstddef>

#include "mlx/array.h"
#include "mlx/backend/cuda/device.h"
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
