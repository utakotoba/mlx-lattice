#include "backends/cuda/runtime_utils.h"

#include <algorithm>

#include "backends/array_utils.h"

namespace mlx_lattice::backend::cuda {

Launch1D launch_1d(std::size_t elements, int block_size) {
    auto blocks = elements == 0
                      ? std::size_t{1}
                      : (elements + static_cast<std::size_t>(block_size) - 1) /
                            static_cast<std::size_t>(block_size);
    return Launch1D{
        .grid = dim3(static_cast<unsigned int>(blocks), 1, 1),
        .block = dim3(static_cast<unsigned int>(block_size), 1, 1),
    };
}

mx::array make_int32_temp(int elements) {
    auto out = mx::array(mx::Shape{std::max(elements, 1)}, mx::int32, nullptr);
    backend::allocate(out);
    return out;
}

mx::array make_float32_temp(int elements) {
    auto out =
        mx::array(mx::Shape{std::max(elements, 1)}, mx::float32, nullptr);
    backend::allocate(out);
    return out;
}

} // namespace mlx_lattice::backend::cuda
