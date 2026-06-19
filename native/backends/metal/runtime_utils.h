#pragma once

#include <cstddef>
#include <string>
#include <utility>

namespace mlx_lattice::metal {

std::string binary_dir();

} // namespace mlx_lattice::metal

#ifdef _METAL_
#include <algorithm>

#include "mlx/backend/metal/device.h"

template <typename Encoder, typename Kernel>
void dispatch_1d(Encoder& encoder, Kernel* kernel, std::size_t elements) {
    auto threads = std::max<std::size_t>(elements, 1);
    auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}

template <typename Encoder> void dispatch_single(Encoder& encoder) {
    encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
}

template <typename Encoder, typename... Values>
void set_bytes_range(Encoder& encoder, int first, Values&&... values) {
    int index = first;
    (encoder.set_bytes(std::forward<Values>(values), index++), ...);
}
#endif
