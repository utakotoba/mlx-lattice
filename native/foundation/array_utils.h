#pragma once

#include <vector>

#include "mlx/allocator.h"
#include "mlx/array.h"

namespace mlx_lattice::backend {

namespace mx = mlx::core;

inline void allocate(mx::array& out) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
}

inline void allocate_all(std::vector<mx::array>& outputs) {
    for (auto& output : outputs) {
        allocate(output);
    }
}

} // namespace mlx_lattice::backend
