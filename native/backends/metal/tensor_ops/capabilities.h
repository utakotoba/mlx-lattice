#pragma once

#include "mlx/stream.h"

namespace mlx_lattice::backend::metal::tensor_ops {

bool is_available(const mlx::core::Stream& stream);
bool has_nax_acceleration(const mlx::core::Stream& stream);

} // namespace mlx_lattice::backend::metal::tensor_ops
