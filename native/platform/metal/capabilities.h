#pragma once

#include "mlx/stream.h"

namespace mlx_lattice::backend::metal::tensor_ops {

enum class CapabilityTier {
    unavailable,
    gpu,
    neural_accelerator,
};

CapabilityTier capability_tier(const mlx::core::Stream& stream);
bool has_neural_acceleration(const mlx::core::Stream& stream);

} // namespace mlx_lattice::backend::metal::tensor_ops
