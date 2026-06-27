#include "platform/metal/capabilities.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace mlx_lattice::backend::metal::tensor_ops {

CapabilityTier capability_tier(const mlx::core::Stream& stream) {
#ifdef _METAL_
    if (!__builtin_available(macOS 26.2, *)) {
        return CapabilityTier::unavailable;
    }
    auto& device = mlx::core::metal::device(stream.device);
    auto* metal_device = device.mtl_device();
    if (metal_device == nullptr ||
        !metal_device->supportsFamily(MTL::GPUFamilyApple7)) {
        return CapabilityTier::unavailable;
    }
    return metal_device->supportsFamily(MTL::GPUFamilyApple10)
               ? CapabilityTier::neural_accelerator
               : CapabilityTier::gpu;
#else
    (void)stream;
    return CapabilityTier::unavailable;
#endif
}

bool has_neural_acceleration(const mlx::core::Stream& stream) {
    return capability_tier(stream) == CapabilityTier::neural_accelerator;
}

} // namespace mlx_lattice::backend::metal::tensor_ops
