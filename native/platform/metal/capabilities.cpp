#include "platform/metal/capabilities.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace mlx_lattice::backend::metal::tensor_ops {

bool is_available(const mlx::core::Stream& stream) {
    (void)stream;
    if (__builtin_available(macOS 26.2, *)) {
        return true;
    }
    return false;
}

bool has_nax_acceleration(const mlx::core::Stream& stream) {
#ifdef _METAL_
    if (!is_available(stream)) {
        return false;
    }
    auto& device = mlx::core::metal::device(stream.device);
    const auto& architecture = device.get_architecture();
    const auto generation = device.get_architecture_gen();
    return !architecture.empty() &&
           generation >= (architecture.back() == 'p' ? 18 : 17);
#else
    (void)stream;
    return false;
#endif
}

} // namespace mlx_lattice::backend::metal::tensor_ops
