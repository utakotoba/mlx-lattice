#pragma once

#include <utility>

#include "backends/cuda/runtime_utils.h"

#if __has_include(<cublasLt.h>) && __has_include(<cudnn.h>)
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/utils.h"
#else
namespace mlx::core {

template <typename T> inline T* gpu_ptr(array&) { return nullptr; }

template <typename T>
inline std::conditional_t<std::is_same_v<T, void>, void*, const T*>
gpu_ptr(const array&) {
    return nullptr;
}

namespace cu {

class CommandEncoder {
  public:
    void set_input_array(const array&) {}
    void set_output_array(const array&) {}

    template <typename F, typename... Params>
    void add_kernel_node(F*, dim3, dim3, Params&&...) {}
};

inline CommandEncoder& get_command_encoder(Stream) {
    static CommandEncoder encoder;
    return encoder;
}

} // namespace cu
} // namespace mlx::core
#endif
