#include "ops/exec/streams.h"

#include <stdexcept>

#include "backends/metal/exec/runtime.h"
#include "mlx/device.h"

namespace mlx_lattice {

namespace {

bool is_gpu_device(const mx::Device& device) {
    return device == mx::Device(mx::Device::gpu);
}

mx::Device sparse_exec_device() {
    auto device = mx::default_device();
    return is_gpu_device(device) ? mx::Device::gpu : mx::Device::cpu;
}

mx::Stream sparse_exec_stream(const mx::Device& device) {
    return mx::default_stream(device);
}

} // namespace

mx::Stream sparse_conv_features_stream(
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    const mx::array& counts
) {
    auto device = sparse_exec_device();
    if (is_gpu_device(device) &&
        (feats.dtype() != mx::float32 || weights.dtype() != mx::float32 ||
         in_rows.dtype() != mx::int32 || out_rows.dtype() != mx::int32 ||
         kernel_ids.dtype() != mx::int32 || counts.dtype() != mx::int32)) {
        throw std::invalid_argument(
            "Metal sparse convolution features require int32 relation arrays "
            "and float32 features/weights."
        );
    }
    return sparse_exec_stream(device);
}

mx::Stream sparse_pool_stream(
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& feats,
    const mx::array& offsets
) {
    auto device = sparse_exec_device();
    if (is_gpu_device(device) && !exec::metal::can_run_sparse_pool(
                                     coords, active_rows, feats, offsets
                                 )) {
        throw std::invalid_argument(
            "Metal sparse pooling requires int32 coords/active_rows/offsets "
            "and float32 features."
        );
    }
    return sparse_exec_stream(device);
}

} // namespace mlx_lattice
