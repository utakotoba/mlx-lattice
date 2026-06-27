#include "features/pooling/streams.h"

#include <stdexcept>

#include "mlx/device.h"

namespace mlx_lattice {

mx::Stream sparse_pool_features_stream(
    const mx::array& feats,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    const mx::array& row_offsets,
    const mx::array& counts
) {
    auto device = mx::default_device() == mx::Device::gpu ? mx::Device::gpu
                                                          : mx::Device::cpu;
    if (device == mx::Device::gpu &&
        (feats.dtype() != mx::float32 || in_rows.dtype() != mx::int32 ||
         out_rows.dtype() != mx::int32 || kernel_ids.dtype() != mx::int32 ||
         row_offsets.dtype() != mx::int32 || counts.dtype() != mx::int32)) {
        throw std::invalid_argument(
            "Metal sparse pooling requires int32 relation arrays and float32 "
            "features."
        );
    }
    return mx::default_stream(device);
}

} // namespace mlx_lattice
