#include "features/coordinates/streams.h"

namespace mlx_lattice {

namespace {

bool is_gpu_device(const mx::Device& device) {
    return device == mx::Device(mx::Device::gpu);
}

} // namespace

mx::Device coord_device() {
    auto device = mx::default_device();
    return is_gpu_device(device) ? mx::Device::gpu : mx::Device::cpu;
}

mx::Stream coord_stream(const mx::Device& device) {
    return mx::default_stream(device);
}

} // namespace mlx_lattice
