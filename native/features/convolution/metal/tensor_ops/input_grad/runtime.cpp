#include "features/convolution/metal/tensor_ops/input_grad/runtime.h"

#include <stdexcept>

#include "platform/metal/capabilities.h"
#include "platform/metal/runtime_utils.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace mlx_lattice::backend::metal::tensor_ops::conv::input_grad {
namespace {

constexpr int kChannels = 16;
constexpr int kMinInputRows = 32768;

int stride_at(const mx::array& array, int dim) {
    return static_cast<int>(array.strides(dim));
}

} // namespace

bool supports(SparseConvShape shape) {
    return shape.in_channels == kChannels && shape.out_channels == kChannels &&
           shape.n_kernels >= 16 && shape.weight_layout == 0;
}

bool is_preferred(SparseConvShape shape, const mx::Stream& stream) {
    return supports(shape) && shape.in_capacity >= kMinInputRows &&
           has_nax_acceleration(stream);
}

void encode(
    SparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    mx::array& out
) {
#ifdef _METAL_
    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);

    auto kernel = device.get_kernel(
        "sparse_relation_conv_input_grad_tensor_ops_f32_i32", library
    );
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[1], 1);
    encoder.set_input_array(inputs[3], 2);
    encoder.set_input_array(inputs[4], 3);
    encoder.set_input_array(inputs[5], 4);
    encoder.set_input_array(inputs[7], 5);
    encoder.set_input_array(inputs[8], 6);
    encoder.set_output_array(out, 7);
    encoder.set_bytes(static_cast<int>(inputs[2].shape(0)), 8);
    encoder.set_bytes(shape.out_capacity, 9);
    encoder.set_bytes(shape.in_capacity, 10);
    encoder.set_bytes(shape.n_kernels, 11);
    encoder.set_bytes(stride_at(inputs[0], 0), 12);
    encoder.set_bytes(stride_at(inputs[0], 1), 13);
    encoder.set_bytes(stride_at(inputs[1], 0), 14);
    encoder.set_bytes(stride_at(inputs[1], 1), 15);
    encoder.set_bytes(stride_at(inputs[1], 2), 16);
    auto total_tiles =
        static_cast<size_t>((shape.in_capacity + kChannels - 1) / kChannels);
    encoder.dispatch_threadgroups(
        MTL::Size(total_tiles, 1, 1), MTL::Size(32, 1, 1)
    );
#else
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)out;
    throw std::runtime_error("Metal support is not available.");
#endif
}

} // namespace mlx_lattice::backend::metal::tensor_ops::conv::input_grad
