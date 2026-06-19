#include "backends/metal/tensor_ops/conv/weight_grad/runtime.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

#include "backends/array_utils.h"
#include "backends/metal/runtime_utils.h"
#include "backends/metal/tensor_ops/capabilities.h"

#ifdef _METAL_
#endif

namespace mlx_lattice::backend::metal::tensor_ops::conv::weight_grad {
namespace {

constexpr int kChannels = 16;
constexpr int kPartitionEdges = 2048;
constexpr int kMaxPartitions = 64;
constexpr int kMinInputRows = 32768;

#ifdef _METAL_
mx::array make_float_temp(std::size_t elements) {
    auto count = std::max<std::size_t>(elements, 1);
    return mx::array(
        mx::allocator::malloc(count * sizeof(float)),
        mx::Shape{static_cast<int>(count)},
        mx::float32
    );
}

bool is_float16(const mx::array& array) { return array.dtype() == mx::float16; }

int partition_count(SparseConvShape shape) {
    auto partitions =
        (shape.in_capacity + kPartitionEdges - 1) / kPartitionEdges;
    return std::clamp(partitions, 1, kMaxPartitions);
}

#endif

} // namespace

bool supports(SparseConvShape shape) {
    return shape.in_channels == shape.out_channels &&
           (shape.in_channels == 16 || shape.in_channels == 32 ||
            shape.in_channels == 64) &&
           shape.n_kernels >= 16;
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
    auto partitions = partition_count(shape);
    auto channel_blocks = shape.in_channels / kChannels;
    auto channel_tiles = channel_blocks * channel_blocks;
    auto partial_values = static_cast<std::size_t>(partitions) *
                          static_cast<std::size_t>(shape.n_kernels) *
                          static_cast<std::size_t>(channel_tiles) * kChannels *
                          kChannels;
    auto partials = make_float_temp(partial_values);

    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);
    encoder.add_temporary(partials);

    auto contract = device.get_kernel(
        is_float16(inputs[0])
            ? "sparse_relation_conv_weight_grad_tensor_ops_f16_i32"
            : "sparse_relation_conv_weight_grad_tensor_ops_f32_i32",
        library
    );
    encoder.set_compute_pipeline_state(contract);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[1], 1);
    encoder.set_input_array(inputs[2], 2);
    encoder.set_input_array(inputs[3], 3);
    encoder.set_input_array(inputs[5], 4);
    encoder.set_input_array(inputs[7], 5);
    encoder.set_input_array(inputs[8], 6);
    encoder.set_output_array(partials, 7);
    set_bytes_range(
        encoder,
        8,
        static_cast<int>(inputs[2].shape(0)),
        shape.out_capacity,
        shape.n_kernels,
        partitions,
        static_cast<int>(inputs[0].strides(0)),
        static_cast<int>(inputs[0].strides(1)),
        static_cast<int>(inputs[1].strides(0)),
        static_cast<int>(inputs[1].strides(1)),
        shape.in_channels,
        shape.out_channels
    );
    encoder.dispatch_threadgroups(
        MTL::Size(
            static_cast<std::size_t>(shape.n_kernels) *
                static_cast<std::size_t>(partitions) *
                static_cast<std::size_t>(channel_tiles),
            1,
            1
        ),
        MTL::Size(32, 1, 1)
    );

    auto reduce = device.get_kernel(
        is_float16(inputs[0])
            ? "sparse_relation_conv_weight_grad_tensor_ops_reduce_f16"
            : "sparse_relation_conv_weight_grad_tensor_ops_reduce_f32",
        library
    );
    encoder.set_compute_pipeline_state(reduce);
    encoder.set_input_array(partials, 0);
    encoder.set_output_array(out, 1);
    set_bytes_range(
        encoder,
        2,
        shape.n_kernels,
        partitions,
        shape.weight_layout,
        shape.kernel_x,
        shape.kernel_y,
        shape.kernel_z,
        shape.in_channels,
        shape.out_channels
    );
    dispatch_1d(
        encoder,
        reduce,
        static_cast<std::size_t>(shape.n_kernels) *
            static_cast<std::size_t>(channel_tiles) * kChannels * kChannels
    );
#else
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)out;
    throw std::runtime_error("Metal support is not available.");
#endif
}

} // namespace mlx_lattice::backend::metal::tensor_ops::conv::weight_grad
