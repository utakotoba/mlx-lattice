#include "backends/metal/tensor_ops/conv/weight_grad/runtime.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

#include "backends/array_utils.h"
#include "backends/metal/runtime_utils.h"
#include "backends/metal/tensor_ops/capabilities.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
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

int partition_count(SparseConvShape shape) {
    auto partitions =
        (shape.in_capacity + kPartitionEdges - 1) / kPartitionEdges;
    return std::clamp(partitions, 1, kMaxPartitions);
}

template <typename Encoder, typename Kernel>
void dispatch_1d(Encoder& encoder, Kernel* kernel, std::size_t elements) {
    auto threads = std::max<std::size_t>(elements, 1);
    auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}
#endif

} // namespace

bool supports(SparseConvShape shape) {
    return shape.in_channels == kChannels && shape.out_channels == kChannels &&
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
    auto partial_values = static_cast<std::size_t>(partitions) *
                          static_cast<std::size_t>(shape.n_kernels) *
                          kChannels * kChannels;
    auto partials = make_float_temp(partial_values);

    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);
    encoder.add_temporary(partials);

    auto contract = device.get_kernel(
        "sparse_relation_conv_weight_grad_tensor_ops_f32_i32", library
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
    encoder.set_bytes(static_cast<int>(inputs[2].shape(0)), 8);
    encoder.set_bytes(shape.out_capacity, 9);
    encoder.set_bytes(shape.n_kernels, 10);
    encoder.set_bytes(partitions, 11);
    encoder.set_bytes(static_cast<int>(inputs[0].strides(0)), 12);
    encoder.set_bytes(static_cast<int>(inputs[0].strides(1)), 13);
    encoder.set_bytes(static_cast<int>(inputs[1].strides(0)), 14);
    encoder.set_bytes(static_cast<int>(inputs[1].strides(1)), 15);
    encoder.dispatch_threadgroups(
        MTL::Size(
            static_cast<std::size_t>(shape.n_kernels) *
                static_cast<std::size_t>(partitions),
            1,
            1
        ),
        MTL::Size(32, 1, 1)
    );

    auto reduce = device.get_kernel(
        "sparse_relation_conv_weight_grad_tensor_ops_reduce_f32", library
    );
    encoder.set_compute_pipeline_state(reduce);
    encoder.set_input_array(partials, 0);
    encoder.set_output_array(out, 1);
    encoder.set_bytes(shape.n_kernels, 2);
    encoder.set_bytes(partitions, 3);
    encoder.set_bytes(shape.weight_layout, 4);
    encoder.set_bytes(shape.kernel_x, 5);
    encoder.set_bytes(shape.kernel_y, 6);
    encoder.set_bytes(shape.kernel_z, 7);
    dispatch_1d(
        encoder,
        reduce,
        static_cast<std::size_t>(shape.n_kernels) * kChannels * kChannels
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
