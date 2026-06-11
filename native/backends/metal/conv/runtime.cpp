#include "backends/metal/conv/runtime.h"

#include <algorithm>
#include <stdexcept>

#include "backends/array_utils.h"
#include "backends/metal/runtime_utils.h"
#include "backends/metal/tensor_ops/conv/weight_grad/runtime.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace mlx_lattice::backend::metal::conv {
namespace {

int stride_at(const mx::array& array, int dim) {
    return static_cast<int>(array.strides(dim));
}

#ifdef _METAL_
template <typename Encoder, typename Kernel>
void dispatch_1d(Encoder& encoder, Kernel* kernel, size_t elements) {
    auto threads = std::max<size_t>(elements, 1);
    auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}

template <typename Encoder>
void bind_common_shape(
    Encoder& encoder,
    const std::vector<mx::array>& inputs,
    SparseConvShape shape,
    int first_index
) {
    auto edge_capacity = static_cast<int>(inputs[2].shape(0));
    encoder.set_bytes(edge_capacity, first_index);
    encoder.set_bytes(shape.out_capacity, first_index + 1);
    encoder.set_bytes(shape.in_channels, first_index + 2);
    encoder.set_bytes(shape.out_channels, first_index + 3);
}

template <typename Encoder>
void bind_weight_shape(
    Encoder& encoder,
    const mx::array& weights,
    SparseConvShape shape,
    int first_index
) {
    encoder.set_bytes(stride_at(weights, 0), first_index);
    encoder.set_bytes(stride_at(weights, 1), first_index + 1);
    encoder.set_bytes(stride_at(weights, 2), first_index + 2);
    encoder.set_bytes(
        weights.ndim() == 5 ? stride_at(weights, 3) : 0, first_index + 3
    );
    encoder.set_bytes(
        weights.ndim() == 5 ? stride_at(weights, 4) : 0, first_index + 4
    );
    encoder.set_bytes(shape.weight_layout, first_index + 5);
    encoder.set_bytes(shape.kernel_x, first_index + 6);
    encoder.set_bytes(shape.kernel_y, first_index + 7);
    encoder.set_bytes(shape.kernel_z, first_index + 8);
}

template <typename Encoder, typename Library>
void clear_output(
    Encoder& encoder,
    mx::metal::Device& device,
    Library library,
    mx::array& out
) {
    auto clear = device.get_kernel("sparse_relation_conv_clear_f32", library);
    encoder.set_compute_pipeline_state(clear);
    encoder.set_output_array(out, 0);
    auto total = static_cast<int>(out.size());
    encoder.set_bytes(total, 1);
    dispatch_1d(encoder, clear, static_cast<size_t>(total));
}

void encode_weight_grad_classic(
    SparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    mx::array& out
) {
    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);

    auto use_block4 = shape.in_channels % 4 == 0 &&
                      shape.out_channels % 4 == 0 && shape.n_kernels >= 16 &&
                      shape.in_capacity >= 50000;
    auto use_cout16 = shape.out_channels == 16 && shape.n_kernels >= 16;
    auto use_gather = shape.n_kernels >= 16;
    if (!use_gather && !use_block4 && !use_cout16) {
        clear_output(encoder, device, library, out);
    }
    auto kernel = device.get_kernel(
        use_block4
            ? "sparse_relation_conv_weight_grad_block4_f32_i32"
            : (use_cout16
                   ? "sparse_relation_conv_weight_grad_cout16_f32_i32"
                   : (use_gather
                          ? "sparse_relation_conv_weight_grad_f32_i32"
                          : "sparse_relation_conv_weight_grad_atomic_f32_i32")),
        library
    );
    encoder.set_compute_pipeline_state(kernel);
    for (int index = 0; index < int(inputs.size()); ++index) {
        encoder.set_input_array(inputs[index], index);
    }
    encoder.set_output_array(out, 9);
    encoder.set_bytes(static_cast<int>(inputs[2].shape(0)), 10);
    encoder.set_bytes(shape.out_capacity, 11);
    encoder.set_bytes(shape.n_kernels, 12);
    encoder.set_bytes(shape.in_channels, 13);
    encoder.set_bytes(shape.out_channels, 14);
    encoder.set_bytes(stride_at(inputs[0], 0), 15);
    encoder.set_bytes(stride_at(inputs[0], 1), 16);
    encoder.set_bytes(stride_at(inputs[1], 0), 17);
    encoder.set_bytes(stride_at(inputs[1], 1), 18);
    encoder.set_bytes(shape.weight_layout, 19);
    encoder.set_bytes(shape.kernel_x, 20);
    encoder.set_bytes(shape.kernel_y, 21);
    encoder.set_bytes(shape.kernel_z, 22);
    if (use_block4) {
        auto total_tiles = static_cast<size_t>(shape.n_kernels) *
                           static_cast<size_t>(shape.in_channels / 4);
        encoder.dispatch_threadgroups(
            MTL::Size(total_tiles, 1, 1), MTL::Size(64, 1, 1)
        );
    } else if (use_cout16) {
        auto total_pairs = static_cast<size_t>(shape.n_kernels) *
                           static_cast<size_t>(shape.in_channels);
        encoder.dispatch_threadgroups(
            MTL::Size(total_pairs, 1, 1), MTL::Size(256, 1, 1)
        );
    } else {
        dispatch_1d(
            encoder,
            kernel,
            (use_gather ? static_cast<size_t>(shape.n_kernels)
                        : static_cast<size_t>(inputs[2].shape(0))) *
                static_cast<size_t>(shape.in_channels) *
                static_cast<size_t>(shape.out_channels)
        );
    }
}

#endif

} // namespace

void eval(
    SparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
#ifdef _METAL_
    auto& out = outputs[0];
    allocate(out);
    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);

    auto use_cout16 = shape.out_channels == 16 &&
                      ((shape.n_kernels >= 16 && shape.out_capacity >= 4096) ||
                       shape.out_capacity >= 50000);
    auto use_vec4 = shape.out_channels % 4 == 0;
    auto use_gather = use_vec4 || shape.n_kernels == 1;
    if (!use_gather) {
        clear_output(encoder, device, library, out);
    }
    auto kernel = device.get_kernel(
        use_cout16
            ? "sparse_relation_conv_f32_i32_cout16"
            : (use_vec4 ? "sparse_relation_conv_f32_i32_vec4"
                        : (use_gather ? "sparse_relation_conv_f32_i32"
                                      : "sparse_relation_conv_atomic_f32_i32")),
        library
    );
    encoder.set_compute_pipeline_state(kernel);
    for (int index = 0; index < 7; ++index) {
        encoder.set_input_array(inputs[index], index);
    }
    encoder.set_output_array(out, 7);
    bind_common_shape(encoder, inputs, shape, 8);
    encoder.set_bytes(stride_at(inputs[0], 0), 12);
    encoder.set_bytes(stride_at(inputs[0], 1), 13);
    bind_weight_shape(encoder, inputs[1], shape, 14);
    dispatch_1d(
        encoder,
        kernel,
        use_cout16
            ? static_cast<size_t>(shape.out_capacity)
            : (use_vec4 ? static_cast<size_t>(shape.out_capacity) *
                              static_cast<size_t>(shape.out_channels / 4)
                        : static_cast<size_t>(
                              use_gather ? shape.out_capacity
                                         : static_cast<int>(inputs[2].shape(0))
                          ) * static_cast<size_t>(shape.out_channels))
    );
#else
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

void eval_input_grad(
    SparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
#ifdef _METAL_
    auto& out = outputs[0];
    allocate(out);
    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);

    auto use_cin16 = shape.in_channels == 16 && shape.in_capacity >= 4096;
    auto use_vec4 = shape.in_channels % 4 == 0;
    auto kernel = device.get_kernel(
        use_cin16 ? "sparse_relation_conv_input_grad_f32_i32_cin16"
                  : (use_vec4 ? "sparse_relation_conv_input_grad_f32_i32_vec4"
                              : "sparse_relation_conv_input_grad_f32_i32"),
        library
    );
    encoder.set_compute_pipeline_state(kernel);
    for (int index = 0; index < int(inputs.size()); ++index) {
        encoder.set_input_array(inputs[index], index);
    }
    encoder.set_output_array(out, 9);
    encoder.set_bytes(static_cast<int>(inputs[2].shape(0)), 10);
    encoder.set_bytes(shape.out_capacity, 11);
    encoder.set_bytes(shape.in_capacity, 12);
    encoder.set_bytes(shape.in_channels, 13);
    encoder.set_bytes(shape.out_channels, 14);
    encoder.set_bytes(stride_at(inputs[0], 0), 15);
    encoder.set_bytes(stride_at(inputs[0], 1), 16);
    bind_weight_shape(encoder, inputs[1], shape, 17);
    dispatch_1d(
        encoder,
        kernel,
        use_cin16 ? static_cast<size_t>(shape.in_capacity)
                  : (use_vec4 ? static_cast<size_t>(shape.in_capacity) *
                                    static_cast<size_t>(shape.in_channels / 4)
                              : static_cast<size_t>(shape.in_capacity) *
                                    static_cast<size_t>(shape.in_channels))
    );
#else
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

void eval_weight_grad(
    SparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
#ifdef _METAL_
    auto& out = outputs[0];
    allocate(out);
    if (tensor_ops::conv::weight_grad::is_preferred(shape, stream)) {
        tensor_ops::conv::weight_grad::encode(shape, stream, inputs, out);
        return;
    }
    encode_weight_grad_classic(shape, stream, inputs, out);
#else
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

} // namespace mlx_lattice::backend::metal::conv
