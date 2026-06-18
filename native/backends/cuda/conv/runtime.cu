#include "backends/cuda/conv/runtime.h"

#include <stdexcept>
#include <utility>

#include "backends/array_utils.h"
#include "backends/cuda/conv/kernels.cuh"
#include "backends/cuda/runtime_utils.h"
#include "mlx/backend/cuda/utils.h"

namespace mlx_lattice::backend::cuda::conv {
namespace {

int stride_at(const mx::array& array, int dim) {
    return dim < array.ndim() ? static_cast<int>(array.strides(dim)) : 0;
}

void require_i32(const mx::array& input, const char* name) {
    if (input.dtype() != mx::int32) {
        throw std::invalid_argument(
            std::string("CUDA sparse convolution requires int32 ") + name + "."
        );
    }
}

void require_features(const mx::array& lhs, const mx::array& rhs) {
    if ((lhs.dtype() != mx::float32 && lhs.dtype() != mx::float16) ||
        rhs.dtype() != lhs.dtype()) {
        throw std::invalid_argument(
            "CUDA sparse convolution requires matching float32 or float16 "
            "feature tensors."
        );
    }
}

ConvShapeArgs shape_args(SparseConvShape shape, const mx::array& in_rows) {
    return ConvShapeArgs{
        .edge_capacity = static_cast<int>(in_rows.shape(0)),
        .in_capacity = shape.in_capacity,
        .out_capacity = shape.out_capacity,
        .n_kernels = shape.n_kernels,
        .in_channels = shape.in_channels,
        .out_channels = shape.out_channels,
        .weight_layout = shape.weight_layout,
        .kernel_x = shape.kernel_x,
        .kernel_y = shape.kernel_y,
        .kernel_z = shape.kernel_z,
    };
}

ConvStrideArgs forward_strides(
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& out
) {
    return ConvStrideArgs{
        .feat_s0 = stride_at(feats, 0),
        .feat_s1 = stride_at(feats, 1),
        .cot_s0 = stride_at(feats, 0),
        .cot_s1 = stride_at(feats, 1),
        .weight_s0 = stride_at(weights, 0),
        .weight_s1 = stride_at(weights, 1),
        .weight_s2 = stride_at(weights, 2),
        .weight_s3 = stride_at(weights, 3),
        .weight_s4 = stride_at(weights, 4),
        .out_s0 = stride_at(out, 0),
        .out_s1 = stride_at(out, 1),
        .out_s2 = stride_at(out, 2),
        .out_s3 = stride_at(out, 3),
        .out_s4 = stride_at(out, 4),
    };
}

ConvStrideArgs grad_strides(
    const mx::array& lhs,
    const mx::array& weights,
    const mx::array& out
) {
    auto strides = forward_strides(lhs, weights, out);
    strides.out_s0 = stride_at(out, 0);
    strides.out_s1 = stride_at(out, 1);
    strides.out_s2 = stride_at(out, 2);
    strides.out_s3 = stride_at(out, 3);
    strides.out_s4 = stride_at(out, 4);
    return strides;
}

template <typename Kernel, typename... Args>
void add_1d(
    const mx::Stream& stream,
    Kernel kernel,
    std::size_t elements,
    Args&&... args
) {
    auto launch = launch_1d(elements);
    mx::cu::get_command_encoder(stream).add_kernel_node(
        kernel, launch.grid, launch.block, std::forward<Args>(args)...
    );
}

template <typename T> T* device_ptr(mx::array& array) {
    return mx::gpu_ptr<T>(array);
}

template <typename T> const T* device_ptr(const mx::array& array) {
    return mx::gpu_ptr<T>(array);
}

bool can_use_square_channel_forward(
    SparseConvShape shape,
    const mx::array& out
) {
    return shape.in_channels == shape.out_channels && out.ndim() == 2 &&
           out.strides(1) == 1 && out.strides(0) == shape.out_channels &&
           (shape.out_channels == 16 || shape.out_channels == 32 ||
            shape.out_channels == 64);
}

} // namespace

void eval(
    SparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    require_features(inputs[0], inputs[1]);
    for (int index = 2; index < 7; ++index) {
        require_i32(inputs[index], "relation array");
    }

    auto& out = outputs[0];
    allocate(out);
    auto& encoder = mx::cu::get_command_encoder(stream);
    for (const auto& input : inputs) {
        encoder.set_input_array(input);
    }
    encoder.set_output_array(out);

    auto shape_arg = shape_args(shape, inputs[2]);
    auto stride_arg = forward_strides(inputs[0], inputs[1], out);
    auto elements = static_cast<std::size_t>(shape.out_capacity) *
                    static_cast<std::size_t>(shape.out_channels);

    if (inputs[0].dtype() == mx::float32) {
        auto kernel = sparse_conv_forward_f32;
        if (can_use_square_channel_forward(shape, out)) {
            if (shape.out_channels == 16) {
                kernel = sparse_conv_forward_f32_c16;
            } else if (shape.out_channels == 32) {
                kernel = sparse_conv_forward_f32_c32;
            } else {
                kernel = sparse_conv_forward_f32_c64;
            }
            elements = static_cast<std::size_t>(shape.out_capacity);
        }
        add_1d(
            stream,
            kernel,
            elements,
            device_ptr<float>(inputs[0]),
            device_ptr<float>(inputs[1]),
            device_ptr<int>(inputs[2]),
            device_ptr<int>(inputs[3]),
            device_ptr<int>(inputs[4]),
            device_ptr<int>(inputs[5]),
            device_ptr<int>(inputs[6]),
            device_ptr<float>(out),
            shape_arg,
            stride_arg
        );
        return;
    }

    auto kernel = sparse_conv_forward_f16;
    if (can_use_square_channel_forward(shape, out)) {
        if (shape.out_channels == 16) {
            kernel = sparse_conv_forward_f16_c16;
        } else if (shape.out_channels == 32) {
            kernel = sparse_conv_forward_f16_c32;
        } else {
            kernel = sparse_conv_forward_f16_c64;
        }
        elements = static_cast<std::size_t>(shape.out_capacity);
    }
    add_1d(
        stream,
        kernel,
        elements,
        device_ptr<__half>(inputs[0]),
        device_ptr<__half>(inputs[1]),
        device_ptr<int>(inputs[2]),
        device_ptr<int>(inputs[3]),
        device_ptr<int>(inputs[4]),
        device_ptr<int>(inputs[5]),
        device_ptr<int>(inputs[6]),
        device_ptr<__half>(out),
        shape_arg,
        stride_arg
    );
}

void eval_input_grad(
    SparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    require_features(inputs[0], inputs[1]);
    for (int index = 2; index < 9; ++index) {
        require_i32(inputs[index], "relation array");
    }

    auto& out = outputs[0];
    allocate(out);
    auto& encoder = mx::cu::get_command_encoder(stream);
    for (const auto& input : inputs) {
        encoder.set_input_array(input);
    }
    encoder.set_output_array(out);

    auto shape_arg = shape_args(shape, inputs[2]);
    auto stride_arg = grad_strides(inputs[0], inputs[1], out);
    auto elements = static_cast<std::size_t>(shape.in_capacity) *
                    static_cast<std::size_t>(shape.in_channels);

    if (inputs[0].dtype() == mx::float32) {
        add_1d(
            stream,
            sparse_conv_input_grad_f32,
            elements,
            device_ptr<float>(inputs[0]),
            device_ptr<float>(inputs[1]),
            device_ptr<int>(inputs[2]),
            device_ptr<int>(inputs[3]),
            device_ptr<int>(inputs[4]),
            device_ptr<int>(inputs[5]),
            device_ptr<int>(inputs[6]),
            device_ptr<int>(inputs[7]),
            device_ptr<int>(inputs[8]),
            device_ptr<float>(out),
            shape_arg,
            stride_arg
        );
        return;
    }

    add_1d(
        stream,
        sparse_conv_input_grad_f16,
        elements,
        device_ptr<__half>(inputs[0]),
        device_ptr<__half>(inputs[1]),
        device_ptr<int>(inputs[2]),
        device_ptr<int>(inputs[3]),
        device_ptr<int>(inputs[4]),
        device_ptr<int>(inputs[5]),
        device_ptr<int>(inputs[6]),
        device_ptr<int>(inputs[7]),
        device_ptr<int>(inputs[8]),
        device_ptr<__half>(out),
        shape_arg,
        stride_arg
    );
}

void eval_weight_grad(
    SparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    require_features(inputs[0], inputs[1]);
    for (int index = 2; index < 9; ++index) {
        require_i32(inputs[index], "relation array");
    }

    auto& out = outputs[0];
    allocate(out);
    auto& encoder = mx::cu::get_command_encoder(stream);
    for (const auto& input : inputs) {
        encoder.set_input_array(input);
    }
    encoder.set_output_array(out);

    auto shape_arg = shape_args(shape, inputs[2]);
    auto stride_arg = grad_strides(inputs[0], out, out);
    auto elements = static_cast<std::size_t>(shape.n_kernels) *
                    static_cast<std::size_t>(shape.in_channels) *
                    static_cast<std::size_t>(shape.out_channels);

    if (inputs[0].dtype() == mx::float32) {
        add_1d(
            stream,
            sparse_conv_weight_grad_f32,
            elements,
            device_ptr<float>(inputs[0]),
            device_ptr<float>(inputs[1]),
            device_ptr<int>(inputs[2]),
            device_ptr<int>(inputs[3]),
            device_ptr<int>(inputs[4]),
            device_ptr<int>(inputs[5]),
            device_ptr<int>(inputs[6]),
            device_ptr<int>(inputs[7]),
            device_ptr<int>(inputs[8]),
            device_ptr<float>(out),
            shape_arg,
            stride_arg
        );
        return;
    }

    add_1d(
        stream,
        sparse_conv_weight_grad_f16,
        elements,
        device_ptr<__half>(inputs[0]),
        device_ptr<__half>(inputs[1]),
        device_ptr<int>(inputs[2]),
        device_ptr<int>(inputs[3]),
        device_ptr<int>(inputs[4]),
        device_ptr<int>(inputs[5]),
        device_ptr<int>(inputs[6]),
        device_ptr<int>(inputs[7]),
        device_ptr<int>(inputs[8]),
        device_ptr<__half>(out),
        shape_arg,
        stride_arg
    );
}

} // namespace mlx_lattice::backend::cuda::conv
