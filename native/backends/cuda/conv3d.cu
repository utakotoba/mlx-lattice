#include "backends/cuda/conv3d.h"

#include <cstdint>

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/utils.h"

namespace mlx_lattice::cuda {

namespace mx = mlx::core;

#include "backends/cuda/conv3d_kernels.cuh"
#include "backends/cuda/launch_utils.cuh"

// MARK: - api

void eval_conv3d_feats(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs,
    mx::Stream stream,
    int rows,
    int in_channels,
    int out_channels
) {
    const auto& feats = inputs[0];
    const auto& weight = inputs[1];
    const auto& maps = inputs[2];
    const auto& kernels = inputs[3];
    auto& out = outputs[0];

    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto& encoder = mx::cu::get_command_encoder(stream);
    encoder.set_input_array(feats);
    encoder.set_input_array(weight);
    encoder.set_input_array(maps);
    encoder.set_input_array(kernels);
    encoder.set_output_array(out);

    int out_size = rows * out_channels;
    launch(
        encoder, fill_zero_float32, out_size, mx::gpu_ptr<float>(out), out_size
    );
    int pair_count = maps.shape(0);
    launch(
        encoder,
        conv3d_feats_float32,
        static_cast<size_t>(pair_count) * out_channels,
        mx::gpu_ptr<float>(feats),
        mx::gpu_ptr<float>(weight),
        mx::gpu_ptr<int32_t>(maps),
        mx::gpu_ptr<int32_t>(kernels),
        mx::gpu_ptr<float>(out),
        pair_count,
        in_channels,
        out_channels
    );
}

void eval_conv3d_subm_feats(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs,
    mx::Stream stream,
    int rows,
    int in_channels,
    int out_channels,
    int center_kernel
) {
    const auto& feats = inputs[0];
    const auto& weight = inputs[1];
    const auto& maps = inputs[2];
    const auto& kernels = inputs[3];
    auto& out = outputs[0];

    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto& encoder = mx::cu::get_command_encoder(stream);
    encoder.set_input_array(feats);
    encoder.set_input_array(weight);
    encoder.set_input_array(maps);
    encoder.set_input_array(kernels);
    encoder.set_output_array(out);

    launch(
        encoder,
        conv3d_subm_center_float32,
        static_cast<size_t>(rows) * out_channels,
        mx::gpu_ptr<float>(feats),
        mx::gpu_ptr<float>(weight),
        mx::gpu_ptr<float>(out),
        center_kernel,
        rows,
        in_channels,
        out_channels
    );
    int pair_count = maps.shape(0);
    launch(
        encoder,
        conv3d_subm_residual_float32,
        static_cast<size_t>(pair_count) * out_channels,
        mx::gpu_ptr<float>(feats),
        mx::gpu_ptr<float>(weight),
        mx::gpu_ptr<int32_t>(maps),
        mx::gpu_ptr<int32_t>(kernels),
        mx::gpu_ptr<float>(out),
        pair_count,
        in_channels,
        out_channels,
        center_kernel
    );
}

void eval_conv3d_residual_feats(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs,
    mx::Stream stream,
    int rows,
    int in_channels,
    int out_channels
) {
    const auto& base = inputs[0];
    const auto& feats = inputs[1];
    const auto& weight = inputs[2];
    const auto& maps = inputs[3];
    const auto& kernels = inputs[4];
    const auto& offsets = inputs[5];
    auto& out = outputs[0];

    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto& encoder = mx::cu::get_command_encoder(stream);
    encoder.set_input_array(base);
    encoder.set_input_array(feats);
    encoder.set_input_array(weight);
    encoder.set_input_array(maps);
    encoder.set_input_array(kernels);
    encoder.set_input_array(offsets);
    encoder.set_output_array(out);

    bool use_vec4 = out_channels % 4 == 0;
    auto elements =
        static_cast<size_t>(rows) *
        static_cast<size_t>(use_vec4 ? out_channels / 4 : out_channels);
    if (use_vec4) {
        launch(
            encoder,
            conv3d_residual_rows_vec4_float32,
            elements,
            mx::gpu_ptr<float>(base),
            mx::gpu_ptr<float>(feats),
            mx::gpu_ptr<float>(weight),
            mx::gpu_ptr<int32_t>(maps),
            mx::gpu_ptr<int32_t>(kernels),
            mx::gpu_ptr<int32_t>(offsets),
            mx::gpu_ptr<float>(out),
            rows,
            in_channels,
            out_channels
        );
        return;
    }
    launch(
        encoder,
        conv3d_residual_rows_float32,
        elements,
        mx::gpu_ptr<float>(base),
        mx::gpu_ptr<float>(feats),
        mx::gpu_ptr<float>(weight),
        mx::gpu_ptr<int32_t>(maps),
        mx::gpu_ptr<int32_t>(kernels),
        mx::gpu_ptr<int32_t>(offsets),
        mx::gpu_ptr<float>(out),
        rows,
        in_channels,
        out_channels
    );
}

void eval_pool3d_feats(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs,
    mx::Stream stream,
    int rows,
    int channels
) {
    const auto& feats = inputs[0];
    const auto& maps = inputs[1];
    const auto& kernels = inputs[2];
    const auto& offsets = inputs[3];
    auto& out = outputs[0];

    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto& encoder = mx::cu::get_command_encoder(stream);
    encoder.set_input_array(feats);
    encoder.set_input_array(maps);
    encoder.set_input_array(kernels);
    encoder.set_input_array(offsets);
    encoder.set_output_array(out);

    launch(
        encoder,
        pool3d_feats_float32,
        static_cast<size_t>(rows) * channels,
        mx::gpu_ptr<float>(feats),
        mx::gpu_ptr<int32_t>(maps),
        mx::gpu_ptr<int32_t>(kernels),
        mx::gpu_ptr<int32_t>(offsets),
        mx::gpu_ptr<float>(out),
        rows,
        channels
    );
}

void eval_pool3d_feats_grad(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs,
    mx::Stream stream,
    int rows,
    int channels
) {
    const auto& grad = inputs[0];
    const auto& maps = inputs[1];
    const auto& kernels = inputs[2];
    auto& out = outputs[0];

    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto& encoder = mx::cu::get_command_encoder(stream);
    encoder.set_input_array(grad);
    encoder.set_input_array(maps);
    encoder.set_input_array(kernels);
    encoder.set_output_array(out);

    int out_size = rows * channels;
    launch(
        encoder, fill_zero_float32, out_size, mx::gpu_ptr<float>(out), out_size
    );
    int pair_count = maps.shape(0);
    launch(
        encoder,
        pool3d_feats_grad_float32,
        static_cast<size_t>(pair_count) * channels,
        mx::gpu_ptr<float>(grad),
        mx::gpu_ptr<int32_t>(maps),
        mx::gpu_ptr<int32_t>(kernels),
        mx::gpu_ptr<float>(out),
        pair_count,
        channels
    );
}

void eval_conv3d_feats_grad(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs,
    mx::Stream stream,
    int rows,
    int in_channels,
    int out_channels
) {
    const auto& grad = inputs[0];
    const auto& weight = inputs[1];
    const auto& maps = inputs[2];
    const auto& kernels = inputs[3];
    auto& out = outputs[0];

    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto& encoder = mx::cu::get_command_encoder(stream);
    encoder.set_input_array(grad);
    encoder.set_input_array(weight);
    encoder.set_input_array(maps);
    encoder.set_input_array(kernels);
    encoder.set_output_array(out);

    int out_size = rows * in_channels;
    launch(
        encoder, fill_zero_float32, out_size, mx::gpu_ptr<float>(out), out_size
    );
    int pair_count = maps.shape(0);
    launch(
        encoder,
        conv3d_feats_grad_float32,
        static_cast<size_t>(pair_count) * in_channels,
        mx::gpu_ptr<float>(grad),
        mx::gpu_ptr<float>(weight),
        mx::gpu_ptr<int32_t>(maps),
        mx::gpu_ptr<int32_t>(kernels),
        mx::gpu_ptr<float>(out),
        pair_count,
        in_channels,
        out_channels
    );
}

void eval_conv3d_weight_grad(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs,
    mx::Stream stream,
    int kernels,
    int in_channels,
    int out_channels
) {
    const auto& feats = inputs[0];
    const auto& grad = inputs[1];
    const auto& maps = inputs[2];
    const auto& kernel_indices = inputs[3];
    auto& out = outputs[0];

    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto& encoder = mx::cu::get_command_encoder(stream);
    encoder.set_input_array(feats);
    encoder.set_input_array(grad);
    encoder.set_input_array(maps);
    encoder.set_input_array(kernel_indices);
    encoder.set_output_array(out);

    int out_size = kernels * in_channels * out_channels;
    launch(
        encoder, fill_zero_float32, out_size, mx::gpu_ptr<float>(out), out_size
    );
    int pair_count = maps.shape(0);
    launch(
        encoder,
        conv3d_weight_grad_float32,
        static_cast<size_t>(pair_count) * in_channels * out_channels,
        mx::gpu_ptr<float>(feats),
        mx::gpu_ptr<float>(grad),
        mx::gpu_ptr<int32_t>(maps),
        mx::gpu_ptr<int32_t>(kernel_indices),
        mx::gpu_ptr<float>(out),
        pair_count,
        in_channels,
        out_channels
    );
}

} // namespace mlx_lattice::cuda
