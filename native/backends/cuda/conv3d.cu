#include "backends/cuda/conv3d.h"

#include <algorithm>
#include <cstdint>

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/utils.h"

namespace mlx_lattice::cuda {

namespace {

namespace mx = mlx::core;

// MARK: - kernels

__global__ void fill_zero_float32(float* out, int size) {
    int elem = blockIdx.x * blockDim.x + threadIdx.x;
    if (elem < size) {
        out[elem] = 0.0f;
    }
}

__global__ void conv3d_feats_float32(
    const float* __restrict__ feats,
    const float* __restrict__ weight,
    const int* __restrict__ maps,
    const int* __restrict__ kernels,
    float* __restrict__ out,
    int pair_count,
    int in_channels,
    int out_channels
) {
    int elem = blockIdx.x * blockDim.x + threadIdx.x;
    int total = pair_count * out_channels;
    if (elem >= total) {
        return;
    }

    int pair = elem / out_channels;
    int out_col = elem - pair * out_channels;
    int kernel = kernels[pair];
    if (kernel < 0) {
        return;
    }

    int in_row = maps[pair * 2];
    int out_row = maps[pair * 2 + 1];
    float acc = 0.0f;
    for (int in_col = 0; in_col < in_channels; ++in_col) {
        acc += feats[in_row * in_channels + in_col] *
               weight[(kernel * in_channels + in_col) * out_channels + out_col];
    }
    atomicAdd(&out[out_row * out_channels + out_col], acc);
}

__global__ void conv3d_subm_center_float32(
    const float* __restrict__ feats,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int center_kernel,
    int rows,
    int in_channels,
    int out_channels
) {
    int elem = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * out_channels;
    if (elem >= total) {
        return;
    }

    int row = elem / out_channels;
    int out_col = elem - row * out_channels;
    float acc = 0.0f;
    for (int in_col = 0; in_col < in_channels; ++in_col) {
        acc += feats[row * in_channels + in_col] *
               weight
                   [(center_kernel * in_channels + in_col) * out_channels +
                    out_col];
    }
    out[elem] = acc;
}

__global__ void conv3d_subm_residual_float32(
    const float* __restrict__ feats,
    const float* __restrict__ weight,
    const int* __restrict__ maps,
    const int* __restrict__ kernels,
    float* __restrict__ out,
    int pair_count,
    int in_channels,
    int out_channels,
    int center_kernel
) {
    int elem = blockIdx.x * blockDim.x + threadIdx.x;
    int total = pair_count * out_channels;
    if (elem >= total) {
        return;
    }

    int pair = elem / out_channels;
    int kernel = kernels[pair];
    if (kernel < 0 || kernel == center_kernel) {
        return;
    }

    int out_col = elem - pair * out_channels;
    int in_row = maps[pair * 2];
    int out_row = maps[pair * 2 + 1];
    float acc = 0.0f;
    for (int in_col = 0; in_col < in_channels; ++in_col) {
        acc += feats[in_row * in_channels + in_col] *
               weight[(kernel * in_channels + in_col) * out_channels + out_col];
    }
    atomicAdd(&out[out_row * out_channels + out_col], acc);
}

__global__ void conv3d_residual_rows_float32(
    const float* __restrict__ base,
    const float* __restrict__ feats,
    const float* __restrict__ weight,
    const int* __restrict__ maps,
    const int* __restrict__ kernels,
    const int* __restrict__ offsets,
    float* __restrict__ out,
    int rows,
    int in_channels,
    int out_channels
) {
    int elem = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * out_channels;
    if (elem >= total) {
        return;
    }

    int row = elem / out_channels;
    int out_col = elem - row * out_channels;
    float acc = base[elem];
    for (int pair = offsets[row]; pair < offsets[row + 1]; ++pair) {
        int kernel = kernels[pair];
        if (kernel < 0) {
            continue;
        }
        int in_row = maps[pair * 2];
        for (int in_col = 0; in_col < in_channels; ++in_col) {
            acc +=
                feats[in_row * in_channels + in_col] *
                weight
                    [(kernel * in_channels + in_col) * out_channels + out_col];
        }
    }
    out[elem] = acc;
}

__global__ void conv3d_residual_rows_vec4_float32(
    const float* __restrict__ base,
    const float* __restrict__ feats,
    const float* __restrict__ weight,
    const int* __restrict__ maps,
    const int* __restrict__ kernels,
    const int* __restrict__ offsets,
    float* __restrict__ out,
    int rows,
    int in_channels,
    int out_channels
) {
    int blocks = out_channels / 4;
    int elem = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * blocks;
    if (elem >= total) {
        return;
    }

    int row = elem / blocks;
    int out_col = (elem - row * blocks) * 4;
    int row_base = row * out_channels + out_col;
    float4 acc = make_float4(
        base[row_base],
        base[row_base + 1],
        base[row_base + 2],
        base[row_base + 3]
    );

    for (int pair = offsets[row]; pair < offsets[row + 1]; ++pair) {
        int kernel = kernels[pair];
        if (kernel < 0) {
            continue;
        }
        int in_row = maps[pair * 2];
        int feat_base = in_row * in_channels;
        for (int in_col = 0; in_col < in_channels; ++in_col) {
            float value = feats[feat_base + in_col];
            int weight_base =
                (kernel * in_channels + in_col) * out_channels + out_col;
            acc.x += value * weight[weight_base];
            acc.y += value * weight[weight_base + 1];
            acc.z += value * weight[weight_base + 2];
            acc.w += value * weight[weight_base + 3];
        }
    }

    out[row_base] = acc.x;
    out[row_base + 1] = acc.y;
    out[row_base + 2] = acc.z;
    out[row_base + 3] = acc.w;
}

__global__ void pool3d_feats_float32(
    const float* __restrict__ feats,
    const int* __restrict__ maps,
    const int* __restrict__ kernels,
    const int* __restrict__ offsets,
    float* __restrict__ out,
    int rows,
    int channels
) {
    int elem = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * channels;
    if (elem >= total) {
        return;
    }

    int row = elem / channels;
    int channel = elem - row * channels;
    float acc = 0.0f;
    for (int pair = offsets[row]; pair < offsets[row + 1]; ++pair) {
        if (kernels[pair] < 0) {
            continue;
        }
        int in_row = maps[pair * 2];
        acc += feats[in_row * channels + channel];
    }
    out[elem] = acc;
}

__global__ void pool3d_feats_grad_float32(
    const float* __restrict__ grad,
    const int* __restrict__ maps,
    const int* __restrict__ kernels,
    float* __restrict__ out,
    int pair_count,
    int channels
) {
    int elem = blockIdx.x * blockDim.x + threadIdx.x;
    int total = pair_count * channels;
    if (elem >= total) {
        return;
    }

    int pair = elem / channels;
    if (kernels[pair] < 0) {
        return;
    }
    int channel = elem - pair * channels;
    int in_row = maps[pair * 2];
    int out_row = maps[pair * 2 + 1];
    atomicAdd(
        &out[in_row * channels + channel], grad[out_row * channels + channel]
    );
}

__global__ void conv3d_feats_grad_float32(
    const float* __restrict__ grad,
    const float* __restrict__ weight,
    const int* __restrict__ maps,
    const int* __restrict__ kernels,
    float* __restrict__ out,
    int pair_count,
    int in_channels,
    int out_channels
) {
    int elem = blockIdx.x * blockDim.x + threadIdx.x;
    int total = pair_count * in_channels;
    if (elem >= total) {
        return;
    }

    int pair = elem / in_channels;
    int in_col = elem - pair * in_channels;
    int kernel = kernels[pair];
    if (kernel < 0) {
        return;
    }
    int in_row = maps[pair * 2];
    int out_row = maps[pair * 2 + 1];
    float acc = 0.0f;
    for (int out_col = 0; out_col < out_channels; ++out_col) {
        acc += grad[out_row * out_channels + out_col] *
               weight[(kernel * in_channels + in_col) * out_channels + out_col];
    }
    atomicAdd(&out[in_row * in_channels + in_col], acc);
}

__global__ void conv3d_weight_grad_float32(
    const float* __restrict__ feats,
    const float* __restrict__ grad,
    const int* __restrict__ maps,
    const int* __restrict__ kernels,
    float* __restrict__ out,
    int pair_count,
    int in_channels,
    int out_channels
) {
    int elem = blockIdx.x * blockDim.x + threadIdx.x;
    int total = pair_count * in_channels * out_channels;
    if (elem >= total) {
        return;
    }

    int pair = elem / (in_channels * out_channels);
    int local = elem - pair * in_channels * out_channels;
    int in_col = local / out_channels;
    int out_col = local - in_col * out_channels;
    int kernel = kernels[pair];
    if (kernel < 0) {
        return;
    }
    int in_row = maps[pair * 2];
    int out_row = maps[pair * 2 + 1];
    float value = feats[in_row * in_channels + in_col] *
                  grad[out_row * out_channels + out_col];
    atomicAdd(
        &out[(kernel * in_channels + in_col) * out_channels + out_col], value
    );
}

// MARK: - helpers

dim3 grid_for(size_t elements, int block = 256) {
    return dim3(static_cast<unsigned int>((elements + block - 1) / block));
}

template <typename Kernel, typename... Args>
void launch(
    mx::cu::CommandEncoder& encoder,
    Kernel kernel,
    size_t elements,
    Args... args
) {
    if (elements == 0) {
        return;
    }
    constexpr int block = 256;
    encoder.add_kernel_node(
        kernel, grid_for(elements, block), dim3(block), args...
    );
}

} // namespace

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
