#include "backends/metal/conv3d.h"

#include <dlfcn.h>

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <string>

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace mlx_lattice::metal {

namespace {

// MARK: - helpers

std::string binary_dir() {
    static std::string dir = [] {
        Dl_info info;
        if (!dladdr(reinterpret_cast<void*>(&binary_dir), &info)) {
            throw std::runtime_error("Unable to resolve native module path.");
        }
        return std::filesystem::path(info.dli_fname).parent_path().string();
    }();
    return dir;
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
#ifdef _METAL_
    const auto& feats = inputs[0];
    const auto& weight = inputs[1];
    const auto& maps = inputs[2];
    const auto& kernels = inputs[3];
    auto& out = outputs[0];

    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto& device = mx::metal::device(stream.device);
    auto library = device.get_library("mlx_lattice", binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);

    auto zero = device.get_kernel("fill_zero_float32", library);
    int out_size = rows * out_channels;
    if (out_size > 0) {
        encoder.set_compute_pipeline_state(zero);
        encoder.set_output_array(out, 0);
        encoder.set_bytes(out_size, 1);
        auto zero_group = std::min(
            static_cast<size_t>(out_size), zero->maxTotalThreadsPerThreadgroup()
        );
        encoder.dispatch_threads(
            MTL::Size(static_cast<size_t>(out_size), 1, 1),
            MTL::Size(zero_group, 1, 1)
        );
    }

    auto conv = device.get_kernel("conv3d_feats_float32", library);
    auto elements =
        static_cast<size_t>(maps.shape(0)) * static_cast<size_t>(out_channels);
    if (elements == 0) {
        return;
    }

    encoder.set_compute_pipeline_state(conv);
    encoder.set_input_array(feats, 0);
    encoder.set_input_array(weight, 1);
    encoder.set_input_array(maps, 2);
    encoder.set_input_array(kernels, 3);
    encoder.set_output_array(out, 4);
    int pair_count = maps.shape(0);
    encoder.set_bytes(pair_count, 5);
    encoder.set_bytes(in_channels, 6);
    encoder.set_bytes(out_channels, 7);
    auto conv_group = std::min(elements, conv->maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threads(
        MTL::Size(elements, 1, 1), MTL::Size(conv_group, 1, 1)
    );
#else
    throw std::runtime_error("Metal support is not available.");
#endif
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
#ifdef _METAL_
    const auto& feats = inputs[0];
    const auto& weight = inputs[1];
    const auto& maps = inputs[2];
    const auto& kernels = inputs[3];
    auto& out = outputs[0];

    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto& device = mx::metal::device(stream.device);
    auto library = device.get_library("mlx_lattice", binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);

    auto center = device.get_kernel("conv3d_subm_center_float32", library);
    auto center_elements =
        static_cast<size_t>(rows) * static_cast<size_t>(out_channels);
    if (center_elements > 0) {
        encoder.set_compute_pipeline_state(center);
        encoder.set_input_array(feats, 0);
        encoder.set_input_array(weight, 1);
        encoder.set_output_array(out, 2);
        encoder.set_bytes(center_kernel, 3);
        encoder.set_bytes(rows, 4);
        encoder.set_bytes(in_channels, 5);
        encoder.set_bytes(out_channels, 6);
        auto center_group =
            std::min(center_elements, center->maxTotalThreadsPerThreadgroup());
        encoder.dispatch_threads(
            MTL::Size(center_elements, 1, 1), MTL::Size(center_group, 1, 1)
        );
    }

    auto residual = device.get_kernel("conv3d_subm_residual_float32", library);
    int pair_count = maps.shape(0);
    if (pair_count == 0 || out_channels == 0) {
        return;
    }

    encoder.set_compute_pipeline_state(residual);
    encoder.set_input_array(feats, 0);
    encoder.set_input_array(weight, 1);
    encoder.set_input_array(maps, 2);
    encoder.set_input_array(kernels, 3);
    encoder.set_output_array(out, 4);
    encoder.set_bytes(pair_count, 5);
    encoder.set_bytes(in_channels, 6);
    encoder.set_bytes(out_channels, 7);
    encoder.set_bytes(center_kernel, 8);
    auto residual_elements =
        static_cast<size_t>(pair_count) * static_cast<size_t>(out_channels);
    auto residual_group =
        std::min(residual_elements, residual->maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threads(
        MTL::Size(residual_elements, 1, 1), MTL::Size(residual_group, 1, 1)
    );
#else
    throw std::runtime_error("Metal support is not available.");
#endif
}

void eval_conv3d_residual_feats(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs,
    mx::Stream stream,
    int rows,
    int in_channels,
    int out_channels
) {
#ifdef _METAL_
    const auto& base = inputs[0];
    const auto& feats = inputs[1];
    const auto& weight = inputs[2];
    const auto& maps = inputs[3];
    const auto& kernels = inputs[4];
    const auto& offsets = inputs[5];
    auto& out = outputs[0];

    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto& device = mx::metal::device(stream.device);
    auto library = device.get_library("mlx_lattice", binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);

    bool use_vec4 = out_channels % 4 == 0;
    auto residual = device.get_kernel(
        use_vec4 ? "conv3d_residual_rows_vec4_float32"
                 : "conv3d_residual_rows_float32",
        library
    );
    auto elements =
        static_cast<size_t>(rows) *
        static_cast<size_t>(use_vec4 ? out_channels / 4 : out_channels);
    if (elements == 0) {
        return;
    }

    encoder.set_compute_pipeline_state(residual);
    encoder.set_input_array(base, 0);
    encoder.set_input_array(feats, 1);
    encoder.set_input_array(weight, 2);
    encoder.set_input_array(maps, 3);
    encoder.set_input_array(kernels, 4);
    encoder.set_input_array(offsets, 5);
    encoder.set_output_array(out, 6);
    encoder.set_bytes(rows, 7);
    encoder.set_bytes(in_channels, 8);
    encoder.set_bytes(out_channels, 9);
    auto group = std::min(elements, residual->maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threads(MTL::Size(elements, 1, 1), MTL::Size(group, 1, 1));
#else
    throw std::runtime_error("Metal support is not available.");
#endif
}

void eval_pool3d_feats(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs,
    mx::Stream stream,
    int rows,
    int channels
) {
#ifdef _METAL_
    const auto& feats = inputs[0];
    const auto& maps = inputs[1];
    const auto& kernels = inputs[2];
    const auto& offsets = inputs[3];
    auto& out = outputs[0];

    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto& device = mx::metal::device(stream.device);
    auto library = device.get_library("mlx_lattice", binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);

    auto pool = device.get_kernel("pool3d_feats_float32", library);
    auto elements = static_cast<size_t>(rows) * static_cast<size_t>(channels);
    if (elements == 0) {
        return;
    }

    encoder.set_compute_pipeline_state(pool);
    encoder.set_input_array(feats, 0);
    encoder.set_input_array(maps, 1);
    encoder.set_input_array(kernels, 2);
    encoder.set_input_array(offsets, 3);
    encoder.set_output_array(out, 4);
    encoder.set_bytes(rows, 5);
    encoder.set_bytes(channels, 6);
    auto group = std::min(elements, pool->maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threads(MTL::Size(elements, 1, 1), MTL::Size(group, 1, 1));
#else
    throw std::runtime_error("Metal support is not available.");
#endif
}

void eval_pool3d_feats_grad(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs,
    mx::Stream stream,
    int rows,
    int channels
) {
#ifdef _METAL_
    const auto& grad = inputs[0];
    const auto& maps = inputs[1];
    const auto& kernels = inputs[2];
    auto& out = outputs[0];

    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto& device = mx::metal::device(stream.device);
    auto library = device.get_library("mlx_lattice", binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);

    auto zero = device.get_kernel("fill_zero_float32", library);
    int out_size = rows * channels;
    if (out_size > 0) {
        encoder.set_compute_pipeline_state(zero);
        encoder.set_output_array(out, 0);
        encoder.set_bytes(out_size, 1);
        auto zero_group = std::min(
            static_cast<size_t>(out_size), zero->maxTotalThreadsPerThreadgroup()
        );
        encoder.dispatch_threads(
            MTL::Size(static_cast<size_t>(out_size), 1, 1),
            MTL::Size(zero_group, 1, 1)
        );
    }

    auto grad_kernel = device.get_kernel("pool3d_feats_grad_float32", library);
    auto elements =
        static_cast<size_t>(maps.shape(0)) * static_cast<size_t>(channels);
    if (elements == 0) {
        return;
    }

    encoder.set_compute_pipeline_state(grad_kernel);
    encoder.set_input_array(grad, 0);
    encoder.set_input_array(maps, 1);
    encoder.set_input_array(kernels, 2);
    encoder.set_output_array(out, 3);
    int pair_count = maps.shape(0);
    encoder.set_bytes(pair_count, 4);
    encoder.set_bytes(channels, 5);
    auto group =
        std::min(elements, grad_kernel->maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threads(MTL::Size(elements, 1, 1), MTL::Size(group, 1, 1));
#else
    throw std::runtime_error("Metal support is not available.");
#endif
}

void eval_conv3d_feats_grad(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs,
    mx::Stream stream,
    int rows,
    int in_channels,
    int out_channels
) {
#ifdef _METAL_
    const auto& grad = inputs[0];
    const auto& weight = inputs[1];
    const auto& maps = inputs[2];
    const auto& kernels = inputs[3];
    auto& out = outputs[0];

    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto& device = mx::metal::device(stream.device);
    auto library = device.get_library("mlx_lattice", binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);

    auto zero = device.get_kernel("fill_zero_float32", library);
    int out_size = rows * in_channels;
    if (out_size > 0) {
        encoder.set_compute_pipeline_state(zero);
        encoder.set_output_array(out, 0);
        encoder.set_bytes(out_size, 1);
        auto zero_group = std::min(
            static_cast<size_t>(out_size), zero->maxTotalThreadsPerThreadgroup()
        );
        encoder.dispatch_threads(
            MTL::Size(static_cast<size_t>(out_size), 1, 1),
            MTL::Size(zero_group, 1, 1)
        );
    }

    auto grad_kernel = device.get_kernel("conv3d_feats_grad_float32", library);
    auto elements =
        static_cast<size_t>(maps.shape(0)) * static_cast<size_t>(in_channels);
    if (elements == 0) {
        return;
    }

    encoder.set_compute_pipeline_state(grad_kernel);
    encoder.set_input_array(grad, 0);
    encoder.set_input_array(weight, 1);
    encoder.set_input_array(maps, 2);
    encoder.set_input_array(kernels, 3);
    encoder.set_output_array(out, 4);
    int pair_count = maps.shape(0);
    encoder.set_bytes(pair_count, 5);
    encoder.set_bytes(in_channels, 6);
    encoder.set_bytes(out_channels, 7);
    auto group =
        std::min(elements, grad_kernel->maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threads(MTL::Size(elements, 1, 1), MTL::Size(group, 1, 1));
#else
    throw std::runtime_error("Metal support is not available.");
#endif
}

void eval_conv3d_weight_grad(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs,
    mx::Stream stream,
    int kernels,
    int in_channels,
    int out_channels
) {
#ifdef _METAL_
    const auto& feats = inputs[0];
    const auto& grad = inputs[1];
    const auto& maps = inputs[2];
    const auto& kernel_indices = inputs[3];
    auto& out = outputs[0];

    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto& device = mx::metal::device(stream.device);
    auto library = device.get_library("mlx_lattice", binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);

    auto zero = device.get_kernel("fill_zero_float32", library);
    int out_size = kernels * in_channels * out_channels;
    if (out_size > 0) {
        encoder.set_compute_pipeline_state(zero);
        encoder.set_output_array(out, 0);
        encoder.set_bytes(out_size, 1);
        auto zero_group = std::min(
            static_cast<size_t>(out_size), zero->maxTotalThreadsPerThreadgroup()
        );
        encoder.dispatch_threads(
            MTL::Size(static_cast<size_t>(out_size), 1, 1),
            MTL::Size(zero_group, 1, 1)
        );
    }

    auto grad_kernel = device.get_kernel("conv3d_weight_grad_float32", library);
    auto elements = static_cast<size_t>(maps.shape(0)) *
                    static_cast<size_t>(in_channels) *
                    static_cast<size_t>(out_channels);
    if (elements == 0) {
        return;
    }

    encoder.set_compute_pipeline_state(grad_kernel);
    encoder.set_input_array(feats, 0);
    encoder.set_input_array(grad, 1);
    encoder.set_input_array(maps, 2);
    encoder.set_input_array(kernel_indices, 3);
    encoder.set_output_array(out, 4);
    int pair_count = maps.shape(0);
    encoder.set_bytes(pair_count, 5);
    encoder.set_bytes(in_channels, 6);
    encoder.set_bytes(out_channels, 7);
    auto group =
        std::min(elements, grad_kernel->maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threads(MTL::Size(elements, 1, 1), MTL::Size(group, 1, 1));
#else
    throw std::runtime_error("Metal support is not available.");
#endif
}

} // namespace mlx_lattice::metal
