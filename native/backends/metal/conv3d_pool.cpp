#include "backends/metal/conv3d.h"

#include <algorithm>
#include <stdexcept>

#include "backends/metal/conv3d_common.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace mlx_lattice::metal {

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

} // namespace mlx_lattice::metal
