#include "backends/cpu/conv3d.h"

#include <algorithm>

#include "mlx/backend/cpu/encoder.h"

namespace mlx_lattice::cpu {

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
    auto& encoder = mx::cpu::get_command_encoder(stream);
    encoder.set_input_array(feats);
    encoder.set_input_array(maps);
    encoder.set_input_array(kernels);
    encoder.set_input_array(offsets);
    encoder.set_output_array(out);

    encoder.dispatch([feats_ptr = feats.data<float>(),
                      maps_ptr = maps.data<int32_t>(),
                      kernels_ptr = kernels.data<int32_t>(),
                      offsets_ptr = offsets.data<int32_t>(),
                      out_ptr = out.data<float>(),
                      rows,
                      channels]() {
        for (int row = 0; row < rows; ++row) {
            for (int channel = 0; channel < channels; ++channel) {
                float acc = 0.0f;
                for (int pair = offsets_ptr[row]; pair < offsets_ptr[row + 1];
                     ++pair) {
                    if (kernels_ptr[pair] < 0) {
                        continue;
                    }
                    int in_row = maps_ptr[pair * 2];
                    acc += feats_ptr[in_row * channels + channel];
                }
                out_ptr[row * channels + channel] = acc;
            }
        }
    });
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
    auto& encoder = mx::cpu::get_command_encoder(stream);
    encoder.set_input_array(grad);
    encoder.set_input_array(maps);
    encoder.set_input_array(kernels);
    encoder.set_output_array(out);

    encoder.dispatch([grad_ptr = grad.data<float>(),
                      maps_ptr = maps.data<int32_t>(),
                      kernels_ptr = kernels.data<int32_t>(),
                      out_ptr = out.data<float>(),
                      pairs = maps.shape(0),
                      rows,
                      channels]() {
        std::fill(out_ptr, out_ptr + rows * channels, 0.0f);
        for (int pair = 0; pair < pairs; ++pair) {
            if (kernels_ptr[pair] < 0) {
                continue;
            }
            int in_row = maps_ptr[pair * 2];
            int out_row = maps_ptr[pair * 2 + 1];
            for (int channel = 0; channel < channels; ++channel) {
                out_ptr[in_row * channels + channel] +=
                    grad_ptr[out_row * channels + channel];
            }
        }
    });
}

} // namespace mlx_lattice::cpu
