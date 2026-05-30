#include "backends/cpu/conv3d.h"

#include <algorithm>

#include "mlx/backend/cpu/encoder.h"

namespace mlx_lattice::cpu {

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
    auto& encoder = mx::cpu::get_command_encoder(stream);
    encoder.set_input_array(feats);
    encoder.set_input_array(weight);
    encoder.set_input_array(maps);
    encoder.set_input_array(kernels);
    encoder.set_output_array(out);

    encoder.dispatch([feats_ptr = feats.data<float>(),
                      weight_ptr = weight.data<float>(),
                      maps_ptr = maps.data<int32_t>(),
                      kernels_ptr = kernels.data<int32_t>(),
                      out_ptr = out.data<float>(),
                      pairs = maps.shape(0),
                      rows,
                      in_channels,
                      out_channels]() {
        std::fill(out_ptr, out_ptr + rows * out_channels, 0.0f);
        for (int pair = 0; pair < pairs; ++pair) {
            int in_row = maps_ptr[pair * 2];
            int out_row = maps_ptr[pair * 2 + 1];
            int kernel = kernels_ptr[pair];
            if (kernel < 0) {
                continue;
            }
            for (int out_col = 0; out_col < out_channels; ++out_col) {
                float acc = 0.0f;
                for (int in_col = 0; in_col < in_channels; ++in_col) {
                    acc += feats_ptr[in_row * in_channels + in_col] *
                           weight_ptr
                               [(kernel * in_channels + in_col) * out_channels +
                                out_col];
                }
                out_ptr[out_row * out_channels + out_col] += acc;
            }
        }
    });
}

} // namespace mlx_lattice::cpu
