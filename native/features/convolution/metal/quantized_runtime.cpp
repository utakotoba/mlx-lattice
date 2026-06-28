#include "features/convolution/metal/runtime.h"

#include "foundation/array_utils.h"
#include "platform/metal/runtime_utils.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace mlx_lattice::backend::metal::conv {

void eval_quantized(
    QuantizedSparseConvShape shape,
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
    auto fp16 = inputs[0].dtype() == mx::float16;
    auto kernel = device.get_kernel(
        fp16 ? (shape.bits == 4 ? "sparse_quantized_conv_f16_i32_b4"
                                : "sparse_quantized_conv_f16_i32_b8")
             : (shape.bits == 4 ? "sparse_quantized_conv_f32_i32_b4"
                                : "sparse_quantized_conv_f32_i32_b8"),
        library
    );
    encoder.set_compute_pipeline_state(kernel);
    for (int index = 0; index < static_cast<int>(inputs.size()); ++index) {
        encoder.set_input_array(inputs[index], index);
    }
    encoder.set_output_array(out, 9);
    set_bytes_range(
        encoder,
        10,
        static_cast<int>(inputs[4].shape(0)),
        shape.out_capacity,
        shape.in_channels,
        shape.out_channels,
        shape.storage_in_channels,
        shape.group_size,
        static_cast<int>(inputs[0].strides(0)),
        static_cast<int>(inputs[0].strides(1))
    );
    auto channel_blocks = (shape.out_channels + 3) / 4;
    dispatch_1d(
        encoder,
        kernel,
        static_cast<size_t>(shape.out_capacity) * channel_blocks
    );
#else
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

} // namespace mlx_lattice::backend::metal::conv
