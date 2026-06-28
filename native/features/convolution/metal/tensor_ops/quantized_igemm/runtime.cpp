#include "features/convolution/metal/tensor_ops/quantized_igemm/runtime.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

#include "foundation/array_utils.h"
#include "platform/metal/capabilities.h"
#include "platform/metal/runtime_utils.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace mlx_lattice::backend::metal::tensor_ops::conv::quantized_igemm {
namespace {

constexpr int kTileRows = 64;
constexpr int kTileChannels = 32;

#ifdef _METAL_
mx::array make_half_temp(std::size_t elements) {
    auto count = std::max<std::size_t>(elements, 1);
    return mx::array(
        mx::allocator::malloc(count * sizeof(mx::float16_t)),
        mx::Shape{static_cast<int>(count)},
        mx::float16
    );
}

const char* kernel_name(QuantizedSparseConvShape shape) {
    if (shape.in_channels == 32 && shape.out_channels == 32) {
        return shape.bits == 4 ? "sparse_quantized_igemm_f16_b4_cin32_cout32"
                               : "sparse_quantized_igemm_f16_b8_cin32_cout32";
    }
    if (shape.in_channels == 32 && shape.out_channels == 64) {
        return shape.bits == 4 ? "sparse_quantized_igemm_f16_b4_cin32_cout64"
                               : "sparse_quantized_igemm_f16_b8_cin32_cout64";
    }
    if (shape.in_channels == 64 && shape.out_channels == 32) {
        return shape.bits == 4 ? "sparse_quantized_igemm_f16_b4_cin64_cout32"
                               : "sparse_quantized_igemm_f16_b8_cin64_cout32";
    }
    if (shape.in_channels == 64 && shape.out_channels == 64) {
        return shape.bits == 4 ? "sparse_quantized_igemm_f16_b4_cin64_cout64"
                               : "sparse_quantized_igemm_f16_b8_cin64_cout64";
    }
    throw std::invalid_argument(
        "quantized implicit GEMM received an unsupported channel shape."
    );
}

const char* tensor_kernel_name(QuantizedSparseConvShape shape) {
    if (shape.in_channels == 32 && shape.out_channels == 32) {
        return shape.bits == 4 ? "sparse_quantized_tensor_f16_b4_cin32_cout32"
                               : "sparse_quantized_tensor_f16_b8_cin32_cout32";
    }
    if (shape.in_channels == 32 && shape.out_channels == 64) {
        return shape.bits == 4 ? "sparse_quantized_tensor_f16_b4_cin32_cout64"
                               : "sparse_quantized_tensor_f16_b8_cin32_cout64";
    }
    if (shape.in_channels == 64 && shape.out_channels == 32) {
        return shape.bits == 4 ? "sparse_quantized_tensor_f16_b4_cin64_cout32"
                               : "sparse_quantized_tensor_f16_b8_cin64_cout32";
    }
    if (shape.in_channels == 64 && shape.out_channels == 64) {
        return shape.bits == 4 ? "sparse_quantized_tensor_f16_b4_cin64_cout64"
                               : "sparse_quantized_tensor_f16_b8_cin64_cout64";
    }
    throw std::invalid_argument(
        "quantized tensor contraction received an unsupported channel shape."
    );
}
#endif

} // namespace

bool is_preferred(QuantizedSparseConvShape shape, const mx::Stream& stream) {
    auto supported_channels =
        (shape.in_channels == 32 || shape.in_channels == 64) &&
        (shape.out_channels == 32 || shape.out_channels == 64);
    return shape.sorted != 0 && shape.n_kernels == 27 &&
           shape.storage_in_channels == shape.in_channels &&
           shape.group_size <= shape.in_channels && supported_channels &&
           capability_tier(stream) != CapabilityTier::unavailable;
}

void encode(
    QuantizedSparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    mx::array& out
) {
#ifdef _METAL_
    if (!is_preferred(shape, stream) || inputs.size() != 12 ||
        inputs[0].dtype() != mx::float16) {
        throw std::invalid_argument(
            "quantized implicit GEMM requires a sorted fp16 K=27 problem "
            "with C_in and C_out in {32, 64}."
        );
    }
    auto sorted = make_half_temp(
        static_cast<std::size_t>(shape.out_capacity) * shape.out_channels
    );
    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);

    if (has_neural_acceleration(stream) &&
        shape.group_size == shape.in_channels) {
        auto dequantized_weights = make_half_temp(
            static_cast<std::size_t>(shape.n_kernels) * shape.in_channels *
            shape.out_channels
        );
        encoder.add_temporary(dequantized_weights);
        auto dequantize = device.get_kernel(
            shape.bits == 4 ? "sparse_quantized_dequantize_f16_b4"
                            : "sparse_quantized_dequantize_f16_b8",
            library
        );
        encoder.set_compute_pipeline_state(dequantize);
        encoder.set_input_array(inputs[1], 0);
        encoder.set_input_array(inputs[2], 1);
        encoder.set_input_array(inputs[3], 2);
        encoder.set_output_array(dequantized_weights, 3);
        set_bytes_range(
            encoder,
            4,
            shape.n_kernels,
            shape.in_channels,
            shape.out_channels,
            shape.group_size
        );
        dispatch_1d(
            encoder,
            dequantize,
            static_cast<size_t>(shape.n_kernels) * shape.in_channels *
                shape.out_channels
        );

        auto tensor_contract =
            device.get_kernel(tensor_kernel_name(shape), library);
        encoder.set_compute_pipeline_state(tensor_contract);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_input_array(dequantized_weights, 1);
        encoder.set_input_array(inputs[9], 2);
        encoder.set_input_array(inputs[11], 3);
        encoder.set_input_array(inputs[10], 4);
        encoder.set_output_array(out, 5);
        set_bytes_range(encoder, 6, shape.out_capacity);
        encoder.dispatch_threadgroups(
            MTL::Size(
                static_cast<size_t>(shape.out_channels / kTileChannels),
                static_cast<size_t>(
                    (shape.out_capacity + kTileRows - 1) / kTileRows
                ),
                1
            ),
            MTL::Size(128, 1, 1)
        );
        return;
    }

    encoder.add_temporary(sorted);

    auto contract = device.get_kernel(kernel_name(shape), library);
    encoder.set_compute_pipeline_state(contract);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[1], 1);
    encoder.set_input_array(inputs[2], 2);
    encoder.set_input_array(inputs[3], 3);
    encoder.set_input_array(inputs[9], 4);
    encoder.set_input_array(inputs[11], 5);
    encoder.set_output_array(sorted, 6);
    set_bytes_range(encoder, 7, shape.out_capacity, shape.group_size);
    encoder.dispatch_threadgroups(
        MTL::Size(
            static_cast<size_t>(shape.out_channels / kTileChannels),
            static_cast<size_t>(
                (shape.out_capacity + kTileRows - 1) / kTileRows
            ),
            1
        ),
        MTL::Size(128, 1, 1)
    );

    auto reorder =
        device.get_kernel("sparse_quantized_igemm_reorder_f16", library);
    encoder.set_compute_pipeline_state(reorder);
    encoder.set_input_array(sorted, 0);
    encoder.set_input_array(inputs[10], 1);
    encoder.set_output_array(out, 2);
    set_bytes_range(encoder, 3, shape.out_capacity, shape.out_channels);
    dispatch_1d(
        encoder,
        reorder,
        static_cast<size_t>(shape.out_capacity) * shape.out_channels
    );
#else
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)out;
    throw std::runtime_error("Metal support is not available.");
#endif
}

} // namespace mlx_lattice::backend::metal::tensor_ops::conv::quantized_igemm
