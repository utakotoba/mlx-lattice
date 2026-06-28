#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
#include <metal_tensor>

using namespace metal;
using namespace mpp::tensor_ops;

template <int bits, int in_channels, int out_channels>
inline void sparse_quantized_tensor_impl(
    device const half* feats,
    device half* dequantized_weights,
    device const int* sorted_kv_out_in_map,
    device const int* tile_masks,
    device const int* reorder_rows,
    device half* out,
    threadgroup half* lhs,
    constant const int& rows,
    uint2 group_id,
    uint tid
) {
    constexpr int tile_rows = 64;
    constexpr int tile_channels = 32;
    constexpr int thread_count = 128;
    (void)bits;

    int co_base = int(group_id.x) * tile_channels;
    int row_start = int(group_id.y) * tile_rows;
    int mask_base = int(group_id.y) * 4;
    uint active_mask = uint(
        tile_masks[mask_base] | tile_masks[mask_base + 1] |
        tile_masks[mask_base + 2] | tile_masks[mask_base + 3]
    );
    constexpr auto descriptor = matmul2d_descriptor(
        tile_rows,
        tile_channels,
        in_channels,
        false,
        false,
        true,
        matmul2d_descriptor::mode::multiply_accumulate
    );
    matmul2d<descriptor, execution_simdgroups<4>> op;
    auto lhs_tensor = tensor<
        threadgroup half,
        extents<int32_t, in_channels, tile_rows>,
        tensor_inline>(lhs, extents<int32_t, in_channels, tile_rows>());
    auto rhs_type = tensor(
        dequantized_weights,
        extents<int32_t, tile_channels, in_channels>(),
        array<int32_t, 2>{1, out_channels}
    );
    auto final_tensor = op.template get_destination_cooperative_tensor<
        decltype(lhs_tensor),
        decltype(rhs_type),
        float>();
#pragma unroll
    for (uint16_t index = 0; index < final_tensor.get_capacity(); ++index) {
        if (final_tensor.is_valid_element(index)) {
            final_tensor[index] = 0.0f;
        }
    }

    while (active_mask != 0) {
        int kernel_id = ctz(active_mask);
        active_mask &= active_mask - 1;

        for (int index = int(tid); index < tile_rows * in_channels;
             index += thread_count) {
            int row_slot = index / in_channels;
            int ci = index - row_slot * in_channels;
            int sorted_row = row_start + row_slot;
            int in_row =
                sorted_row < rows
                    ? sorted_kv_out_in_map[kernel_id * rows + sorted_row]
                    : -1;
            lhs[index] =
                in_row >= 0 ? feats[in_row * in_channels + ci] : half(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto rhs_tensor = tensor(
            dequantized_weights + kernel_id * in_channels * out_channels +
                co_base,
            extents<int32_t, tile_channels, in_channels>(),
            array<int32_t, 2>{1, out_channels}
        );
        op.run(lhs_tensor, rhs_tensor, final_tensor);
        threadgroup_barrier(mem_flags::mem_none);
    }

#pragma unroll
    for (uint16_t index = 0; index < final_tensor.get_capacity(); ++index) {
        if (!final_tensor.is_valid_element(index)) {
            continue;
        }
        auto coord = final_tensor.get_multidimensional_index(index);
        int co = co_base + int(coord[0]);
        int sorted_row = row_start + int(coord[1]);
        if (co < out_channels && sorted_row < rows) {
            int out_row = reorder_rows[sorted_row];
            out[out_row * out_channels + co] = half(final_tensor[index]);
        }
    }
}

#define instantiate_quantized_tensor(name, bits_value, in_value, out_value)    \
    [[kernel, max_total_threads_per_threadgroup(128)]] void name(              \
        device const half* feats [[buffer(0)]],                                \
        device half* dequantized_weights [[buffer(1)]],                        \
        device const int* sorted_kv_out_in_map [[buffer(2)]],                  \
        device const int* tile_masks [[buffer(3)]],                            \
        device const int* reorder_rows [[buffer(4)]],                          \
        device half* out [[buffer(5)]],                                        \
        constant const int& rows [[buffer(6)]],                                \
        uint2 group_id [[threadgroup_position_in_grid]],                       \
        uint tid [[thread_index_in_threadgroup]]                               \
    ) {                                                                        \
        threadgroup half lhs[64 * in_value];                                   \
        sparse_quantized_tensor_impl<bits_value, in_value, out_value>(         \
            feats,                                                             \
            dequantized_weights,                                               \
            sorted_kv_out_in_map,                                              \
            tile_masks,                                                        \
            reorder_rows,                                                      \
            out,                                                               \
            lhs,                                                               \
            rows,                                                              \
            group_id,                                                          \
            tid                                                                \
        );                                                                     \
    }

instantiate_quantized_tensor(sparse_quantized_tensor_f16_b4_cin32_cout32, 4, 32, 32) instantiate_quantized_tensor(
    sparse_quantized_tensor_f16_b8_cin32_cout32,
    8,
    32,
    32
) instantiate_quantized_tensor(sparse_quantized_tensor_f16_b4_cin32_cout64, 4, 32, 64)

    template <int bits>
    inline void sparse_quantized_dequantize_impl(
        device const uint* weights,
        device const half* scales,
        device const half* biases,
        device half* out,
        constant const int& kernel_count,
        constant const int& in_channels,
        constant const int& out_channels,
        constant const int& group_size,
        uint elem
    ) {
    int total = kernel_count * in_channels * out_channels;
    if (elem >= uint(total)) {
        return;
    }
    constexpr int values_per_word = 32 / bits;
    constexpr uint quant_mask = (1u << bits) - 1u;
    int kernel_stride = in_channels * out_channels;
    int kernel_id = int(elem) / kernel_stride;
    int within_kernel = int(elem) - kernel_id * kernel_stride;
    int ci = within_kernel / out_channels;
    int co = within_kernel - ci * out_channels;
    int packed_words = in_channels * bits / 32;
    int word = ci / values_per_word;
    int shift = (ci - word * values_per_word) * bits;
    uint packed =
        weights[(kernel_id * packed_words + word) * out_channels + co];
    uint quantized = (packed >> shift) & quant_mask;
    int groups = in_channels / group_size;
    int group = ci / group_size;
    int quant_index = (kernel_id * groups + group) * out_channels + co;
    out[elem] = half(
        float(quantized) * float(scales[quant_index]) +
        float(biases[quant_index])
    );
}

#define instantiate_dequantize(name, bits_value)                               \
    [[kernel]] void name(                                                      \
        device const uint* weights [[buffer(0)]],                              \
        device const half* scales [[buffer(1)]],                               \
        device const half* biases [[buffer(2)]],                               \
        device half* out [[buffer(3)]],                                        \
        constant const int& kernel_count [[buffer(4)]],                        \
        constant const int& in_channels [[buffer(5)]],                         \
        constant const int& out_channels [[buffer(6)]],                        \
        constant const int& group_size [[buffer(7)]],                          \
        uint elem [[thread_position_in_grid]]                                  \
    ) {                                                                        \
        sparse_quantized_dequantize_impl<bits_value>(                          \
            weights,                                                           \
            scales,                                                            \
            biases,                                                            \
            out,                                                               \
            kernel_count,                                                      \
            in_channels,                                                       \
            out_channels,                                                      \
            group_size,                                                        \
            elem                                                               \
        );                                                                     \
    }

instantiate_dequantize(sparse_quantized_dequantize_f16_b4, 4) instantiate_dequantize(sparse_quantized_dequantize_f16_b8, 8) instantiate_quantized_tensor(sparse_quantized_tensor_f16_b8_cin32_cout64, 8, 32, 64) instantiate_quantized_tensor(
    sparse_quantized_tensor_f16_b4_cin64_cout32,
    4,
    64,
    32
) instantiate_quantized_tensor(sparse_quantized_tensor_f16_b8_cin64_cout32, 8, 64, 32) instantiate_quantized_tensor(sparse_quantized_tensor_f16_b4_cin64_cout64, 4, 64, 64) instantiate_quantized_tensor(sparse_quantized_tensor_f16_b8_cin64_cout64, 8, 64, 64)
