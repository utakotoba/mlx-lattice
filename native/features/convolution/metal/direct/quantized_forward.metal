#include <metal_stdlib>

using namespace metal;

template <typename T, int bits>
inline void sparse_quantized_conv_impl(
    device const T* feats,
    device const uint* weights,
    device const T* scales,
    device const T* biases,
    device const int* in_rows,
    device const int* kernel_ids,
    device const int* counts,
    device const int* row_offsets,
    device T* out,
    constant const int& edge_capacity,
    constant const int& out_capacity,
    constant const int& in_channels,
    constant const int& out_channels,
    constant const int& storage_in_channels,
    constant const int& group_size,
    constant const int& feat_s0,
    constant const int& feat_s1,
    uint elem
) {
    constexpr int vector_width = 4;
    constexpr int output_tile = 8;
    int channel_blocks = (out_channels + output_tile - 1) / output_tile;
    int total = out_capacity * channel_blocks;
    if (elem >= uint(total)) {
        return;
    }

    int out_row = int(elem) / channel_blocks;
    int co_base = (int(elem) - out_row * channel_blocks) * output_tile;
    int out_base = out_row * out_channels + co_base;
    int lanes = min(output_tile, out_channels - co_base);
    int first_lanes = min(vector_width, lanes);
    int second_lanes = max(0, lanes - vector_width);
    int out_count = min(counts[1], out_capacity);
    if (out_row >= out_count) {
        for (int lane = 0; lane < lanes; ++lane) {
            out[out_base + lane] = T(0);
        }
        return;
    }

    int packed_words = storage_in_channels * bits / 32;
    int groups = storage_in_channels / group_size;
    int edge_count = min(counts[0], edge_capacity);
    float4 first_acc = float4(0.0f);
    float4 second_acc = float4(0.0f);
    for (int edge = row_offsets[out_row]; edge < row_offsets[out_row + 1];
         ++edge) {
        if (edge < 0 || edge >= edge_count) {
            continue;
        }
        int in_row = in_rows[edge];
        int kernel_id = kernel_ids[edge];
        if (in_row < 0 || kernel_id < 0) {
            continue;
        }
        for (int group = 0; group < groups; ++group) {
            float4 first_scale = float4(0.0f);
            float4 second_scale = float4(0.0f);
            float4 first_bias = float4(0.0f);
            float4 second_bias = float4(0.0f);
            int quant_base =
                (kernel_id * groups + group) * out_channels + co_base;
            for (int lane = 0; lane < lanes; ++lane) {
                if (lane < vector_width) {
                    first_scale[lane] = float(scales[quant_base + lane]);
                    first_bias[lane] = float(biases[quant_base + lane]);
                } else {
                    int vector_lane = lane - vector_width;
                    second_scale[vector_lane] =
                        float(scales[quant_base + lane]);
                    second_bias[vector_lane] = float(biases[quant_base + lane]);
                }
            }
            int first_ci = group * group_size;
            int last_ci = min(first_ci + group_size, in_channels);
            constexpr int values_per_word = 32 / bits;
            float feature_sum = 0.0f;
            float4 first_quantized_acc = float4(0.0f);
            float4 second_quantized_acc = float4(0.0f);
            for (int first = first_ci; first < last_ci;
                 first += values_per_word) {
                int bit = first * bits;
                int packed_base =
                    (kernel_id * packed_words + (bit >> 5)) * out_channels +
                    co_base;
                uint4 first_packed = uint4(0u);
                uint4 second_packed = uint4(0u);
                if (first_lanes == vector_width) {
                    first_packed = *reinterpret_cast<device const uint4*>(
                        weights + packed_base
                    );
                } else {
                    for (int lane = 0; lane < first_lanes; ++lane) {
                        first_packed[lane] = weights[packed_base + lane];
                    }
                }
                if (second_lanes == vector_width) {
                    second_packed = *reinterpret_cast<device const uint4*>(
                        weights + packed_base + vector_width
                    );
                } else {
                    for (int lane = 0; lane < second_lanes; ++lane) {
                        second_packed[lane] =
                            weights[packed_base + vector_width + lane];
                    }
                }
                constexpr uint mask = (1u << bits) - 1u;
                int packed_values = min(values_per_word, last_ci - first);
                for (int value = 0; value < packed_values; ++value) {
                    int ci = first + value;
                    float feature =
                        float(feats[in_row * feat_s0 + ci * feat_s1]);
                    feature_sum += feature;
                    first_quantized_acc +=
                        feature *
                        float4((first_packed >> (value * bits)) & mask);
                    second_quantized_acc +=
                        feature *
                        float4((second_packed >> (value * bits)) & mask);
                }
            }
            first_acc +=
                first_quantized_acc * first_scale + feature_sum * first_bias;
            second_acc +=
                second_quantized_acc * second_scale + feature_sum * second_bias;
        }
    }
    for (int lane = 0; lane < first_lanes; ++lane) {
        out[out_base + lane] = T(first_acc[lane]);
    }
    for (int lane = 0; lane < second_lanes; ++lane) {
        out[out_base + vector_width + lane] = T(second_acc[lane]);
    }
}

#define instantiate_quantized_conv(name, type, bits_value)                     \
    [[kernel]] void name(                                                      \
        device const type* feats [[buffer(0)]],                                \
        device const uint* weights [[buffer(1)]],                              \
        device const type* scales [[buffer(2)]],                               \
        device const type* biases [[buffer(3)]],                               \
        device const int* in_rows [[buffer(4)]],                               \
        device const int* out_rows [[buffer(5)]],                              \
        device const int* kernel_ids [[buffer(6)]],                            \
        device const int* counts [[buffer(7)]],                                \
        device const int* row_offsets [[buffer(8)]],                           \
        device type* out [[buffer(9)]],                                        \
        constant const int& edge_capacity [[buffer(10)]],                      \
        constant const int& out_capacity [[buffer(11)]],                       \
        constant const int& in_channels [[buffer(12)]],                        \
        constant const int& out_channels [[buffer(13)]],                       \
        constant const int& storage_in_channels [[buffer(14)]],                \
        constant const int& group_size [[buffer(15)]],                         \
        constant const int& feat_s0 [[buffer(16)]],                            \
        constant const int& feat_s1 [[buffer(17)]],                            \
        uint elem [[thread_position_in_grid]]                                  \
    ) {                                                                        \
        sparse_quantized_conv_impl<type, bits_value>(                          \
            feats,                                                             \
            weights,                                                           \
            scales,                                                            \
            biases,                                                            \
            in_rows,                                                           \
            kernel_ids,                                                        \
            counts,                                                            \
            row_offsets,                                                       \
            out,                                                               \
            edge_capacity,                                                     \
            out_capacity,                                                      \
            in_channels,                                                       \
            out_channels,                                                      \
            storage_in_channels,                                               \
            group_size,                                                        \
            feat_s0,                                                           \
            feat_s1,                                                           \
            elem                                                               \
        );                                                                     \
        (void)out_rows;                                                        \
    }

instantiate_quantized_conv(sparse_quantized_conv_f16_i32_b4, half, 4) instantiate_quantized_conv(
    sparse_quantized_conv_f16_i32_b8,
    half,
    8
) instantiate_quantized_conv(sparse_quantized_conv_f32_i32_b4, float, 4) instantiate_quantized_conv(sparse_quantized_conv_f32_i32_b8, float, 8)
