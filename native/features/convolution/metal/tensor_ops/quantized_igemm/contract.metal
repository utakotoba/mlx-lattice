#include <metal_stdlib>

#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/complex.h"
#include "mlx/backend/metal/kernels/steel/gemm/mma.h"

using namespace metal;
using namespace mlx::steel;

template <int bits, int in_channels, int out_channels>
inline void sparse_quantized_igemm_impl(
    device const half* feats,
    device const uint* weights,
    device const half* scales,
    device const half* biases,
    device const int* sorted_kv_out_in_map,
    device const int* tile_masks,
    device half* sorted_out,
    threadgroup half* lhs,
    threadgroup half* rhs,
    constant const int& rows,
    constant const int& group_size,
    uint2 group_id,
    uint tid,
    uint simd_group_id,
    uint simd_lane_id
) {
    constexpr int tile_rows = 64;
    constexpr int tile_channels = 32;
    constexpr int thread_count = 128;
    constexpr int lhs_stride = in_channels + 8;
    constexpr int rhs_stride = tile_channels + 8;
    constexpr int values_per_word = 32 / bits;
    constexpr uint quant_mask = (1u << bits) - 1u;

    using mma_type = BlockMMA<
        half,
        half,
        tile_rows,
        tile_channels,
        in_channels,
        2,
        2,
        false,
        false,
        lhs_stride,
        rhs_stride>;
    mma_type mma(simd_group_id, simd_lane_id);

    int co_base = int(group_id.x) * tile_channels;
    int row_start = int(group_id.y) * tile_rows;
    int mask_base = int(group_id.y) * 4;
    uint active_mask = uint(
        tile_masks[mask_base] | tile_masks[mask_base + 1] |
        tile_masks[mask_base + 2] | tile_masks[mask_base + 3]
    );
    int packed_words = in_channels * bits / 32;
    int groups = in_channels / group_size;

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
            lhs[row_slot * lhs_stride + ci] =
                in_row >= 0 ? feats[in_row * in_channels + ci] : half(0.0h);
        }
        for (int index = int(tid); index < in_channels * tile_channels;
             index += thread_count) {
            int ci = index / tile_channels;
            int co_local = index - ci * tile_channels;
            int co = co_base + co_local;
            half value = half(0.0h);
            if (co < out_channels) {
                int word = ci / values_per_word;
                int shift = (ci - word * values_per_word) * bits;
                uint packed = weights
                    [(kernel_id * packed_words + word) * out_channels + co];
                uint quantized = (packed >> shift) & quant_mask;
                int group = ci / group_size;
                int quant_index =
                    (kernel_id * groups + group) * out_channels + co;
                value = half(
                    float(quantized) * float(scales[quant_index]) +
                    float(biases[quant_index])
                );
            }
            rhs[ci * rhs_stride + co_local] = value;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma.mma(lhs, rhs);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    int valid_rows = min(tile_rows, rows - row_start);
    int valid_channels = min(tile_channels, out_channels - co_base);
    mma.store_result_safe(
        sorted_out + row_start * out_channels + co_base,
        out_channels,
        short2(valid_channels, valid_rows)
    );
}

#define instantiate_quantized_igemm(name, bits_value, in_value, out_value)     \
    [[kernel, max_total_threads_per_threadgroup(128)]] void name(              \
        device const half* feats [[buffer(0)]],                                \
        device const uint* weights [[buffer(1)]],                              \
        device const half* scales [[buffer(2)]],                               \
        device const half* biases [[buffer(3)]],                               \
        device const int* sorted_kv_out_in_map [[buffer(4)]],                  \
        device const int* tile_masks [[buffer(5)]],                            \
        device half* sorted_out [[buffer(6)]],                                 \
        constant const int& rows [[buffer(7)]],                                \
        constant const int& group_size [[buffer(8)]],                          \
        uint2 group_id [[threadgroup_position_in_grid]],                       \
        uint tid [[thread_index_in_threadgroup]],                              \
        uint simd_group_id [[simdgroup_index_in_threadgroup]],                 \
        uint simd_lane_id [[thread_index_in_simdgroup]]                        \
    ) {                                                                        \
        threadgroup half lhs[64 * (in_value + 8)];                             \
        threadgroup half rhs[in_value * (32 + 8)];                             \
        sparse_quantized_igemm_impl<bits_value, in_value, out_value>(          \
            feats,                                                             \
            weights,                                                           \
            scales,                                                            \
            biases,                                                            \
            sorted_kv_out_in_map,                                              \
            tile_masks,                                                        \
            sorted_out,                                                        \
            lhs,                                                               \
            rhs,                                                               \
            rows,                                                              \
            group_size,                                                        \
            group_id,                                                          \
            tid,                                                               \
            simd_group_id,                                                     \
            simd_lane_id                                                       \
        );                                                                     \
    }

instantiate_quantized_igemm(sparse_quantized_igemm_f16_b4_cin32_cout32, 4, 32, 32) instantiate_quantized_igemm(sparse_quantized_igemm_f16_b8_cin32_cout32, 8, 32, 32) instantiate_quantized_igemm(sparse_quantized_igemm_f16_b4_cin32_cout64, 4, 32, 64) instantiate_quantized_igemm(sparse_quantized_igemm_f16_b8_cin32_cout64, 8, 32, 64) instantiate_quantized_igemm(
    sparse_quantized_igemm_f16_b4_cin64_cout32,
    4,
    64,
    32
) instantiate_quantized_igemm(sparse_quantized_igemm_f16_b8_cin64_cout32, 8, 64, 32) instantiate_quantized_igemm(sparse_quantized_igemm_f16_b4_cin64_cout64, 4, 64, 64) instantiate_quantized_igemm(sparse_quantized_igemm_f16_b8_cin64_cout64, 8, 64, 64)

    [[kernel]] void sparse_quantized_igemm_reorder_f16(
        device const half* sorted [[buffer(0)]],
        device const int* reorder_rows [[buffer(1)]],
        device half* out [[buffer(2)]],
        constant const int& rows [[buffer(3)]],
        constant const int& channels [[buffer(4)]],
        uint elem [[thread_position_in_grid]]
    ) {
    int total = rows * channels;
    if (elem >= uint(total)) {
        return;
    }
    int sorted_row = int(elem) / channels;
    int channel = int(elem) - sorted_row * channels;
    int out_row = reorder_rows[sorted_row];
    out[out_row * channels + channel] = sorted[elem];
}
