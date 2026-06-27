#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
#include <metal_tensor>

using namespace metal;
using namespace mpp::tensor_ops;

#include "native/features/convolution/metal/common.metal"

template <typename T>
inline void sparse_relation_conv_weight_grad_tensor_ops_impl(
    device const T* feats,
    device const T* cotangent,
    device const int* in_rows,
    device const int* out_rows,
    device const int* counts,
    device const int* kernel_row_offsets,
    device const int* kernel_edge_ids,
    device float* partials,
    constant const int& edge_capacity,
    constant const int& out_capacity,
    constant const int& n_kernels,
    constant const int& partitions,
    constant const int& feat_s0,
    constant const int& feat_s1,
    constant const int& cotangent_s0,
    constant const int& cotangent_s1,
    constant const int& in_channels,
    constant const int& out_channels,
    uint group_id,
    uint lane,
    threadgroup T* lhs_tile,
    threadgroup T* rhs_tile,
    threadgroup float* out_tile
) {
    const int channel_blocks = in_channels / 16;
    const int channel_tiles = channel_blocks * channel_blocks;
    const int kernel_tile = int(group_id) % (n_kernels * channel_tiles);
    const int kernel_id = kernel_tile / channel_tiles;
    const int channel_tile = kernel_tile - kernel_id * channel_tiles;
    const int ci_base = (channel_tile / channel_blocks) * 16;
    const int co_base = (channel_tile - (ci_base / 16) * channel_blocks) * 16;
    const int partition = int(group_id) / (n_kernels * channel_tiles);
    const int edge_count = min(counts[0], edge_capacity);
    const int kernel_start = kernel_row_offsets[kernel_id];
    const int kernel_stop = kernel_row_offsets[kernel_id + 1];
    const int kernel_edges = max(kernel_stop - kernel_start, 0);
    const int partition_edges = (kernel_edges + partitions - 1) / partitions;
    const int start =
        min(kernel_start + partition * partition_edges, kernel_stop);
    const int stop = min(start + partition_edges, kernel_stop);

    for (uint index = lane; index < 16 * 16; index += 32) {
        out_tile[index] = 0.0f;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    constexpr auto desc = matmul2d_descriptor(
        16,
        16,
        16,
        false,
        false,
        false,
        matmul2d_descriptor::mode::multiply_accumulate
    );
    matmul2d<desc, execution_simdgroup> op;
    auto lhs_tensor =
        tensor<threadgroup T, extents<int32_t, 16, 16>, tensor_inline>(
            lhs_tile, extents<int32_t, 16, 16>()
        );
    auto rhs_tensor =
        tensor<threadgroup T, extents<int32_t, 16, 16>, tensor_inline>(
            rhs_tile, extents<int32_t, 16, 16>()
        );
    auto out_tensor =
        tensor<threadgroup float, extents<int32_t, 16, 16>, tensor_inline>(
            out_tile, extents<int32_t, 16, 16>()
        );

    for (int base = start; base < stop; base += 16) {
        for (uint index = lane; index < 16 * 16; index += 32) {
            const int ci_offset = int(index) / 16;
            const int ci = ci_base + ci_offset;
            const int edge_slot = int(index) - ci_offset * 16;
            const int cursor = base + edge_slot;
            T value = T(0);
            if (cursor < stop) {
                const int edge = kernel_edge_ids[cursor];
                if (edge >= 0 && edge < edge_count) {
                    const int in_row = in_rows[edge];
                    if (in_row >= 0) {
                        value = feats[in_row * feat_s0 + ci * feat_s1];
                    }
                }
            }
            lhs_tile[index] = value;
        }

        for (uint index = lane; index < 16 * 16; index += 32) {
            const int edge_slot = int(index) / 16;
            const int co = co_base + int(index) - edge_slot * 16;
            const int cursor = base + edge_slot;
            T value = T(0);
            if (co < out_channels && cursor < stop) {
                const int edge = kernel_edge_ids[cursor];
                if (edge >= 0 && edge < edge_count) {
                    const int out_row = out_rows[edge];
                    if (out_row >= 0 && out_row < out_capacity) {
                        value = cotangent
                            [out_row * cotangent_s0 + co * cotangent_s1];
                    }
                }
            }
            rhs_tile[index] = value;
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);
        op.run(lhs_tensor, rhs_tensor, out_tensor);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int partial_base =
        ((partition * n_kernels + kernel_id) * channel_tiles + channel_tile) *
        16 * 16;
    for (uint index = lane; index < 16 * 16; index += 32) {
        const int ci = int(index) / 16;
        const int co = int(index) - ci * 16;
        partials[partial_base + int(index)] = out_tile[ci * 16 + co];
    }
}

[[kernel, max_total_threads_per_threadgroup(32)]] void
sparse_relation_conv_weight_grad_tensor_ops_f32_i32(
    device const float* feats [[buffer(0)]],
    device const float* cotangent [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* counts [[buffer(4)]],
    device const int* kernel_row_offsets [[buffer(5)]],
    device const int* kernel_edge_ids [[buffer(6)]],
    device float* partials [[buffer(7)]],
    constant const int& edge_capacity [[buffer(8)]],
    constant const int& out_capacity [[buffer(9)]],
    constant const int& n_kernels [[buffer(10)]],
    constant const int& partitions [[buffer(11)]],
    constant const int& feat_s0 [[buffer(12)]],
    constant const int& feat_s1 [[buffer(13)]],
    constant const int& cotangent_s0 [[buffer(14)]],
    constant const int& cotangent_s1 [[buffer(15)]],
    constant const int& in_channels [[buffer(16)]],
    constant const int& out_channels [[buffer(17)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    threadgroup float lhs_tile[16 * 16];
    threadgroup float rhs_tile[16 * 16];
    threadgroup float out_tile[16 * 16];
    sparse_relation_conv_weight_grad_tensor_ops_impl<float>(
        feats,
        cotangent,
        in_rows,
        out_rows,
        counts,
        kernel_row_offsets,
        kernel_edge_ids,
        partials,
        edge_capacity,
        out_capacity,
        n_kernels,
        partitions,
        feat_s0,
        feat_s1,
        cotangent_s0,
        cotangent_s1,
        in_channels,
        out_channels,
        group_id,
        lane,
        lhs_tile,
        rhs_tile,
        out_tile
    );
}

[[kernel, max_total_threads_per_threadgroup(32)]] void
sparse_relation_conv_weight_grad_tensor_ops_f16_i32(
    device const half* feats [[buffer(0)]],
    device const half* cotangent [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* counts [[buffer(4)]],
    device const int* kernel_row_offsets [[buffer(5)]],
    device const int* kernel_edge_ids [[buffer(6)]],
    device float* partials [[buffer(7)]],
    constant const int& edge_capacity [[buffer(8)]],
    constant const int& out_capacity [[buffer(9)]],
    constant const int& n_kernels [[buffer(10)]],
    constant const int& partitions [[buffer(11)]],
    constant const int& feat_s0 [[buffer(12)]],
    constant const int& feat_s1 [[buffer(13)]],
    constant const int& cotangent_s0 [[buffer(14)]],
    constant const int& cotangent_s1 [[buffer(15)]],
    constant const int& in_channels [[buffer(16)]],
    constant const int& out_channels [[buffer(17)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    threadgroup half lhs_tile[16 * 16];
    threadgroup half rhs_tile[16 * 16];
    threadgroup float out_tile[16 * 16];
    sparse_relation_conv_weight_grad_tensor_ops_impl<half>(
        feats,
        cotangent,
        in_rows,
        out_rows,
        counts,
        kernel_row_offsets,
        kernel_edge_ids,
        partials,
        edge_capacity,
        out_capacity,
        n_kernels,
        partitions,
        feat_s0,
        feat_s1,
        cotangent_s0,
        cotangent_s1,
        in_channels,
        out_channels,
        group_id,
        lane,
        lhs_tile,
        rhs_tile,
        out_tile
    );
}
