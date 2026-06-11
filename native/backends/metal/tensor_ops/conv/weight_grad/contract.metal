#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
#include <metal_tensor>

using namespace metal;
using namespace mpp::tensor_ops;

#include "native/backends/metal/conv/common.metal"

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
    uint group_id [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    threadgroup float lhs_tile[16 * 16];
    threadgroup float rhs_tile[16 * 32];
    threadgroup float out_tile[16 * 32];

    const int kernel_id = int(group_id) % n_kernels;
    const int partition = int(group_id) / n_kernels;
    const int edge_count = min(counts[0], edge_capacity);
    const int kernel_start = kernel_row_offsets[kernel_id];
    const int kernel_stop = kernel_row_offsets[kernel_id + 1];
    const int kernel_edges = max(kernel_stop - kernel_start, 0);
    const int partition_edges = (kernel_edges + partitions - 1) / partitions;
    const int start =
        min(kernel_start + partition * partition_edges, kernel_stop);
    const int stop = min(start + partition_edges, kernel_stop);

    for (uint index = lane; index < 16 * 32; index += 32) {
        out_tile[index] = 0.0f;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    constexpr auto desc = matmul2d_descriptor(
        16,
        32,
        16,
        false,
        false,
        false,
        matmul2d_descriptor::mode::multiply_accumulate
    );
    matmul2d<desc, execution_simdgroup> op;
    auto lhs_tensor =
        tensor<threadgroup float, extents<int32_t, 16, 16>, tensor_inline>(
            lhs_tile, extents<int32_t, 16, 16>()
        );
    auto rhs_tensor =
        tensor<threadgroup float, extents<int32_t, 32, 16>, tensor_inline>(
            rhs_tile, extents<int32_t, 32, 16>()
        );
    auto out_tensor =
        tensor<threadgroup float, extents<int32_t, 32, 16>, tensor_inline>(
            out_tile, extents<int32_t, 32, 16>()
        );

    for (int base = start; base < stop; base += 16) {
        for (uint index = lane; index < 16 * 16; index += 32) {
            const int ci = int(index) / 16;
            const int edge_slot = int(index) - ci * 16;
            const int cursor = base + edge_slot;
            float value = 0.0f;
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

        for (uint index = lane; index < 16 * 32; index += 32) {
            const int edge_slot = int(index) / 32;
            const int co = int(index) - edge_slot * 32;
            const int cursor = base + edge_slot;
            float value = 0.0f;
            if (co < 16 && cursor < stop) {
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

    const int partial_base = (partition * n_kernels + kernel_id) * 16 * 16;
    for (uint index = lane; index < 16 * 16; index += 32) {
        const int ci = int(index) / 16;
        const int co = int(index) - ci * 16;
        partials[partial_base + int(index)] = out_tile[ci * 32 + co];
    }
}
