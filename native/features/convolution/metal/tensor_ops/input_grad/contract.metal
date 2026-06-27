#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
#include <metal_tensor>

using namespace metal;
using namespace mpp::tensor_ops;

[[kernel, max_total_threads_per_threadgroup(32)]] void
sparse_relation_conv_input_grad_tensor_ops_f32_i32(
    device const float* cotangent [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* out_rows [[buffer(2)]],
    device const int* kernel_ids [[buffer(3)]],
    device const int* counts [[buffer(4)]],
    device const int* in_row_offsets [[buffer(5)]],
    device const int* in_edge_ids [[buffer(6)]],
    device float* grad [[buffer(7)]],
    constant const int& edge_capacity [[buffer(8)]],
    constant const int& out_capacity [[buffer(9)]],
    constant const int& in_capacity [[buffer(10)]],
    constant const int& n_kernels [[buffer(11)]],
    constant const int& cot_s0 [[buffer(12)]],
    constant const int& cot_s1 [[buffer(13)]],
    constant const int& weight_s0 [[buffer(14)]],
    constant const int& weight_s1 [[buffer(15)]],
    constant const int& weight_s2 [[buffer(16)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    constexpr int tile_rows = 16;
    threadgroup float lhs_tile[tile_rows * 16];
    threadgroup float rhs_tile[16 * 16];
    threadgroup float out_tile[tile_rows * 16];

    const int row_start = int(group_id) * tile_rows;
    const int edge_count = min(counts[0], edge_capacity);

    for (uint index = lane; index < tile_rows * 16; index += 32) {
        out_tile[index] = 0.0f;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    constexpr auto desc = matmul2d_descriptor(
        tile_rows,
        16,
        16,
        false,
        false,
        false,
        matmul2d_descriptor::mode::multiply_accumulate
    );
    matmul2d<desc, execution_simdgroup> op;
    auto lhs_tensor = tensor<
        threadgroup float,
        extents<int32_t, 16, tile_rows>,
        tensor_inline>(lhs_tile, extents<int32_t, 16, tile_rows>());
    auto rhs_tensor =
        tensor<threadgroup float, extents<int32_t, 16, 16>, tensor_inline>(
            rhs_tile, extents<int32_t, 16, 16>()
        );
    auto out_tensor = tensor<
        threadgroup float,
        extents<int32_t, 16, tile_rows>,
        tensor_inline>(out_tile, extents<int32_t, 16, tile_rows>());

    for (int kernel_id = 0; kernel_id < n_kernels; ++kernel_id) {
        for (uint index = lane; index < tile_rows * 16; index += 32) {
            const int row_slot = int(index) / 16;
            const int co = int(index) - row_slot * 16;
            const int in_row = row_start + row_slot;
            float value = 0.0f;
            if (in_row < in_capacity) {
                const int begin = in_row_offsets[in_row];
                const int end = in_row_offsets[in_row + 1];
                for (int cursor = begin; cursor < end; ++cursor) {
                    const int edge = in_edge_ids[cursor];
                    if (edge < 0 || edge >= edge_count ||
                        kernel_ids[edge] != kernel_id) {
                        continue;
                    }
                    const int out_row = out_rows[edge];
                    if (out_row >= 0 && out_row < out_capacity) {
                        value = cotangent[out_row * cot_s0 + co * cot_s1];
                    }
                    break;
                }
            }
            lhs_tile[index] = value;
        }

        for (uint index = lane; index < 16 * 16; index += 32) {
            const int co = int(index) / 16;
            const int ci = int(index) - co * 16;
            rhs_tile[index] = weights
                [kernel_id * weight_s0 + ci * weight_s1 + co * weight_s2];
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);
        op.run(lhs_tensor, rhs_tensor, out_tensor);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint index = lane; index < tile_rows * 16; index += 32) {
        const int row_slot = int(index) / 16;
        const int ci = int(index) - row_slot * 16;
        const int in_row = row_start + row_slot;
        if (in_row < in_capacity) {
            grad[in_row * 16 + ci] = out_tile[row_slot * 16 + ci];
        }
    }
}
