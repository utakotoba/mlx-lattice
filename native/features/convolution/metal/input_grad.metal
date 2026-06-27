#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
#include <metal_tensor>

using namespace metal;
using namespace mpp::tensor_ops;

#include "native/features/convolution/metal/common.metal"

// Specialized kernels share the generic convolution binding ABI, so some
// bound buffers are intentionally unused by a given specialization.
#pragma clang diagnostic ignored "-Wunused-parameter"

[[kernel]] void sparse_relation_conv_input_grad_f32_i32(
    device const float* cotangent [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device const int* in_row_offsets [[buffer(7)]],
    device const int* in_edge_ids [[buffer(8)]],
    device float* grad [[buffer(9)]],
    constant const int& edge_capacity [[buffer(10)]],
    constant const int& out_capacity [[buffer(11)]],
    constant const int& in_capacity [[buffer(12)]],
    constant const int& in_channels [[buffer(13)]],
    constant const int& out_channels [[buffer(14)]],
    constant const int& cotangent_s0 [[buffer(15)]],
    constant const int& cotangent_s1 [[buffer(16)]],
    constant const int& weight_s0 [[buffer(17)]],
    constant const int& weight_s1 [[buffer(18)]],
    constant const int& weight_s2 [[buffer(19)]],
    constant const int& weight_s3 [[buffer(20)]],
    constant const int& weight_s4 [[buffer(21)]],
    constant const int& weight_layout [[buffer(22)]],
    constant const int& kernel_x [[buffer(23)]],
    constant const int& kernel_y [[buffer(24)]],
    constant const int& kernel_z [[buffer(25)]],
    uint elem [[thread_position_in_grid]]
) {
    int total = in_capacity * in_channels;
    if (elem >= uint(total)) {
        return;
    }

    int in_row = int(elem) / in_channels;
    int ci = int(elem) - in_row * in_channels;
    int edge_count = min(counts[0], edge_capacity);
    float acc = 0.0f;
    for (int cursor = in_row_offsets[in_row];
         cursor < in_row_offsets[in_row + 1];
         ++cursor) {
        int edge = in_edge_ids[cursor];
        if (edge < 0 || edge >= edge_count) {
            continue;
        }
        int out_row = out_rows[edge];
        int kernel_id = kernel_ids[edge];
        if (out_row < 0 || out_row >= out_capacity || kernel_id < 0) {
            continue;
        }
        for (int co = 0; co < out_channels; ++co) {
            acc += cotangent[out_row * cotangent_s0 + co * cotangent_s1] *
                   weights[sparse_conv_weight_offset(
                       kernel_id,
                       ci,
                       co,
                       weight_layout,
                       kernel_x,
                       kernel_y,
                       kernel_z,
                       weight_s0,
                       weight_s1,
                       weight_s2,
                       weight_s3,
                       weight_s4
                   )];
        }
    }
    grad[in_row * in_channels + ci] = acc;
    (void)in_rows;
    (void)row_offsets;
}

[[kernel]] void sparse_relation_conv_input_grad_f32_i32_vec4(
    device const float* cotangent [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device const int* in_row_offsets [[buffer(7)]],
    device const int* in_edge_ids [[buffer(8)]],
    device float* grad [[buffer(9)]],
    constant const int& edge_capacity [[buffer(10)]],
    constant const int& out_capacity [[buffer(11)]],
    constant const int& in_capacity [[buffer(12)]],
    constant const int& in_channels [[buffer(13)]],
    constant const int& out_channels [[buffer(14)]],
    constant const int& cotangent_s0 [[buffer(15)]],
    constant const int& cotangent_s1 [[buffer(16)]],
    constant const int& weight_s0 [[buffer(17)]],
    constant const int& weight_s1 [[buffer(18)]],
    constant const int& weight_s2 [[buffer(19)]],
    constant const int& weight_s3 [[buffer(20)]],
    constant const int& weight_s4 [[buffer(21)]],
    constant const int& weight_layout [[buffer(22)]],
    constant const int& kernel_x [[buffer(23)]],
    constant const int& kernel_y [[buffer(24)]],
    constant const int& kernel_z [[buffer(25)]],
    uint elem [[thread_position_in_grid]]
) {
    int blocks = in_channels / 4;
    int total = in_capacity * blocks;
    if (elem >= uint(total)) {
        return;
    }

    int in_row = int(elem) / blocks;
    int ci = (int(elem) - in_row * blocks) * 4;
    int edge_count = min(counts[0], edge_capacity);
    float4 acc = float4(0.0f);
    for (int cursor = in_row_offsets[in_row];
         cursor < in_row_offsets[in_row + 1];
         ++cursor) {
        int edge = in_edge_ids[cursor];
        if (edge < 0 || edge >= edge_count) {
            continue;
        }
        int out_row = out_rows[edge];
        int kernel_id = kernel_ids[edge];
        if (out_row < 0 || out_row >= out_capacity || kernel_id < 0) {
            continue;
        }
        for (int co = 0; co < out_channels; ++co) {
            float value = cotangent[out_row * cotangent_s0 + co * cotangent_s1];
            acc += value * float4(
                               weights[sparse_conv_weight_offset(
                                   kernel_id,
                                   ci,
                                   co,
                                   weight_layout,
                                   kernel_x,
                                   kernel_y,
                                   kernel_z,
                                   weight_s0,
                                   weight_s1,
                                   weight_s2,
                                   weight_s3,
                                   weight_s4
                               )],
                               weights[sparse_conv_weight_offset(
                                   kernel_id,
                                   ci + 1,
                                   co,
                                   weight_layout,
                                   kernel_x,
                                   kernel_y,
                                   kernel_z,
                                   weight_s0,
                                   weight_s1,
                                   weight_s2,
                                   weight_s3,
                                   weight_s4
                               )],
                               weights[sparse_conv_weight_offset(
                                   kernel_id,
                                   ci + 2,
                                   co,
                                   weight_layout,
                                   kernel_x,
                                   kernel_y,
                                   kernel_z,
                                   weight_s0,
                                   weight_s1,
                                   weight_s2,
                                   weight_s3,
                                   weight_s4
                               )],
                               weights[sparse_conv_weight_offset(
                                   kernel_id,
                                   ci + 3,
                                   co,
                                   weight_layout,
                                   kernel_x,
                                   kernel_y,
                                   kernel_z,
                                   weight_s0,
                                   weight_s1,
                                   weight_s2,
                                   weight_s3,
                                   weight_s4
                               )]
                           );
        }
    }
    int grad_base = in_row * in_channels + ci;
    grad[grad_base] = acc.x;
    grad[grad_base + 1] = acc.y;
    grad[grad_base + 2] = acc.z;
    grad[grad_base + 3] = acc.w;
    (void)in_rows;
    (void)row_offsets;
}

[[kernel]] void sparse_relation_conv_input_grad_f32_i32_cin16(
    device const float* cotangent [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device const int* in_row_offsets [[buffer(7)]],
    device const int* in_edge_ids [[buffer(8)]],
    device float* grad [[buffer(9)]],
    constant const int& edge_capacity [[buffer(10)]],
    constant const int& out_capacity [[buffer(11)]],
    constant const int& in_capacity [[buffer(12)]],
    constant const int& in_channels [[buffer(13)]],
    constant const int& out_channels [[buffer(14)]],
    constant const int& cotangent_s0 [[buffer(15)]],
    constant const int& cotangent_s1 [[buffer(16)]],
    constant const int& weight_s0 [[buffer(17)]],
    constant const int& weight_s1 [[buffer(18)]],
    constant const int& weight_s2 [[buffer(19)]],
    constant const int& weight_s3 [[buffer(20)]],
    constant const int& weight_s4 [[buffer(21)]],
    constant const int& weight_layout [[buffer(22)]],
    constant const int& kernel_x [[buffer(23)]],
    constant const int& kernel_y [[buffer(24)]],
    constant const int& kernel_z [[buffer(25)]],
    uint in_row_id [[thread_position_in_grid]]
) {
    if (in_row_id >= uint(in_capacity)) {
        return;
    }

    int in_row = int(in_row_id);
    int edge_count = min(counts[0], edge_capacity);
    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);
    float4 acc3 = float4(0.0f);
    for (int cursor = in_row_offsets[in_row];
         cursor < in_row_offsets[in_row + 1];
         ++cursor) {
        int edge = in_edge_ids[cursor];
        if (edge < 0 || edge >= edge_count) {
            continue;
        }
        int out_row = out_rows[edge];
        int kernel_id = kernel_ids[edge];
        if (out_row < 0 || out_row >= out_capacity || kernel_id < 0) {
            continue;
        }
        for (int co = 0; co < out_channels; ++co) {
            float value = cotangent[out_row * cotangent_s0 + co * cotangent_s1];
            acc0 += value * float4(
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    0,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )],
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    1,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )],
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    2,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )],
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    3,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )]
                            );
            acc1 += value * float4(
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    4,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )],
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    5,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )],
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    6,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )],
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    7,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )]
                            );
            acc2 += value * float4(
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    8,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )],
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    9,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )],
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    10,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )],
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    11,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )]
                            );
            acc3 += value * float4(
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    12,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )],
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    13,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )],
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    14,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )],
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    15,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )]
                            );
        }
    }
    int grad_base = in_row * 16;
    grad[grad_base] = acc0.x;
    grad[grad_base + 1] = acc0.y;
    grad[grad_base + 2] = acc0.z;
    grad[grad_base + 3] = acc0.w;
    grad[grad_base + 4] = acc1.x;
    grad[grad_base + 5] = acc1.y;
    grad[grad_base + 6] = acc1.z;
    grad[grad_base + 7] = acc1.w;
    grad[grad_base + 8] = acc2.x;
    grad[grad_base + 9] = acc2.y;
    grad[grad_base + 10] = acc2.z;
    grad[grad_base + 11] = acc2.w;
    grad[grad_base + 12] = acc3.x;
    grad[grad_base + 13] = acc3.y;
    grad[grad_base + 14] = acc3.z;
    grad[grad_base + 15] = acc3.w;
    (void)in_rows;
    (void)row_offsets;
    (void)in_channels;
}

[[kernel]] void sparse_relation_conv_input_grad_f16_i32(
    device const half* cotangent [[buffer(0)]],
    device const half* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device const int* in_row_offsets [[buffer(7)]],
    device const int* in_edge_ids [[buffer(8)]],
    device half* grad [[buffer(9)]],
    constant const int& edge_capacity [[buffer(10)]],
    constant const int& out_capacity [[buffer(11)]],
    constant const int& in_capacity [[buffer(12)]],
    constant const int& in_channels [[buffer(13)]],
    constant const int& out_channels [[buffer(14)]],
    constant const int& cotangent_s0 [[buffer(15)]],
    constant const int& cotangent_s1 [[buffer(16)]],
    constant const int& weight_s0 [[buffer(17)]],
    constant const int& weight_s1 [[buffer(18)]],
    constant const int& weight_s2 [[buffer(19)]],
    constant const int& weight_s3 [[buffer(20)]],
    constant const int& weight_s4 [[buffer(21)]],
    constant const int& weight_layout [[buffer(22)]],
    constant const int& kernel_x [[buffer(23)]],
    constant const int& kernel_y [[buffer(24)]],
    constant const int& kernel_z [[buffer(25)]],
    uint elem [[thread_position_in_grid]]
) {
    int total = in_capacity * in_channels;
    if (elem >= uint(total)) {
        return;
    }

    int in_row = int(elem) / in_channels;
    int ci = int(elem) - in_row * in_channels;
    int edge_count = min(counts[0], edge_capacity);
    float acc = 0.0f;
    for (int cursor = in_row_offsets[in_row];
         cursor < in_row_offsets[in_row + 1];
         ++cursor) {
        int edge = in_edge_ids[cursor];
        if (edge < 0 || edge >= edge_count) {
            continue;
        }
        int out_row = out_rows[edge];
        int kernel_id = kernel_ids[edge];
        if (out_row < 0 || out_row >= out_capacity || kernel_id < 0) {
            continue;
        }
        for (int co = 0; co < out_channels; ++co) {
            acc +=
                float(cotangent[out_row * cotangent_s0 + co * cotangent_s1]) *
                float(weights[sparse_conv_weight_offset(
                    kernel_id,
                    ci,
                    co,
                    weight_layout,
                    kernel_x,
                    kernel_y,
                    kernel_z,
                    weight_s0,
                    weight_s1,
                    weight_s2,
                    weight_s3,
                    weight_s4
                )]);
        }
    }
    grad[in_row * in_channels + ci] = half(acc);
    (void)in_rows;
    (void)row_offsets;
}

[[kernel]] void sparse_relation_conv_input_grad_f16_i32_cin16(
    device const half* cotangent [[buffer(0)]],
    device const half* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device const int* in_row_offsets [[buffer(7)]],
    device const int* in_edge_ids [[buffer(8)]],
    device half* grad [[buffer(9)]],
    constant const int& edge_capacity [[buffer(10)]],
    constant const int& out_capacity [[buffer(11)]],
    constant const int& in_capacity [[buffer(12)]],
    constant const int& in_channels [[buffer(13)]],
    constant const int& out_channels [[buffer(14)]],
    constant const int& cotangent_s0 [[buffer(15)]],
    constant const int& cotangent_s1 [[buffer(16)]],
    constant const int& weight_s0 [[buffer(17)]],
    constant const int& weight_s1 [[buffer(18)]],
    constant const int& weight_s2 [[buffer(19)]],
    constant const int& weight_s3 [[buffer(20)]],
    constant const int& weight_s4 [[buffer(21)]],
    constant const int& weight_layout [[buffer(22)]],
    constant const int& kernel_x [[buffer(23)]],
    constant const int& kernel_y [[buffer(24)]],
    constant const int& kernel_z [[buffer(25)]],
    uint in_row_id [[thread_position_in_grid]]
) {
    if (in_row_id >= uint(in_capacity)) {
        return;
    }

    int in_row = int(in_row_id);
    int edge_count = min(counts[0], edge_capacity);
    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);
    float4 acc3 = float4(0.0f);
    for (int cursor = in_row_offsets[in_row];
         cursor < in_row_offsets[in_row + 1];
         ++cursor) {
        int edge = in_edge_ids[cursor];
        if (edge < 0 || edge >= edge_count) {
            continue;
        }
        int out_row = out_rows[edge];
        int kernel_id = kernel_ids[edge];
        if (out_row < 0 || out_row >= out_capacity || kernel_id < 0) {
            continue;
        }
        for (int co = 0; co < out_channels; ++co) {
            float value =
                float(cotangent[out_row * cotangent_s0 + co * cotangent_s1]);
            acc0 += value * float4(
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    0,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )]),
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    1,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )]),
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    2,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )]),
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    3,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )])
                            );
            acc1 += value * float4(
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    4,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )]),
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    5,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )]),
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    6,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )]),
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    7,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )])
                            );
            acc2 += value * float4(
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    8,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )]),
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    9,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )]),
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    10,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )]),
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    11,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )])
                            );
            acc3 += value * float4(
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    12,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )]),
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    13,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )]),
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    14,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )]),
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    15,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )])
                            );
        }
    }
    int grad_base = in_row * 16;
    grad[grad_base] = half(acc0.x);
    grad[grad_base + 1] = half(acc0.y);
    grad[grad_base + 2] = half(acc0.z);
    grad[grad_base + 3] = half(acc0.w);
    grad[grad_base + 4] = half(acc1.x);
    grad[grad_base + 5] = half(acc1.y);
    grad[grad_base + 6] = half(acc1.z);
    grad[grad_base + 7] = half(acc1.w);
    grad[grad_base + 8] = half(acc2.x);
    grad[grad_base + 9] = half(acc2.y);
    grad[grad_base + 10] = half(acc2.z);
    grad[grad_base + 11] = half(acc2.w);
    grad[grad_base + 12] = half(acc3.x);
    grad[grad_base + 13] = half(acc3.y);
    grad[grad_base + 14] = half(acc3.z);
    grad[grad_base + 15] = half(acc3.w);
    (void)in_rows;
    (void)row_offsets;
    (void)in_channels;
}
