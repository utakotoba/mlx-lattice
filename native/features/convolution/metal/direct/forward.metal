#include <metal_stdlib>

using namespace metal;

#include "native/features/convolution/metal/common.metal"

// Specialized kernels share the generic convolution binding ABI, so some
// bound buffers are intentionally unused by a given specialization.
#pragma clang diagnostic ignored "-Wunused-parameter"

[[kernel]] void sparse_relation_conv_clear_f32(
    device float* out [[buffer(0)]],
    constant const int& total [[buffer(1)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem < uint(total)) {
        out[elem] = 0.0f;
    }
}

[[kernel]] void sparse_relation_conv_f32_i32(
    device const float* feats [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device float* out [[buffer(7)]],
    constant const int& edge_capacity [[buffer(8)]],
    constant const int& out_capacity [[buffer(9)]],
    constant const int& in_channels [[buffer(10)]],
    constant const int& out_channels [[buffer(11)]],
    constant const int& feat_s0 [[buffer(12)]],
    constant const int& feat_s1 [[buffer(13)]],
    constant const int& weight_s0 [[buffer(14)]],
    constant const int& weight_s1 [[buffer(15)]],
    constant const int& weight_s2 [[buffer(16)]],
    constant const int& weight_s3 [[buffer(17)]],
    constant const int& weight_s4 [[buffer(18)]],
    constant const int& weight_layout [[buffer(19)]],
    constant const int& kernel_x [[buffer(20)]],
    constant const int& kernel_y [[buffer(21)]],
    constant const int& kernel_z [[buffer(22)]],
    uint elem [[thread_position_in_grid]]
) {
    int total = out_capacity * out_channels;
    if (elem >= uint(total)) {
        return;
    }

    int out_row = int(elem) / out_channels;
    int co = int(elem) - out_row * out_channels;
    int out_count = min(counts[1], out_capacity);
    if (out_row >= out_count) {
        out[elem] = 0.0f;
        return;
    }

    int edge_count = min(counts[0], edge_capacity);
    float acc = 0.0f;
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
        for (int ci = 0; ci < in_channels; ++ci) {
            acc += feats[in_row * feat_s0 + ci * feat_s1] *
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
    out[elem] = acc;
    (void)out_rows;
}

[[kernel]] void sparse_relation_conv_f32_i32_vec4(
    device const float* feats [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device float* out [[buffer(7)]],
    constant const int& edge_capacity [[buffer(8)]],
    constant const int& out_capacity [[buffer(9)]],
    constant const int& in_channels [[buffer(10)]],
    constant const int& out_channels [[buffer(11)]],
    constant const int& feat_s0 [[buffer(12)]],
    constant const int& feat_s1 [[buffer(13)]],
    constant const int& weight_s0 [[buffer(14)]],
    constant const int& weight_s1 [[buffer(15)]],
    constant const int& weight_s2 [[buffer(16)]],
    constant const int& weight_s3 [[buffer(17)]],
    constant const int& weight_s4 [[buffer(18)]],
    constant const int& weight_layout [[buffer(19)]],
    constant const int& kernel_x [[buffer(20)]],
    constant const int& kernel_y [[buffer(21)]],
    constant const int& kernel_z [[buffer(22)]],
    uint elem [[thread_position_in_grid]]
) {
    int blocks = out_channels / 4;
    int total = out_capacity * blocks;
    if (elem >= uint(total)) {
        return;
    }

    int out_row = int(elem) / blocks;
    int co = (int(elem) - out_row * blocks) * 4;
    int out_base = out_row * out_channels + co;
    int out_count = min(counts[1], out_capacity);
    if (out_row >= out_count) {
        out[out_base] = 0.0f;
        out[out_base + 1] = 0.0f;
        out[out_base + 2] = 0.0f;
        out[out_base + 3] = 0.0f;
        return;
    }

    int edge_count = min(counts[0], edge_capacity);
    float4 acc = float4(0.0f);
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
        for (int ci = 0; ci < in_channels; ++ci) {
            float value = feats[in_row * feat_s0 + ci * feat_s1];
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
                                   ci,
                                   co + 1,
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
                                   ci,
                                   co + 2,
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
                                   ci,
                                   co + 3,
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
    out[out_base] = acc.x;
    out[out_base + 1] = acc.y;
    out[out_base + 2] = acc.z;
    out[out_base + 3] = acc.w;
    (void)out_rows;
}

[[kernel]] void sparse_relation_conv_f32_i32_cout16(
    device const float* feats [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device float* out [[buffer(7)]],
    constant const int& edge_capacity [[buffer(8)]],
    constant const int& out_capacity [[buffer(9)]],
    constant const int& in_channels [[buffer(10)]],
    constant const int& out_channels [[buffer(11)]],
    constant const int& feat_s0 [[buffer(12)]],
    constant const int& feat_s1 [[buffer(13)]],
    constant const int& weight_s0 [[buffer(14)]],
    constant const int& weight_s1 [[buffer(15)]],
    constant const int& weight_s2 [[buffer(16)]],
    constant const int& weight_s3 [[buffer(17)]],
    constant const int& weight_s4 [[buffer(18)]],
    constant const int& weight_layout [[buffer(19)]],
    constant const int& kernel_x [[buffer(20)]],
    constant const int& kernel_y [[buffer(21)]],
    constant const int& kernel_z [[buffer(22)]],
    uint out_row_id [[thread_position_in_grid]]
) {
    if (out_row_id >= uint(out_capacity)) {
        return;
    }

    int out_row = int(out_row_id);
    int out_base = out_row * 16;
    int out_count = min(counts[1], out_capacity);
    if (out_row >= out_count) {
        for (int co = 0; co < 16; ++co) {
            out[out_base + co] = 0.0f;
        }
        return;
    }

    int edge_count = min(counts[0], edge_capacity);
    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);
    float4 acc3 = float4(0.0f);
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
        for (int ci = 0; ci < in_channels; ++ci) {
            float value = feats[in_row * feat_s0 + ci * feat_s1];
            acc0 += value * float4(
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    ci,
                                    0,
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
                                    ci,
                                    1,
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
                                    ci,
                                    2,
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
                                    ci,
                                    3,
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
                                    ci,
                                    4,
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
                                    ci,
                                    5,
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
                                    ci,
                                    6,
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
                                    ci,
                                    7,
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
                                    ci,
                                    8,
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
                                    ci,
                                    9,
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
                                    ci,
                                    10,
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
                                    ci,
                                    11,
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
                                    ci,
                                    12,
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
                                    ci,
                                    13,
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
                                    ci,
                                    14,
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
                                    ci,
                                    15,
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
    out[out_base] = acc0.x;
    out[out_base + 1] = acc0.y;
    out[out_base + 2] = acc0.z;
    out[out_base + 3] = acc0.w;
    out[out_base + 4] = acc1.x;
    out[out_base + 5] = acc1.y;
    out[out_base + 6] = acc1.z;
    out[out_base + 7] = acc1.w;
    out[out_base + 8] = acc2.x;
    out[out_base + 9] = acc2.y;
    out[out_base + 10] = acc2.z;
    out[out_base + 11] = acc2.w;
    out[out_base + 12] = acc3.x;
    out[out_base + 13] = acc3.y;
    out[out_base + 14] = acc3.z;
    out[out_base + 15] = acc3.w;
    (void)out_rows;
    (void)out_channels;
}

[[kernel]] void sparse_relation_conv_atomic_f32_i32(
    device const float* feats [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device atomic_float* out [[buffer(7)]],
    constant const int& edge_capacity [[buffer(8)]],
    constant const int& out_capacity [[buffer(9)]],
    constant const int& in_channels [[buffer(10)]],
    constant const int& out_channels [[buffer(11)]],
    constant const int& feat_s0 [[buffer(12)]],
    constant const int& feat_s1 [[buffer(13)]],
    constant const int& weight_s0 [[buffer(14)]],
    constant const int& weight_s1 [[buffer(15)]],
    constant const int& weight_s2 [[buffer(16)]],
    constant const int& weight_s3 [[buffer(17)]],
    constant const int& weight_s4 [[buffer(18)]],
    constant const int& weight_layout [[buffer(19)]],
    constant const int& kernel_x [[buffer(20)]],
    constant const int& kernel_y [[buffer(21)]],
    constant const int& kernel_z [[buffer(22)]],
    uint elem [[thread_position_in_grid]]
) {
    int edge_count = min(counts[0], edge_capacity);
    int total = edge_count * out_channels;
    if (elem >= uint(total)) {
        return;
    }

    int edge = int(elem) / out_channels;
    int co = int(elem) - edge * out_channels;
    int in_row = in_rows[edge];
    int out_row = out_rows[edge];
    int kernel_id = kernel_ids[edge];
    if (in_row < 0 || out_row < 0 || out_row >= out_capacity || kernel_id < 0) {
        return;
    }

    float acc = 0.0f;
    for (int ci = 0; ci < in_channels; ++ci) {
        acc += feats[in_row * feat_s0 + ci * feat_s1] *
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
    atomic_fetch_add_explicit(
        &out[out_row * out_channels + co], acc, memory_order_relaxed
    );
    (void)row_offsets;
}

[[kernel]] void sparse_relation_conv_f16_i32(
    device const half* feats [[buffer(0)]],
    device const half* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device half* out [[buffer(7)]],
    constant const int& edge_capacity [[buffer(8)]],
    constant const int& out_capacity [[buffer(9)]],
    constant const int& in_channels [[buffer(10)]],
    constant const int& out_channels [[buffer(11)]],
    constant const int& feat_s0 [[buffer(12)]],
    constant const int& feat_s1 [[buffer(13)]],
    constant const int& weight_s0 [[buffer(14)]],
    constant const int& weight_s1 [[buffer(15)]],
    constant const int& weight_s2 [[buffer(16)]],
    constant const int& weight_s3 [[buffer(17)]],
    constant const int& weight_s4 [[buffer(18)]],
    constant const int& weight_layout [[buffer(19)]],
    constant const int& kernel_x [[buffer(20)]],
    constant const int& kernel_y [[buffer(21)]],
    constant const int& kernel_z [[buffer(22)]],
    uint elem [[thread_position_in_grid]]
) {
    int total = out_capacity * out_channels;
    if (elem >= uint(total)) {
        return;
    }

    int out_row = int(elem) / out_channels;
    int co = int(elem) - out_row * out_channels;
    int out_count = min(counts[1], out_capacity);
    if (out_row >= out_count) {
        out[elem] = half(0.0h);
        return;
    }

    int edge_count = min(counts[0], edge_capacity);
    float acc = 0.0f;
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
        for (int ci = 0; ci < in_channels; ++ci) {
            acc += float(feats[in_row * feat_s0 + ci * feat_s1]) *
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
    out[elem] = half(acc);
    (void)out_rows;
}

[[kernel]] void sparse_relation_conv_f16_i32_cout16(
    device const half* feats [[buffer(0)]],
    device const half* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device half* out [[buffer(7)]],
    constant const int& edge_capacity [[buffer(8)]],
    constant const int& out_capacity [[buffer(9)]],
    constant const int& in_channels [[buffer(10)]],
    constant const int& out_channels [[buffer(11)]],
    constant const int& feat_s0 [[buffer(12)]],
    constant const int& feat_s1 [[buffer(13)]],
    constant const int& weight_s0 [[buffer(14)]],
    constant const int& weight_s1 [[buffer(15)]],
    constant const int& weight_s2 [[buffer(16)]],
    constant const int& weight_s3 [[buffer(17)]],
    constant const int& weight_s4 [[buffer(18)]],
    constant const int& weight_layout [[buffer(19)]],
    constant const int& kernel_x [[buffer(20)]],
    constant const int& kernel_y [[buffer(21)]],
    constant const int& kernel_z [[buffer(22)]],
    uint out_row_id [[thread_position_in_grid]]
) {
    if (out_row_id >= uint(out_capacity)) {
        return;
    }

    int out_row = int(out_row_id);
    int out_base = out_row * 16;
    int out_count = min(counts[1], out_capacity);
    if (out_row >= out_count) {
        for (int co = 0; co < 16; ++co) {
            out[out_base + co] = half(0.0h);
        }
        return;
    }

    int edge_count = min(counts[0], edge_capacity);
    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);
    float4 acc3 = float4(0.0f);
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
        for (int ci = 0; ci < in_channels; ++ci) {
            float value = float(feats[in_row * feat_s0 + ci * feat_s1]);
            acc0 += value * float4(
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    ci,
                                    0,
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
                                    ci,
                                    1,
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
                                    ci,
                                    2,
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
                                    ci,
                                    3,
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
                                    ci,
                                    4,
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
                                    ci,
                                    5,
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
                                    ci,
                                    6,
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
                                    ci,
                                    7,
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
                                    ci,
                                    8,
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
                                    ci,
                                    9,
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
                                    ci,
                                    10,
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
                                    ci,
                                    11,
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
                                    ci,
                                    12,
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
                                    ci,
                                    13,
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
                                    ci,
                                    14,
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
                                    ci,
                                    15,
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
    out[out_base] = half(acc0.x);
    out[out_base + 1] = half(acc0.y);
    out[out_base + 2] = half(acc0.z);
    out[out_base + 3] = half(acc0.w);
    out[out_base + 4] = half(acc1.x);
    out[out_base + 5] = half(acc1.y);
    out[out_base + 6] = half(acc1.z);
    out[out_base + 7] = half(acc1.w);
    out[out_base + 8] = half(acc2.x);
    out[out_base + 9] = half(acc2.y);
    out[out_base + 10] = half(acc2.z);
    out[out_base + 11] = half(acc2.w);
    out[out_base + 12] = half(acc3.x);
    out[out_base + 13] = half(acc3.y);
    out[out_base + 14] = half(acc3.z);
    out[out_base + 15] = half(acc3.w);
    (void)out_rows;
    (void)out_channels;
}
