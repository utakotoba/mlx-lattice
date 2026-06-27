#include <metal_stdlib>

using namespace metal;

#include "native/features/convolution/metal/common.metal"
#include "native/features/convolution/metal/direct/dense_weight_grad.metal"
#include "native/features/convolution/metal/direct/vector_io.metal"

// Specialized kernels share the generic convolution binding ABI, so some
// bound buffers are intentionally unused by a given specialization.
#pragma clang diagnostic ignored "-Wunused-parameter"

[[kernel]] void sparse_relation_conv_weight_grad_f32_i32(
    device const float* feats [[buffer(0)]],
    device const float* cotangent [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device const int* kernel_row_offsets [[buffer(7)]],
    device const int* kernel_edge_ids [[buffer(8)]],
    device float* grad [[buffer(9)]],
    constant const int& edge_capacity [[buffer(10)]],
    constant const int& out_capacity [[buffer(11)]],
    constant const int& n_kernels [[buffer(12)]],
    constant const int& in_channels [[buffer(13)]],
    constant const int& out_channels [[buffer(14)]],
    constant const int& feat_s0 [[buffer(15)]],
    constant const int& feat_s1 [[buffer(16)]],
    constant const int& cotangent_s0 [[buffer(17)]],
    constant const int& cotangent_s1 [[buffer(18)]],
    constant const int& weight_layout [[buffer(19)]],
    constant const int& kernel_x [[buffer(20)]],
    constant const int& kernel_y [[buffer(21)]],
    constant const int& kernel_z [[buffer(22)]],
    uint elem [[thread_position_in_grid]]
) {
    int edge_count = min(counts[0], edge_capacity);
    int total = n_kernels * in_channels * out_channels;
    if (elem >= uint(total)) {
        return;
    }

    int channel = int(elem) % (in_channels * out_channels);
    int kernel_id = int(elem) / (in_channels * out_channels);
    int ci = channel / out_channels;
    int co = channel - ci * out_channels;

    float acc = 0.0f;
    for (int cursor = kernel_row_offsets[kernel_id];
         cursor < kernel_row_offsets[kernel_id + 1];
         ++cursor) {
        int edge = kernel_edge_ids[cursor];
        if (edge < 0 || edge >= edge_count) {
            continue;
        }
        int in_row = in_rows[edge];
        int out_row = out_rows[edge];
        if (in_row < 0 || out_row < 0 || out_row >= out_capacity) {
            continue;
        }
        acc += feats[in_row * feat_s0 + ci * feat_s1] *
               cotangent[out_row * cotangent_s0 + co * cotangent_s1];
    }
    grad[sparse_conv_dense_weight_offset(
        kernel_id,
        ci,
        co,
        weight_layout,
        kernel_x,
        kernel_y,
        kernel_z,
        in_channels,
        out_channels
    )] = acc;
    (void)kernel_ids;
    (void)row_offsets;
}

[[kernel]] void sparse_relation_conv_weight_grad_cout16_f32_i32(
    device const float* feats [[buffer(0)]],
    device const float* cotangent [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device const int* kernel_row_offsets [[buffer(7)]],
    device const int* kernel_edge_ids [[buffer(8)]],
    device float* grad [[buffer(9)]],
    constant const int& edge_capacity [[buffer(10)]],
    constant const int& out_capacity [[buffer(11)]],
    constant const int& n_kernels [[buffer(12)]],
    constant const int& in_channels [[buffer(13)]],
    constant const int& out_channels [[buffer(14)]],
    constant const int& feat_s0 [[buffer(15)]],
    constant const int& feat_s1 [[buffer(16)]],
    constant const int& cotangent_s0 [[buffer(17)]],
    constant const int& cotangent_s1 [[buffer(18)]],
    constant const int& weight_layout [[buffer(19)]],
    constant const int& kernel_x [[buffer(20)]],
    constant const int& kernel_y [[buffer(21)]],
    constant const int& kernel_z [[buffer(22)]],
    uint pair_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float partial[4096];
    int total_pairs = n_kernels * in_channels;
    if (pair_id >= uint(total_pairs) || tid >= 256) {
        return;
    }

    int kernel_id = int(pair_id) / in_channels;
    int ci = int(pair_id) - kernel_id * in_channels;
    int edge_count = min(counts[0], edge_capacity);
    int start = kernel_row_offsets[kernel_id];
    int stop = kernel_row_offsets[kernel_id + 1];
    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);
    float4 acc3 = float4(0.0f);
    for (int cursor = start + int(tid); cursor < stop; cursor += 256) {
        int edge = kernel_edge_ids[cursor];
        if (edge < 0 || edge >= edge_count) {
            continue;
        }
        int in_row = in_rows[edge];
        int out_row = out_rows[edge];
        if (in_row < 0 || out_row < 0 || out_row >= out_capacity) {
            continue;
        }
        float value = feats[in_row * feat_s0 + ci * feat_s1];
        int cotangent_base = out_row * cotangent_s0;
        acc0 += value * float4(
                            cotangent[cotangent_base],
                            cotangent[cotangent_base + cotangent_s1],
                            cotangent[cotangent_base + cotangent_s1 * 2],
                            cotangent[cotangent_base + cotangent_s1 * 3]
                        );
        acc1 += value * float4(
                            cotangent[cotangent_base + cotangent_s1 * 4],
                            cotangent[cotangent_base + cotangent_s1 * 5],
                            cotangent[cotangent_base + cotangent_s1 * 6],
                            cotangent[cotangent_base + cotangent_s1 * 7]
                        );
        acc2 += value * float4(
                            cotangent[cotangent_base + cotangent_s1 * 8],
                            cotangent[cotangent_base + cotangent_s1 * 9],
                            cotangent[cotangent_base + cotangent_s1 * 10],
                            cotangent[cotangent_base + cotangent_s1 * 11]
                        );
        acc3 += value * float4(
                            cotangent[cotangent_base + cotangent_s1 * 12],
                            cotangent[cotangent_base + cotangent_s1 * 13],
                            cotangent[cotangent_base + cotangent_s1 * 14],
                            cotangent[cotangent_base + cotangent_s1 * 15]
                        );
    }

    int base = int(tid) * 16;
    partial[base] = acc0.x;
    partial[base + 1] = acc0.y;
    partial[base + 2] = acc0.z;
    partial[base + 3] = acc0.w;
    partial[base + 4] = acc1.x;
    partial[base + 5] = acc1.y;
    partial[base + 6] = acc1.z;
    partial[base + 7] = acc1.w;
    partial[base + 8] = acc2.x;
    partial[base + 9] = acc2.y;
    partial[base + 10] = acc2.z;
    partial[base + 11] = acc2.w;
    partial[base + 12] = acc3.x;
    partial[base + 13] = acc3.y;
    partial[base + 14] = acc3.z;
    partial[base + 15] = acc3.w;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            int lhs = int(tid) * 16;
            int rhs = int(tid + stride) * 16;
            for (int co = 0; co < 16; ++co) {
                partial[lhs + co] += partial[rhs + co];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        for (int co = 0; co < 16; ++co) {
            grad[sparse_conv_dense_weight_offset(
                kernel_id,
                ci,
                co,
                weight_layout,
                kernel_x,
                kernel_y,
                kernel_z,
                in_channels,
                out_channels
            )] = partial[co];
        }
    }
    (void)kernel_ids;
    (void)row_offsets;
}

[[kernel]] void sparse_relation_conv_weight_grad_block4_f32_i32(
    device const float* feats [[buffer(0)]],
    device const float* cotangent [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device const int* kernel_row_offsets [[buffer(7)]],
    device const int* kernel_edge_ids [[buffer(8)]],
    device float* grad [[buffer(9)]],
    constant const int& edge_capacity [[buffer(10)]],
    constant const int& out_capacity [[buffer(11)]],
    constant const int& n_kernels [[buffer(12)]],
    constant const int& in_channels [[buffer(13)]],
    constant const int& out_channels [[buffer(14)]],
    constant const int& feat_s0 [[buffer(15)]],
    constant const int& feat_s1 [[buffer(16)]],
    constant const int& cotangent_s0 [[buffer(17)]],
    constant const int& cotangent_s1 [[buffer(18)]],
    constant const int& weight_layout [[buffer(19)]],
    constant const int& kernel_x [[buffer(20)]],
    constant const int& kernel_y [[buffer(21)]],
    constant const int& kernel_z [[buffer(22)]],
    uint tile_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float partial[1024];
    weight_grad_block4_impl<float>(
        feats,
        cotangent,
        in_rows,
        out_rows,
        kernel_ids,
        counts,
        row_offsets,
        kernel_row_offsets,
        kernel_edge_ids,
        grad,
        edge_capacity,
        out_capacity,
        n_kernels,
        in_channels,
        out_channels,
        feat_s0,
        feat_s1,
        cotangent_s0,
        cotangent_s1,
        weight_layout,
        kernel_x,
        kernel_y,
        kernel_z,
        tile_id,
        tid,
        partial
    );
}

[[kernel]] void sparse_relation_conv_weight_grad_block4_f16_i32(
    device const half* feats [[buffer(0)]],
    device const half* cotangent [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device const int* kernel_row_offsets [[buffer(7)]],
    device const int* kernel_edge_ids [[buffer(8)]],
    device half* grad [[buffer(9)]],
    constant const int& edge_capacity [[buffer(10)]],
    constant const int& out_capacity [[buffer(11)]],
    constant const int& n_kernels [[buffer(12)]],
    constant const int& in_channels [[buffer(13)]],
    constant const int& out_channels [[buffer(14)]],
    constant const int& feat_s0 [[buffer(15)]],
    constant const int& feat_s1 [[buffer(16)]],
    constant const int& cotangent_s0 [[buffer(17)]],
    constant const int& cotangent_s1 [[buffer(18)]],
    constant const int& weight_layout [[buffer(19)]],
    constant const int& kernel_x [[buffer(20)]],
    constant const int& kernel_y [[buffer(21)]],
    constant const int& kernel_z [[buffer(22)]],
    uint tile_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float partial[1024];
    weight_grad_block4_impl<half>(
        feats,
        cotangent,
        in_rows,
        out_rows,
        kernel_ids,
        counts,
        row_offsets,
        kernel_row_offsets,
        kernel_edge_ids,
        grad,
        edge_capacity,
        out_capacity,
        n_kernels,
        in_channels,
        out_channels,
        feat_s0,
        feat_s1,
        cotangent_s0,
        cotangent_s1,
        weight_layout,
        kernel_x,
        kernel_y,
        kernel_z,
        tile_id,
        tid,
        partial
    );
}

[[kernel]] void sparse_relation_conv_weight_grad_cout16_f16_i32(
    device const half* feats [[buffer(0)]],
    device const half* cotangent [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device const int* kernel_row_offsets [[buffer(7)]],
    device const int* kernel_edge_ids [[buffer(8)]],
    device half* grad [[buffer(9)]],
    constant const int& edge_capacity [[buffer(10)]],
    constant const int& out_capacity [[buffer(11)]],
    constant const int& n_kernels [[buffer(12)]],
    constant const int& in_channels [[buffer(13)]],
    constant const int& out_channels [[buffer(14)]],
    constant const int& feat_s0 [[buffer(15)]],
    constant const int& feat_s1 [[buffer(16)]],
    constant const int& cotangent_s0 [[buffer(17)]],
    constant const int& cotangent_s1 [[buffer(18)]],
    constant const int& weight_layout [[buffer(19)]],
    constant const int& kernel_x [[buffer(20)]],
    constant const int& kernel_y [[buffer(21)]],
    constant const int& kernel_z [[buffer(22)]],
    uint pair_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float partial[4096];
    int total_pairs = n_kernels * in_channels;
    if (pair_id >= uint(total_pairs) || tid >= 256) {
        return;
    }

    int kernel_id = int(pair_id) / in_channels;
    int ci = int(pair_id) - kernel_id * in_channels;
    int edge_count = min(counts[0], edge_capacity);
    int start = kernel_row_offsets[kernel_id];
    int stop = kernel_row_offsets[kernel_id + 1];
    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);
    float4 acc3 = float4(0.0f);
    for (int cursor = start + int(tid); cursor < stop; cursor += 256) {
        int edge = kernel_edge_ids[cursor];
        if (edge < 0 || edge >= edge_count) {
            continue;
        }
        int in_row = in_rows[edge];
        int out_row = out_rows[edge];
        if (in_row < 0 || out_row < 0 || out_row >= out_capacity) {
            continue;
        }
        float value = float(feats[in_row * feat_s0 + ci * feat_s1]);
        int cotangent_base = out_row * cotangent_s0;
        acc0 += value * float4(
                            float(cotangent[cotangent_base]),
                            float(cotangent[cotangent_base + cotangent_s1]),
                            float(cotangent[cotangent_base + cotangent_s1 * 2]),
                            float(cotangent[cotangent_base + cotangent_s1 * 3])
                        );
        acc1 += value * float4(
                            float(cotangent[cotangent_base + cotangent_s1 * 4]),
                            float(cotangent[cotangent_base + cotangent_s1 * 5]),
                            float(cotangent[cotangent_base + cotangent_s1 * 6]),
                            float(cotangent[cotangent_base + cotangent_s1 * 7])
                        );
        acc2 +=
            value * float4(
                        float(cotangent[cotangent_base + cotangent_s1 * 8]),
                        float(cotangent[cotangent_base + cotangent_s1 * 9]),
                        float(cotangent[cotangent_base + cotangent_s1 * 10]),
                        float(cotangent[cotangent_base + cotangent_s1 * 11])
                    );
        acc3 +=
            value * float4(
                        float(cotangent[cotangent_base + cotangent_s1 * 12]),
                        float(cotangent[cotangent_base + cotangent_s1 * 13]),
                        float(cotangent[cotangent_base + cotangent_s1 * 14]),
                        float(cotangent[cotangent_base + cotangent_s1 * 15])
                    );
    }

    int base = int(tid) * 16;
    partial[base] = acc0.x;
    partial[base + 1] = acc0.y;
    partial[base + 2] = acc0.z;
    partial[base + 3] = acc0.w;
    partial[base + 4] = acc1.x;
    partial[base + 5] = acc1.y;
    partial[base + 6] = acc1.z;
    partial[base + 7] = acc1.w;
    partial[base + 8] = acc2.x;
    partial[base + 9] = acc2.y;
    partial[base + 10] = acc2.z;
    partial[base + 11] = acc2.w;
    partial[base + 12] = acc3.x;
    partial[base + 13] = acc3.y;
    partial[base + 14] = acc3.z;
    partial[base + 15] = acc3.w;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            int lhs = int(tid) * 16;
            int rhs = int(tid + stride) * 16;
            for (int co = 0; co < 16; ++co) {
                partial[lhs + co] += partial[rhs + co];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        for (int co = 0; co < 16; ++co) {
            grad[sparse_conv_dense_weight_offset(
                kernel_id,
                ci,
                co,
                weight_layout,
                kernel_x,
                kernel_y,
                kernel_z,
                in_channels,
                out_channels
            )] = half(partial[co]);
        }
    }
    (void)kernel_ids;
    (void)row_offsets;
    (void)out_channels;
}

[[kernel]] void sparse_relation_conv_weight_grad_atomic_f32_i32(
    device const float* feats [[buffer(0)]],
    device const float* cotangent [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device const int* kernel_row_offsets [[buffer(7)]],
    device const int* kernel_edge_ids [[buffer(8)]],
    device atomic_float* grad [[buffer(9)]],
    constant const int& edge_capacity [[buffer(10)]],
    constant const int& out_capacity [[buffer(11)]],
    constant const int& n_kernels [[buffer(12)]],
    constant const int& in_channels [[buffer(13)]],
    constant const int& out_channels [[buffer(14)]],
    constant const int& feat_s0 [[buffer(15)]],
    constant const int& feat_s1 [[buffer(16)]],
    constant const int& cotangent_s0 [[buffer(17)]],
    constant const int& cotangent_s1 [[buffer(18)]],
    constant const int& weight_layout [[buffer(19)]],
    constant const int& kernel_x [[buffer(20)]],
    constant const int& kernel_y [[buffer(21)]],
    constant const int& kernel_z [[buffer(22)]],
    uint elem [[thread_position_in_grid]]
) {
    int edge_count = min(counts[0], edge_capacity);
    int total = edge_count * in_channels * out_channels;
    if (elem >= uint(total)) {
        return;
    }

    int channel = int(elem) % (in_channels * out_channels);
    int edge = int(elem) / (in_channels * out_channels);
    int ci = channel / out_channels;
    int co = channel - ci * out_channels;

    int in_row = in_rows[edge];
    int out_row = out_rows[edge];
    int kernel_id = kernel_ids[edge];
    if (in_row < 0 || out_row < 0 || out_row >= out_capacity || kernel_id < 0) {
        return;
    }
    float value = feats[in_row * feat_s0 + ci * feat_s1] *
                  cotangent[out_row * cotangent_s0 + co * cotangent_s1];
    atomic_fetch_add_explicit(
        &grad[sparse_conv_dense_weight_offset(
            kernel_id,
            ci,
            co,
            weight_layout,
            kernel_x,
            kernel_y,
            kernel_z,
            in_channels,
            out_channels
        )],
        value,
        memory_order_relaxed
    );
    (void)row_offsets;
    (void)kernel_row_offsets;
    (void)kernel_edge_ids;
    (void)n_kernels;
}

[[kernel]] void sparse_relation_conv_weight_grad_f16_i32(
    device const half* feats [[buffer(0)]],
    device const half* cotangent [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device const int* kernel_row_offsets [[buffer(7)]],
    device const int* kernel_edge_ids [[buffer(8)]],
    device half* grad [[buffer(9)]],
    constant const int& edge_capacity [[buffer(10)]],
    constant const int& out_capacity [[buffer(11)]],
    constant const int& n_kernels [[buffer(12)]],
    constant const int& in_channels [[buffer(13)]],
    constant const int& out_channels [[buffer(14)]],
    constant const int& feat_s0 [[buffer(15)]],
    constant const int& feat_s1 [[buffer(16)]],
    constant const int& cotangent_s0 [[buffer(17)]],
    constant const int& cotangent_s1 [[buffer(18)]],
    constant const int& weight_layout [[buffer(19)]],
    constant const int& kernel_x [[buffer(20)]],
    constant const int& kernel_y [[buffer(21)]],
    constant const int& kernel_z [[buffer(22)]],
    uint elem [[thread_position_in_grid]]
) {
    int edge_count = min(counts[0], edge_capacity);
    int total = n_kernels * in_channels * out_channels;
    if (elem >= uint(total)) {
        return;
    }

    int channel = int(elem) % (in_channels * out_channels);
    int kernel_id = int(elem) / (in_channels * out_channels);
    int ci = channel / out_channels;
    int co = channel - ci * out_channels;

    float acc = 0.0f;
    for (int cursor = kernel_row_offsets[kernel_id];
         cursor < kernel_row_offsets[kernel_id + 1];
         ++cursor) {
        int edge = kernel_edge_ids[cursor];
        if (edge < 0 || edge >= edge_count) {
            continue;
        }
        int in_row = in_rows[edge];
        int out_row = out_rows[edge];
        if (in_row < 0 || out_row < 0 || out_row >= out_capacity) {
            continue;
        }
        acc += float(feats[in_row * feat_s0 + ci * feat_s1]) *
               float(cotangent[out_row * cotangent_s0 + co * cotangent_s1]);
    }
    grad[sparse_conv_dense_weight_offset(
        kernel_id,
        ci,
        co,
        weight_layout,
        kernel_x,
        kernel_y,
        kernel_z,
        in_channels,
        out_channels
    )] = half(acc);
    (void)kernel_ids;
    (void)row_offsets;
}
