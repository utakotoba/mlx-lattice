#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

[[kernel]] void fill_zero_float32(
    device float* out [[buffer(0)]],
    constant const int& size [[buffer(1)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem < uint(size)) {
        out[elem] = 0.0f;
    }
}

[[kernel]] void conv3d_feats_float32(
    device const float* feats [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const int* maps [[buffer(2)]],
    device const int* kernels [[buffer(3)]],
    device atomic_float* out [[buffer(4)]],
    constant const int& pair_count [[buffer(5)]],
    constant const int& in_channels [[buffer(6)]],
    constant const int& out_channels [[buffer(7)]],
    uint elem [[thread_position_in_grid]]
) {
    uint total = uint(pair_count * out_channels);
    if (elem >= total) {
        return;
    }

    int pair = int(elem / uint(out_channels));
    int out_col = int(elem - uint(pair * out_channels));
    int in_row = maps[pair * 2];
    int out_row = maps[pair * 2 + 1];
    int kernel_index = kernels[pair];
    if (kernel_index < 0) {
        return;
    }

    float acc = 0.0f;
    for (int in_col = 0; in_col < in_channels; ++in_col) {
        acc += feats[in_row * in_channels + in_col] *
               weight
                   [(kernel_index * in_channels + in_col) * out_channels +
                    out_col];
    }
    atomic_fetch_add_explicit(
        &out[out_row * out_channels + out_col], acc, memory_order_relaxed
    );
}

[[kernel]] void conv3d_subm_center_float32(
    device const float* feats [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant const int& center_kernel [[buffer(3)]],
    constant const int& rows [[buffer(4)]],
    constant const int& in_channels [[buffer(5)]],
    constant const int& out_channels [[buffer(6)]],
    uint elem [[thread_position_in_grid]]
) {
    uint total = uint(rows * out_channels);
    if (elem >= total) {
        return;
    }

    int row = int(elem / uint(out_channels));
    int out_col = int(elem - uint(row * out_channels));
    float acc = 0.0f;
    for (int in_col = 0; in_col < in_channels; ++in_col) {
        acc += feats[row * in_channels + in_col] *
               weight
                   [(center_kernel * in_channels + in_col) * out_channels +
                    out_col];
    }
    out[elem] = acc;
}

[[kernel]] void conv3d_subm_residual_float32(
    device const float* feats [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const int* maps [[buffer(2)]],
    device const int* kernels [[buffer(3)]],
    device atomic_float* out [[buffer(4)]],
    constant const int& pair_count [[buffer(5)]],
    constant const int& in_channels [[buffer(6)]],
    constant const int& out_channels [[buffer(7)]],
    constant const int& center_kernel [[buffer(8)]],
    uint elem [[thread_position_in_grid]]
) {
    uint total = uint(pair_count * out_channels);
    if (elem >= total) {
        return;
    }

    int pair = int(elem / uint(out_channels));
    int kernel_index = kernels[pair];
    if (kernel_index < 0 || kernel_index == center_kernel) {
        return;
    }

    int out_col = int(elem - uint(pair * out_channels));
    int in_row = maps[pair * 2];
    int out_row = maps[pair * 2 + 1];
    float acc = 0.0f;
    for (int in_col = 0; in_col < in_channels; ++in_col) {
        acc += feats[in_row * in_channels + in_col] *
               weight
                   [(kernel_index * in_channels + in_col) * out_channels +
                    out_col];
    }
    atomic_fetch_add_explicit(
        &out[out_row * out_channels + out_col], acc, memory_order_relaxed
    );
}

[[kernel]] void conv3d_residual_rows_float32(
    device const float* base [[buffer(0)]],
    device const float* feats [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device const int* maps [[buffer(3)]],
    device const int* kernels [[buffer(4)]],
    device const int* offsets [[buffer(5)]],
    device float* out [[buffer(6)]],
    constant const int& rows [[buffer(7)]],
    constant const int& in_channels [[buffer(8)]],
    constant const int& out_channels [[buffer(9)]],
    uint elem [[thread_position_in_grid]]
) {
    uint total = uint(rows * out_channels);
    if (elem >= total) {
        return;
    }

    int row = int(elem / uint(out_channels));
    int out_col = int(elem - uint(row * out_channels));
    float acc = base[elem];
    for (int pair = offsets[row]; pair < offsets[row + 1]; ++pair) {
        int kernel_index = kernels[pair];
        if (kernel_index < 0) {
            continue;
        }
        int in_row = maps[pair * 2];
        for (int in_col = 0; in_col < in_channels; ++in_col) {
            acc += feats[in_row * in_channels + in_col] *
                   weight
                       [(kernel_index * in_channels + in_col) * out_channels +
                        out_col];
        }
    }
    out[elem] = acc;
}

[[kernel]] void conv3d_residual_rows_vec4_float32(
    device const float* base [[buffer(0)]],
    device const float* feats [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device const int* maps [[buffer(3)]],
    device const int* kernels [[buffer(4)]],
    device const int* offsets [[buffer(5)]],
    device float* out [[buffer(6)]],
    constant const int& rows [[buffer(7)]],
    constant const int& in_channels [[buffer(8)]],
    constant const int& out_channels [[buffer(9)]],
    uint elem [[thread_position_in_grid]]
) {
    int blocks = out_channels / 4;
    uint total = uint(rows * blocks);
    if (elem >= total) {
        return;
    }

    int row = int(elem / uint(blocks));
    int out_col = int(elem - uint(row * blocks)) * 4;
    int row_base = row * out_channels + out_col;
    float4 acc = float4(
        base[row_base],
        base[row_base + 1],
        base[row_base + 2],
        base[row_base + 3]
    );

    for (int pair = offsets[row]; pair < offsets[row + 1]; ++pair) {
        int kernel_index = kernels[pair];
        if (kernel_index < 0) {
            continue;
        }
        int in_row = maps[pair * 2];
        int feat_base = in_row * in_channels;
        for (int in_col = 0; in_col < in_channels; ++in_col) {
            float value = feats[feat_base + in_col];
            int weight_base =
                (kernel_index * in_channels + in_col) * out_channels + out_col;
            acc += value * float4(
                               weight[weight_base],
                               weight[weight_base + 1],
                               weight[weight_base + 2],
                               weight[weight_base + 3]
                           );
        }
    }

    out[row_base] = acc.x;
    out[row_base + 1] = acc.y;
    out[row_base + 2] = acc.z;
    out[row_base + 3] = acc.w;
}

[[kernel]] void pool3d_feats_float32(
    device const float* feats [[buffer(0)]],
    device const int* maps [[buffer(1)]],
    device const int* kernels [[buffer(2)]],
    device const int* offsets [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant const int& rows [[buffer(5)]],
    constant const int& channels [[buffer(6)]],
    uint elem [[thread_position_in_grid]]
) {
    uint total = uint(rows * channels);
    if (elem >= total) {
        return;
    }

    int row = int(elem / uint(channels));
    int channel = int(elem - uint(row * channels));
    float acc = 0.0f;
    for (int pair = offsets[row]; pair < offsets[row + 1]; ++pair) {
        if (kernels[pair] < 0) {
            continue;
        }
        int in_row = maps[pair * 2];
        acc += feats[in_row * channels + channel];
    }
    out[elem] = acc;
}

[[kernel]] void pool3d_feats_grad_float32(
    device const float* grad [[buffer(0)]],
    device const int* maps [[buffer(1)]],
    device const int* kernels [[buffer(2)]],
    device atomic_float* out [[buffer(3)]],
    constant const int& pair_count [[buffer(4)]],
    constant const int& channels [[buffer(5)]],
    uint elem [[thread_position_in_grid]]
) {
    uint total = uint(pair_count * channels);
    if (elem >= total) {
        return;
    }

    int pair = int(elem / uint(channels));
    int channel = int(elem - uint(pair * channels));
    if (kernels[pair] < 0) {
        return;
    }
    int in_row = maps[pair * 2];
    int out_row = maps[pair * 2 + 1];
    atomic_fetch_add_explicit(
        &out[in_row * channels + channel],
        grad[out_row * channels + channel],
        memory_order_relaxed
    );
}

[[kernel]] void conv3d_feats_grad_float32(
    device const float* grad [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const int* maps [[buffer(2)]],
    device const int* kernels [[buffer(3)]],
    device atomic_float* out [[buffer(4)]],
    constant const int& pair_count [[buffer(5)]],
    constant const int& in_channels [[buffer(6)]],
    constant const int& out_channels [[buffer(7)]],
    uint elem [[thread_position_in_grid]]
) {
    uint total = uint(pair_count * in_channels);
    if (elem >= total) {
        return;
    }

    int pair = int(elem / uint(in_channels));
    int in_col = int(elem - uint(pair * in_channels));
    int kernel_index = kernels[pair];
    if (kernel_index < 0) {
        return;
    }
    int in_row = maps[pair * 2];
    int out_row = maps[pair * 2 + 1];
    float acc = 0.0f;
    for (int out_col = 0; out_col < out_channels; ++out_col) {
        acc += grad[out_row * out_channels + out_col] *
               weight
                   [(kernel_index * in_channels + in_col) * out_channels +
                    out_col];
    }
    atomic_fetch_add_explicit(
        &out[in_row * in_channels + in_col], acc, memory_order_relaxed
    );
}

[[kernel]] void conv3d_weight_grad_float32(
    device const float* feats [[buffer(0)]],
    device const float* grad [[buffer(1)]],
    device const int* maps [[buffer(2)]],
    device const int* kernels [[buffer(3)]],
    device atomic_float* out [[buffer(4)]],
    constant const int& pair_count [[buffer(5)]],
    constant const int& in_channels [[buffer(6)]],
    constant const int& out_channels [[buffer(7)]],
    uint elem [[thread_position_in_grid]]
) {
    uint total = uint(pair_count * in_channels * out_channels);
    if (elem >= total) {
        return;
    }

    int pair = int(elem / uint(in_channels * out_channels));
    int local = int(elem - uint(pair * in_channels * out_channels));
    int in_col = local / out_channels;
    int out_col = local - in_col * out_channels;
    int kernel_index = kernels[pair];
    if (kernel_index < 0) {
        return;
    }
    int in_row = maps[pair * 2];
    int out_row = maps[pair * 2 + 1];
    float value = feats[in_row * in_channels + in_col] *
                  grad[out_row * out_channels + out_col];
    atomic_fetch_add_explicit(
        &out[(kernel_index * in_channels + in_col) * out_channels + out_col],
        value,
        memory_order_relaxed
    );
}
