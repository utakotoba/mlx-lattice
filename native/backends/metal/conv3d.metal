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
