#include <metal_stdlib>

using namespace metal;

[[kernel]] void spmm_edges_f32_serial(
    device const float* feats [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant const int& edge_count [[buffer(6)]],
    constant const int& in_channels [[buffer(7)]],
    constant const int& out_channels [[buffer(8)]],
    constant const int& n_out_rows [[buffer(9)]],
    constant const int& n_in_rows [[buffer(10)]],
    constant const int& n_kernels [[buffer(11)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem != 0) {
        return;
    }

    int out_size = n_out_rows * out_channels;
    for (int index = 0; index < out_size; ++index) {
        out[index] = 0.0f;
    }

    for (int edge = 0; edge < edge_count; ++edge) {
        int in_row = in_rows[edge];
        int out_row = out_rows[edge];
        int kernel_id = kernel_ids[edge];
        if (in_row < 0 || in_row >= n_in_rows || out_row < 0 ||
            out_row >= n_out_rows || kernel_id < 0 || kernel_id >= n_kernels) {
            continue;
        }

        for (int ci = 0; ci < in_channels; ++ci) {
            float value = feats[in_row * in_channels + ci];
            int weight_base = (kernel_id * in_channels + ci) * out_channels;
            int out_base = out_row * out_channels;
            for (int co = 0; co < out_channels; ++co) {
                out[out_base + co] += value * weights[weight_base + co];
            }
        }
    }
}
