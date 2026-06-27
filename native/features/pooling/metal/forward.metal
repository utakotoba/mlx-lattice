#include <metal_stdlib>

using namespace metal;

[[kernel]] void sparse_pool_relation_f32_i32(
    device const float* feats [[buffer(0)]],
    device const int* in_rows [[buffer(1)]],
    device const int* out_rows [[buffer(2)]],
    device const int* kernel_ids [[buffer(3)]],
    device const int* row_offsets [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device float* out [[buffer(6)]],
    constant const int& reduce [[buffer(7)]],
    constant const int& out_capacity [[buffer(8)]],
    constant const int& channels [[buffer(9)]],
    constant const int& feat_s0 [[buffer(10)]],
    constant const int& feat_s1 [[buffer(11)]],
    uint elem [[thread_position_in_grid]]
) {
    (void)out_rows;
    (void)kernel_ids;
    int total = out_capacity * channels;
    if (elem >= uint(total)) {
        return;
    }

    int out_row = int(elem) / channels;
    int channel = int(elem) - out_row * channels;
    if (out_row >= counts[1]) {
        out[elem] = reduce == 1 ? -INFINITY : 0.0f;
        return;
    }

    float acc = reduce == 1 ? -INFINITY : 0.0f;
    int degree = 0;
    for (int edge = row_offsets[out_row]; edge < row_offsets[out_row + 1];
         ++edge) {
        int in_row = in_rows[edge];
        float value = feats[in_row * feat_s0 + channel * feat_s1];
        acc = reduce == 1 ? max(acc, value) : acc + value;
        ++degree;
    }
    out[elem] = reduce == 2 ? acc / float(max(degree, 1)) : acc;
}
