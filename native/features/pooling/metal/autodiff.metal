#include <metal_stdlib>

using namespace metal;

inline int pool_edge_rank(
    device const int* in_rows,
    device const int* kernel_ids,
    int edge,
    int n_kernels
) {
    return in_rows[edge] * n_kernels + kernel_ids[edge];
}

inline int pool_max_tie_count(
    device const float* feats,
    device const int* in_rows,
    device const int* row_offsets,
    int out_row,
    int channel,
    float pooled_value,
    int feat_s0,
    int feat_s1
) {
    int count = 0;
    for (int edge = row_offsets[out_row]; edge < row_offsets[out_row + 1];
         ++edge) {
        int in_row = in_rows[edge];
        if (feats[in_row * feat_s0 + channel * feat_s1] == pooled_value) {
            ++count;
        }
    }
    return count;
}

[[kernel]] void sparse_pool_relation_sum_avg_input_grad_f32_i32(
    device const float* cotangent [[buffer(0)]],
    device const float* feats [[buffer(1)]],
    device const float* pooled [[buffer(2)]],
    device const int* in_rows [[buffer(3)]],
    device const int* out_rows [[buffer(4)]],
    device const int* kernel_ids [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device const int* counts [[buffer(7)]],
    device const int* in_row_offsets [[buffer(8)]],
    device const int* in_edge_ids [[buffer(9)]],
    device float* grad [[buffer(10)]],
    constant const int& reduce [[buffer(11)]],
    constant const int& in_capacity [[buffer(12)]],
    constant const int& out_capacity [[buffer(13)]],
    constant const int& n_kernels [[buffer(14)]],
    constant const int& channels [[buffer(15)]],
    constant const int& cotangent_s0 [[buffer(16)]],
    constant const int& cotangent_s1 [[buffer(17)]],
    constant const int& feat_s0 [[buffer(18)]],
    constant const int& feat_s1 [[buffer(19)]],
    constant const int& pooled_s0 [[buffer(20)]],
    constant const int& pooled_s1 [[buffer(21)]],
    uint elem [[thread_position_in_grid]]
) {
    (void)feats;
    (void)in_rows;
    (void)kernel_ids;
    (void)n_kernels;
    (void)feat_s0;
    (void)feat_s1;
    (void)pooled;
    (void)pooled_s0;
    (void)pooled_s1;
    int total = in_capacity * channels;
    if (elem >= uint(total)) {
        return;
    }

    int in_row = int(elem) / channels;
    int channel = int(elem) - in_row * channels;
    float value = 0.0f;
    int out_count = min(counts[1], out_capacity);
    for (int cursor = in_row_offsets[in_row];
         cursor < in_row_offsets[in_row + 1];
         ++cursor) {
        int edge = in_edge_ids[cursor];
        if (edge < 0) {
            continue;
        }
        int out_row = out_rows[edge];
        if (out_row < 0 || out_row >= out_count) {
            continue;
        }
        int degree = row_offsets[out_row + 1] - row_offsets[out_row];
        float scale = reduce == 2 ? 1.0f / float(max(degree, 1)) : 1.0f;
        value +=
            cotangent[out_row * cotangent_s0 + channel * cotangent_s1] * scale;
    }
    grad[elem] = value;
}

[[kernel]] void sparse_pool_relation_exclusive_input_grad_f32_i32(
    device const float* cotangent [[buffer(0)]],
    device const float* feats [[buffer(1)]],
    device const float* pooled [[buffer(2)]],
    device const int* in_rows [[buffer(3)]],
    device const int* out_rows [[buffer(4)]],
    device const int* kernel_ids [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device const int* counts [[buffer(7)]],
    device const int* output_row_offsets [[buffer(8)]],
    device const int* in_edge_ids [[buffer(9)]],
    device float* grad [[buffer(10)]],
    constant const int& reduce [[buffer(11)]],
    constant const int& in_capacity [[buffer(12)]],
    constant const int& out_capacity [[buffer(13)]],
    constant const int& n_kernels [[buffer(14)]],
    constant const int& channels [[buffer(15)]],
    constant const int& cotangent_s0 [[buffer(16)]],
    constant const int& cotangent_s1 [[buffer(17)]],
    constant const int& feat_s0 [[buffer(18)]],
    constant const int& feat_s1 [[buffer(19)]],
    constant const int& pooled_s0 [[buffer(20)]],
    constant const int& pooled_s1 [[buffer(21)]],
    uint elem [[thread_position_in_grid]]
) {
    (void)in_rows;
    (void)kernel_ids;
    (void)n_kernels;
    (void)output_row_offsets;
    int total = in_capacity * channels;
    if (elem >= uint(total)) {
        return;
    }

    int in_row = int(elem) / channels;
    int channel = int(elem) - in_row * channels;
    int edge = in_edge_ids[in_row];
    int out_count = min(counts[1], out_capacity);
    if (edge < 0) {
        grad[elem] = 0.0f;
        return;
    }
    int out_row = out_rows[edge];
    if (out_row < 0 || out_row >= out_count) {
        grad[elem] = 0.0f;
        return;
    }

    int degree = row_offsets[out_row + 1] - row_offsets[out_row];
    float scale = 1.0f;
    if (reduce == 1) {
        float feat_value = feats[in_row * feat_s0 + channel * feat_s1];
        float pooled_value = pooled[out_row * pooled_s0 + channel * pooled_s1];
        if (feat_value != pooled_value) {
            grad[elem] = 0.0f;
            return;
        }
        int tie_count = pool_max_tie_count(
            feats,
            in_rows,
            row_offsets,
            out_row,
            channel,
            pooled_value,
            feat_s0,
            feat_s1
        );
        scale = 1.0f / float(max(tie_count, 1));
    } else if (reduce == 2) {
        scale = 1.0f / float(max(degree, 1));
    }
    grad[elem] =
        cotangent[out_row * cotangent_s0 + channel * cotangent_s1] * scale;
}

[[kernel]] void sparse_pool_relation_max_input_grad_f32_i32(
    device const float* cotangent [[buffer(0)]],
    device const float* feats [[buffer(1)]],
    device const float* pooled [[buffer(2)]],
    device const int* in_rows [[buffer(3)]],
    device const int* out_rows [[buffer(4)]],
    device const int* kernel_ids [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device const int* counts [[buffer(7)]],
    device const int* in_row_offsets [[buffer(8)]],
    device const int* in_edge_ids [[buffer(9)]],
    device float* grad [[buffer(10)]],
    constant const int& reduce [[buffer(11)]],
    constant const int& in_capacity [[buffer(12)]],
    constant const int& out_capacity [[buffer(13)]],
    constant const int& n_kernels [[buffer(14)]],
    constant const int& channels [[buffer(15)]],
    constant const int& cotangent_s0 [[buffer(16)]],
    constant const int& cotangent_s1 [[buffer(17)]],
    constant const int& feat_s0 [[buffer(18)]],
    constant const int& feat_s1 [[buffer(19)]],
    constant const int& pooled_s0 [[buffer(20)]],
    constant const int& pooled_s1 [[buffer(21)]],
    uint elem [[thread_position_in_grid]]
) {
    (void)kernel_ids;
    (void)reduce;
    (void)n_kernels;
    int total = in_capacity * channels;
    if (elem >= uint(total)) {
        return;
    }

    int in_row = int(elem) / channels;
    int channel = int(elem) - in_row * channels;
    float value = 0.0f;
    int out_count = min(counts[1], out_capacity);
    float feat_value = feats[in_row * feat_s0 + channel * feat_s1];
    for (int cursor = in_row_offsets[in_row];
         cursor < in_row_offsets[in_row + 1];
         ++cursor) {
        int edge = in_edge_ids[cursor];
        if (edge < 0) {
            continue;
        }
        int out_row = out_rows[edge];
        if (out_row < 0 || out_row >= out_count) {
            continue;
        }
        float pooled_value = pooled[out_row * pooled_s0 + channel * pooled_s1];
        if (feat_value != pooled_value) {
            continue;
        }
        int tie_count = pool_max_tie_count(
            feats,
            in_rows,
            row_offsets,
            out_row,
            channel,
            pooled_value,
            feat_s0,
            feat_s1
        );
        if (tie_count == 0) {
            continue;
        }
        value += cotangent[out_row * cotangent_s0 + channel * cotangent_s1] /
                 float(tie_count);
    }
    grad[elem] = value;
}

[[kernel]] void sparse_pool_relation_jvp_f32_i32(
    device const float* tangent [[buffer(0)]],
    device const float* feats [[buffer(1)]],
    device const float* pooled [[buffer(2)]],
    device const int* in_rows [[buffer(3)]],
    device const int* out_rows [[buffer(4)]],
    device const int* kernel_ids [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device const int* counts [[buffer(7)]],
    device float* out [[buffer(8)]],
    constant const int& reduce [[buffer(9)]],
    constant const int& in_capacity [[buffer(10)]],
    constant const int& out_capacity [[buffer(11)]],
    constant const int& n_kernels [[buffer(12)]],
    constant const int& channels [[buffer(13)]],
    constant const int& tangent_s0 [[buffer(14)]],
    constant const int& tangent_s1 [[buffer(15)]],
    constant const int& feat_s0 [[buffer(16)]],
    constant const int& feat_s1 [[buffer(17)]],
    constant const int& pooled_s0 [[buffer(18)]],
    constant const int& pooled_s1 [[buffer(19)]],
    uint elem [[thread_position_in_grid]]
) {
    (void)out_rows;
    (void)in_capacity;
    int total = out_capacity * channels;
    if (elem >= uint(total)) {
        return;
    }

    int out_row = int(elem) / channels;
    int channel = int(elem) - out_row * channels;
    if (out_row >= counts[1]) {
        out[elem] = 0.0f;
        return;
    }

    float pooled_value = pooled[out_row * pooled_s0 + channel * pooled_s1];
    float value = 0.0f;
    int degree = 0;
    int first_rank = in_capacity * n_kernels;
    float first_tangent = 0.0f;
    for (int edge = row_offsets[out_row]; edge < row_offsets[out_row + 1];
         ++edge) {
        int in_row = in_rows[edge];
        float tangent_value =
            tangent[in_row * tangent_s0 + channel * tangent_s1];
        if (reduce == 1) {
            if (feats[in_row * feat_s0 + channel * feat_s1] != pooled_value) {
                continue;
            }
            int rank = pool_edge_rank(in_rows, kernel_ids, edge, n_kernels);
            if (rank < first_rank) {
                first_rank = rank;
                first_tangent = tangent_value;
            }
            continue;
        }
        value += tangent_value;
        ++degree;
    }

    if (reduce == 1) {
        value = first_tangent;
    } else if (reduce == 2) {
        value /= float(max(degree, 1));
    }
    out[elem] = value;
}
