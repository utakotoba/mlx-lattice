template <typename T>
inline float4 load_contiguous4(device const T* values, int base) {
    return float4(
        float(values[base + 0]),
        float(values[base + 1]),
        float(values[base + 2]),
        float(values[base + 3])
    );
}

inline float4 load_contiguous4(device const half* values, int base) {
    const half4 packed = *reinterpret_cast<device const half4*>(values + base);
    return float4(
        float(packed.x), float(packed.y), float(packed.z), float(packed.w)
    );
}

template <typename T>
inline float4 load_strided4(device const T* values, int base, int stride) {
    return float4(
        float(values[base + 0 * stride]),
        float(values[base + 1 * stride]),
        float(values[base + 2 * stride]),
        float(values[base + 3 * stride])
    );
}

inline float4 load_strided4(device const half* values, int base, int stride) {
    if (stride == 1) {
        return load_contiguous4(values, base);
    }
    return float4(
        float(values[base + 0 * stride]),
        float(values[base + 1 * stride]),
        float(values[base + 2 * stride]),
        float(values[base + 3 * stride])
    );
}

inline void store_contiguous4(device float* out, int base, float4 value) {
    out[base + 0] = value.x;
    out[base + 1] = value.y;
    out[base + 2] = value.z;
    out[base + 3] = value.w;
}

inline void store_contiguous4(device half* out, int base, float4 value) {
    *reinterpret_cast<device half4*>(out + base) = half4(value);
}

template <typename T>
inline float4 load_dense_weight4(
    device const T* weights,
    int weight_s0,
    int channels,
    int kernel_id,
    int ci,
    int co_base
) {
    return float4(
        float(weights[(co_base + 0) * weight_s0 + kernel_id * channels + ci]),
        float(weights[(co_base + 1) * weight_s0 + kernel_id * channels + ci]),
        float(weights[(co_base + 2) * weight_s0 + kernel_id * channels + ci]),
        float(weights[(co_base + 3) * weight_s0 + kernel_id * channels + ci])
    );
}

template <typename T>
inline float4 load_dense_weight_ci4(
    device const T* weights,
    int weight_s0,
    int channels,
    int kernel_id,
    int co,
    int ci_base
) {
    return load_contiguous4(
        weights, co * weight_s0 + kernel_id * channels + ci_base
    );
}

inline void store4(device float* out, int base, float4 value) {
    store_contiguous4(out, base, value);
}

inline void store4(device half* out, int base, float4 value) {
    store_contiguous4(out, base, value);
}

struct DenseForwardParams {
    int edge_capacity;
    int out_capacity;
    int feat_s0;
    int feat_s1;
    int weight_s0;
};

struct DenseInputGradParams {
    int edge_capacity;
    int out_capacity;
    int in_capacity;
    int cotangent_s0;
    int cotangent_s1;
    int weight_s0;
};

template <typename T, int in_channels, int out_channels>
inline void dense_forward_cout4_impl(
    device const T* feats,
    device const T* weights,
    device const int* in_rows,
    device const int* kernel_ids,
    device const int* counts,
    device const int* row_offsets,
    device T* out,
    DenseForwardParams params,
    uint elem
) {
    const int blocks = out_channels / 4;
    const int total = params.out_capacity * blocks;
    if (elem >= uint(total)) {
        return;
    }

    const int out_row = int(elem) / blocks;
    const int co = (int(elem) - out_row * blocks) * 4;
    const int out_base = out_row * out_channels + co;
    const int out_count = min(counts[1], params.out_capacity);
    if (out_row >= out_count) {
        store4(out, out_base, float4(0.0f));
        return;
    }

    const int edge_count = min(counts[0], params.edge_capacity);
    float4 acc = float4(0.0f);
    const int edge_begin = max(row_offsets[out_row], 0);
    const int edge_end = min(row_offsets[out_row + 1], edge_count);
    for (int edge = edge_begin; edge < edge_end; ++edge) {
        const int in_row = in_rows[edge];
        const int kernel_id = kernel_ids[edge];
        if (in_row < 0 || kernel_id < 0) {
            continue;
        }
        const int feat_base = in_row * params.feat_s0;
        for (int ci = 0; ci < in_channels; ++ci) {
            const float value = float(feats[feat_base + ci * params.feat_s1]);
            acc += value *
                   load_dense_weight4(
                       weights, params.weight_s0, in_channels, kernel_id, ci, co
                   );
        }
    }
    store4(out, out_base, acc);
}

template <typename T, int in_channels, int out_channels>
inline void dense_forward_cout16_impl(
    device const T* feats,
    device const T* weights,
    device const int* in_rows,
    device const int* kernel_ids,
    device const int* counts,
    device const int* row_offsets,
    device T* out,
    DenseForwardParams params,
    uint elem
) {
    const int blocks = out_channels / 16;
    const int total = params.out_capacity * blocks;
    if (elem >= uint(total)) {
        return;
    }

    const int out_row = int(elem) / blocks;
    const int co = (int(elem) - out_row * blocks) * 16;
    const int out_base = out_row * out_channels + co;
    const int out_count = min(counts[1], params.out_capacity);
    if (out_row >= out_count) {
        for (int offset = 0; offset < 16; ++offset) {
            out[out_base + offset] = T(0);
        }
        return;
    }

    const int edge_count = min(counts[0], params.edge_capacity);
    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);
    float4 acc3 = float4(0.0f);
    const int edge_begin = max(row_offsets[out_row], 0);
    const int edge_end = min(row_offsets[out_row + 1], edge_count);
    for (int edge = edge_begin; edge < edge_end; ++edge) {
        const int in_row = in_rows[edge];
        const int kernel_id = kernel_ids[edge];
        if (in_row < 0 || kernel_id < 0) {
            continue;
        }
        const int feat_base = in_row * params.feat_s0;
        for (int ci = 0; ci < in_channels; ++ci) {
            const float value = float(feats[feat_base + ci * params.feat_s1]);
            acc0 +=
                value *
                load_dense_weight4(
                    weights, params.weight_s0, in_channels, kernel_id, ci, co
                );
            acc1 += value * load_dense_weight4(
                                weights,
                                params.weight_s0,
                                in_channels,
                                kernel_id,
                                ci,
                                co + 4
                            );
            acc2 += value * load_dense_weight4(
                                weights,
                                params.weight_s0,
                                in_channels,
                                kernel_id,
                                ci,
                                co + 8
                            );
            acc3 += value * load_dense_weight4(
                                weights,
                                params.weight_s0,
                                in_channels,
                                kernel_id,
                                ci,
                                co + 12
                            );
        }
    }
    store4(out, out_base, acc0);
    store4(out, out_base + 4, acc1);
    store4(out, out_base + 8, acc2);
    store4(out, out_base + 12, acc3);
}

template <int in_channels, int out_channels>
inline void dense_forward_cout16_ci4_f16_impl(
    device const half* feats,
    device const half* weights,
    device const int* in_rows,
    device const int* kernel_ids,
    device const int* counts,
    device const int* row_offsets,
    device half* out,
    DenseForwardParams params,
    uint elem
) {
    const int blocks = out_channels / 16;
    const int total = params.out_capacity * blocks;
    if (elem >= uint(total)) {
        return;
    }

    const int out_row = int(elem) / blocks;
    const int co = (int(elem) - out_row * blocks) * 16;
    const int out_base = out_row * out_channels + co;
    const int out_count = min(counts[1], params.out_capacity);
    if (out_row >= out_count) {
        for (int offset = 0; offset < 16; offset += 4) {
            store4(out, out_base + offset, float4(0.0f));
        }
        return;
    }

    const int edge_count = min(counts[0], params.edge_capacity);
    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);
    float4 acc3 = float4(0.0f);
    for (int edge = row_offsets[out_row]; edge < row_offsets[out_row + 1];
         ++edge) {
        if (edge < 0 || edge >= edge_count) {
            continue;
        }
        const int in_row = in_rows[edge];
        const int kernel_id = kernel_ids[edge];
        if (in_row < 0 || kernel_id < 0) {
            continue;
        }
        const int feat_base = in_row * params.feat_s0;
        for (int ci = 0; ci < in_channels; ci += 4) {
            const float4 feat4 = load_strided4(
                feats, feat_base + ci * params.feat_s1, params.feat_s1
            );
            acc0.x +=
                dot(feat4,
                    load_dense_weight_ci4(
                        weights,
                        params.weight_s0,
                        in_channels,
                        kernel_id,
                        co + 0,
                        ci
                    ));
            acc0.y +=
                dot(feat4,
                    load_dense_weight_ci4(
                        weights,
                        params.weight_s0,
                        in_channels,
                        kernel_id,
                        co + 1,
                        ci
                    ));
            acc0.z +=
                dot(feat4,
                    load_dense_weight_ci4(
                        weights,
                        params.weight_s0,
                        in_channels,
                        kernel_id,
                        co + 2,
                        ci
                    ));
            acc0.w +=
                dot(feat4,
                    load_dense_weight_ci4(
                        weights,
                        params.weight_s0,
                        in_channels,
                        kernel_id,
                        co + 3,
                        ci
                    ));
            acc1.x +=
                dot(feat4,
                    load_dense_weight_ci4(
                        weights,
                        params.weight_s0,
                        in_channels,
                        kernel_id,
                        co + 4,
                        ci
                    ));
            acc1.y +=
                dot(feat4,
                    load_dense_weight_ci4(
                        weights,
                        params.weight_s0,
                        in_channels,
                        kernel_id,
                        co + 5,
                        ci
                    ));
            acc1.z +=
                dot(feat4,
                    load_dense_weight_ci4(
                        weights,
                        params.weight_s0,
                        in_channels,
                        kernel_id,
                        co + 6,
                        ci
                    ));
            acc1.w +=
                dot(feat4,
                    load_dense_weight_ci4(
                        weights,
                        params.weight_s0,
                        in_channels,
                        kernel_id,
                        co + 7,
                        ci
                    ));
            acc2.x +=
                dot(feat4,
                    load_dense_weight_ci4(
                        weights,
                        params.weight_s0,
                        in_channels,
                        kernel_id,
                        co + 8,
                        ci
                    ));
            acc2.y +=
                dot(feat4,
                    load_dense_weight_ci4(
                        weights,
                        params.weight_s0,
                        in_channels,
                        kernel_id,
                        co + 9,
                        ci
                    ));
            acc2.z +=
                dot(feat4,
                    load_dense_weight_ci4(
                        weights,
                        params.weight_s0,
                        in_channels,
                        kernel_id,
                        co + 10,
                        ci
                    ));
            acc2.w +=
                dot(feat4,
                    load_dense_weight_ci4(
                        weights,
                        params.weight_s0,
                        in_channels,
                        kernel_id,
                        co + 11,
                        ci
                    ));
            acc3.x +=
                dot(feat4,
                    load_dense_weight_ci4(
                        weights,
                        params.weight_s0,
                        in_channels,
                        kernel_id,
                        co + 12,
                        ci
                    ));
            acc3.y +=
                dot(feat4,
                    load_dense_weight_ci4(
                        weights,
                        params.weight_s0,
                        in_channels,
                        kernel_id,
                        co + 13,
                        ci
                    ));
            acc3.z +=
                dot(feat4,
                    load_dense_weight_ci4(
                        weights,
                        params.weight_s0,
                        in_channels,
                        kernel_id,
                        co + 14,
                        ci
                    ));
            acc3.w +=
                dot(feat4,
                    load_dense_weight_ci4(
                        weights,
                        params.weight_s0,
                        in_channels,
                        kernel_id,
                        co + 15,
                        ci
                    ));
        }
    }
    store4(out, out_base, acc0);
    store4(out, out_base + 4, acc1);
    store4(out, out_base + 8, acc2);
    store4(out, out_base + 12, acc3);
}

template <int in_channels, int out_channels>
inline void dense_forward_cout16_ci4_f32_impl(
    device const float* feats,
    device const float* weights,
    device const int* in_rows,
    device const int* kernel_ids,
    device const int* counts,
    device const int* row_offsets,
    device float* out,
    DenseForwardParams params,
    uint elem
) {
    const int blocks = out_channels / 16;
    const int total = params.out_capacity * blocks;
    if (elem >= uint(total)) {
        return;
    }

    const int out_row = int(elem) / blocks;
    const int co = (int(elem) - out_row * blocks) * 16;
    const int out_base = out_row * out_channels + co;
    const int out_count = min(counts[1], params.out_capacity);
    if (out_row >= out_count) {
        for (int offset = 0; offset < 16; offset += 4) {
            store4(out, out_base + offset, float4(0.0f));
        }
        return;
    }

    const int edge_count = min(counts[0], params.edge_capacity);
    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);
    float4 acc3 = float4(0.0f);
    const int edge_begin = max(row_offsets[out_row], 0);
    const int edge_end = min(row_offsets[out_row + 1], edge_count);
    for (int edge = edge_begin; edge < edge_end; ++edge) {
        const int in_row = in_rows[edge];
        const int kernel_id = kernel_ids[edge];
        if (in_row < 0 || kernel_id < 0) {
            continue;
        }
        const int feat_base = in_row * params.feat_s0;
        for (int ci = 0; ci < in_channels; ci += 4) {
            const float4 feat4 = load_strided4(
                feats, feat_base + ci * params.feat_s1, params.feat_s1
            );
            acc0.x += dot(
                feat4,
                load_dense_weight_ci4(
                    weights, params.weight_s0, in_channels, kernel_id, co, ci
                )
            );
            acc0.y +=
                dot(feat4,
                    load_dense_weight_ci4(
                        weights,
                        params.weight_s0,
                        in_channels,
                        kernel_id,
                        co + 1,
                        ci
                    ));
            acc0.z +=
                dot(feat4,
                    load_dense_weight_ci4(
                        weights,
                        params.weight_s0,
                        in_channels,
                        kernel_id,
                        co + 2,
                        ci
                    ));
            acc0.w +=
                dot(feat4,
                    load_dense_weight_ci4(
                        weights,
                        params.weight_s0,
                        in_channels,
                        kernel_id,
                        co + 3,
                        ci
                    ));
            acc1.x +=
                dot(feat4,
                    load_dense_weight_ci4(
                        weights,
                        params.weight_s0,
                        in_channels,
                        kernel_id,
                        co + 4,
                        ci
                    ));
            acc1.y +=
                dot(feat4,
                    load_dense_weight_ci4(
                        weights,
                        params.weight_s0,
                        in_channels,
                        kernel_id,
                        co + 5,
                        ci
                    ));
            acc1.z +=
                dot(feat4,
                    load_dense_weight_ci4(
                        weights,
                        params.weight_s0,
                        in_channels,
                        kernel_id,
                        co + 6,
                        ci
                    ));
            acc1.w +=
                dot(feat4,
                    load_dense_weight_ci4(
                        weights,
                        params.weight_s0,
                        in_channels,
                        kernel_id,
                        co + 7,
                        ci
                    ));
            acc2.x +=
                dot(feat4,
                    load_dense_weight_ci4(
                        weights,
                        params.weight_s0,
                        in_channels,
                        kernel_id,
                        co + 8,
                        ci
                    ));
            acc2.y +=
                dot(feat4,
                    load_dense_weight_ci4(
                        weights,
                        params.weight_s0,
                        in_channels,
                        kernel_id,
                        co + 9,
                        ci
                    ));
            acc2.z +=
                dot(feat4,
                    load_dense_weight_ci4(
                        weights,
                        params.weight_s0,
                        in_channels,
                        kernel_id,
                        co + 10,
                        ci
                    ));
            acc2.w +=
                dot(feat4,
                    load_dense_weight_ci4(
                        weights,
                        params.weight_s0,
                        in_channels,
                        kernel_id,
                        co + 11,
                        ci
                    ));
            acc3.x +=
                dot(feat4,
                    load_dense_weight_ci4(
                        weights,
                        params.weight_s0,
                        in_channels,
                        kernel_id,
                        co + 12,
                        ci
                    ));
            acc3.y +=
                dot(feat4,
                    load_dense_weight_ci4(
                        weights,
                        params.weight_s0,
                        in_channels,
                        kernel_id,
                        co + 13,
                        ci
                    ));
            acc3.z +=
                dot(feat4,
                    load_dense_weight_ci4(
                        weights,
                        params.weight_s0,
                        in_channels,
                        kernel_id,
                        co + 14,
                        ci
                    ));
            acc3.w +=
                dot(feat4,
                    load_dense_weight_ci4(
                        weights,
                        params.weight_s0,
                        in_channels,
                        kernel_id,
                        co + 15,
                        ci
                    ));
        }
    }
    store4(out, out_base, acc0);
    store4(out, out_base + 4, acc1);
    store4(out, out_base + 8, acc2);
    store4(out, out_base + 12, acc3);
}

template <typename T, int in_channels, int out_channels>
inline void dense_input_grad_cin16_grouped_impl(
    device const T* cotangent,
    device const T* weights,
    device const int* out_rows,
    device const int* kernel_ids,
    device const int* counts,
    device const int* in_row_offsets,
    device const int* in_edge_ids,
    device T* grad,
    DenseInputGradParams params,
    uint elem
) {
    const int groups = in_channels / 16;
    const int total = params.in_capacity * groups;
    if (elem >= uint(total)) {
        return;
    }
    const int in_row = int(elem) / groups;
    const int ci_base = (int(elem) - in_row * groups) * 16;
    const int edge_count = min(counts[0], params.edge_capacity);
    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);
    float4 acc3 = float4(0.0f);
    for (int cursor = in_row_offsets[in_row];
         cursor < in_row_offsets[in_row + 1];
         ++cursor) {
        const int edge = in_edge_ids[cursor];
        if (edge < 0 || edge >= edge_count) {
            continue;
        }
        const int out_row = out_rows[edge];
        const int kernel_id = kernel_ids[edge];
        if (out_row < 0 || out_row >= params.out_capacity || kernel_id < 0) {
            continue;
        }
        const int cot_base = out_row * params.cotangent_s0;
        for (int co = 0; co < out_channels; ++co) {
            const float value =
                float(cotangent[cot_base + co * params.cotangent_s1]);
            acc0 += value * load_dense_weight_ci4(
                                weights,
                                params.weight_s0,
                                in_channels,
                                kernel_id,
                                co,
                                ci_base
                            );
            acc1 += value * load_dense_weight_ci4(
                                weights,
                                params.weight_s0,
                                in_channels,
                                kernel_id,
                                co,
                                ci_base + 4
                            );
            acc2 += value * load_dense_weight_ci4(
                                weights,
                                params.weight_s0,
                                in_channels,
                                kernel_id,
                                co,
                                ci_base + 8
                            );
            acc3 += value * load_dense_weight_ci4(
                                weights,
                                params.weight_s0,
                                in_channels,
                                kernel_id,
                                co,
                                ci_base + 12
                            );
        }
    }
    const int grad_base = in_row * in_channels + ci_base;
    store4(grad, grad_base, acc0);
    store4(grad, grad_base + 4, acc1);
    store4(grad, grad_base + 8, acc2);
    store4(grad, grad_base + 12, acc3);
}

template <typename T, int in_channels, int out_channels>
inline void dense_input_grad_cin16_impl(
    device const T* cotangent,
    device const T* weights,
    device const int* out_rows,
    device const int* kernel_ids,
    device const int* counts,
    device const int* in_row_offsets,
    device const int* in_edge_ids,
    device T* grad,
    DenseInputGradParams params,
    uint in_row_id
) {
    if (in_row_id >= uint(params.in_capacity)) {
        return;
    }
    const int in_row = int(in_row_id);
    const int edge_count = min(counts[0], params.edge_capacity);
    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);
    float4 acc3 = float4(0.0f);
    for (int cursor = in_row_offsets[in_row];
         cursor < in_row_offsets[in_row + 1];
         ++cursor) {
        const int edge = in_edge_ids[cursor];
        if (edge < 0 || edge >= edge_count) {
            continue;
        }
        const int out_row = out_rows[edge];
        const int kernel_id = kernel_ids[edge];
        if (out_row < 0 || out_row >= params.out_capacity || kernel_id < 0) {
            continue;
        }
        const int cot_base = out_row * params.cotangent_s0;
        for (int co = 0; co < out_channels; ++co) {
            const float value =
                float(cotangent[cot_base + co * params.cotangent_s1]);
            acc0 += value *
                    load_dense_weight_ci4(
                        weights, params.weight_s0, in_channels, kernel_id, co, 0
                    );
            acc1 += value *
                    load_dense_weight_ci4(
                        weights, params.weight_s0, in_channels, kernel_id, co, 4
                    );
            acc2 += value *
                    load_dense_weight_ci4(
                        weights, params.weight_s0, in_channels, kernel_id, co, 8
                    );
            acc3 +=
                value *
                load_dense_weight_ci4(
                    weights, params.weight_s0, in_channels, kernel_id, co, 12
                );
        }
    }
    const int grad_base = in_row * in_channels;
    store4(grad, grad_base, acc0);
    store4(grad, grad_base + 4, acc1);
    store4(grad, grad_base + 8, acc2);
    store4(grad, grad_base + 12, acc3);
}

template <typename T, int in_channels, int out_channels>
inline void dense_input_grad_cin4_impl(
    device const T* cotangent,
    device const T* weights,
    device const int* out_rows,
    device const int* kernel_ids,
    device const int* counts,
    device const int* in_row_offsets,
    device const int* in_edge_ids,
    device T* grad,
    DenseInputGradParams params,
    uint elem
) {
    const int blocks = in_channels / 4;
    const int total = params.in_capacity * blocks;
    if (elem >= uint(total)) {
        return;
    }
    const int in_row = int(elem) / blocks;
    const int ci = (int(elem) - in_row * blocks) * 4;
    const int edge_count = min(counts[0], params.edge_capacity);
    float4 acc = float4(0.0f);
    for (int cursor = in_row_offsets[in_row];
         cursor < in_row_offsets[in_row + 1];
         ++cursor) {
        const int edge = in_edge_ids[cursor];
        if (edge < 0 || edge >= edge_count) {
            continue;
        }
        const int out_row = out_rows[edge];
        const int kernel_id = kernel_ids[edge];
        if (out_row < 0 || out_row >= params.out_capacity || kernel_id < 0) {
            continue;
        }
        const int cot_base = out_row * params.cotangent_s0;
        for (int co = 0; co < out_channels; ++co) {
            const float value =
                float(cotangent[cot_base + co * params.cotangent_s1]);
            acc += value *
                   load_dense_weight_ci4(
                       weights, params.weight_s0, in_channels, kernel_id, co, ci
                   );
        }
    }
    store4(grad, in_row * in_channels + ci, acc);
}

template <typename T, int in_channels, int out_channels>
inline void dense_weight_grad_ci4_co4_impl(
    device const T* feats,
    device const T* cotangent,
    device const int* in_rows,
    device const int* out_rows,
    device const int* counts,
    device const int* kernel_row_offsets,
    device const int* kernel_edge_ids,
    device T* grad,
    int edge_capacity,
    int out_capacity,
    int n_kernels,
    int feat_s0,
    int feat_s1,
    int cotangent_s0,
    int cotangent_s1,
    uint tile_id,
    uint tid,
    threadgroup float* partial
) {
    const int in_blocks = in_channels / 4;
    const int out_blocks = out_channels / 4;
    const int tiles_per_kernel = in_blocks * out_blocks;
    const int total_tiles = n_kernels * tiles_per_kernel;
    if (tile_id >= uint(total_tiles) || tid >= 64) {
        return;
    }

    const int tile = int(tile_id);
    const int kernel_id = tile / tiles_per_kernel;
    const int rem = tile - kernel_id * tiles_per_kernel;
    const int ci = (rem / out_blocks) * 4;
    const int co = (rem - (ci / 4) * out_blocks) * 4;
    const int edge_count = min(counts[0], edge_capacity);
    const int start = kernel_row_offsets[kernel_id];
    const int stop = kernel_row_offsets[kernel_id + 1];
    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);
    float4 acc3 = float4(0.0f);
    for (int cursor = start + int(tid); cursor < stop; cursor += 64) {
        const int edge = kernel_edge_ids[cursor];
        if (edge < 0 || edge >= edge_count) {
            continue;
        }
        const int in_row = in_rows[edge];
        const int out_row = out_rows[edge];
        if (in_row < 0 || out_row < 0 || out_row >= out_capacity) {
            continue;
        }
        const int feat_base = in_row * feat_s0 + ci * feat_s1;
        const float4 feat4 = float4(
            float(feats[feat_base]),
            float(feats[feat_base + feat_s1]),
            float(feats[feat_base + feat_s1 * 2]),
            float(feats[feat_base + feat_s1 * 3])
        );
        const int cotangent_base = out_row * cotangent_s0 + co * cotangent_s1;
        acc0 += feat4 * float(cotangent[cotangent_base]);
        acc1 += feat4 * float(cotangent[cotangent_base + cotangent_s1]);
        acc2 += feat4 * float(cotangent[cotangent_base + cotangent_s1 * 2]);
        acc3 += feat4 * float(cotangent[cotangent_base + cotangent_s1 * 3]);
    }

    const int thread_base = int(tid) * 16;
    partial[thread_base] = acc0.x;
    partial[thread_base + 1] = acc0.y;
    partial[thread_base + 2] = acc0.z;
    partial[thread_base + 3] = acc0.w;
    partial[thread_base + 4] = acc1.x;
    partial[thread_base + 5] = acc1.y;
    partial[thread_base + 6] = acc1.z;
    partial[thread_base + 7] = acc1.w;
    partial[thread_base + 8] = acc2.x;
    partial[thread_base + 9] = acc2.y;
    partial[thread_base + 10] = acc2.z;
    partial[thread_base + 11] = acc2.w;
    partial[thread_base + 12] = acc3.x;
    partial[thread_base + 13] = acc3.y;
    partial[thread_base + 14] = acc3.z;
    partial[thread_base + 15] = acc3.w;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 32; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const int lhs = int(tid) * 16;
            const int rhs = int(tid + stride) * 16;
            for (int value = 0; value < 16; ++value) {
                partial[lhs + value] += partial[rhs + value];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        const int kernel_base = kernel_id * in_channels + ci;
        store4(
            grad,
            ((co + 0) * n_kernels * in_channels) + kernel_base,
            float4(partial[0], partial[1], partial[2], partial[3])
        );
        store4(
            grad,
            ((co + 1) * n_kernels * in_channels) + kernel_base,
            float4(partial[4], partial[5], partial[6], partial[7])
        );
        store4(
            grad,
            ((co + 2) * n_kernels * in_channels) + kernel_base,
            float4(partial[8], partial[9], partial[10], partial[11])
        );
        store4(
            grad,
            ((co + 3) * n_kernels * in_channels) + kernel_base,
            float4(partial[12], partial[13], partial[14], partial[15])
        );
    }
}

template <typename T>
inline void weight_grad_block4_impl(
    device const T* feats,
    device const T* cotangent,
    device const int* in_rows,
    device const int* out_rows,
    device const int* kernel_ids,
    device const int* counts,
    device const int* row_offsets,
    device const int* kernel_row_offsets,
    device const int* kernel_edge_ids,
    device T* grad,
    int edge_capacity,
    int out_capacity,
    int n_kernels,
    int in_channels,
    int out_channels,
    int feat_s0,
    int feat_s1,
    int cotangent_s0,
    int cotangent_s1,
    int weight_layout,
    int kernel_x,
    int kernel_y,
    int kernel_z,
    uint tile_id,
    uint tid,
    threadgroup float* partial
) {
    const int in_blocks = in_channels / 4;
    const int total_tiles = n_kernels * in_blocks;
    if (tile_id >= uint(total_tiles) || tid >= 64) {
        return;
    }

    const int kernel_id = int(tile_id) / in_blocks;
    const int ci = (int(tile_id) - kernel_id * in_blocks) * 4;
    const int edge_count = min(counts[0], edge_capacity);
    const int start = kernel_row_offsets[kernel_id];
    const int stop = kernel_row_offsets[kernel_id + 1];
    const int thread_base = int(tid) * 16;
    for (int co = 0; co < out_channels; co += 4) {
        float4 acc0 = float4(0.0f);
        float4 acc1 = float4(0.0f);
        float4 acc2 = float4(0.0f);
        float4 acc3 = float4(0.0f);
        for (int cursor = start + int(tid); cursor < stop; cursor += 64) {
            const int edge = kernel_edge_ids[cursor];
            if (edge < 0 || edge >= edge_count) {
                continue;
            }
            const int in_row = in_rows[edge];
            const int out_row = out_rows[edge];
            if (in_row < 0 || out_row < 0 || out_row >= out_capacity) {
                continue;
            }
            const int feat_base = in_row * feat_s0 + ci * feat_s1;
            const int cotangent_base =
                out_row * cotangent_s0 + co * cotangent_s1;
            const float4 grad4 = float4(
                float(cotangent[cotangent_base]),
                float(cotangent[cotangent_base + cotangent_s1]),
                float(cotangent[cotangent_base + cotangent_s1 * 2]),
                float(cotangent[cotangent_base + cotangent_s1 * 3])
            );
            acc0 += float(feats[feat_base]) * grad4;
            acc1 += float(feats[feat_base + feat_s1]) * grad4;
            acc2 += float(feats[feat_base + feat_s1 * 2]) * grad4;
            acc3 += float(feats[feat_base + feat_s1 * 3]) * grad4;
        }

        partial[thread_base] = acc0.x;
        partial[thread_base + 1] = acc0.y;
        partial[thread_base + 2] = acc0.z;
        partial[thread_base + 3] = acc0.w;
        partial[thread_base + 4] = acc1.x;
        partial[thread_base + 5] = acc1.y;
        partial[thread_base + 6] = acc1.z;
        partial[thread_base + 7] = acc1.w;
        partial[thread_base + 8] = acc2.x;
        partial[thread_base + 9] = acc2.y;
        partial[thread_base + 10] = acc2.z;
        partial[thread_base + 11] = acc2.w;
        partial[thread_base + 12] = acc3.x;
        partial[thread_base + 13] = acc3.y;
        partial[thread_base + 14] = acc3.z;
        partial[thread_base + 15] = acc3.w;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const int lhs = int(tid) * 16;
                const int rhs = int(tid + stride) * 16;
                for (int value = 0; value < 16; ++value) {
                    partial[lhs + value] += partial[rhs + value];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0) {
            for (int ci_offset = 0; ci_offset < 4; ++ci_offset) {
                for (int co_offset = 0; co_offset < 4; ++co_offset) {
                    grad[sparse_conv_dense_weight_offset(
                        kernel_id,
                        ci + ci_offset,
                        co + co_offset,
                        weight_layout,
                        kernel_x,
                        kernel_y,
                        kernel_z,
                        in_channels,
                        out_channels
                    )] = T(partial[ci_offset * 4 + co_offset]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    (void)kernel_ids;
    (void)row_offsets;
}
