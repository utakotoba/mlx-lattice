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

template <typename T, int in_channels>
inline void accumulate_dense_forward_cout16_ci4_f16(
    device const T* weights,
    int weight_s0,
    int kernel_id,
    int co,
    int ci,
    float4 feat4,
    thread float4& acc0,
    thread float4& acc1,
    thread float4& acc2,
    thread float4& acc3
) {
    acc0.x +=
        dot(feat4,
            load_dense_weight_ci4(
                weights, weight_s0, in_channels, kernel_id, co + 0, ci
            ));
    acc0.y +=
        dot(feat4,
            load_dense_weight_ci4(
                weights, weight_s0, in_channels, kernel_id, co + 1, ci
            ));
    acc0.z +=
        dot(feat4,
            load_dense_weight_ci4(
                weights, weight_s0, in_channels, kernel_id, co + 2, ci
            ));
    acc0.w +=
        dot(feat4,
            load_dense_weight_ci4(
                weights, weight_s0, in_channels, kernel_id, co + 3, ci
            ));
    acc1.x +=
        dot(feat4,
            load_dense_weight_ci4(
                weights, weight_s0, in_channels, kernel_id, co + 4, ci
            ));
    acc1.y +=
        dot(feat4,
            load_dense_weight_ci4(
                weights, weight_s0, in_channels, kernel_id, co + 5, ci
            ));
    acc1.z +=
        dot(feat4,
            load_dense_weight_ci4(
                weights, weight_s0, in_channels, kernel_id, co + 6, ci
            ));
    acc1.w +=
        dot(feat4,
            load_dense_weight_ci4(
                weights, weight_s0, in_channels, kernel_id, co + 7, ci
            ));
    acc2.x +=
        dot(feat4,
            load_dense_weight_ci4(
                weights, weight_s0, in_channels, kernel_id, co + 8, ci
            ));
    acc2.y +=
        dot(feat4,
            load_dense_weight_ci4(
                weights, weight_s0, in_channels, kernel_id, co + 9, ci
            ));
    acc2.z +=
        dot(feat4,
            load_dense_weight_ci4(
                weights, weight_s0, in_channels, kernel_id, co + 10, ci
            ));
    acc2.w +=
        dot(feat4,
            load_dense_weight_ci4(
                weights, weight_s0, in_channels, kernel_id, co + 11, ci
            ));
    acc3.x +=
        dot(feat4,
            load_dense_weight_ci4(
                weights, weight_s0, in_channels, kernel_id, co + 12, ci
            ));
    acc3.y +=
        dot(feat4,
            load_dense_weight_ci4(
                weights, weight_s0, in_channels, kernel_id, co + 13, ci
            ));
    acc3.z +=
        dot(feat4,
            load_dense_weight_ci4(
                weights, weight_s0, in_channels, kernel_id, co + 14, ci
            ));
    acc3.w +=
        dot(feat4,
            load_dense_weight_ci4(
                weights, weight_s0, in_channels, kernel_id, co + 15, ci
            ));
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
            accumulate_dense_forward_cout16_ci4_f16<float, in_channels>(
                weights,
                params.weight_s0,
                kernel_id,
                co,
                ci,
                feat4,
                acc0,
                acc1,
                acc2,
                acc3
            );
        }
    }
    store4(out, out_base, acc0);
    store4(out, out_base + 4, acc1);
    store4(out, out_base + 8, acc2);
    store4(out, out_base + 12, acc3);
}
