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
