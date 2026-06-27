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
