#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
#include <metal_tensor>

using namespace metal;
using namespace mpp::tensor_ops;

#include "native/backends/metal/conv/common.metal"
#include "native/backends/metal/conv/dense_kernels.metal"

// Specialized kernels share the generic convolution binding ABI, so some
// bound buffers are intentionally unused by a given specialization.
#pragma clang diagnostic ignored "-Wunused-parameter"

static inline float4
load_packed_weight4(device const half* weights_kci_co, int kv, int ci, int co) {
    constexpr int channels = 32;
    const int offset = kv * channels * channels + ci * channels + co;
    return float4(
        *reinterpret_cast<device const half4*>(weights_kci_co + offset)
    );
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void row_stationary_direct_packedw_c32_m64(
    device const half* feats [[buffer(0)]],
    device half* weights_kci_co [[buffer(1)]],
    device const int* sorted_out_in_map [[buffer(2)]],
    device const int* reorder_rows [[buffer(3)]],
    device const int* tile_masks [[buffer(4)]],
    device half* out [[buffer(5)]],
    constant const int& rows [[buffer(6)]],
    constant const int& store_sorted [[buffer(7)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    constexpr int tile_rows = 64;
    constexpr int channels = 32;

    const int row_slot = int(tid) / 2;
    const int co = (int(tid) - row_slot * 2) * 16;
    const int sorted_row = int(group_id) * tile_rows + row_slot;
    if (sorted_row >= rows) {
        return;
    }

    const int mask_base = int(group_id) * 4;
    uint active_mask = uint(
        tile_masks[mask_base + 0] | tile_masks[mask_base + 1] |
        tile_masks[mask_base + 2] | tile_masks[mask_base + 3]
    );

    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);
    float4 acc3 = float4(0.0f);

    while (active_mask != 0) {
        const int kv = ctz(active_mask);
        active_mask &= active_mask - 1;
        const int in_row = sorted_out_in_map[sorted_row * 27 + kv];
        if (in_row < 0) {
            continue;
        }
        const int feat_base = in_row * channels;
        for (int ci = 0; ci < channels; ++ci) {
            const float feat = float(feats[feat_base + ci]);
            acc0 += feat * load_packed_weight4(weights_kci_co, kv, ci, co + 0);
            acc1 += feat * load_packed_weight4(weights_kci_co, kv, ci, co + 4);
            acc2 += feat * load_packed_weight4(weights_kci_co, kv, ci, co + 8);
            acc3 += feat * load_packed_weight4(weights_kci_co, kv, ci, co + 12);
        }
    }

    const int out_row =
        store_sorted != 0 ? sorted_row : reorder_rows[sorted_row];
    const int out_base = out_row * channels + co;
    *reinterpret_cast<device half4*>(out + out_base + 0) = half4(acc0);
    *reinterpret_cast<device half4*>(out + out_base + 4) = half4(acc1);
    *reinterpret_cast<device half4*>(out + out_base + 8) = half4(acc2);
    *reinterpret_cast<device half4*>(out + out_base + 12) = half4(acc3);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void row_stationary_direct_packedw_c64_m64(
    device const half* feats [[buffer(0)]],
    device half* weights_kci_co [[buffer(1)]],
    device const int* sorted_out_in_map [[buffer(2)]],
    device const int* reorder_rows [[buffer(3)]],
    device const int* tile_masks [[buffer(4)]],
    device half* out [[buffer(5)]],
    constant const int& rows [[buffer(6)]],
    constant const int& store_sorted [[buffer(7)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    constexpr int tile_rows = 64;
    constexpr int channels = 64;

    const int linear = int(tid);
    const int row_slot = linear / 2;
    const int co_base = (linear - row_slot * 2) * 32;
    const int sorted_row = int(group_id) * tile_rows + row_slot;
    if (sorted_row >= rows) {
        return;
    }

    const int mask_base = int(group_id) * 4;
    uint active_mask = uint(
        tile_masks[mask_base + 0] | tile_masks[mask_base + 1] |
        tile_masks[mask_base + 2] | tile_masks[mask_base + 3]
    );
    float acc[32];
    for (int index = 0; index < 32; ++index) {
        acc[index] = 0.0f;
    }

    while (active_mask != 0) {
        const int kv = ctz(active_mask);
        active_mask &= active_mask - 1;
        const int in_row = sorted_out_in_map[sorted_row * 27 + kv];
        if (in_row < 0) {
            continue;
        }
        for (int ci = 0; ci < channels; ++ci) {
            const float feat = float(feats[in_row * channels + ci]);
            const int weight_base =
                kv * channels * channels + ci * channels + co_base;
            for (int co = 0; co < 32; ++co) {
                acc[co] += feat * float(weights_kci_co[weight_base + co]);
            }
        }
    }

    const int out_row =
        store_sorted != 0 ? sorted_row : reorder_rows[sorted_row];
    const int out_base = out_row * channels + co_base;
    for (int co = 0; co < 32; co += 4) {
        *reinterpret_cast<device half4*>(out + out_base + co) =
            half4(acc[co + 0], acc[co + 1], acc[co + 2], acc[co + 3]);
    }
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void row_stationary_tensor_coop_devicew_rowfill4x2_kvmap_f16acc_sorted_c32_m64(
    device const half* feats [[buffer(0)]],
    device half* weights_kci_co [[buffer(1)]],
    device const int* sorted_kv_out_in_map [[buffer(2)]],
    device const int* reorder_rows [[buffer(3)]],
    device const int* tile_masks [[buffer(4)]],
    device half* out [[buffer(5)]],
    constant const int& rows [[buffer(6)]],
    constant const int& store_sorted [[buffer(7)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    constexpr int tile_rows = 64;
    constexpr int channels = 32;
    constexpr int chunks_per_row = channels / 4;

    threadgroup half lhs_tile[tile_rows * channels];

    const int row_start = int(group_id) * tile_rows;
    const int mask_base = int(group_id) * 4;
    uint active_mask = uint(
        tile_masks[mask_base + 0] | tile_masks[mask_base + 1] |
        tile_masks[mask_base + 2] | tile_masks[mask_base + 3]
    );
    const uint original_mask = active_mask;

    constexpr auto desc = matmul2d_descriptor(
        tile_rows,
        channels,
        channels,
        false,
        false,
        true,
        matmul2d_descriptor::mode::multiply_accumulate
    );
    matmul2d<desc, execution_simdgroups<4>> op;
    auto lhs_tensor = tensor<
        threadgroup half,
        extents<int32_t, channels, tile_rows>,
        tensor_inline>(lhs_tile, extents<int32_t, channels, tile_rows>());
    auto out_tensor = op.get_destination_cooperative_tensor<
        decltype(lhs_tensor),
        decltype(tensor(
            weights_kci_co,
            extents<int32_t, channels, channels>(),
            array<int32_t, 2>{1, channels}
        )),
        half>();

#pragma unroll
    for (uint16_t i = 0; i < out_tensor.get_capacity(); ++i) {
        if (out_tensor.is_valid_element(i)) {
            out_tensor[i] = half(0.0h);
        }
    }

    while (active_mask != 0) {
        const int kv = ctz(active_mask);
        active_mask &= active_mask - 1;
        const int kv_base = kv * rows;

        if (tid < tile_rows * 2) {
            const int row_slot = int(tid >> 1);
            const int half_slot = int(tid & 1);
            const int sorted_row = row_start + row_slot;
            const int in_row = sorted_row < rows
                                   ? sorted_kv_out_in_map[kv_base + sorted_row]
                                   : -1;
            const int tile_base = row_slot * channels + half_slot * 16;
            if (in_row >= 0) {
                const int feat_base = in_row * channels + half_slot * 16;
#pragma unroll
                for (int chunk = 0; chunk < chunks_per_row / 2; ++chunk) {
                    const int ci = chunk * 4;
                    *reinterpret_cast<threadgroup half4*>(
                        lhs_tile + tile_base + ci
                    ) =
                        *reinterpret_cast<device const half4*>(
                            feats + feat_base + ci
                        );
                }
            } else {
#pragma unroll
                for (int chunk = 0; chunk < chunks_per_row / 2; ++chunk) {
                    const int ci = chunk * 4;
                    *reinterpret_cast<threadgroup half4*>(
                        lhs_tile + tile_base + ci
                    ) = half4(0.0h);
                }
            }
        }

        auto rhs_tensor = tensor(
            weights_kci_co + kv * channels * channels,
            extents<int32_t, channels, channels>(),
            array<int32_t, 2>{1, channels}
        );

        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(lhs_tensor, rhs_tensor, out_tensor);
        threadgroup_barrier(mem_flags::mem_none);
    }

    if (store_sorted != 0 && row_start + tile_rows <= rows) {
        auto device_out = tensor(
            out + row_start * channels,
            extents<int32_t, channels, tile_rows>(),
            array<int32_t, 2>{1, channels}
        );
        out_tensor.store(device_out);
    } else {
#pragma unroll
        for (uint16_t i = 0; i < out_tensor.get_capacity(); ++i) {
            if (!out_tensor.is_valid_element(i)) {
                continue;
            }
            const auto idx = out_tensor.get_multidimensional_index(i);
            const int co = int(idx[0]);
            const int row_slot = int(idx[1]);
            const int sorted_row = row_start + row_slot;
            if (sorted_row < rows) {
                const int out_row =
                    store_sorted != 0 ? sorted_row : reorder_rows[sorted_row];
                out[out_row * channels + co] =
                    original_mask == 0 ? half(0.0h) : out_tensor[i];
            }
        }
    }
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void row_stationary_tensor_coop_devicew_full64_kvmap_f16acc_sorted_c64_m64(
    device const half* feats [[buffer(0)]],
    device half* weights_kci_co [[buffer(1)]],
    device const int* sorted_kv_out_in_map [[buffer(2)]],
    device const int* reorder_rows [[buffer(3)]],
    device const int* tile_masks [[buffer(4)]],
    device half* out [[buffer(5)]],
    constant const int& rows [[buffer(6)]],
    constant const int& store_sorted [[buffer(7)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    constexpr int tile_rows = 64;
    constexpr int channels = 64;
    constexpr int chunks_per_row = channels / 4;

    threadgroup half lhs_tile[tile_rows * channels];

    const int row_start = int(group_id) * tile_rows;
    const int mask_base = int(group_id) * 4;
    uint active_mask = uint(
        tile_masks[mask_base + 0] | tile_masks[mask_base + 1] |
        tile_masks[mask_base + 2] | tile_masks[mask_base + 3]
    );
    const uint original_mask = active_mask;

    constexpr auto desc = matmul2d_descriptor(
        tile_rows,
        channels,
        channels,
        false,
        false,
        true,
        matmul2d_descriptor::mode::multiply_accumulate
    );
    matmul2d<desc, execution_simdgroups<4>> op;
    auto lhs_tensor = tensor<
        threadgroup half,
        extents<int32_t, channels, tile_rows>,
        tensor_inline>(lhs_tile, extents<int32_t, channels, tile_rows>());
    auto out_tensor = op.get_destination_cooperative_tensor<
        decltype(lhs_tensor),
        decltype(tensor(
            weights_kci_co,
            extents<int32_t, channels, channels>(),
            array<int32_t, 2>{1, channels}
        )),
        half>();

#pragma unroll
    for (uint16_t i = 0; i < out_tensor.get_capacity(); ++i) {
        if (out_tensor.is_valid_element(i)) {
            out_tensor[i] = half(0.0h);
        }
    }

    while (active_mask != 0) {
        const int kv = ctz(active_mask);
        active_mask &= active_mask - 1;
        const int kv_base = kv * rows;

        if (tid < tile_rows * 2) {
            const int row_slot = int(tid >> 1);
            const int half_slot = int(tid & 1);
            const int sorted_row = row_start + row_slot;
            const int in_row = sorted_row < rows
                                   ? sorted_kv_out_in_map[kv_base + sorted_row]
                                   : -1;
            const int tile_base = row_slot * channels + half_slot * 32;
            if (in_row >= 0) {
                const int feat_base = in_row * channels + half_slot * 32;
#pragma unroll
                for (int chunk = 0; chunk < chunks_per_row / 2; ++chunk) {
                    const int ci = chunk * 4;
                    *reinterpret_cast<threadgroup half4*>(
                        lhs_tile + tile_base + ci
                    ) =
                        *reinterpret_cast<device const half4*>(
                            feats + feat_base + ci
                        );
                }
            } else {
#pragma unroll
                for (int chunk = 0; chunk < chunks_per_row / 2; ++chunk) {
                    const int ci = chunk * 4;
                    *reinterpret_cast<threadgroup half4*>(
                        lhs_tile + tile_base + ci
                    ) = half4(0.0h);
                }
            }
        }

        auto rhs_tensor = tensor(
            weights_kci_co + kv * channels * channels,
            extents<int32_t, channels, channels>(),
            array<int32_t, 2>{1, channels}
        );

        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(lhs_tensor, rhs_tensor, out_tensor);
        threadgroup_barrier(mem_flags::mem_none);
    }

    if (store_sorted != 0 && row_start + tile_rows <= rows) {
        auto device_out = tensor(
            out + row_start * channels,
            extents<int32_t, channels, tile_rows>(),
            array<int32_t, 2>{1, channels}
        );
        out_tensor.store(device_out);
    } else {
#pragma unroll
        for (uint16_t i = 0; i < out_tensor.get_capacity(); ++i) {
            if (!out_tensor.is_valid_element(i)) {
                continue;
            }
            const auto idx = out_tensor.get_multidimensional_index(i);
            const int co = int(idx[0]);
            const int row_slot = int(idx[1]);
            const int sorted_row = row_start + row_slot;
            if (sorted_row < rows) {
                const int out_row =
                    store_sorted != 0 ? sorted_row : reorder_rows[sorted_row];
                out[out_row * channels + co] =
                    original_mask == 0 ? half(0.0h) : out_tensor[i];
            }
        }
    }
}

template <int in_channels, int out_channels>
[[kernel]] void sparse_relation_conv_f32_i32_cout16_dense(
    device const float* feats [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device float* out [[buffer(7)]],
    constant const int& edge_capacity [[buffer(8)]],
    constant const int& out_capacity [[buffer(9)]],
    constant const int& runtime_in_channels [[buffer(10)]],
    constant const int& runtime_out_channels [[buffer(11)]],
    constant const int& feat_s0 [[buffer(12)]],
    constant const int& feat_s1 [[buffer(13)]],
    constant const int& weight_s0 [[buffer(14)]],
    constant const int& weight_s1 [[buffer(15)]],
    constant const int& weight_s2 [[buffer(16)]],
    constant const int& weight_s3 [[buffer(17)]],
    constant const int& weight_s4 [[buffer(18)]],
    constant const int& weight_layout [[buffer(19)]],
    constant const int& kernel_x [[buffer(20)]],
    constant const int& kernel_y [[buffer(21)]],
    constant const int& kernel_z [[buffer(22)]],
    uint elem [[thread_position_in_grid]]
) {
    dense_forward_cout16_impl<float, in_channels, out_channels>(
        feats,
        weights,
        in_rows,
        kernel_ids,
        counts,
        row_offsets,
        out,
        DenseForwardParams{
            edge_capacity, out_capacity, feat_s0, feat_s1, weight_s0
        },
        elem
    );
}

[[kernel]] void sparse_relation_conv_f32_i32_cout16_dense_cin64_cout64(
    device const float* feats [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device float* out [[buffer(7)]],
    constant const int& edge_capacity [[buffer(8)]],
    constant const int& out_capacity [[buffer(9)]],
    constant const int& runtime_in_channels [[buffer(10)]],
    constant const int& runtime_out_channels [[buffer(11)]],
    constant const int& feat_s0 [[buffer(12)]],
    constant const int& feat_s1 [[buffer(13)]],
    constant const int& weight_s0 [[buffer(14)]],
    constant const int& weight_s1 [[buffer(15)]],
    constant const int& weight_s2 [[buffer(16)]],
    constant const int& weight_s3 [[buffer(17)]],
    constant const int& weight_s4 [[buffer(18)]],
    constant const int& weight_layout [[buffer(19)]],
    constant const int& kernel_x [[buffer(20)]],
    constant const int& kernel_y [[buffer(21)]],
    constant const int& kernel_z [[buffer(22)]],
    uint elem [[thread_position_in_grid]]
) {
    dense_forward_cout16_ci4_f32_impl<64, 64>(
        feats,
        weights,
        in_rows,
        kernel_ids,
        counts,
        row_offsets,
        out,
        DenseForwardParams{
            edge_capacity, out_capacity, feat_s0, feat_s1, weight_s0
        },
        elem
    );
}

template <int in_channels, int out_channels>
[[kernel]] void sparse_relation_conv_f32_i32_cout16_dense_ci4(
    device const float* feats [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device float* out [[buffer(7)]],
    constant const int& edge_capacity [[buffer(8)]],
    constant const int& out_capacity [[buffer(9)]],
    constant const int& runtime_in_channels [[buffer(10)]],
    constant const int& runtime_out_channels [[buffer(11)]],
    constant const int& feat_s0 [[buffer(12)]],
    constant const int& feat_s1 [[buffer(13)]],
    constant const int& weight_s0 [[buffer(14)]],
    constant const int& weight_s1 [[buffer(15)]],
    constant const int& weight_s2 [[buffer(16)]],
    constant const int& weight_s3 [[buffer(17)]],
    constant const int& weight_s4 [[buffer(18)]],
    constant const int& weight_layout [[buffer(19)]],
    constant const int& kernel_x [[buffer(20)]],
    constant const int& kernel_y [[buffer(21)]],
    constant const int& kernel_z [[buffer(22)]],
    uint elem [[thread_position_in_grid]]
) {
    dense_forward_cout16_ci4_f32_impl<in_channels, out_channels>(
        feats,
        weights,
        in_rows,
        kernel_ids,
        counts,
        row_offsets,
        out,
        DenseForwardParams{
            edge_capacity, out_capacity, feat_s0, feat_s1, weight_s0
        },
        elem
    );
}

template <int in_channels, int out_channels>
[[kernel]] void sparse_relation_conv_f16_i32_cout16_dense(
    device const half* feats [[buffer(0)]],
    device const half* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device half* out [[buffer(7)]],
    constant const int& edge_capacity [[buffer(8)]],
    constant const int& out_capacity [[buffer(9)]],
    constant const int& runtime_in_channels [[buffer(10)]],
    constant const int& runtime_out_channels [[buffer(11)]],
    constant const int& feat_s0 [[buffer(12)]],
    constant const int& feat_s1 [[buffer(13)]],
    constant const int& weight_s0 [[buffer(14)]],
    constant const int& weight_s1 [[buffer(15)]],
    constant const int& weight_s2 [[buffer(16)]],
    constant const int& weight_s3 [[buffer(17)]],
    constant const int& weight_s4 [[buffer(18)]],
    constant const int& weight_layout [[buffer(19)]],
    constant const int& kernel_x [[buffer(20)]],
    constant const int& kernel_y [[buffer(21)]],
    constant const int& kernel_z [[buffer(22)]],
    uint elem [[thread_position_in_grid]]
) {
    dense_forward_cout16_ci4_f16_impl<in_channels, out_channels>(
        feats,
        weights,
        in_rows,
        kernel_ids,
        counts,
        row_offsets,
        out,
        DenseForwardParams{
            edge_capacity, out_capacity, feat_s0, feat_s1, weight_s0
        },
        elem
    );
}

[[kernel]] void
sparse_relation_conv_f16_i32_cout16_dense_contiguous_cin64_cout64(
    device const half* feats [[buffer(0)]],
    device const half* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device half* out [[buffer(7)]],
    constant const int& edge_capacity [[buffer(8)]],
    constant const int& out_capacity [[buffer(9)]],
    constant const int& runtime_in_channels [[buffer(10)]],
    constant const int& runtime_out_channels [[buffer(11)]],
    constant const int& feat_s0 [[buffer(12)]],
    constant const int& feat_s1 [[buffer(13)]],
    constant const int& weight_s0 [[buffer(14)]],
    constant const int& weight_s1 [[buffer(15)]],
    constant const int& weight_s2 [[buffer(16)]],
    constant const int& weight_s3 [[buffer(17)]],
    constant const int& weight_s4 [[buffer(18)]],
    constant const int& weight_layout [[buffer(19)]],
    constant const int& kernel_x [[buffer(20)]],
    constant const int& kernel_y [[buffer(21)]],
    constant const int& kernel_z [[buffer(22)]],
    uint elem [[thread_position_in_grid]]
) {
    dense_forward_cout16_ci4_f16_contiguous_impl<64, 64>(
        feats,
        weights,
        in_rows,
        kernel_ids,
        counts,
        row_offsets,
        out,
        DenseForwardParams{
            edge_capacity, out_capacity, feat_s0, feat_s1, weight_s0
        },
        elem
    );
}

template [[host_name("sparse_relation_conv_f32_i32_cout16_dense_cin16_cout16")]]
[[kernel]] void
sparse_relation_conv_f32_i32_cout16_dense<16, 16>(
    device const float* feats [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device float* out [[buffer(7)]],
    constant const int& edge_capacity [[buffer(8)]],
    constant const int& out_capacity [[buffer(9)]],
    constant const int& runtime_in_channels [[buffer(10)]],
    constant const int& runtime_out_channels [[buffer(11)]],
    constant const int& feat_s0 [[buffer(12)]],
    constant const int& feat_s1 [[buffer(13)]],
    constant const int& weight_s0 [[buffer(14)]],
    constant const int& weight_s1 [[buffer(15)]],
    constant const int& weight_s2 [[buffer(16)]],
    constant const int& weight_s3 [[buffer(17)]],
    constant const int& weight_s4 [[buffer(18)]],
    constant const int& weight_layout [[buffer(19)]],
    constant const int& kernel_x [[buffer(20)]],
    constant const int& kernel_y [[buffer(21)]],
    constant const int& kernel_z [[buffer(22)]],
    uint elem [[thread_position_in_grid]]
);

template
    [[host_name("sparse_relation_conv_f32_i32_cout16_dense_ci4_cin16_cout16")]]
    [[kernel]] void
    sparse_relation_conv_f32_i32_cout16_dense_ci4<16, 16>(
        device const float* feats [[buffer(0)]],
        device const float* weights [[buffer(1)]],
        device const int* in_rows [[buffer(2)]],
        device const int* out_rows [[buffer(3)]],
        device const int* kernel_ids [[buffer(4)]],
        device const int* counts [[buffer(5)]],
        device const int* row_offsets [[buffer(6)]],
        device float* out [[buffer(7)]],
        constant const int& edge_capacity [[buffer(8)]],
        constant const int& out_capacity [[buffer(9)]],
        constant const int& runtime_in_channels [[buffer(10)]],
        constant const int& runtime_out_channels [[buffer(11)]],
        constant const int& feat_s0 [[buffer(12)]],
        constant const int& feat_s1 [[buffer(13)]],
        constant const int& weight_s0 [[buffer(14)]],
        constant const int& weight_s1 [[buffer(15)]],
        constant const int& weight_s2 [[buffer(16)]],
        constant const int& weight_s3 [[buffer(17)]],
        constant const int& weight_s4 [[buffer(18)]],
        constant const int& weight_layout [[buffer(19)]],
        constant const int& kernel_x [[buffer(20)]],
        constant const int& kernel_y [[buffer(21)]],
        constant const int& kernel_z [[buffer(22)]],
        uint elem [[thread_position_in_grid]]
    );

template [[host_name("sparse_relation_conv_f32_i32_cout16_dense_cin16_cout32")]]
[[kernel]] void
sparse_relation_conv_f32_i32_cout16_dense<16, 32>(
    device const float* feats [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device float* out [[buffer(7)]],
    constant const int& edge_capacity [[buffer(8)]],
    constant const int& out_capacity [[buffer(9)]],
    constant const int& runtime_in_channels [[buffer(10)]],
    constant const int& runtime_out_channels [[buffer(11)]],
    constant const int& feat_s0 [[buffer(12)]],
    constant const int& feat_s1 [[buffer(13)]],
    constant const int& weight_s0 [[buffer(14)]],
    constant const int& weight_s1 [[buffer(15)]],
    constant const int& weight_s2 [[buffer(16)]],
    constant const int& weight_s3 [[buffer(17)]],
    constant const int& weight_s4 [[buffer(18)]],
    constant const int& weight_layout [[buffer(19)]],
    constant const int& kernel_x [[buffer(20)]],
    constant const int& kernel_y [[buffer(21)]],
    constant const int& kernel_z [[buffer(22)]],
    uint elem [[thread_position_in_grid]]
);

template
    [[host_name("sparse_relation_conv_f32_i32_cout16_dense_ci4_cin16_cout32")]]
    [[kernel]] void
    sparse_relation_conv_f32_i32_cout16_dense_ci4<16, 32>(
        device const float* feats [[buffer(0)]],
        device const float* weights [[buffer(1)]],
        device const int* in_rows [[buffer(2)]],
        device const int* out_rows [[buffer(3)]],
        device const int* kernel_ids [[buffer(4)]],
        device const int* counts [[buffer(5)]],
        device const int* row_offsets [[buffer(6)]],
        device float* out [[buffer(7)]],
        constant const int& edge_capacity [[buffer(8)]],
        constant const int& out_capacity [[buffer(9)]],
        constant const int& runtime_in_channels [[buffer(10)]],
        constant const int& runtime_out_channels [[buffer(11)]],
        constant const int& feat_s0 [[buffer(12)]],
        constant const int& feat_s1 [[buffer(13)]],
        constant const int& weight_s0 [[buffer(14)]],
        constant const int& weight_s1 [[buffer(15)]],
        constant const int& weight_s2 [[buffer(16)]],
        constant const int& weight_s3 [[buffer(17)]],
        constant const int& weight_s4 [[buffer(18)]],
        constant const int& weight_layout [[buffer(19)]],
        constant const int& kernel_x [[buffer(20)]],
        constant const int& kernel_y [[buffer(21)]],
        constant const int& kernel_z [[buffer(22)]],
        uint elem [[thread_position_in_grid]]
    );

template [[host_name("sparse_relation_conv_f32_i32_cout16_dense_cin16_cout64")]]
[[kernel]] void
sparse_relation_conv_f32_i32_cout16_dense<16, 64>(
    device const float* feats [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device float* out [[buffer(7)]],
    constant const int& edge_capacity [[buffer(8)]],
    constant const int& out_capacity [[buffer(9)]],
    constant const int& runtime_in_channels [[buffer(10)]],
    constant const int& runtime_out_channels [[buffer(11)]],
    constant const int& feat_s0 [[buffer(12)]],
    constant const int& feat_s1 [[buffer(13)]],
    constant const int& weight_s0 [[buffer(14)]],
    constant const int& weight_s1 [[buffer(15)]],
    constant const int& weight_s2 [[buffer(16)]],
    constant const int& weight_s3 [[buffer(17)]],
    constant const int& weight_s4 [[buffer(18)]],
    constant const int& weight_layout [[buffer(19)]],
    constant const int& kernel_x [[buffer(20)]],
    constant const int& kernel_y [[buffer(21)]],
    constant const int& kernel_z [[buffer(22)]],
    uint elem [[thread_position_in_grid]]
);

template
    [[host_name("sparse_relation_conv_f32_i32_cout16_dense_ci4_cin16_cout64")]]
    [[kernel]] void
    sparse_relation_conv_f32_i32_cout16_dense_ci4<16, 64>(
        device const float* feats [[buffer(0)]],
        device const float* weights [[buffer(1)]],
        device const int* in_rows [[buffer(2)]],
        device const int* out_rows [[buffer(3)]],
        device const int* kernel_ids [[buffer(4)]],
        device const int* counts [[buffer(5)]],
        device const int* row_offsets [[buffer(6)]],
        device float* out [[buffer(7)]],
        constant const int& edge_capacity [[buffer(8)]],
        constant const int& out_capacity [[buffer(9)]],
        constant const int& runtime_in_channels [[buffer(10)]],
        constant const int& runtime_out_channels [[buffer(11)]],
        constant const int& feat_s0 [[buffer(12)]],
        constant const int& feat_s1 [[buffer(13)]],
        constant const int& weight_s0 [[buffer(14)]],
        constant const int& weight_s1 [[buffer(15)]],
        constant const int& weight_s2 [[buffer(16)]],
        constant const int& weight_s3 [[buffer(17)]],
        constant const int& weight_s4 [[buffer(18)]],
        constant const int& weight_layout [[buffer(19)]],
        constant const int& kernel_x [[buffer(20)]],
        constant const int& kernel_y [[buffer(21)]],
        constant const int& kernel_z [[buffer(22)]],
        uint elem [[thread_position_in_grid]]
    );

template [[host_name("sparse_relation_conv_f32_i32_cout16_dense_cin32_cout16")]]
[[kernel]] void
sparse_relation_conv_f32_i32_cout16_dense<32, 16>(
    device const float* feats [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device float* out [[buffer(7)]],
    constant const int& edge_capacity [[buffer(8)]],
    constant const int& out_capacity [[buffer(9)]],
    constant const int& runtime_in_channels [[buffer(10)]],
    constant const int& runtime_out_channels [[buffer(11)]],
    constant const int& feat_s0 [[buffer(12)]],
    constant const int& feat_s1 [[buffer(13)]],
    constant const int& weight_s0 [[buffer(14)]],
    constant const int& weight_s1 [[buffer(15)]],
    constant const int& weight_s2 [[buffer(16)]],
    constant const int& weight_s3 [[buffer(17)]],
    constant const int& weight_s4 [[buffer(18)]],
    constant const int& weight_layout [[buffer(19)]],
    constant const int& kernel_x [[buffer(20)]],
    constant const int& kernel_y [[buffer(21)]],
    constant const int& kernel_z [[buffer(22)]],
    uint elem [[thread_position_in_grid]]
);

template
    [[host_name("sparse_relation_conv_f32_i32_cout16_dense_ci4_cin32_cout16")]]
    [[kernel]] void
    sparse_relation_conv_f32_i32_cout16_dense_ci4<32, 16>(
        device const float* feats [[buffer(0)]],
        device const float* weights [[buffer(1)]],
        device const int* in_rows [[buffer(2)]],
        device const int* out_rows [[buffer(3)]],
        device const int* kernel_ids [[buffer(4)]],
        device const int* counts [[buffer(5)]],
        device const int* row_offsets [[buffer(6)]],
        device float* out [[buffer(7)]],
        constant const int& edge_capacity [[buffer(8)]],
        constant const int& out_capacity [[buffer(9)]],
        constant const int& runtime_in_channels [[buffer(10)]],
        constant const int& runtime_out_channels [[buffer(11)]],
        constant const int& feat_s0 [[buffer(12)]],
        constant const int& feat_s1 [[buffer(13)]],
        constant const int& weight_s0 [[buffer(14)]],
        constant const int& weight_s1 [[buffer(15)]],
        constant const int& weight_s2 [[buffer(16)]],
        constant const int& weight_s3 [[buffer(17)]],
        constant const int& weight_s4 [[buffer(18)]],
        constant const int& weight_layout [[buffer(19)]],
        constant const int& kernel_x [[buffer(20)]],
        constant const int& kernel_y [[buffer(21)]],
        constant const int& kernel_z [[buffer(22)]],
        uint elem [[thread_position_in_grid]]
    );

template [[host_name("sparse_relation_conv_f32_i32_cout16_dense_cin32_cout32")]]
[[kernel]] void
sparse_relation_conv_f32_i32_cout16_dense<32, 32>(
    device const float* feats [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device float* out [[buffer(7)]],
    constant const int& edge_capacity [[buffer(8)]],
    constant const int& out_capacity [[buffer(9)]],
    constant const int& runtime_in_channels [[buffer(10)]],
    constant const int& runtime_out_channels [[buffer(11)]],
    constant const int& feat_s0 [[buffer(12)]],
    constant const int& feat_s1 [[buffer(13)]],
    constant const int& weight_s0 [[buffer(14)]],
    constant const int& weight_s1 [[buffer(15)]],
    constant const int& weight_s2 [[buffer(16)]],
    constant const int& weight_s3 [[buffer(17)]],
    constant const int& weight_s4 [[buffer(18)]],
    constant const int& weight_layout [[buffer(19)]],
    constant const int& kernel_x [[buffer(20)]],
    constant const int& kernel_y [[buffer(21)]],
    constant const int& kernel_z [[buffer(22)]],
    uint elem [[thread_position_in_grid]]
);

template
    [[host_name("sparse_relation_conv_f32_i32_cout16_dense_ci4_cin32_cout32")]]
    [[kernel]] void
    sparse_relation_conv_f32_i32_cout16_dense_ci4<32, 32>(
        device const float* feats [[buffer(0)]],
        device const float* weights [[buffer(1)]],
        device const int* in_rows [[buffer(2)]],
        device const int* out_rows [[buffer(3)]],
        device const int* kernel_ids [[buffer(4)]],
        device const int* counts [[buffer(5)]],
        device const int* row_offsets [[buffer(6)]],
        device float* out [[buffer(7)]],
        constant const int& edge_capacity [[buffer(8)]],
        constant const int& out_capacity [[buffer(9)]],
        constant const int& runtime_in_channels [[buffer(10)]],
        constant const int& runtime_out_channels [[buffer(11)]],
        constant const int& feat_s0 [[buffer(12)]],
        constant const int& feat_s1 [[buffer(13)]],
        constant const int& weight_s0 [[buffer(14)]],
        constant const int& weight_s1 [[buffer(15)]],
        constant const int& weight_s2 [[buffer(16)]],
        constant const int& weight_s3 [[buffer(17)]],
        constant const int& weight_s4 [[buffer(18)]],
        constant const int& weight_layout [[buffer(19)]],
        constant const int& kernel_x [[buffer(20)]],
        constant const int& kernel_y [[buffer(21)]],
        constant const int& kernel_z [[buffer(22)]],
        uint elem [[thread_position_in_grid]]
    );

template [[host_name("sparse_relation_conv_f32_i32_cout16_dense_cin32_cout64")]]
[[kernel]] void
sparse_relation_conv_f32_i32_cout16_dense<32, 64>(
    device const float* feats [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device float* out [[buffer(7)]],
    constant const int& edge_capacity [[buffer(8)]],
    constant const int& out_capacity [[buffer(9)]],
    constant const int& runtime_in_channels [[buffer(10)]],
    constant const int& runtime_out_channels [[buffer(11)]],
    constant const int& feat_s0 [[buffer(12)]],
    constant const int& feat_s1 [[buffer(13)]],
    constant const int& weight_s0 [[buffer(14)]],
    constant const int& weight_s1 [[buffer(15)]],
    constant const int& weight_s2 [[buffer(16)]],
    constant const int& weight_s3 [[buffer(17)]],
    constant const int& weight_s4 [[buffer(18)]],
    constant const int& weight_layout [[buffer(19)]],
    constant const int& kernel_x [[buffer(20)]],
    constant const int& kernel_y [[buffer(21)]],
    constant const int& kernel_z [[buffer(22)]],
    uint elem [[thread_position_in_grid]]
);

template
    [[host_name("sparse_relation_conv_f32_i32_cout16_dense_ci4_cin32_cout64")]]
    [[kernel]] void
    sparse_relation_conv_f32_i32_cout16_dense_ci4<32, 64>(
        device const float* feats [[buffer(0)]],
        device const float* weights [[buffer(1)]],
        device const int* in_rows [[buffer(2)]],
        device const int* out_rows [[buffer(3)]],
        device const int* kernel_ids [[buffer(4)]],
        device const int* counts [[buffer(5)]],
        device const int* row_offsets [[buffer(6)]],
        device float* out [[buffer(7)]],
        constant const int& edge_capacity [[buffer(8)]],
        constant const int& out_capacity [[buffer(9)]],
        constant const int& runtime_in_channels [[buffer(10)]],
        constant const int& runtime_out_channels [[buffer(11)]],
        constant const int& feat_s0 [[buffer(12)]],
        constant const int& feat_s1 [[buffer(13)]],
        constant const int& weight_s0 [[buffer(14)]],
        constant const int& weight_s1 [[buffer(15)]],
        constant const int& weight_s2 [[buffer(16)]],
        constant const int& weight_s3 [[buffer(17)]],
        constant const int& weight_s4 [[buffer(18)]],
        constant const int& weight_layout [[buffer(19)]],
        constant const int& kernel_x [[buffer(20)]],
        constant const int& kernel_y [[buffer(21)]],
        constant const int& kernel_z [[buffer(22)]],
        uint elem [[thread_position_in_grid]]
    );

template [[host_name("sparse_relation_conv_f32_i32_cout16_dense_cin64_cout16")]]
[[kernel]] void
sparse_relation_conv_f32_i32_cout16_dense<64, 16>(
    device const float* feats [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device float* out [[buffer(7)]],
    constant const int& edge_capacity [[buffer(8)]],
    constant const int& out_capacity [[buffer(9)]],
    constant const int& runtime_in_channels [[buffer(10)]],
    constant const int& runtime_out_channels [[buffer(11)]],
    constant const int& feat_s0 [[buffer(12)]],
    constant const int& feat_s1 [[buffer(13)]],
    constant const int& weight_s0 [[buffer(14)]],
    constant const int& weight_s1 [[buffer(15)]],
    constant const int& weight_s2 [[buffer(16)]],
    constant const int& weight_s3 [[buffer(17)]],
    constant const int& weight_s4 [[buffer(18)]],
    constant const int& weight_layout [[buffer(19)]],
    constant const int& kernel_x [[buffer(20)]],
    constant const int& kernel_y [[buffer(21)]],
    constant const int& kernel_z [[buffer(22)]],
    uint elem [[thread_position_in_grid]]
);

template
    [[host_name("sparse_relation_conv_f32_i32_cout16_dense_ci4_cin64_cout16")]]
    [[kernel]] void
    sparse_relation_conv_f32_i32_cout16_dense_ci4<64, 16>(
        device const float* feats [[buffer(0)]],
        device const float* weights [[buffer(1)]],
        device const int* in_rows [[buffer(2)]],
        device const int* out_rows [[buffer(3)]],
        device const int* kernel_ids [[buffer(4)]],
        device const int* counts [[buffer(5)]],
        device const int* row_offsets [[buffer(6)]],
        device float* out [[buffer(7)]],
        constant const int& edge_capacity [[buffer(8)]],
        constant const int& out_capacity [[buffer(9)]],
        constant const int& runtime_in_channels [[buffer(10)]],
        constant const int& runtime_out_channels [[buffer(11)]],
        constant const int& feat_s0 [[buffer(12)]],
        constant const int& feat_s1 [[buffer(13)]],
        constant const int& weight_s0 [[buffer(14)]],
        constant const int& weight_s1 [[buffer(15)]],
        constant const int& weight_s2 [[buffer(16)]],
        constant const int& weight_s3 [[buffer(17)]],
        constant const int& weight_s4 [[buffer(18)]],
        constant const int& weight_layout [[buffer(19)]],
        constant const int& kernel_x [[buffer(20)]],
        constant const int& kernel_y [[buffer(21)]],
        constant const int& kernel_z [[buffer(22)]],
        uint elem [[thread_position_in_grid]]
    );

template [[host_name("sparse_relation_conv_f32_i32_cout16_dense_cin64_cout32")]]
[[kernel]] void
sparse_relation_conv_f32_i32_cout16_dense<64, 32>(
    device const float* feats [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device float* out [[buffer(7)]],
    constant const int& edge_capacity [[buffer(8)]],
    constant const int& out_capacity [[buffer(9)]],
    constant const int& runtime_in_channels [[buffer(10)]],
    constant const int& runtime_out_channels [[buffer(11)]],
    constant const int& feat_s0 [[buffer(12)]],
    constant const int& feat_s1 [[buffer(13)]],
    constant const int& weight_s0 [[buffer(14)]],
    constant const int& weight_s1 [[buffer(15)]],
    constant const int& weight_s2 [[buffer(16)]],
    constant const int& weight_s3 [[buffer(17)]],
    constant const int& weight_s4 [[buffer(18)]],
    constant const int& weight_layout [[buffer(19)]],
    constant const int& kernel_x [[buffer(20)]],
    constant const int& kernel_y [[buffer(21)]],
    constant const int& kernel_z [[buffer(22)]],
    uint elem [[thread_position_in_grid]]
);

template
    [[host_name("sparse_relation_conv_f32_i32_cout16_dense_ci4_cin64_cout32")]]
    [[kernel]] void
    sparse_relation_conv_f32_i32_cout16_dense_ci4<64, 32>(
        device const float* feats [[buffer(0)]],
        device const float* weights [[buffer(1)]],
        device const int* in_rows [[buffer(2)]],
        device const int* out_rows [[buffer(3)]],
        device const int* kernel_ids [[buffer(4)]],
        device const int* counts [[buffer(5)]],
        device const int* row_offsets [[buffer(6)]],
        device float* out [[buffer(7)]],
        constant const int& edge_capacity [[buffer(8)]],
        constant const int& out_capacity [[buffer(9)]],
        constant const int& runtime_in_channels [[buffer(10)]],
        constant const int& runtime_out_channels [[buffer(11)]],
        constant const int& feat_s0 [[buffer(12)]],
        constant const int& feat_s1 [[buffer(13)]],
        constant const int& weight_s0 [[buffer(14)]],
        constant const int& weight_s1 [[buffer(15)]],
        constant const int& weight_s2 [[buffer(16)]],
        constant const int& weight_s3 [[buffer(17)]],
        constant const int& weight_s4 [[buffer(18)]],
        constant const int& weight_layout [[buffer(19)]],
        constant const int& kernel_x [[buffer(20)]],
        constant const int& kernel_y [[buffer(21)]],
        constant const int& kernel_z [[buffer(22)]],
        uint elem [[thread_position_in_grid]]
    );

template [[host_name("sparse_relation_conv_f16_i32_cout16_dense_cin16_cout16")]]
[[kernel]] void
sparse_relation_conv_f16_i32_cout16_dense<16, 16>(
    device const half* feats [[buffer(0)]],
    device const half* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device half* out [[buffer(7)]],
    constant const int& edge_capacity [[buffer(8)]],
    constant const int& out_capacity [[buffer(9)]],
    constant const int& runtime_in_channels [[buffer(10)]],
    constant const int& runtime_out_channels [[buffer(11)]],
    constant const int& feat_s0 [[buffer(12)]],
    constant const int& feat_s1 [[buffer(13)]],
    constant const int& weight_s0 [[buffer(14)]],
    constant const int& weight_s1 [[buffer(15)]],
    constant const int& weight_s2 [[buffer(16)]],
    constant const int& weight_s3 [[buffer(17)]],
    constant const int& weight_s4 [[buffer(18)]],
    constant const int& weight_layout [[buffer(19)]],
    constant const int& kernel_x [[buffer(20)]],
    constant const int& kernel_y [[buffer(21)]],
    constant const int& kernel_z [[buffer(22)]],
    uint elem [[thread_position_in_grid]]
);

template [[host_name("sparse_relation_conv_f16_i32_cout16_dense_cin16_cout32")]]
[[kernel]] void
sparse_relation_conv_f16_i32_cout16_dense<16, 32>(
    device const half* feats [[buffer(0)]],
    device const half* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device half* out [[buffer(7)]],
    constant const int& edge_capacity [[buffer(8)]],
    constant const int& out_capacity [[buffer(9)]],
    constant const int& runtime_in_channels [[buffer(10)]],
    constant const int& runtime_out_channels [[buffer(11)]],
    constant const int& feat_s0 [[buffer(12)]],
    constant const int& feat_s1 [[buffer(13)]],
    constant const int& weight_s0 [[buffer(14)]],
    constant const int& weight_s1 [[buffer(15)]],
    constant const int& weight_s2 [[buffer(16)]],
    constant const int& weight_s3 [[buffer(17)]],
    constant const int& weight_s4 [[buffer(18)]],
    constant const int& weight_layout [[buffer(19)]],
    constant const int& kernel_x [[buffer(20)]],
    constant const int& kernel_y [[buffer(21)]],
    constant const int& kernel_z [[buffer(22)]],
    uint elem [[thread_position_in_grid]]
);

template [[host_name("sparse_relation_conv_f16_i32_cout16_dense_cin16_cout64")]]
[[kernel]] void
sparse_relation_conv_f16_i32_cout16_dense<16, 64>(
    device const half* feats [[buffer(0)]],
    device const half* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device half* out [[buffer(7)]],
    constant const int& edge_capacity [[buffer(8)]],
    constant const int& out_capacity [[buffer(9)]],
    constant const int& runtime_in_channels [[buffer(10)]],
    constant const int& runtime_out_channels [[buffer(11)]],
    constant const int& feat_s0 [[buffer(12)]],
    constant const int& feat_s1 [[buffer(13)]],
    constant const int& weight_s0 [[buffer(14)]],
    constant const int& weight_s1 [[buffer(15)]],
    constant const int& weight_s2 [[buffer(16)]],
    constant const int& weight_s3 [[buffer(17)]],
    constant const int& weight_s4 [[buffer(18)]],
    constant const int& weight_layout [[buffer(19)]],
    constant const int& kernel_x [[buffer(20)]],
    constant const int& kernel_y [[buffer(21)]],
    constant const int& kernel_z [[buffer(22)]],
    uint elem [[thread_position_in_grid]]
);

template [[host_name("sparse_relation_conv_f16_i32_cout16_dense_cin32_cout16")]]
[[kernel]] void
sparse_relation_conv_f16_i32_cout16_dense<32, 16>(
    device const half* feats [[buffer(0)]],
    device const half* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device half* out [[buffer(7)]],
    constant const int& edge_capacity [[buffer(8)]],
    constant const int& out_capacity [[buffer(9)]],
    constant const int& runtime_in_channels [[buffer(10)]],
    constant const int& runtime_out_channels [[buffer(11)]],
    constant const int& feat_s0 [[buffer(12)]],
    constant const int& feat_s1 [[buffer(13)]],
    constant const int& weight_s0 [[buffer(14)]],
    constant const int& weight_s1 [[buffer(15)]],
    constant const int& weight_s2 [[buffer(16)]],
    constant const int& weight_s3 [[buffer(17)]],
    constant const int& weight_s4 [[buffer(18)]],
    constant const int& weight_layout [[buffer(19)]],
    constant const int& kernel_x [[buffer(20)]],
    constant const int& kernel_y [[buffer(21)]],
    constant const int& kernel_z [[buffer(22)]],
    uint elem [[thread_position_in_grid]]
);

template [[host_name("sparse_relation_conv_f16_i32_cout16_dense_cin32_cout32")]]
[[kernel]] void
sparse_relation_conv_f16_i32_cout16_dense<32, 32>(
    device const half* feats [[buffer(0)]],
    device const half* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device half* out [[buffer(7)]],
    constant const int& edge_capacity [[buffer(8)]],
    constant const int& out_capacity [[buffer(9)]],
    constant const int& runtime_in_channels [[buffer(10)]],
    constant const int& runtime_out_channels [[buffer(11)]],
    constant const int& feat_s0 [[buffer(12)]],
    constant const int& feat_s1 [[buffer(13)]],
    constant const int& weight_s0 [[buffer(14)]],
    constant const int& weight_s1 [[buffer(15)]],
    constant const int& weight_s2 [[buffer(16)]],
    constant const int& weight_s3 [[buffer(17)]],
    constant const int& weight_s4 [[buffer(18)]],
    constant const int& weight_layout [[buffer(19)]],
    constant const int& kernel_x [[buffer(20)]],
    constant const int& kernel_y [[buffer(21)]],
    constant const int& kernel_z [[buffer(22)]],
    uint elem [[thread_position_in_grid]]
);

template [[host_name("sparse_relation_conv_f16_i32_cout16_dense_cin32_cout64")]]
[[kernel]] void
sparse_relation_conv_f16_i32_cout16_dense<32, 64>(
    device const half* feats [[buffer(0)]],
    device const half* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device half* out [[buffer(7)]],
    constant const int& edge_capacity [[buffer(8)]],
    constant const int& out_capacity [[buffer(9)]],
    constant const int& runtime_in_channels [[buffer(10)]],
    constant const int& runtime_out_channels [[buffer(11)]],
    constant const int& feat_s0 [[buffer(12)]],
    constant const int& feat_s1 [[buffer(13)]],
    constant const int& weight_s0 [[buffer(14)]],
    constant const int& weight_s1 [[buffer(15)]],
    constant const int& weight_s2 [[buffer(16)]],
    constant const int& weight_s3 [[buffer(17)]],
    constant const int& weight_s4 [[buffer(18)]],
    constant const int& weight_layout [[buffer(19)]],
    constant const int& kernel_x [[buffer(20)]],
    constant const int& kernel_y [[buffer(21)]],
    constant const int& kernel_z [[buffer(22)]],
    uint elem [[thread_position_in_grid]]
);

template [[host_name("sparse_relation_conv_f16_i32_cout16_dense_cin64_cout16")]]
[[kernel]] void
sparse_relation_conv_f16_i32_cout16_dense<64, 16>(
    device const half* feats [[buffer(0)]],
    device const half* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device half* out [[buffer(7)]],
    constant const int& edge_capacity [[buffer(8)]],
    constant const int& out_capacity [[buffer(9)]],
    constant const int& runtime_in_channels [[buffer(10)]],
    constant const int& runtime_out_channels [[buffer(11)]],
    constant const int& feat_s0 [[buffer(12)]],
    constant const int& feat_s1 [[buffer(13)]],
    constant const int& weight_s0 [[buffer(14)]],
    constant const int& weight_s1 [[buffer(15)]],
    constant const int& weight_s2 [[buffer(16)]],
    constant const int& weight_s3 [[buffer(17)]],
    constant const int& weight_s4 [[buffer(18)]],
    constant const int& weight_layout [[buffer(19)]],
    constant const int& kernel_x [[buffer(20)]],
    constant const int& kernel_y [[buffer(21)]],
    constant const int& kernel_z [[buffer(22)]],
    uint elem [[thread_position_in_grid]]
);

template [[host_name("sparse_relation_conv_f16_i32_cout16_dense_cin64_cout32")]]
[[kernel]] void
sparse_relation_conv_f16_i32_cout16_dense<64, 32>(
    device const half* feats [[buffer(0)]],
    device const half* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device half* out [[buffer(7)]],
    constant const int& edge_capacity [[buffer(8)]],
    constant const int& out_capacity [[buffer(9)]],
    constant const int& runtime_in_channels [[buffer(10)]],
    constant const int& runtime_out_channels [[buffer(11)]],
    constant const int& feat_s0 [[buffer(12)]],
    constant const int& feat_s1 [[buffer(13)]],
    constant const int& weight_s0 [[buffer(14)]],
    constant const int& weight_s1 [[buffer(15)]],
    constant const int& weight_s2 [[buffer(16)]],
    constant const int& weight_s3 [[buffer(17)]],
    constant const int& weight_s4 [[buffer(18)]],
    constant const int& weight_layout [[buffer(19)]],
    constant const int& kernel_x [[buffer(20)]],
    constant const int& kernel_y [[buffer(21)]],
    constant const int& kernel_z [[buffer(22)]],
    uint elem [[thread_position_in_grid]]
);

template [[host_name("sparse_relation_conv_f16_i32_cout16_dense_cin64_cout64")]]
[[kernel]] void
sparse_relation_conv_f16_i32_cout16_dense<64, 64>(
    device const half* feats [[buffer(0)]],
    device const half* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device half* out [[buffer(7)]],
    constant const int& edge_capacity [[buffer(8)]],
    constant const int& out_capacity [[buffer(9)]],
    constant const int& runtime_in_channels [[buffer(10)]],
    constant const int& runtime_out_channels [[buffer(11)]],
    constant const int& feat_s0 [[buffer(12)]],
    constant const int& feat_s1 [[buffer(13)]],
    constant const int& weight_s0 [[buffer(14)]],
    constant const int& weight_s1 [[buffer(15)]],
    constant const int& weight_s2 [[buffer(16)]],
    constant const int& weight_s3 [[buffer(17)]],
    constant const int& weight_s4 [[buffer(18)]],
    constant const int& weight_layout [[buffer(19)]],
    constant const int& kernel_x [[buffer(20)]],
    constant const int& kernel_y [[buffer(21)]],
    constant const int& kernel_z [[buffer(22)]],
    uint elem [[thread_position_in_grid]]
);

template <typename T, int in_channels, int out_channels>
[[kernel]] void sparse_relation_conv_input_grad_cin16_dense(
    device const T* cotangent [[buffer(0)]],
    device const T* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device const int* in_row_offsets [[buffer(7)]],
    device const int* in_edge_ids [[buffer(8)]],
    device T* grad [[buffer(9)]],
    constant const int& edge_capacity [[buffer(10)]],
    constant const int& out_capacity [[buffer(11)]],
    constant const int& in_capacity [[buffer(12)]],
    constant const int& runtime_in_channels [[buffer(13)]],
    constant const int& runtime_out_channels [[buffer(14)]],
    constant const int& cotangent_s0 [[buffer(15)]],
    constant const int& cotangent_s1 [[buffer(16)]],
    constant const int& weight_s0 [[buffer(17)]],
    constant const int& weight_s1 [[buffer(18)]],
    constant const int& weight_s2 [[buffer(19)]],
    constant const int& weight_s3 [[buffer(20)]],
    constant const int& weight_s4 [[buffer(21)]],
    constant const int& weight_layout [[buffer(22)]],
    constant const int& kernel_x [[buffer(23)]],
    constant const int& kernel_y [[buffer(24)]],
    constant const int& kernel_z [[buffer(25)]],
    uint elem [[thread_position_in_grid]]
) {
    dense_input_grad_cin16_impl<T, in_channels, out_channels>(
        cotangent,
        weights,
        out_rows,
        kernel_ids,
        counts,
        in_row_offsets,
        in_edge_ids,
        grad,
        DenseInputGradParams{
            edge_capacity,
            out_capacity,
            in_capacity,
            cotangent_s0,
            cotangent_s1,
            weight_s0,
        },
        elem
    );
}

template <typename T, int in_channels, int out_channels>
[[kernel]] void sparse_relation_conv_input_grad_cin16_grouped_dense(
    device const T* cotangent [[buffer(0)]],
    device const T* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device const int* in_row_offsets [[buffer(7)]],
    device const int* in_edge_ids [[buffer(8)]],
    device T* grad [[buffer(9)]],
    constant const int& edge_capacity [[buffer(10)]],
    constant const int& out_capacity [[buffer(11)]],
    constant const int& in_capacity [[buffer(12)]],
    constant const int& runtime_in_channels [[buffer(13)]],
    constant const int& runtime_out_channels [[buffer(14)]],
    constant const int& cotangent_s0 [[buffer(15)]],
    constant const int& cotangent_s1 [[buffer(16)]],
    constant const int& weight_s0 [[buffer(17)]],
    constant const int& weight_s1 [[buffer(18)]],
    constant const int& weight_s2 [[buffer(19)]],
    constant const int& weight_s3 [[buffer(20)]],
    constant const int& weight_s4 [[buffer(21)]],
    constant const int& weight_layout [[buffer(22)]],
    constant const int& kernel_x [[buffer(23)]],
    constant const int& kernel_y [[buffer(24)]],
    constant const int& kernel_z [[buffer(25)]],
    uint elem [[thread_position_in_grid]]
) {
    dense_input_grad_cin16_grouped_impl<T, in_channels, out_channels>(
        cotangent,
        weights,
        out_rows,
        kernel_ids,
        counts,
        in_row_offsets,
        in_edge_ids,
        grad,
        DenseInputGradParams{
            edge_capacity,
            out_capacity,
            in_capacity,
            cotangent_s0,
            cotangent_s1,
            weight_s0,
        },
        elem
    );
}

template <typename T, int in_channels, int out_channels>
[[kernel]] void sparse_relation_conv_input_grad_cin4_dense(
    device const T* cotangent [[buffer(0)]],
    device const T* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device const int* in_row_offsets [[buffer(7)]],
    device const int* in_edge_ids [[buffer(8)]],
    device T* grad [[buffer(9)]],
    constant const int& edge_capacity [[buffer(10)]],
    constant const int& out_capacity [[buffer(11)]],
    constant const int& in_capacity [[buffer(12)]],
    constant const int& runtime_in_channels [[buffer(13)]],
    constant const int& runtime_out_channels [[buffer(14)]],
    constant const int& cotangent_s0 [[buffer(15)]],
    constant const int& cotangent_s1 [[buffer(16)]],
    constant const int& weight_s0 [[buffer(17)]],
    constant const int& weight_s1 [[buffer(18)]],
    constant const int& weight_s2 [[buffer(19)]],
    constant const int& weight_s3 [[buffer(20)]],
    constant const int& weight_s4 [[buffer(21)]],
    constant const int& weight_layout [[buffer(22)]],
    constant const int& kernel_x [[buffer(23)]],
    constant const int& kernel_y [[buffer(24)]],
    constant const int& kernel_z [[buffer(25)]],
    uint elem [[thread_position_in_grid]]
) {
    dense_input_grad_cin4_impl<T, in_channels, out_channels>(
        cotangent,
        weights,
        out_rows,
        kernel_ids,
        counts,
        in_row_offsets,
        in_edge_ids,
        grad,
        DenseInputGradParams{
            edge_capacity,
            out_capacity,
            in_capacity,
            cotangent_s0,
            cotangent_s1,
            weight_s0,
        },
        elem
    );
}

template <typename T, int in_channels, int out_channels>
[[kernel]] void sparse_relation_conv_weight_grad_c4_dense(
    device const T* feats [[buffer(0)]],
    device const T* cotangent [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device const int* kernel_row_offsets [[buffer(7)]],
    device const int* kernel_edge_ids [[buffer(8)]],
    device T* grad [[buffer(9)]],
    constant const int& edge_capacity [[buffer(10)]],
    constant const int& out_capacity [[buffer(11)]],
    constant const int& n_kernels [[buffer(12)]],
    constant const int& runtime_in_channels [[buffer(13)]],
    constant const int& runtime_out_channels [[buffer(14)]],
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
    dense_weight_grad_ci4_co4_impl<T, in_channels, out_channels>(
        feats,
        cotangent,
        in_rows,
        out_rows,
        counts,
        kernel_row_offsets,
        kernel_edge_ids,
        grad,
        edge_capacity,
        out_capacity,
        n_kernels,
        feat_s0,
        feat_s1,
        cotangent_s0,
        cotangent_s1,
        tile_id,
        tid,
        partial
    );
}

template
    [[host_name("sparse_relation_conv_input_grad_f32_i32_cin16_dense_c16")]]
    [[kernel]] void
    sparse_relation_conv_input_grad_cin16_dense<float, 16, 16>(
        device const float* cotangent [[buffer(0)]],
        device const float* weights [[buffer(1)]],
        device const int* in_rows [[buffer(2)]],
        device const int* out_rows [[buffer(3)]],
        device const int* kernel_ids [[buffer(4)]],
        device const int* counts [[buffer(5)]],
        device const int* row_offsets [[buffer(6)]],
        device const int* in_row_offsets [[buffer(7)]],
        device const int* in_edge_ids [[buffer(8)]],
        device float* grad [[buffer(9)]],
        constant const int& edge_capacity [[buffer(10)]],
        constant const int& out_capacity [[buffer(11)]],
        constant const int& in_capacity [[buffer(12)]],
        constant const int& runtime_in_channels [[buffer(13)]],
        constant const int& runtime_out_channels [[buffer(14)]],
        constant const int& cotangent_s0 [[buffer(15)]],
        constant const int& cotangent_s1 [[buffer(16)]],
        constant const int& weight_s0 [[buffer(17)]],
        constant const int& weight_s1 [[buffer(18)]],
        constant const int& weight_s2 [[buffer(19)]],
        constant const int& weight_s3 [[buffer(20)]],
        constant const int& weight_s4 [[buffer(21)]],
        constant const int& weight_layout [[buffer(22)]],
        constant const int& kernel_x [[buffer(23)]],
        constant const int& kernel_y [[buffer(24)]],
        constant const int& kernel_z [[buffer(25)]],
        uint elem [[thread_position_in_grid]]
    );

template
    [[host_name("sparse_relation_conv_input_grad_f16_i32_cin16_dense_c16")]]
    [[kernel]] void
    sparse_relation_conv_input_grad_cin16_dense<half, 16, 16>(
        device const half* cotangent [[buffer(0)]],
        device const half* weights [[buffer(1)]],
        device const int* in_rows [[buffer(2)]],
        device const int* out_rows [[buffer(3)]],
        device const int* kernel_ids [[buffer(4)]],
        device const int* counts [[buffer(5)]],
        device const int* row_offsets [[buffer(6)]],
        device const int* in_row_offsets [[buffer(7)]],
        device const int* in_edge_ids [[buffer(8)]],
        device half* grad [[buffer(9)]],
        constant const int& edge_capacity [[buffer(10)]],
        constant const int& out_capacity [[buffer(11)]],
        constant const int& in_capacity [[buffer(12)]],
        constant const int& runtime_in_channels [[buffer(13)]],
        constant const int& runtime_out_channels [[buffer(14)]],
        constant const int& cotangent_s0 [[buffer(15)]],
        constant const int& cotangent_s1 [[buffer(16)]],
        constant const int& weight_s0 [[buffer(17)]],
        constant const int& weight_s1 [[buffer(18)]],
        constant const int& weight_s2 [[buffer(19)]],
        constant const int& weight_s3 [[buffer(20)]],
        constant const int& weight_s4 [[buffer(21)]],
        constant const int& weight_layout [[buffer(22)]],
        constant const int& kernel_x [[buffer(23)]],
        constant const int& kernel_y [[buffer(24)]],
        constant const int& kernel_z [[buffer(25)]],
        uint elem [[thread_position_in_grid]]
    );

template [[host_name(
    "sparse_relation_conv_input_grad_f32_i32_cin16_dense_cin16_cout32"
)]] [[kernel]] void
sparse_relation_conv_input_grad_cin16_dense<float, 16, 32>(
    device const float* cotangent [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device const int* in_row_offsets [[buffer(7)]],
    device const int* in_edge_ids [[buffer(8)]],
    device float* grad [[buffer(9)]],
    constant const int& edge_capacity [[buffer(10)]],
    constant const int& out_capacity [[buffer(11)]],
    constant const int& in_capacity [[buffer(12)]],
    constant const int& runtime_in_channels [[buffer(13)]],
    constant const int& runtime_out_channels [[buffer(14)]],
    constant const int& cotangent_s0 [[buffer(15)]],
    constant const int& cotangent_s1 [[buffer(16)]],
    constant const int& weight_s0 [[buffer(17)]],
    constant const int& weight_s1 [[buffer(18)]],
    constant const int& weight_s2 [[buffer(19)]],
    constant const int& weight_s3 [[buffer(20)]],
    constant const int& weight_s4 [[buffer(21)]],
    constant const int& weight_layout [[buffer(22)]],
    constant const int& kernel_x [[buffer(23)]],
    constant const int& kernel_y [[buffer(24)]],
    constant const int& kernel_z [[buffer(25)]],
    uint elem [[thread_position_in_grid]]
);

template [[host_name(
    "sparse_relation_conv_input_grad_f16_i32_cin16_dense_cin16_cout32"
)]] [[kernel]] void
sparse_relation_conv_input_grad_cin16_dense<half, 16, 32>(
    device const half* cotangent [[buffer(0)]],
    device const half* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device const int* in_row_offsets [[buffer(7)]],
    device const int* in_edge_ids [[buffer(8)]],
    device half* grad [[buffer(9)]],
    constant const int& edge_capacity [[buffer(10)]],
    constant const int& out_capacity [[buffer(11)]],
    constant const int& in_capacity [[buffer(12)]],
    constant const int& runtime_in_channels [[buffer(13)]],
    constant const int& runtime_out_channels [[buffer(14)]],
    constant const int& cotangent_s0 [[buffer(15)]],
    constant const int& cotangent_s1 [[buffer(16)]],
    constant const int& weight_s0 [[buffer(17)]],
    constant const int& weight_s1 [[buffer(18)]],
    constant const int& weight_s2 [[buffer(19)]],
    constant const int& weight_s3 [[buffer(20)]],
    constant const int& weight_s4 [[buffer(21)]],
    constant const int& weight_layout [[buffer(22)]],
    constant const int& kernel_x [[buffer(23)]],
    constant const int& kernel_y [[buffer(24)]],
    constant const int& kernel_z [[buffer(25)]],
    uint elem [[thread_position_in_grid]]
);

template [[host_name(
    "sparse_relation_conv_input_grad_f32_i32_cin16_dense_cin16_cout64"
)]] [[kernel]] void
sparse_relation_conv_input_grad_cin16_dense<float, 16, 64>(
    device const float* cotangent [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device const int* in_row_offsets [[buffer(7)]],
    device const int* in_edge_ids [[buffer(8)]],
    device float* grad [[buffer(9)]],
    constant const int& edge_capacity [[buffer(10)]],
    constant const int& out_capacity [[buffer(11)]],
    constant const int& in_capacity [[buffer(12)]],
    constant const int& runtime_in_channels [[buffer(13)]],
    constant const int& runtime_out_channels [[buffer(14)]],
    constant const int& cotangent_s0 [[buffer(15)]],
    constant const int& cotangent_s1 [[buffer(16)]],
    constant const int& weight_s0 [[buffer(17)]],
    constant const int& weight_s1 [[buffer(18)]],
    constant const int& weight_s2 [[buffer(19)]],
    constant const int& weight_s3 [[buffer(20)]],
    constant const int& weight_s4 [[buffer(21)]],
    constant const int& weight_layout [[buffer(22)]],
    constant const int& kernel_x [[buffer(23)]],
    constant const int& kernel_y [[buffer(24)]],
    constant const int& kernel_z [[buffer(25)]],
    uint elem [[thread_position_in_grid]]
);

template [[host_name(
    "sparse_relation_conv_input_grad_f16_i32_cin16_dense_cin16_cout64"
)]] [[kernel]] void
sparse_relation_conv_input_grad_cin16_dense<half, 16, 64>(
    device const half* cotangent [[buffer(0)]],
    device const half* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device const int* in_row_offsets [[buffer(7)]],
    device const int* in_edge_ids [[buffer(8)]],
    device half* grad [[buffer(9)]],
    constant const int& edge_capacity [[buffer(10)]],
    constant const int& out_capacity [[buffer(11)]],
    constant const int& in_capacity [[buffer(12)]],
    constant const int& runtime_in_channels [[buffer(13)]],
    constant const int& runtime_out_channels [[buffer(14)]],
    constant const int& cotangent_s0 [[buffer(15)]],
    constant const int& cotangent_s1 [[buffer(16)]],
    constant const int& weight_s0 [[buffer(17)]],
    constant const int& weight_s1 [[buffer(18)]],
    constant const int& weight_s2 [[buffer(19)]],
    constant const int& weight_s3 [[buffer(20)]],
    constant const int& weight_s4 [[buffer(21)]],
    constant const int& weight_layout [[buffer(22)]],
    constant const int& kernel_x [[buffer(23)]],
    constant const int& kernel_y [[buffer(24)]],
    constant const int& kernel_z [[buffer(25)]],
    uint elem [[thread_position_in_grid]]
);

template [[host_name("sparse_relation_conv_input_grad_f32_i32_cin4_dense_c32")]]
[[kernel]] void
sparse_relation_conv_input_grad_cin4_dense<float, 32, 32>(
    device const float* cotangent [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device const int* in_row_offsets [[buffer(7)]],
    device const int* in_edge_ids [[buffer(8)]],
    device float* grad [[buffer(9)]],
    constant const int& edge_capacity [[buffer(10)]],
    constant const int& out_capacity [[buffer(11)]],
    constant const int& in_capacity [[buffer(12)]],
    constant const int& runtime_in_channels [[buffer(13)]],
    constant const int& runtime_out_channels [[buffer(14)]],
    constant const int& cotangent_s0 [[buffer(15)]],
    constant const int& cotangent_s1 [[buffer(16)]],
    constant const int& weight_s0 [[buffer(17)]],
    constant const int& weight_s1 [[buffer(18)]],
    constant const int& weight_s2 [[buffer(19)]],
    constant const int& weight_s3 [[buffer(20)]],
    constant const int& weight_s4 [[buffer(21)]],
    constant const int& weight_layout [[buffer(22)]],
    constant const int& kernel_x [[buffer(23)]],
    constant const int& kernel_y [[buffer(24)]],
    constant const int& kernel_z [[buffer(25)]],
    uint elem [[thread_position_in_grid]]
);

template [[host_name("sparse_relation_conv_input_grad_f16_i32_cin4_dense_c32")]]
[[kernel]] void
sparse_relation_conv_input_grad_cin4_dense<half, 32, 32>(
    device const half* cotangent [[buffer(0)]],
    device const half* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device const int* in_row_offsets [[buffer(7)]],
    device const int* in_edge_ids [[buffer(8)]],
    device half* grad [[buffer(9)]],
    constant const int& edge_capacity [[buffer(10)]],
    constant const int& out_capacity [[buffer(11)]],
    constant const int& in_capacity [[buffer(12)]],
    constant const int& runtime_in_channels [[buffer(13)]],
    constant const int& runtime_out_channels [[buffer(14)]],
    constant const int& cotangent_s0 [[buffer(15)]],
    constant const int& cotangent_s1 [[buffer(16)]],
    constant const int& weight_s0 [[buffer(17)]],
    constant const int& weight_s1 [[buffer(18)]],
    constant const int& weight_s2 [[buffer(19)]],
    constant const int& weight_s3 [[buffer(20)]],
    constant const int& weight_s4 [[buffer(21)]],
    constant const int& weight_layout [[buffer(22)]],
    constant const int& kernel_x [[buffer(23)]],
    constant const int& kernel_y [[buffer(24)]],
    constant const int& kernel_z [[buffer(25)]],
    uint elem [[thread_position_in_grid]]
);

template [[host_name("sparse_relation_conv_input_grad_f32_i32_cin4_dense_c64")]]
[[kernel]] void
sparse_relation_conv_input_grad_cin4_dense<float, 64, 64>(
    device const float* cotangent [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device const int* in_row_offsets [[buffer(7)]],
    device const int* in_edge_ids [[buffer(8)]],
    device float* grad [[buffer(9)]],
    constant const int& edge_capacity [[buffer(10)]],
    constant const int& out_capacity [[buffer(11)]],
    constant const int& in_capacity [[buffer(12)]],
    constant const int& runtime_in_channels [[buffer(13)]],
    constant const int& runtime_out_channels [[buffer(14)]],
    constant const int& cotangent_s0 [[buffer(15)]],
    constant const int& cotangent_s1 [[buffer(16)]],
    constant const int& weight_s0 [[buffer(17)]],
    constant const int& weight_s1 [[buffer(18)]],
    constant const int& weight_s2 [[buffer(19)]],
    constant const int& weight_s3 [[buffer(20)]],
    constant const int& weight_s4 [[buffer(21)]],
    constant const int& weight_layout [[buffer(22)]],
    constant const int& kernel_x [[buffer(23)]],
    constant const int& kernel_y [[buffer(24)]],
    constant const int& kernel_z [[buffer(25)]],
    uint elem [[thread_position_in_grid]]
);

template [[host_name("sparse_relation_conv_input_grad_f16_i32_cin4_dense_c64")]]
[[kernel]] void
sparse_relation_conv_input_grad_cin4_dense<half, 64, 64>(
    device const half* cotangent [[buffer(0)]],
    device const half* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device const int* in_row_offsets [[buffer(7)]],
    device const int* in_edge_ids [[buffer(8)]],
    device half* grad [[buffer(9)]],
    constant const int& edge_capacity [[buffer(10)]],
    constant const int& out_capacity [[buffer(11)]],
    constant const int& in_capacity [[buffer(12)]],
    constant const int& runtime_in_channels [[buffer(13)]],
    constant const int& runtime_out_channels [[buffer(14)]],
    constant const int& cotangent_s0 [[buffer(15)]],
    constant const int& cotangent_s1 [[buffer(16)]],
    constant const int& weight_s0 [[buffer(17)]],
    constant const int& weight_s1 [[buffer(18)]],
    constant const int& weight_s2 [[buffer(19)]],
    constant const int& weight_s3 [[buffer(20)]],
    constant const int& weight_s4 [[buffer(21)]],
    constant const int& weight_layout [[buffer(22)]],
    constant const int& kernel_x [[buffer(23)]],
    constant const int& kernel_y [[buffer(24)]],
    constant const int& kernel_z [[buffer(25)]],
    uint elem [[thread_position_in_grid]]
);

template
    [[host_name("sparse_relation_conv_input_grad_f32_i32_cin16_dense_c32")]]
    [[kernel]] void
    sparse_relation_conv_input_grad_cin16_grouped_dense<float, 32, 32>(
        device const float* cotangent [[buffer(0)]],
        device const float* weights [[buffer(1)]],
        device const int* in_rows [[buffer(2)]],
        device const int* out_rows [[buffer(3)]],
        device const int* kernel_ids [[buffer(4)]],
        device const int* counts [[buffer(5)]],
        device const int* row_offsets [[buffer(6)]],
        device const int* in_row_offsets [[buffer(7)]],
        device const int* in_edge_ids [[buffer(8)]],
        device float* grad [[buffer(9)]],
        constant const int& edge_capacity [[buffer(10)]],
        constant const int& out_capacity [[buffer(11)]],
        constant const int& in_capacity [[buffer(12)]],
        constant const int& runtime_in_channels [[buffer(13)]],
        constant const int& runtime_out_channels [[buffer(14)]],
        constant const int& cotangent_s0 [[buffer(15)]],
        constant const int& cotangent_s1 [[buffer(16)]],
        constant const int& weight_s0 [[buffer(17)]],
        constant const int& weight_s1 [[buffer(18)]],
        constant const int& weight_s2 [[buffer(19)]],
        constant const int& weight_s3 [[buffer(20)]],
        constant const int& weight_s4 [[buffer(21)]],
        constant const int& weight_layout [[buffer(22)]],
        constant const int& kernel_x [[buffer(23)]],
        constant const int& kernel_y [[buffer(24)]],
        constant const int& kernel_z [[buffer(25)]],
        uint elem [[thread_position_in_grid]]
    );

template
    [[host_name("sparse_relation_conv_input_grad_f16_i32_cin16_dense_c32")]]
    [[kernel]] void
    sparse_relation_conv_input_grad_cin16_grouped_dense<half, 32, 32>(
        device const half* cotangent [[buffer(0)]],
        device const half* weights [[buffer(1)]],
        device const int* in_rows [[buffer(2)]],
        device const int* out_rows [[buffer(3)]],
        device const int* kernel_ids [[buffer(4)]],
        device const int* counts [[buffer(5)]],
        device const int* row_offsets [[buffer(6)]],
        device const int* in_row_offsets [[buffer(7)]],
        device const int* in_edge_ids [[buffer(8)]],
        device half* grad [[buffer(9)]],
        constant const int& edge_capacity [[buffer(10)]],
        constant const int& out_capacity [[buffer(11)]],
        constant const int& in_capacity [[buffer(12)]],
        constant const int& runtime_in_channels [[buffer(13)]],
        constant const int& runtime_out_channels [[buffer(14)]],
        constant const int& cotangent_s0 [[buffer(15)]],
        constant const int& cotangent_s1 [[buffer(16)]],
        constant const int& weight_s0 [[buffer(17)]],
        constant const int& weight_s1 [[buffer(18)]],
        constant const int& weight_s2 [[buffer(19)]],
        constant const int& weight_s3 [[buffer(20)]],
        constant const int& weight_s4 [[buffer(21)]],
        constant const int& weight_layout [[buffer(22)]],
        constant const int& kernel_x [[buffer(23)]],
        constant const int& kernel_y [[buffer(24)]],
        constant const int& kernel_z [[buffer(25)]],
        uint elem [[thread_position_in_grid]]
    );

template
    [[host_name("sparse_relation_conv_input_grad_f16_i32_cin16_dense_c64")]]
    [[kernel]] void
    sparse_relation_conv_input_grad_cin16_grouped_dense<half, 64, 64>(
        device const half* cotangent [[buffer(0)]],
        device const half* weights [[buffer(1)]],
        device const int* in_rows [[buffer(2)]],
        device const int* out_rows [[buffer(3)]],
        device const int* kernel_ids [[buffer(4)]],
        device const int* counts [[buffer(5)]],
        device const int* row_offsets [[buffer(6)]],
        device const int* in_row_offsets [[buffer(7)]],
        device const int* in_edge_ids [[buffer(8)]],
        device half* grad [[buffer(9)]],
        constant const int& edge_capacity [[buffer(10)]],
        constant const int& out_capacity [[buffer(11)]],
        constant const int& in_capacity [[buffer(12)]],
        constant const int& runtime_in_channels [[buffer(13)]],
        constant const int& runtime_out_channels [[buffer(14)]],
        constant const int& cotangent_s0 [[buffer(15)]],
        constant const int& cotangent_s1 [[buffer(16)]],
        constant const int& weight_s0 [[buffer(17)]],
        constant const int& weight_s1 [[buffer(18)]],
        constant const int& weight_s2 [[buffer(19)]],
        constant const int& weight_s3 [[buffer(20)]],
        constant const int& weight_s4 [[buffer(21)]],
        constant const int& weight_layout [[buffer(22)]],
        constant const int& kernel_x [[buffer(23)]],
        constant const int& kernel_y [[buffer(24)]],
        constant const int& kernel_z [[buffer(25)]],
        uint elem [[thread_position_in_grid]]
    );

template [[host_name(
    "sparse_relation_conv_input_grad_f32_i32_cin16_dense_cin64_cout16"
)]] [[kernel]] void
sparse_relation_conv_input_grad_cin16_grouped_dense<float, 64, 16>(
    device const float* cotangent [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device const int* in_row_offsets [[buffer(7)]],
    device const int* in_edge_ids [[buffer(8)]],
    device float* grad [[buffer(9)]],
    constant const int& edge_capacity [[buffer(10)]],
    constant const int& out_capacity [[buffer(11)]],
    constant const int& in_capacity [[buffer(12)]],
    constant const int& runtime_in_channels [[buffer(13)]],
    constant const int& runtime_out_channels [[buffer(14)]],
    constant const int& cotangent_s0 [[buffer(15)]],
    constant const int& cotangent_s1 [[buffer(16)]],
    constant const int& weight_s0 [[buffer(17)]],
    constant const int& weight_s1 [[buffer(18)]],
    constant const int& weight_s2 [[buffer(19)]],
    constant const int& weight_s3 [[buffer(20)]],
    constant const int& weight_s4 [[buffer(21)]],
    constant const int& weight_layout [[buffer(22)]],
    constant const int& kernel_x [[buffer(23)]],
    constant const int& kernel_y [[buffer(24)]],
    constant const int& kernel_z [[buffer(25)]],
    uint elem [[thread_position_in_grid]]
);

template [[host_name(
    "sparse_relation_conv_input_grad_f16_i32_cin16_dense_cin64_cout16"
)]] [[kernel]] void
sparse_relation_conv_input_grad_cin16_grouped_dense<half, 64, 16>(
    device const half* cotangent [[buffer(0)]],
    device const half* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device const int* in_row_offsets [[buffer(7)]],
    device const int* in_edge_ids [[buffer(8)]],
    device half* grad [[buffer(9)]],
    constant const int& edge_capacity [[buffer(10)]],
    constant const int& out_capacity [[buffer(11)]],
    constant const int& in_capacity [[buffer(12)]],
    constant const int& runtime_in_channels [[buffer(13)]],
    constant const int& runtime_out_channels [[buffer(14)]],
    constant const int& cotangent_s0 [[buffer(15)]],
    constant const int& cotangent_s1 [[buffer(16)]],
    constant const int& weight_s0 [[buffer(17)]],
    constant const int& weight_s1 [[buffer(18)]],
    constant const int& weight_s2 [[buffer(19)]],
    constant const int& weight_s3 [[buffer(20)]],
    constant const int& weight_s4 [[buffer(21)]],
    constant const int& weight_layout [[buffer(22)]],
    constant const int& kernel_x [[buffer(23)]],
    constant const int& kernel_y [[buffer(24)]],
    constant const int& kernel_z [[buffer(25)]],
    uint elem [[thread_position_in_grid]]
);

template [[host_name("sparse_relation_conv_weight_grad_f32_i32_c4_dense_c16")]]
[[kernel]] void
sparse_relation_conv_weight_grad_c4_dense<float, 16, 16>(
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
    constant const int& runtime_in_channels [[buffer(13)]],
    constant const int& runtime_out_channels [[buffer(14)]],
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
);

template [[host_name("sparse_relation_conv_weight_grad_f16_i32_c4_dense_c16")]]
[[kernel]] void
sparse_relation_conv_weight_grad_c4_dense<half, 16, 16>(
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
    constant const int& runtime_in_channels [[buffer(13)]],
    constant const int& runtime_out_channels [[buffer(14)]],
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
);

template [[host_name("sparse_relation_conv_weight_grad_f32_i32_c4_dense_c32")]]
[[kernel]] void
sparse_relation_conv_weight_grad_c4_dense<float, 32, 32>(
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
    constant const int& runtime_in_channels [[buffer(13)]],
    constant const int& runtime_out_channels [[buffer(14)]],
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
);

template [[host_name("sparse_relation_conv_weight_grad_f16_i32_c4_dense_c32")]]
[[kernel]] void
sparse_relation_conv_weight_grad_c4_dense<half, 32, 32>(
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
    constant const int& runtime_in_channels [[buffer(13)]],
    constant const int& runtime_out_channels [[buffer(14)]],
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
);

template [[host_name("sparse_relation_conv_weight_grad_f32_i32_c4_dense_c64")]]
[[kernel]] void
sparse_relation_conv_weight_grad_c4_dense<float, 64, 64>(
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
    constant const int& runtime_in_channels [[buffer(13)]],
    constant const int& runtime_out_channels [[buffer(14)]],
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
);

template [[host_name("sparse_relation_conv_weight_grad_f16_i32_c4_dense_c64")]]
[[kernel]] void
sparse_relation_conv_weight_grad_c4_dense<half, 64, 64>(
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
    constant const int& runtime_in_channels [[buffer(13)]],
    constant const int& runtime_out_channels [[buffer(14)]],
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
);

[[kernel]] void sparse_relation_conv_clear_f32(
    device float* out [[buffer(0)]],
    constant const int& total [[buffer(1)]],
    uint elem [[thread_position_in_grid]]
) {
    if (elem < uint(total)) {
        out[elem] = 0.0f;
    }
}

[[kernel]] void sparse_relation_conv_f32_i32(
    device const float* feats [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device float* out [[buffer(7)]],
    constant const int& edge_capacity [[buffer(8)]],
    constant const int& out_capacity [[buffer(9)]],
    constant const int& in_channels [[buffer(10)]],
    constant const int& out_channels [[buffer(11)]],
    constant const int& feat_s0 [[buffer(12)]],
    constant const int& feat_s1 [[buffer(13)]],
    constant const int& weight_s0 [[buffer(14)]],
    constant const int& weight_s1 [[buffer(15)]],
    constant const int& weight_s2 [[buffer(16)]],
    constant const int& weight_s3 [[buffer(17)]],
    constant const int& weight_s4 [[buffer(18)]],
    constant const int& weight_layout [[buffer(19)]],
    constant const int& kernel_x [[buffer(20)]],
    constant const int& kernel_y [[buffer(21)]],
    constant const int& kernel_z [[buffer(22)]],
    uint elem [[thread_position_in_grid]]
) {
    int total = out_capacity * out_channels;
    if (elem >= uint(total)) {
        return;
    }

    int out_row = int(elem) / out_channels;
    int co = int(elem) - out_row * out_channels;
    int out_count = min(counts[1], out_capacity);
    if (out_row >= out_count) {
        out[elem] = 0.0f;
        return;
    }

    int edge_count = min(counts[0], edge_capacity);
    float acc = 0.0f;
    for (int edge = row_offsets[out_row]; edge < row_offsets[out_row + 1];
         ++edge) {
        if (edge < 0 || edge >= edge_count) {
            continue;
        }
        int in_row = in_rows[edge];
        int kernel_id = kernel_ids[edge];
        if (in_row < 0 || kernel_id < 0) {
            continue;
        }
        for (int ci = 0; ci < in_channels; ++ci) {
            acc += feats[in_row * feat_s0 + ci * feat_s1] *
                   weights[sparse_conv_weight_offset(
                       kernel_id,
                       ci,
                       co,
                       weight_layout,
                       kernel_x,
                       kernel_y,
                       kernel_z,
                       weight_s0,
                       weight_s1,
                       weight_s2,
                       weight_s3,
                       weight_s4
                   )];
        }
    }
    out[elem] = acc;
    (void)out_rows;
}

[[kernel]] void sparse_relation_conv_f32_i32_vec4(
    device const float* feats [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device float* out [[buffer(7)]],
    constant const int& edge_capacity [[buffer(8)]],
    constant const int& out_capacity [[buffer(9)]],
    constant const int& in_channels [[buffer(10)]],
    constant const int& out_channels [[buffer(11)]],
    constant const int& feat_s0 [[buffer(12)]],
    constant const int& feat_s1 [[buffer(13)]],
    constant const int& weight_s0 [[buffer(14)]],
    constant const int& weight_s1 [[buffer(15)]],
    constant const int& weight_s2 [[buffer(16)]],
    constant const int& weight_s3 [[buffer(17)]],
    constant const int& weight_s4 [[buffer(18)]],
    constant const int& weight_layout [[buffer(19)]],
    constant const int& kernel_x [[buffer(20)]],
    constant const int& kernel_y [[buffer(21)]],
    constant const int& kernel_z [[buffer(22)]],
    uint elem [[thread_position_in_grid]]
) {
    int blocks = out_channels / 4;
    int total = out_capacity * blocks;
    if (elem >= uint(total)) {
        return;
    }

    int out_row = int(elem) / blocks;
    int co = (int(elem) - out_row * blocks) * 4;
    int out_base = out_row * out_channels + co;
    int out_count = min(counts[1], out_capacity);
    if (out_row >= out_count) {
        out[out_base] = 0.0f;
        out[out_base + 1] = 0.0f;
        out[out_base + 2] = 0.0f;
        out[out_base + 3] = 0.0f;
        return;
    }

    int edge_count = min(counts[0], edge_capacity);
    float4 acc = float4(0.0f);
    for (int edge = row_offsets[out_row]; edge < row_offsets[out_row + 1];
         ++edge) {
        if (edge < 0 || edge >= edge_count) {
            continue;
        }
        int in_row = in_rows[edge];
        int kernel_id = kernel_ids[edge];
        if (in_row < 0 || kernel_id < 0) {
            continue;
        }
        for (int ci = 0; ci < in_channels; ++ci) {
            float value = feats[in_row * feat_s0 + ci * feat_s1];
            acc += value * float4(
                               weights[sparse_conv_weight_offset(
                                   kernel_id,
                                   ci,
                                   co,
                                   weight_layout,
                                   kernel_x,
                                   kernel_y,
                                   kernel_z,
                                   weight_s0,
                                   weight_s1,
                                   weight_s2,
                                   weight_s3,
                                   weight_s4
                               )],
                               weights[sparse_conv_weight_offset(
                                   kernel_id,
                                   ci,
                                   co + 1,
                                   weight_layout,
                                   kernel_x,
                                   kernel_y,
                                   kernel_z,
                                   weight_s0,
                                   weight_s1,
                                   weight_s2,
                                   weight_s3,
                                   weight_s4
                               )],
                               weights[sparse_conv_weight_offset(
                                   kernel_id,
                                   ci,
                                   co + 2,
                                   weight_layout,
                                   kernel_x,
                                   kernel_y,
                                   kernel_z,
                                   weight_s0,
                                   weight_s1,
                                   weight_s2,
                                   weight_s3,
                                   weight_s4
                               )],
                               weights[sparse_conv_weight_offset(
                                   kernel_id,
                                   ci,
                                   co + 3,
                                   weight_layout,
                                   kernel_x,
                                   kernel_y,
                                   kernel_z,
                                   weight_s0,
                                   weight_s1,
                                   weight_s2,
                                   weight_s3,
                                   weight_s4
                               )]
                           );
        }
    }
    out[out_base] = acc.x;
    out[out_base + 1] = acc.y;
    out[out_base + 2] = acc.z;
    out[out_base + 3] = acc.w;
    (void)out_rows;
}

[[kernel]] void sparse_relation_conv_f32_i32_cout16(
    device const float* feats [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device float* out [[buffer(7)]],
    constant const int& edge_capacity [[buffer(8)]],
    constant const int& out_capacity [[buffer(9)]],
    constant const int& in_channels [[buffer(10)]],
    constant const int& out_channels [[buffer(11)]],
    constant const int& feat_s0 [[buffer(12)]],
    constant const int& feat_s1 [[buffer(13)]],
    constant const int& weight_s0 [[buffer(14)]],
    constant const int& weight_s1 [[buffer(15)]],
    constant const int& weight_s2 [[buffer(16)]],
    constant const int& weight_s3 [[buffer(17)]],
    constant const int& weight_s4 [[buffer(18)]],
    constant const int& weight_layout [[buffer(19)]],
    constant const int& kernel_x [[buffer(20)]],
    constant const int& kernel_y [[buffer(21)]],
    constant const int& kernel_z [[buffer(22)]],
    uint out_row_id [[thread_position_in_grid]]
) {
    if (out_row_id >= uint(out_capacity)) {
        return;
    }

    int out_row = int(out_row_id);
    int out_base = out_row * 16;
    int out_count = min(counts[1], out_capacity);
    if (out_row >= out_count) {
        for (int co = 0; co < 16; ++co) {
            out[out_base + co] = 0.0f;
        }
        return;
    }

    int edge_count = min(counts[0], edge_capacity);
    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);
    float4 acc3 = float4(0.0f);
    for (int edge = row_offsets[out_row]; edge < row_offsets[out_row + 1];
         ++edge) {
        if (edge < 0 || edge >= edge_count) {
            continue;
        }
        int in_row = in_rows[edge];
        int kernel_id = kernel_ids[edge];
        if (in_row < 0 || kernel_id < 0) {
            continue;
        }
        for (int ci = 0; ci < in_channels; ++ci) {
            float value = feats[in_row * feat_s0 + ci * feat_s1];
            acc0 += value * float4(
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    ci,
                                    0,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )],
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    ci,
                                    1,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )],
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    ci,
                                    2,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )],
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    ci,
                                    3,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )]
                            );
            acc1 += value * float4(
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    ci,
                                    4,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )],
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    ci,
                                    5,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )],
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    ci,
                                    6,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )],
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    ci,
                                    7,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )]
                            );
            acc2 += value * float4(
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    ci,
                                    8,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )],
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    ci,
                                    9,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )],
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    ci,
                                    10,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )],
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    ci,
                                    11,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )]
                            );
            acc3 += value * float4(
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    ci,
                                    12,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )],
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    ci,
                                    13,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )],
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    ci,
                                    14,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )],
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    ci,
                                    15,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )]
                            );
        }
    }
    out[out_base] = acc0.x;
    out[out_base + 1] = acc0.y;
    out[out_base + 2] = acc0.z;
    out[out_base + 3] = acc0.w;
    out[out_base + 4] = acc1.x;
    out[out_base + 5] = acc1.y;
    out[out_base + 6] = acc1.z;
    out[out_base + 7] = acc1.w;
    out[out_base + 8] = acc2.x;
    out[out_base + 9] = acc2.y;
    out[out_base + 10] = acc2.z;
    out[out_base + 11] = acc2.w;
    out[out_base + 12] = acc3.x;
    out[out_base + 13] = acc3.y;
    out[out_base + 14] = acc3.z;
    out[out_base + 15] = acc3.w;
    (void)out_rows;
    (void)out_channels;
}

[[kernel]] void sparse_relation_conv_atomic_f32_i32(
    device const float* feats [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device atomic_float* out [[buffer(7)]],
    constant const int& edge_capacity [[buffer(8)]],
    constant const int& out_capacity [[buffer(9)]],
    constant const int& in_channels [[buffer(10)]],
    constant const int& out_channels [[buffer(11)]],
    constant const int& feat_s0 [[buffer(12)]],
    constant const int& feat_s1 [[buffer(13)]],
    constant const int& weight_s0 [[buffer(14)]],
    constant const int& weight_s1 [[buffer(15)]],
    constant const int& weight_s2 [[buffer(16)]],
    constant const int& weight_s3 [[buffer(17)]],
    constant const int& weight_s4 [[buffer(18)]],
    constant const int& weight_layout [[buffer(19)]],
    constant const int& kernel_x [[buffer(20)]],
    constant const int& kernel_y [[buffer(21)]],
    constant const int& kernel_z [[buffer(22)]],
    uint elem [[thread_position_in_grid]]
) {
    int edge_count = min(counts[0], edge_capacity);
    int total = edge_count * out_channels;
    if (elem >= uint(total)) {
        return;
    }

    int edge = int(elem) / out_channels;
    int co = int(elem) - edge * out_channels;
    int in_row = in_rows[edge];
    int out_row = out_rows[edge];
    int kernel_id = kernel_ids[edge];
    if (in_row < 0 || out_row < 0 || out_row >= out_capacity || kernel_id < 0) {
        return;
    }

    float acc = 0.0f;
    for (int ci = 0; ci < in_channels; ++ci) {
        acc += feats[in_row * feat_s0 + ci * feat_s1] *
               weights[sparse_conv_weight_offset(
                   kernel_id,
                   ci,
                   co,
                   weight_layout,
                   kernel_x,
                   kernel_y,
                   kernel_z,
                   weight_s0,
                   weight_s1,
                   weight_s2,
                   weight_s3,
                   weight_s4
               )];
    }
    atomic_fetch_add_explicit(
        &out[out_row * out_channels + co], acc, memory_order_relaxed
    );
    (void)row_offsets;
}

[[kernel]] void sparse_relation_conv_f16_i32(
    device const half* feats [[buffer(0)]],
    device const half* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device half* out [[buffer(7)]],
    constant const int& edge_capacity [[buffer(8)]],
    constant const int& out_capacity [[buffer(9)]],
    constant const int& in_channels [[buffer(10)]],
    constant const int& out_channels [[buffer(11)]],
    constant const int& feat_s0 [[buffer(12)]],
    constant const int& feat_s1 [[buffer(13)]],
    constant const int& weight_s0 [[buffer(14)]],
    constant const int& weight_s1 [[buffer(15)]],
    constant const int& weight_s2 [[buffer(16)]],
    constant const int& weight_s3 [[buffer(17)]],
    constant const int& weight_s4 [[buffer(18)]],
    constant const int& weight_layout [[buffer(19)]],
    constant const int& kernel_x [[buffer(20)]],
    constant const int& kernel_y [[buffer(21)]],
    constant const int& kernel_z [[buffer(22)]],
    uint elem [[thread_position_in_grid]]
) {
    int total = out_capacity * out_channels;
    if (elem >= uint(total)) {
        return;
    }

    int out_row = int(elem) / out_channels;
    int co = int(elem) - out_row * out_channels;
    int out_count = min(counts[1], out_capacity);
    if (out_row >= out_count) {
        out[elem] = half(0.0h);
        return;
    }

    int edge_count = min(counts[0], edge_capacity);
    float acc = 0.0f;
    for (int edge = row_offsets[out_row]; edge < row_offsets[out_row + 1];
         ++edge) {
        if (edge < 0 || edge >= edge_count) {
            continue;
        }
        int in_row = in_rows[edge];
        int kernel_id = kernel_ids[edge];
        if (in_row < 0 || kernel_id < 0) {
            continue;
        }
        for (int ci = 0; ci < in_channels; ++ci) {
            acc += float(feats[in_row * feat_s0 + ci * feat_s1]) *
                   float(weights[sparse_conv_weight_offset(
                       kernel_id,
                       ci,
                       co,
                       weight_layout,
                       kernel_x,
                       kernel_y,
                       kernel_z,
                       weight_s0,
                       weight_s1,
                       weight_s2,
                       weight_s3,
                       weight_s4
                   )]);
        }
    }
    out[elem] = half(acc);
    (void)out_rows;
}

[[kernel]] void sparse_relation_conv_f16_i32_cout16(
    device const half* feats [[buffer(0)]],
    device const half* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device half* out [[buffer(7)]],
    constant const int& edge_capacity [[buffer(8)]],
    constant const int& out_capacity [[buffer(9)]],
    constant const int& in_channels [[buffer(10)]],
    constant const int& out_channels [[buffer(11)]],
    constant const int& feat_s0 [[buffer(12)]],
    constant const int& feat_s1 [[buffer(13)]],
    constant const int& weight_s0 [[buffer(14)]],
    constant const int& weight_s1 [[buffer(15)]],
    constant const int& weight_s2 [[buffer(16)]],
    constant const int& weight_s3 [[buffer(17)]],
    constant const int& weight_s4 [[buffer(18)]],
    constant const int& weight_layout [[buffer(19)]],
    constant const int& kernel_x [[buffer(20)]],
    constant const int& kernel_y [[buffer(21)]],
    constant const int& kernel_z [[buffer(22)]],
    uint out_row_id [[thread_position_in_grid]]
) {
    if (out_row_id >= uint(out_capacity)) {
        return;
    }

    int out_row = int(out_row_id);
    int out_base = out_row * 16;
    int out_count = min(counts[1], out_capacity);
    if (out_row >= out_count) {
        for (int co = 0; co < 16; ++co) {
            out[out_base + co] = half(0.0h);
        }
        return;
    }

    int edge_count = min(counts[0], edge_capacity);
    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);
    float4 acc3 = float4(0.0f);
    for (int edge = row_offsets[out_row]; edge < row_offsets[out_row + 1];
         ++edge) {
        if (edge < 0 || edge >= edge_count) {
            continue;
        }
        int in_row = in_rows[edge];
        int kernel_id = kernel_ids[edge];
        if (in_row < 0 || kernel_id < 0) {
            continue;
        }
        for (int ci = 0; ci < in_channels; ++ci) {
            float value = float(feats[in_row * feat_s0 + ci * feat_s1]);
            acc0 += value * float4(
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    ci,
                                    0,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )]),
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    ci,
                                    1,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )]),
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    ci,
                                    2,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )]),
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    ci,
                                    3,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )])
                            );
            acc1 += value * float4(
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    ci,
                                    4,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )]),
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    ci,
                                    5,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )]),
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    ci,
                                    6,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )]),
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    ci,
                                    7,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )])
                            );
            acc2 += value * float4(
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    ci,
                                    8,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )]),
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    ci,
                                    9,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )]),
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    ci,
                                    10,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )]),
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    ci,
                                    11,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )])
                            );
            acc3 += value * float4(
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    ci,
                                    12,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )]),
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    ci,
                                    13,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )]),
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    ci,
                                    14,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )]),
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    ci,
                                    15,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )])
                            );
        }
    }
    out[out_base] = half(acc0.x);
    out[out_base + 1] = half(acc0.y);
    out[out_base + 2] = half(acc0.z);
    out[out_base + 3] = half(acc0.w);
    out[out_base + 4] = half(acc1.x);
    out[out_base + 5] = half(acc1.y);
    out[out_base + 6] = half(acc1.z);
    out[out_base + 7] = half(acc1.w);
    out[out_base + 8] = half(acc2.x);
    out[out_base + 9] = half(acc2.y);
    out[out_base + 10] = half(acc2.z);
    out[out_base + 11] = half(acc2.w);
    out[out_base + 12] = half(acc3.x);
    out[out_base + 13] = half(acc3.y);
    out[out_base + 14] = half(acc3.z);
    out[out_base + 15] = half(acc3.w);
    (void)out_rows;
    (void)out_channels;
}

[[kernel]] void sparse_relation_conv_input_grad_f32_i32(
    device const float* cotangent [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device const int* in_row_offsets [[buffer(7)]],
    device const int* in_edge_ids [[buffer(8)]],
    device float* grad [[buffer(9)]],
    constant const int& edge_capacity [[buffer(10)]],
    constant const int& out_capacity [[buffer(11)]],
    constant const int& in_capacity [[buffer(12)]],
    constant const int& in_channels [[buffer(13)]],
    constant const int& out_channels [[buffer(14)]],
    constant const int& cotangent_s0 [[buffer(15)]],
    constant const int& cotangent_s1 [[buffer(16)]],
    constant const int& weight_s0 [[buffer(17)]],
    constant const int& weight_s1 [[buffer(18)]],
    constant const int& weight_s2 [[buffer(19)]],
    constant const int& weight_s3 [[buffer(20)]],
    constant const int& weight_s4 [[buffer(21)]],
    constant const int& weight_layout [[buffer(22)]],
    constant const int& kernel_x [[buffer(23)]],
    constant const int& kernel_y [[buffer(24)]],
    constant const int& kernel_z [[buffer(25)]],
    uint elem [[thread_position_in_grid]]
) {
    int total = in_capacity * in_channels;
    if (elem >= uint(total)) {
        return;
    }

    int in_row = int(elem) / in_channels;
    int ci = int(elem) - in_row * in_channels;
    int edge_count = min(counts[0], edge_capacity);
    float acc = 0.0f;
    for (int cursor = in_row_offsets[in_row];
         cursor < in_row_offsets[in_row + 1];
         ++cursor) {
        int edge = in_edge_ids[cursor];
        if (edge < 0 || edge >= edge_count) {
            continue;
        }
        int out_row = out_rows[edge];
        int kernel_id = kernel_ids[edge];
        if (out_row < 0 || out_row >= out_capacity || kernel_id < 0) {
            continue;
        }
        for (int co = 0; co < out_channels; ++co) {
            acc += cotangent[out_row * cotangent_s0 + co * cotangent_s1] *
                   weights[sparse_conv_weight_offset(
                       kernel_id,
                       ci,
                       co,
                       weight_layout,
                       kernel_x,
                       kernel_y,
                       kernel_z,
                       weight_s0,
                       weight_s1,
                       weight_s2,
                       weight_s3,
                       weight_s4
                   )];
        }
    }
    grad[in_row * in_channels + ci] = acc;
    (void)in_rows;
    (void)row_offsets;
}

[[kernel]] void sparse_relation_conv_input_grad_f32_i32_vec4(
    device const float* cotangent [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device const int* in_row_offsets [[buffer(7)]],
    device const int* in_edge_ids [[buffer(8)]],
    device float* grad [[buffer(9)]],
    constant const int& edge_capacity [[buffer(10)]],
    constant const int& out_capacity [[buffer(11)]],
    constant const int& in_capacity [[buffer(12)]],
    constant const int& in_channels [[buffer(13)]],
    constant const int& out_channels [[buffer(14)]],
    constant const int& cotangent_s0 [[buffer(15)]],
    constant const int& cotangent_s1 [[buffer(16)]],
    constant const int& weight_s0 [[buffer(17)]],
    constant const int& weight_s1 [[buffer(18)]],
    constant const int& weight_s2 [[buffer(19)]],
    constant const int& weight_s3 [[buffer(20)]],
    constant const int& weight_s4 [[buffer(21)]],
    constant const int& weight_layout [[buffer(22)]],
    constant const int& kernel_x [[buffer(23)]],
    constant const int& kernel_y [[buffer(24)]],
    constant const int& kernel_z [[buffer(25)]],
    uint elem [[thread_position_in_grid]]
) {
    int blocks = in_channels / 4;
    int total = in_capacity * blocks;
    if (elem >= uint(total)) {
        return;
    }

    int in_row = int(elem) / blocks;
    int ci = (int(elem) - in_row * blocks) * 4;
    int edge_count = min(counts[0], edge_capacity);
    float4 acc = float4(0.0f);
    for (int cursor = in_row_offsets[in_row];
         cursor < in_row_offsets[in_row + 1];
         ++cursor) {
        int edge = in_edge_ids[cursor];
        if (edge < 0 || edge >= edge_count) {
            continue;
        }
        int out_row = out_rows[edge];
        int kernel_id = kernel_ids[edge];
        if (out_row < 0 || out_row >= out_capacity || kernel_id < 0) {
            continue;
        }
        for (int co = 0; co < out_channels; ++co) {
            float value = cotangent[out_row * cotangent_s0 + co * cotangent_s1];
            acc += value * float4(
                               weights[sparse_conv_weight_offset(
                                   kernel_id,
                                   ci,
                                   co,
                                   weight_layout,
                                   kernel_x,
                                   kernel_y,
                                   kernel_z,
                                   weight_s0,
                                   weight_s1,
                                   weight_s2,
                                   weight_s3,
                                   weight_s4
                               )],
                               weights[sparse_conv_weight_offset(
                                   kernel_id,
                                   ci + 1,
                                   co,
                                   weight_layout,
                                   kernel_x,
                                   kernel_y,
                                   kernel_z,
                                   weight_s0,
                                   weight_s1,
                                   weight_s2,
                                   weight_s3,
                                   weight_s4
                               )],
                               weights[sparse_conv_weight_offset(
                                   kernel_id,
                                   ci + 2,
                                   co,
                                   weight_layout,
                                   kernel_x,
                                   kernel_y,
                                   kernel_z,
                                   weight_s0,
                                   weight_s1,
                                   weight_s2,
                                   weight_s3,
                                   weight_s4
                               )],
                               weights[sparse_conv_weight_offset(
                                   kernel_id,
                                   ci + 3,
                                   co,
                                   weight_layout,
                                   kernel_x,
                                   kernel_y,
                                   kernel_z,
                                   weight_s0,
                                   weight_s1,
                                   weight_s2,
                                   weight_s3,
                                   weight_s4
                               )]
                           );
        }
    }
    int grad_base = in_row * in_channels + ci;
    grad[grad_base] = acc.x;
    grad[grad_base + 1] = acc.y;
    grad[grad_base + 2] = acc.z;
    grad[grad_base + 3] = acc.w;
    (void)in_rows;
    (void)row_offsets;
}

[[kernel]] void sparse_relation_conv_input_grad_f32_i32_cin16(
    device const float* cotangent [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device const int* in_row_offsets [[buffer(7)]],
    device const int* in_edge_ids [[buffer(8)]],
    device float* grad [[buffer(9)]],
    constant const int& edge_capacity [[buffer(10)]],
    constant const int& out_capacity [[buffer(11)]],
    constant const int& in_capacity [[buffer(12)]],
    constant const int& in_channels [[buffer(13)]],
    constant const int& out_channels [[buffer(14)]],
    constant const int& cotangent_s0 [[buffer(15)]],
    constant const int& cotangent_s1 [[buffer(16)]],
    constant const int& weight_s0 [[buffer(17)]],
    constant const int& weight_s1 [[buffer(18)]],
    constant const int& weight_s2 [[buffer(19)]],
    constant const int& weight_s3 [[buffer(20)]],
    constant const int& weight_s4 [[buffer(21)]],
    constant const int& weight_layout [[buffer(22)]],
    constant const int& kernel_x [[buffer(23)]],
    constant const int& kernel_y [[buffer(24)]],
    constant const int& kernel_z [[buffer(25)]],
    uint in_row_id [[thread_position_in_grid]]
) {
    if (in_row_id >= uint(in_capacity)) {
        return;
    }

    int in_row = int(in_row_id);
    int edge_count = min(counts[0], edge_capacity);
    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);
    float4 acc3 = float4(0.0f);
    for (int cursor = in_row_offsets[in_row];
         cursor < in_row_offsets[in_row + 1];
         ++cursor) {
        int edge = in_edge_ids[cursor];
        if (edge < 0 || edge >= edge_count) {
            continue;
        }
        int out_row = out_rows[edge];
        int kernel_id = kernel_ids[edge];
        if (out_row < 0 || out_row >= out_capacity || kernel_id < 0) {
            continue;
        }
        for (int co = 0; co < out_channels; ++co) {
            float value = cotangent[out_row * cotangent_s0 + co * cotangent_s1];
            acc0 += value * float4(
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    0,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )],
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    1,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )],
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    2,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )],
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    3,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )]
                            );
            acc1 += value * float4(
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    4,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )],
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    5,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )],
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    6,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )],
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    7,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )]
                            );
            acc2 += value * float4(
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    8,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )],
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    9,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )],
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    10,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )],
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    11,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )]
                            );
            acc3 += value * float4(
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    12,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )],
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    13,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )],
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    14,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )],
                                weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    15,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )]
                            );
        }
    }
    int grad_base = in_row * 16;
    grad[grad_base] = acc0.x;
    grad[grad_base + 1] = acc0.y;
    grad[grad_base + 2] = acc0.z;
    grad[grad_base + 3] = acc0.w;
    grad[grad_base + 4] = acc1.x;
    grad[grad_base + 5] = acc1.y;
    grad[grad_base + 6] = acc1.z;
    grad[grad_base + 7] = acc1.w;
    grad[grad_base + 8] = acc2.x;
    grad[grad_base + 9] = acc2.y;
    grad[grad_base + 10] = acc2.z;
    grad[grad_base + 11] = acc2.w;
    grad[grad_base + 12] = acc3.x;
    grad[grad_base + 13] = acc3.y;
    grad[grad_base + 14] = acc3.z;
    grad[grad_base + 15] = acc3.w;
    (void)in_rows;
    (void)row_offsets;
    (void)in_channels;
}

[[kernel]] void sparse_relation_conv_input_grad_f16_i32(
    device const half* cotangent [[buffer(0)]],
    device const half* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device const int* in_row_offsets [[buffer(7)]],
    device const int* in_edge_ids [[buffer(8)]],
    device half* grad [[buffer(9)]],
    constant const int& edge_capacity [[buffer(10)]],
    constant const int& out_capacity [[buffer(11)]],
    constant const int& in_capacity [[buffer(12)]],
    constant const int& in_channels [[buffer(13)]],
    constant const int& out_channels [[buffer(14)]],
    constant const int& cotangent_s0 [[buffer(15)]],
    constant const int& cotangent_s1 [[buffer(16)]],
    constant const int& weight_s0 [[buffer(17)]],
    constant const int& weight_s1 [[buffer(18)]],
    constant const int& weight_s2 [[buffer(19)]],
    constant const int& weight_s3 [[buffer(20)]],
    constant const int& weight_s4 [[buffer(21)]],
    constant const int& weight_layout [[buffer(22)]],
    constant const int& kernel_x [[buffer(23)]],
    constant const int& kernel_y [[buffer(24)]],
    constant const int& kernel_z [[buffer(25)]],
    uint elem [[thread_position_in_grid]]
) {
    int total = in_capacity * in_channels;
    if (elem >= uint(total)) {
        return;
    }

    int in_row = int(elem) / in_channels;
    int ci = int(elem) - in_row * in_channels;
    int edge_count = min(counts[0], edge_capacity);
    float acc = 0.0f;
    for (int cursor = in_row_offsets[in_row];
         cursor < in_row_offsets[in_row + 1];
         ++cursor) {
        int edge = in_edge_ids[cursor];
        if (edge < 0 || edge >= edge_count) {
            continue;
        }
        int out_row = out_rows[edge];
        int kernel_id = kernel_ids[edge];
        if (out_row < 0 || out_row >= out_capacity || kernel_id < 0) {
            continue;
        }
        for (int co = 0; co < out_channels; ++co) {
            acc +=
                float(cotangent[out_row * cotangent_s0 + co * cotangent_s1]) *
                float(weights[sparse_conv_weight_offset(
                    kernel_id,
                    ci,
                    co,
                    weight_layout,
                    kernel_x,
                    kernel_y,
                    kernel_z,
                    weight_s0,
                    weight_s1,
                    weight_s2,
                    weight_s3,
                    weight_s4
                )]);
        }
    }
    grad[in_row * in_channels + ci] = half(acc);
    (void)in_rows;
    (void)row_offsets;
}

[[kernel]] void sparse_relation_conv_input_grad_f16_i32_cin16(
    device const half* cotangent [[buffer(0)]],
    device const half* weights [[buffer(1)]],
    device const int* in_rows [[buffer(2)]],
    device const int* out_rows [[buffer(3)]],
    device const int* kernel_ids [[buffer(4)]],
    device const int* counts [[buffer(5)]],
    device const int* row_offsets [[buffer(6)]],
    device const int* in_row_offsets [[buffer(7)]],
    device const int* in_edge_ids [[buffer(8)]],
    device half* grad [[buffer(9)]],
    constant const int& edge_capacity [[buffer(10)]],
    constant const int& out_capacity [[buffer(11)]],
    constant const int& in_capacity [[buffer(12)]],
    constant const int& in_channels [[buffer(13)]],
    constant const int& out_channels [[buffer(14)]],
    constant const int& cotangent_s0 [[buffer(15)]],
    constant const int& cotangent_s1 [[buffer(16)]],
    constant const int& weight_s0 [[buffer(17)]],
    constant const int& weight_s1 [[buffer(18)]],
    constant const int& weight_s2 [[buffer(19)]],
    constant const int& weight_s3 [[buffer(20)]],
    constant const int& weight_s4 [[buffer(21)]],
    constant const int& weight_layout [[buffer(22)]],
    constant const int& kernel_x [[buffer(23)]],
    constant const int& kernel_y [[buffer(24)]],
    constant const int& kernel_z [[buffer(25)]],
    uint in_row_id [[thread_position_in_grid]]
) {
    if (in_row_id >= uint(in_capacity)) {
        return;
    }

    int in_row = int(in_row_id);
    int edge_count = min(counts[0], edge_capacity);
    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);
    float4 acc3 = float4(0.0f);
    for (int cursor = in_row_offsets[in_row];
         cursor < in_row_offsets[in_row + 1];
         ++cursor) {
        int edge = in_edge_ids[cursor];
        if (edge < 0 || edge >= edge_count) {
            continue;
        }
        int out_row = out_rows[edge];
        int kernel_id = kernel_ids[edge];
        if (out_row < 0 || out_row >= out_capacity || kernel_id < 0) {
            continue;
        }
        for (int co = 0; co < out_channels; ++co) {
            float value =
                float(cotangent[out_row * cotangent_s0 + co * cotangent_s1]);
            acc0 += value * float4(
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    0,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )]),
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    1,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )]),
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    2,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )]),
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    3,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )])
                            );
            acc1 += value * float4(
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    4,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )]),
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    5,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )]),
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    6,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )]),
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    7,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )])
                            );
            acc2 += value * float4(
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    8,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )]),
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    9,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )]),
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    10,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )]),
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    11,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )])
                            );
            acc3 += value * float4(
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    12,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )]),
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    13,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )]),
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    14,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )]),
                                float(weights[sparse_conv_weight_offset(
                                    kernel_id,
                                    15,
                                    co,
                                    weight_layout,
                                    kernel_x,
                                    kernel_y,
                                    kernel_z,
                                    weight_s0,
                                    weight_s1,
                                    weight_s2,
                                    weight_s3,
                                    weight_s4
                                )])
                            );
        }
    }
    int grad_base = in_row * 16;
    grad[grad_base] = half(acc0.x);
    grad[grad_base + 1] = half(acc0.y);
    grad[grad_base + 2] = half(acc0.z);
    grad[grad_base + 3] = half(acc0.w);
    grad[grad_base + 4] = half(acc1.x);
    grad[grad_base + 5] = half(acc1.y);
    grad[grad_base + 6] = half(acc1.z);
    grad[grad_base + 7] = half(acc1.w);
    grad[grad_base + 8] = half(acc2.x);
    grad[grad_base + 9] = half(acc2.y);
    grad[grad_base + 10] = half(acc2.z);
    grad[grad_base + 11] = half(acc2.w);
    grad[grad_base + 12] = half(acc3.x);
    grad[grad_base + 13] = half(acc3.y);
    grad[grad_base + 14] = half(acc3.z);
    grad[grad_base + 15] = half(acc3.w);
    (void)in_rows;
    (void)row_offsets;
    (void)in_channels;
}

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
