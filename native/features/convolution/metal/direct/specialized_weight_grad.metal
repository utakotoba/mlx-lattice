#include <metal_stdlib>

using namespace metal;

// clang-format off
#include "native/features/convolution/metal/common.metal"
#include "native/features/convolution/metal/direct/vector_io.metal"
#include "native/features/convolution/metal/direct/dense_weight_grad.metal"
// clang-format on

// Specialized kernels share the generic convolution binding ABI, so some
// bound buffers are intentionally unused by a given specialization.
#pragma clang diagnostic ignored "-Wunused-parameter"

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
