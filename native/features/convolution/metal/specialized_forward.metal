#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
#include <metal_tensor>

using namespace metal;
using namespace mpp::tensor_ops;

// clang-format off
#include "native/features/convolution/metal/common.metal"
#include "native/features/convolution/metal/vector_io.metal"
#include "native/features/convolution/metal/dense_forward.metal"
// clang-format on

// Specialized kernels share the generic convolution binding ABI, so some
// bound buffers are intentionally unused by a given specialization.
#pragma clang diagnostic ignored "-Wunused-parameter"

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
