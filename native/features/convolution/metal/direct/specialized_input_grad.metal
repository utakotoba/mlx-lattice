#include <metal_stdlib>

using namespace metal;

// clang-format off
#include "native/features/convolution/metal/common.metal"
#include "native/features/convolution/metal/direct/vector_io.metal"
#include "native/features/convolution/metal/direct/dense_input_grad.metal"
// clang-format on

// Specialized kernels share the generic convolution binding ABI, so some
// bound buffers are intentionally unused by a given specialization.
#pragma clang diagnostic ignored "-Wunused-parameter"

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
