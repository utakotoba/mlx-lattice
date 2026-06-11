#include <metal_stdlib>

using namespace metal;

#include "native/backends/metal/conv/common.metal"

[[kernel]] void sparse_relation_conv_weight_grad_tensor_ops_reduce_f32(
    device const float* partials [[buffer(0)]],
    device float* grad [[buffer(1)]],
    constant const int& n_kernels [[buffer(2)]],
    constant const int& partitions [[buffer(3)]],
    constant const int& weight_layout [[buffer(4)]],
    constant const int& kernel_x [[buffer(5)]],
    constant const int& kernel_y [[buffer(6)]],
    constant const int& kernel_z [[buffer(7)]],
    constant const int& in_channels [[buffer(8)]],
    constant const int& out_channels [[buffer(9)]],
    uint elem [[thread_position_in_grid]]
) {
    const int values_per_kernel = 16 * 16;
    const int channel_blocks = in_channels / 16;
    const int channel_tiles = channel_blocks * channel_blocks;
    const int total = n_kernels * channel_tiles * values_per_kernel;
    if (elem >= uint(total)) {
        return;
    }

    const int kernel_tile = int(elem) / values_per_kernel;
    const int kernel_id = kernel_tile / channel_tiles;
    const int channel_tile = kernel_tile - kernel_id * channel_tiles;
    const int channel = int(elem) - kernel_tile * values_per_kernel;
    const int ci_base = (channel_tile / channel_blocks) * 16;
    const int co_base = (channel_tile - (ci_base / 16) * channel_blocks) * 16;
    const int ci = ci_base + channel / 16;
    const int co = co_base + channel - (channel / 16) * 16;
    float value = 0.0f;
    for (int partition = 0; partition < partitions; ++partition) {
        value += partials
            [((partition * n_kernels + kernel_id) * channel_tiles +
              channel_tile) *
                 values_per_kernel +
             channel];
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
    )] = value;
}

[[kernel]] void sparse_relation_conv_weight_grad_tensor_ops_reduce_f16(
    device const float* partials [[buffer(0)]],
    device half* grad [[buffer(1)]],
    constant const int& n_kernels [[buffer(2)]],
    constant const int& partitions [[buffer(3)]],
    constant const int& weight_layout [[buffer(4)]],
    constant const int& kernel_x [[buffer(5)]],
    constant const int& kernel_y [[buffer(6)]],
    constant const int& kernel_z [[buffer(7)]],
    constant const int& in_channels [[buffer(8)]],
    constant const int& out_channels [[buffer(9)]],
    uint elem [[thread_position_in_grid]]
) {
    const int values_per_kernel = 16 * 16;
    const int channel_blocks = in_channels / 16;
    const int channel_tiles = channel_blocks * channel_blocks;
    const int total = n_kernels * channel_tiles * values_per_kernel;
    if (elem >= uint(total)) {
        return;
    }

    const int kernel_tile = int(elem) / values_per_kernel;
    const int kernel_id = kernel_tile / channel_tiles;
    const int channel_tile = kernel_tile - kernel_id * channel_tiles;
    const int channel = int(elem) - kernel_tile * values_per_kernel;
    const int ci_base = (channel_tile / channel_blocks) * 16;
    const int co_base = (channel_tile - (ci_base / 16) * channel_blocks) * 16;
    const int ci = ci_base + channel / 16;
    const int co = co_base + channel - (channel / 16) * 16;
    float value = 0.0f;
    for (int partition = 0; partition < partitions; ++partition) {
        value += partials
            [((partition * n_kernels + kernel_id) * channel_tiles +
              channel_tile) *
                 values_per_kernel +
             channel];
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
    )] = half(value);
}
