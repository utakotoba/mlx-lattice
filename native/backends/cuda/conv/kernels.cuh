#pragma once

#if __has_include(<cuda_fp16.h>)
#include <cuda_fp16.h>
#else
struct __half {
    unsigned short value;
};
#endif

#ifdef __CUDACC__
#define MLX_LATTICE_CUDA_KERNEL __global__
#else
#define MLX_LATTICE_CUDA_KERNEL
#endif

namespace mlx_lattice::backend::cuda::conv {

struct ConvShapeArgs {
    int edge_capacity;
    int in_capacity;
    int out_capacity;
    int n_kernels;
    int in_channels;
    int out_channels;
    int weight_layout;
    int kernel_x;
    int kernel_y;
    int kernel_z;
};

struct ConvStrideArgs {
    int feat_s0;
    int feat_s1;
    int cot_s0;
    int cot_s1;
    int weight_s0;
    int weight_s1;
    int weight_s2;
    int weight_s3;
    int weight_s4;
    int out_s0;
    int out_s1;
    int out_s2;
    int out_s3;
    int out_s4;
};

MLX_LATTICE_CUDA_KERNEL void sparse_conv_forward_f32(
    const float* feats,
    const float* weights,
    const int* in_rows,
    const int* out_rows,
    const int* kernel_ids,
    const int* counts,
    const int* row_offsets,
    float* out,
    ConvShapeArgs shape,
    ConvStrideArgs strides
);

MLX_LATTICE_CUDA_KERNEL void sparse_conv_forward_f16(
    const __half* feats,
    const __half* weights,
    const int* in_rows,
    const int* out_rows,
    const int* kernel_ids,
    const int* counts,
    const int* row_offsets,
    __half* out,
    ConvShapeArgs shape,
    ConvStrideArgs strides
);

MLX_LATTICE_CUDA_KERNEL void sparse_conv_forward_f32_c16(
    const float* feats,
    const float* weights,
    const int* in_rows,
    const int* out_rows,
    const int* kernel_ids,
    const int* counts,
    const int* row_offsets,
    float* out,
    ConvShapeArgs shape,
    ConvStrideArgs strides
);

MLX_LATTICE_CUDA_KERNEL void sparse_conv_forward_f32_c32(
    const float* feats,
    const float* weights,
    const int* in_rows,
    const int* out_rows,
    const int* kernel_ids,
    const int* counts,
    const int* row_offsets,
    float* out,
    ConvShapeArgs shape,
    ConvStrideArgs strides
);

MLX_LATTICE_CUDA_KERNEL void sparse_conv_forward_f32_c64(
    const float* feats,
    const float* weights,
    const int* in_rows,
    const int* out_rows,
    const int* kernel_ids,
    const int* counts,
    const int* row_offsets,
    float* out,
    ConvShapeArgs shape,
    ConvStrideArgs strides
);

MLX_LATTICE_CUDA_KERNEL void sparse_conv_forward_f16_c16(
    const __half* feats,
    const __half* weights,
    const int* in_rows,
    const int* out_rows,
    const int* kernel_ids,
    const int* counts,
    const int* row_offsets,
    __half* out,
    ConvShapeArgs shape,
    ConvStrideArgs strides
);

MLX_LATTICE_CUDA_KERNEL void sparse_conv_forward_f16_c32(
    const __half* feats,
    const __half* weights,
    const int* in_rows,
    const int* out_rows,
    const int* kernel_ids,
    const int* counts,
    const int* row_offsets,
    __half* out,
    ConvShapeArgs shape,
    ConvStrideArgs strides
);

MLX_LATTICE_CUDA_KERNEL void sparse_conv_forward_f16_c64(
    const __half* feats,
    const __half* weights,
    const int* in_rows,
    const int* out_rows,
    const int* kernel_ids,
    const int* counts,
    const int* row_offsets,
    __half* out,
    ConvShapeArgs shape,
    ConvStrideArgs strides
);

MLX_LATTICE_CUDA_KERNEL void sparse_conv_input_grad_f32(
    const float* cotangent,
    const float* weights,
    const int* in_rows,
    const int* out_rows,
    const int* kernel_ids,
    const int* counts,
    const int* row_offsets,
    const int* in_row_offsets,
    const int* in_edge_ids,
    float* out,
    ConvShapeArgs shape,
    ConvStrideArgs strides
);

MLX_LATTICE_CUDA_KERNEL void sparse_conv_input_grad_f16(
    const __half* cotangent,
    const __half* weights,
    const int* in_rows,
    const int* out_rows,
    const int* kernel_ids,
    const int* counts,
    const int* row_offsets,
    const int* in_row_offsets,
    const int* in_edge_ids,
    __half* out,
    ConvShapeArgs shape,
    ConvStrideArgs strides
);

MLX_LATTICE_CUDA_KERNEL void sparse_conv_weight_grad_f32(
    const float* feats,
    const float* cotangent,
    const int* in_rows,
    const int* out_rows,
    const int* kernel_ids,
    const int* counts,
    const int* row_offsets,
    const int* kernel_row_offsets,
    const int* kernel_edge_ids,
    float* out,
    ConvShapeArgs shape,
    ConvStrideArgs strides
);

MLX_LATTICE_CUDA_KERNEL void sparse_conv_weight_grad_f16(
    const __half* feats,
    const __half* cotangent,
    const int* in_rows,
    const int* out_rows,
    const int* kernel_ids,
    const int* counts,
    const int* row_offsets,
    const int* kernel_row_offsets,
    const int* kernel_edge_ids,
    __half* out,
    ConvShapeArgs shape,
    ConvStrideArgs strides
);

} // namespace mlx_lattice::backend::cuda::conv
