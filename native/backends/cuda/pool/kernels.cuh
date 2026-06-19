#pragma once

#if __has_include(<cuda_runtime_api.h>)
#include <cuda_runtime_api.h>
#endif

#ifdef __CUDACC__
#define MLX_LATTICE_CUDA_KERNEL __global__
#else
#define MLX_LATTICE_CUDA_KERNEL
#endif

namespace mlx_lattice::backend::cuda::pool {

MLX_LATTICE_CUDA_KERNEL void sparse_pool_relation_f32_i32(
    const float* feats,
    const int* in_rows,
    const int* out_rows,
    const int* kernel_ids,
    const int* row_offsets,
    const int* counts,
    float* out,
    int reduce,
    int out_capacity,
    int channels,
    int feat_s0,
    int feat_s1
);

MLX_LATTICE_CUDA_KERNEL void sparse_pool_relation_block_sum_f32_i32(
    const float* feats,
    const int* in_rows,
    const int* row_offsets,
    const int* counts,
    float* out,
    int out_capacity,
    int channels,
    int feat_s0,
    int feat_s1
);

MLX_LATTICE_CUDA_KERNEL void sparse_pool_relation_block_max_f32_i32(
    const float* feats,
    const int* in_rows,
    const int* row_offsets,
    const int* counts,
    float* out,
    int out_capacity,
    int channels,
    int feat_s0,
    int feat_s1
);

MLX_LATTICE_CUDA_KERNEL void sparse_pool_relation_block_avg_f32_i32(
    const float* feats,
    const int* in_rows,
    const int* row_offsets,
    const int* counts,
    float* out,
    int out_capacity,
    int channels,
    int feat_s0,
    int feat_s1
);

MLX_LATTICE_CUDA_KERNEL void sparse_pool_relation_sum_avg_input_grad_f32_i32(
    const float* cotangent,
    const float* feats,
    const float* pooled,
    const int* in_rows,
    const int* out_rows,
    const int* kernel_ids,
    const int* row_offsets,
    const int* counts,
    const int* in_row_offsets,
    const int* in_edge_ids,
    float* out,
    int reduce,
    int in_capacity,
    int out_capacity,
    int n_kernels,
    int channels,
    int cot_s0,
    int cot_s1,
    int feat_s0,
    int feat_s1,
    int pooled_s0,
    int pooled_s1
);

MLX_LATTICE_CUDA_KERNEL void sparse_pool_relation_max_input_grad_f32_i32(
    const float* cotangent,
    const float* feats,
    const float* pooled,
    const int* in_rows,
    const int* out_rows,
    const int* kernel_ids,
    const int* row_offsets,
    const int* counts,
    const int* in_row_offsets,
    const int* in_edge_ids,
    float* out,
    int reduce,
    int in_capacity,
    int out_capacity,
    int n_kernels,
    int channels,
    int cot_s0,
    int cot_s1,
    int feat_s0,
    int feat_s1,
    int pooled_s0,
    int pooled_s1
);

MLX_LATTICE_CUDA_KERNEL void sparse_pool_relation_exclusive_input_grad_f32_i32(
    const float* cotangent,
    const float* feats,
    const float* pooled,
    const int* in_rows,
    const int* out_rows,
    const int* kernel_ids,
    const int* row_offsets,
    const int* counts,
    const int* in_row_offsets,
    const int* in_edge_ids,
    float* out,
    int reduce,
    int in_capacity,
    int out_capacity,
    int n_kernels,
    int channels,
    int cot_s0,
    int cot_s1,
    int feat_s0,
    int feat_s1,
    int pooled_s0,
    int pooled_s1
);

MLX_LATTICE_CUDA_KERNEL void sparse_pool_relation_jvp_f32_i32(
    const float* tangent,
    const float* feats,
    const float* pooled,
    const int* in_rows,
    const int* out_rows,
    const int* kernel_ids,
    const int* row_offsets,
    const int* counts,
    float* out,
    int reduce,
    int in_capacity,
    int out_capacity,
    int n_kernels,
    int channels,
    int tan_s0,
    int tan_s1,
    int feat_s0,
    int feat_s1,
    int pooled_s0,
    int pooled_s1
);

} // namespace mlx_lattice::backend::cuda::pool
