#include "backends/cuda/conv/kernels.cuh"

#include <cuda_runtime.h>

namespace mlx_lattice::backend::cuda::conv {
namespace {

__device__ int elem_1d() { return int(blockIdx.x * blockDim.x + threadIdx.x); }

__device__ int weight_offset(
    ConvShapeArgs shape,
    ConvStrideArgs strides,
    int kernel,
    int in_channel,
    int out_channel
) {
    if (shape.weight_layout == 0) {
        return kernel * strides.weight_s0 + in_channel * strides.weight_s1 +
               out_channel * strides.weight_s2;
    }

    int xy = shape.kernel_y * shape.kernel_z;
    int kx = kernel / xy;
    int rem = kernel % xy;
    int ky = rem / shape.kernel_z;
    int kz = rem % shape.kernel_z;
    return out_channel * strides.weight_s0 + kx * strides.weight_s1 +
           ky * strides.weight_s2 + kz * strides.weight_s3 +
           in_channel * strides.weight_s4;
}

__device__ int out_weight_offset(
    ConvShapeArgs shape,
    ConvStrideArgs strides,
    int kernel,
    int in_channel,
    int out_channel
) {
    if (shape.weight_layout == 0) {
        return kernel * strides.out_s0 + in_channel * strides.out_s1 +
               out_channel * strides.out_s2;
    }

    int xy = shape.kernel_y * shape.kernel_z;
    int kx = kernel / xy;
    int rem = kernel % xy;
    int ky = rem / shape.kernel_z;
    int kz = rem % shape.kernel_z;
    return out_channel * strides.out_s0 + kx * strides.out_s1 +
           ky * strides.out_s2 + kz * strides.out_s3 +
           in_channel * strides.out_s4;
}

template <typename T> __device__ float load_value(const T* ptr, int offset) {
    return float(ptr[offset]);
}

template <> __device__ float load_value<__half>(const __half* ptr, int offset) {
    return __half2float(ptr[offset]);
}

template <typename T> __device__ void store_value(T* ptr, int offset, float v) {
    ptr[offset] = T(v);
}

template <>
__device__ void store_value<__half>(__half* ptr, int offset, float v) {
    ptr[offset] = __float2half(v);
}

template <typename T>
__device__ void forward_kernel(
    const T* feats,
    const T* weights,
    const int* in_rows,
    const int* kernel_ids,
    const int* counts,
    const int* row_offsets,
    T* out,
    ConvShapeArgs shape,
    ConvStrideArgs strides
) {
    int elem = elem_1d();
    int total = shape.out_capacity * shape.out_channels;
    if (elem >= total) {
        return;
    }

    int out_row = elem / shape.out_channels;
    int out_channel = elem - out_row * shape.out_channels;
    int active_out = min(counts[1], shape.out_capacity);
    int active_edges = min(counts[0], shape.edge_capacity);
    float acc = 0.0f;
    if (out_row < active_out) {
        for (int edge = row_offsets[out_row]; edge < row_offsets[out_row + 1];
             ++edge) {
            if (edge < 0 || edge >= active_edges) {
                continue;
            }
            int in_row = in_rows[edge];
            int kernel = kernel_ids[edge];
            if (in_row < 0 || kernel < 0) {
                continue;
            }
            for (int ci = 0; ci < shape.in_channels; ++ci) {
                float feat = load_value(
                    feats, in_row * strides.feat_s0 + ci * strides.feat_s1
                );
                float weight = load_value(
                    weights,
                    weight_offset(shape, strides, kernel, ci, out_channel)
                );
                acc += feat * weight;
            }
        }
    }
    store_value(out, elem, acc);
}

template <typename T, int InChannels, int OutChannels>
__device__ void forward_channels_kernel(
    const T* feats,
    const T* weights,
    const int* in_rows,
    const int* kernel_ids,
    const int* counts,
    const int* row_offsets,
    T* out,
    ConvShapeArgs shape,
    ConvStrideArgs strides
) {
    int out_row = elem_1d();
    if (out_row >= shape.out_capacity) {
        return;
    }

    float acc[OutChannels];
    for (int co = 0; co < OutChannels; ++co) {
        acc[co] = 0.0f;
    }

    int active_out = min(counts[1], shape.out_capacity);
    int active_edges = min(counts[0], shape.edge_capacity);
    if (out_row < active_out) {
        for (int edge = row_offsets[out_row]; edge < row_offsets[out_row + 1];
             ++edge) {
            if (edge < 0 || edge >= active_edges) {
                continue;
            }
            int in_row = in_rows[edge];
            int kernel = kernel_ids[edge];
            if (in_row < 0 || kernel < 0) {
                continue;
            }
            for (int ci = 0; ci < InChannels; ++ci) {
                float feat = load_value(
                    feats, in_row * strides.feat_s0 + ci * strides.feat_s1
                );
                for (int co = 0; co < OutChannels; ++co) {
                    acc[co] +=
                        feat * load_value(
                                   weights,
                                   weight_offset(shape, strides, kernel, ci, co)
                               );
                }
            }
        }
    }

    for (int co = 0; co < OutChannels; ++co) {
        store_value(out, out_row * OutChannels + co, acc[co]);
    }
}

template <typename T>
__device__ void input_grad_kernel(
    const T* cotangent,
    const T* weights,
    const int* out_rows,
    const int* kernel_ids,
    const int* counts,
    const int* in_row_offsets,
    const int* in_edge_ids,
    T* out,
    ConvShapeArgs shape,
    ConvStrideArgs strides
) {
    int elem = elem_1d();
    int total = shape.in_capacity * shape.in_channels;
    if (elem >= total) {
        return;
    }

    int in_row = elem / shape.in_channels;
    int in_channel = elem - in_row * shape.in_channels;
    int active_edges = min(counts[0], shape.edge_capacity);
    float acc = 0.0f;
    for (int cursor = in_row_offsets[in_row];
         cursor < in_row_offsets[in_row + 1];
         ++cursor) {
        int edge = in_edge_ids[cursor];
        if (edge < 0 || edge >= active_edges) {
            continue;
        }
        int out_row = out_rows[edge];
        int kernel = kernel_ids[edge];
        if (out_row < 0 || kernel < 0 || out_row >= shape.out_capacity) {
            continue;
        }
        for (int co = 0; co < shape.out_channels; ++co) {
            float cot = load_value(
                cotangent, out_row * strides.cot_s0 + co * strides.cot_s1
            );
            float weight = load_value(
                weights, weight_offset(shape, strides, kernel, in_channel, co)
            );
            acc += cot * weight;
        }
    }
    store_value(out, elem, acc);
}

template <typename T>
__device__ void weight_grad_kernel(
    const T* feats,
    const T* cotangent,
    const int* in_rows,
    const int* out_rows,
    const int* counts,
    const int* kernel_row_offsets,
    const int* kernel_edge_ids,
    T* out,
    ConvShapeArgs shape,
    ConvStrideArgs strides
) {
    int elem = elem_1d();
    int total = shape.n_kernels * shape.in_channels * shape.out_channels;
    if (elem >= total) {
        return;
    }

    int out_channel = elem % shape.out_channels;
    int in_channel = (elem / shape.out_channels) % shape.in_channels;
    int kernel = elem / (shape.in_channels * shape.out_channels);
    int active_edges = min(counts[0], shape.edge_capacity);
    float acc = 0.0f;
    for (int cursor = kernel_row_offsets[kernel];
         cursor < kernel_row_offsets[kernel + 1];
         ++cursor) {
        int edge = kernel_edge_ids[cursor];
        if (edge < 0 || edge >= active_edges) {
            continue;
        }
        int in_row = in_rows[edge];
        int out_row = out_rows[edge];
        if (in_row < 0 || out_row < 0 || out_row >= shape.out_capacity) {
            continue;
        }
        float feat = load_value(
            feats, in_row * strides.feat_s0 + in_channel * strides.feat_s1
        );
        float cot = load_value(
            cotangent, out_row * strides.cot_s0 + out_channel * strides.cot_s1
        );
        acc += feat * cot;
    }
    store_value(
        out,
        out_weight_offset(shape, strides, kernel, in_channel, out_channel),
        acc
    );
}

} // namespace

__global__ void sparse_conv_forward_f32(
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
) {
    (void)out_rows;
    forward_kernel(
        feats,
        weights,
        in_rows,
        kernel_ids,
        counts,
        row_offsets,
        out,
        shape,
        strides
    );
}

__global__ void sparse_conv_forward_f16(
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
) {
    (void)out_rows;
    forward_kernel(
        feats,
        weights,
        in_rows,
        kernel_ids,
        counts,
        row_offsets,
        out,
        shape,
        strides
    );
}

template <typename T, int Channels>
__device__ void forward_square_channels_entry(
    const T* feats,
    const T* weights,
    const int* in_rows,
    const int* out_rows,
    const int* kernel_ids,
    const int* counts,
    const int* row_offsets,
    T* out,
    ConvShapeArgs shape,
    ConvStrideArgs strides
) {
    (void)out_rows;
    forward_channels_kernel<T, Channels, Channels>(
        feats,
        weights,
        in_rows,
        kernel_ids,
        counts,
        row_offsets,
        out,
        shape,
        strides
    );
}

__global__ void sparse_conv_forward_f32_c16(
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
) {
    forward_square_channels_entry<float, 16>(
        feats,
        weights,
        in_rows,
        out_rows,
        kernel_ids,
        counts,
        row_offsets,
        out,
        shape,
        strides
    );
}

__global__ void sparse_conv_forward_f32_c32(
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
) {
    forward_square_channels_entry<float, 32>(
        feats,
        weights,
        in_rows,
        out_rows,
        kernel_ids,
        counts,
        row_offsets,
        out,
        shape,
        strides
    );
}

__global__ void sparse_conv_forward_f32_c64(
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
) {
    forward_square_channels_entry<float, 64>(
        feats,
        weights,
        in_rows,
        out_rows,
        kernel_ids,
        counts,
        row_offsets,
        out,
        shape,
        strides
    );
}

__global__ void sparse_conv_forward_f16_c16(
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
) {
    forward_square_channels_entry<__half, 16>(
        feats,
        weights,
        in_rows,
        out_rows,
        kernel_ids,
        counts,
        row_offsets,
        out,
        shape,
        strides
    );
}

__global__ void sparse_conv_forward_f16_c32(
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
) {
    forward_square_channels_entry<__half, 32>(
        feats,
        weights,
        in_rows,
        out_rows,
        kernel_ids,
        counts,
        row_offsets,
        out,
        shape,
        strides
    );
}

__global__ void sparse_conv_forward_f16_c64(
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
) {
    forward_square_channels_entry<__half, 64>(
        feats,
        weights,
        in_rows,
        out_rows,
        kernel_ids,
        counts,
        row_offsets,
        out,
        shape,
        strides
    );
}

__global__ void sparse_conv_input_grad_f32(
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
) {
    (void)in_rows;
    (void)row_offsets;
    input_grad_kernel(
        cotangent,
        weights,
        out_rows,
        kernel_ids,
        counts,
        in_row_offsets,
        in_edge_ids,
        out,
        shape,
        strides
    );
}

__global__ void sparse_conv_input_grad_f16(
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
) {
    (void)in_rows;
    (void)row_offsets;
    input_grad_kernel(
        cotangent,
        weights,
        out_rows,
        kernel_ids,
        counts,
        in_row_offsets,
        in_edge_ids,
        out,
        shape,
        strides
    );
}

__global__ void sparse_conv_weight_grad_f32(
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
) {
    (void)kernel_ids;
    (void)row_offsets;
    weight_grad_kernel(
        feats,
        cotangent,
        in_rows,
        out_rows,
        counts,
        kernel_row_offsets,
        kernel_edge_ids,
        out,
        shape,
        strides
    );
}

__global__ void sparse_conv_weight_grad_f16(
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
) {
    (void)kernel_ids;
    (void)row_offsets;
    weight_grad_kernel(
        feats,
        cotangent,
        in_rows,
        out_rows,
        counts,
        kernel_row_offsets,
        kernel_edge_ids,
        out,
        shape,
        strides
    );
}

} // namespace mlx_lattice::backend::cuda::conv
