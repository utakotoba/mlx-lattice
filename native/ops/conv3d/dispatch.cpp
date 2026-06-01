#include "ops/conv3d/dispatch.h"

#include <stdexcept>

#if MLX_LATTICE_HAS_CUDA
#include "backends/cuda/conv3d.h"
#include "mlx/backend/cuda/cuda.h"
#endif
#include "backends/metal/conv3d.h"

namespace mlx_lattice {

namespace {

void require_gpu_backend() {
    throw std::runtime_error("No GPU backend is available.");
}

} // namespace

void eval_gpu_conv3d_feats(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs,
    mx::Stream stream,
    int rows,
    int in_channels,
    int out_channels
) {
#if MLX_LATTICE_HAS_CUDA
    if (mx::cu::is_available()) {
        cuda::eval_conv3d_feats(
            inputs, outputs, stream, rows, in_channels, out_channels
        );
        return;
    }
#endif
#if MLX_LATTICE_HAS_METAL
    metal::eval_conv3d_feats(
        inputs, outputs, stream, rows, in_channels, out_channels
    );
    return;
#endif
    require_gpu_backend();
}

void eval_gpu_conv3d_subm_feats(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs,
    mx::Stream stream,
    int rows,
    int in_channels,
    int out_channels,
    int center_kernel
) {
#if MLX_LATTICE_HAS_CUDA
    if (mx::cu::is_available()) {
        cuda::eval_conv3d_subm_feats(
            inputs,
            outputs,
            stream,
            rows,
            in_channels,
            out_channels,
            center_kernel
        );
        return;
    }
#endif
#if MLX_LATTICE_HAS_METAL
    metal::eval_conv3d_subm_feats(
        inputs, outputs, stream, rows, in_channels, out_channels, center_kernel
    );
    return;
#endif
    require_gpu_backend();
}

void eval_gpu_conv3d_residual_feats(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs,
    mx::Stream stream,
    int rows,
    int in_channels,
    int out_channels
) {
#if MLX_LATTICE_HAS_CUDA
    if (mx::cu::is_available()) {
        cuda::eval_conv3d_residual_feats(
            inputs, outputs, stream, rows, in_channels, out_channels
        );
        return;
    }
#endif
#if MLX_LATTICE_HAS_METAL
    metal::eval_conv3d_residual_feats(
        inputs, outputs, stream, rows, in_channels, out_channels
    );
    return;
#endif
    require_gpu_backend();
}

void eval_gpu_pool3d_feats(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs,
    mx::Stream stream,
    int rows,
    int channels
) {
#if MLX_LATTICE_HAS_CUDA
    if (mx::cu::is_available()) {
        cuda::eval_pool3d_feats(inputs, outputs, stream, rows, channels);
        return;
    }
#endif
#if MLX_LATTICE_HAS_METAL
    metal::eval_pool3d_feats(inputs, outputs, stream, rows, channels);
    return;
#endif
    require_gpu_backend();
}

void eval_gpu_pool3d_feats_grad(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs,
    mx::Stream stream,
    int rows,
    int channels
) {
#if MLX_LATTICE_HAS_CUDA
    if (mx::cu::is_available()) {
        cuda::eval_pool3d_feats_grad(inputs, outputs, stream, rows, channels);
        return;
    }
#endif
#if MLX_LATTICE_HAS_METAL
    metal::eval_pool3d_feats_grad(inputs, outputs, stream, rows, channels);
    return;
#endif
    require_gpu_backend();
}

void eval_gpu_conv3d_feats_grad(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs,
    mx::Stream stream,
    int rows,
    int in_channels,
    int out_channels
) {
#if MLX_LATTICE_HAS_CUDA
    if (mx::cu::is_available()) {
        cuda::eval_conv3d_feats_grad(
            inputs, outputs, stream, rows, in_channels, out_channels
        );
        return;
    }
#endif
#if MLX_LATTICE_HAS_METAL
    metal::eval_conv3d_feats_grad(
        inputs, outputs, stream, rows, in_channels, out_channels
    );
    return;
#endif
    require_gpu_backend();
}

void eval_gpu_conv3d_weight_grad(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs,
    mx::Stream stream,
    int kernels,
    int in_channels,
    int out_channels
) {
#if MLX_LATTICE_HAS_CUDA
    if (mx::cu::is_available()) {
        cuda::eval_conv3d_weight_grad(
            inputs, outputs, stream, kernels, in_channels, out_channels
        );
        return;
    }
#endif
#if MLX_LATTICE_HAS_METAL
    metal::eval_conv3d_weight_grad(
        inputs, outputs, stream, kernels, in_channels, out_channels
    );
    return;
#endif
    require_gpu_backend();
}

} // namespace mlx_lattice
