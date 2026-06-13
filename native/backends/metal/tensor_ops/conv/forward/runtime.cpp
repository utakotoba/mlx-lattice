#include "backends/metal/tensor_ops/conv/forward/runtime.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

#include "backends/array_utils.h"
#include "backends/metal/runtime_utils.h"
#include "backends/metal/tensor_ops/capabilities.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace mlx_lattice::backend::metal::tensor_ops::conv::forward {
namespace {

constexpr int kChannels = 16;

#ifdef _METAL_
struct KernelNames {
    int in_channels;
    int out_channels;
    const char* fp32;
    const char* fp16;
    int co_blocks_per_threadgroup;
};

constexpr KernelNames kKernels[] = {
    {16,
     16,
     "sparse_relation_conv_forward_implicit_gemm_f32_i32_cin16_cout16_sg1",
     "sparse_relation_conv_forward_implicit_gemm_f16_i32_cin16_cout16_sg1",
     1},
    {16,
     32,
     "sparse_relation_conv_forward_implicit_gemm_f32_i32_cin16_cout32_sg1",
     "sparse_relation_conv_forward_implicit_gemm_f16_i32_cin16_cout32_sg1",
     1},
    {16,
     64,
     "sparse_relation_conv_forward_implicit_gemm_f32_i32_cin16_cout64_sg1",
     "sparse_relation_conv_forward_implicit_gemm_f16_i32_cin16_cout64_sg1",
     1},
    {32,
     16,
     "sparse_relation_conv_forward_implicit_gemm_f32_i32_cin32_cout16_sg1",
     "sparse_relation_conv_forward_implicit_gemm_f16_i32_cin32_cout16_sg1",
     1},
    {32,
     32,
     "sparse_relation_conv_forward_implicit_gemm_f32_i32_cin32_cout32_sg1",
     "sparse_relation_conv_forward_implicit_gemm_f16_i32_cin32_cout32_sg1",
     1},
    {32,
     64,
     "sparse_relation_conv_forward_implicit_gemm_f32_i32_cin32_cout64_sg1",
     "sparse_relation_conv_forward_implicit_gemm_f16_i32_cin32_cout64_sg1",
     1},
    {64,
     16,
     "sparse_relation_conv_forward_implicit_gemm_f32_i32_cin64_cout16_sg1",
     "sparse_relation_conv_forward_implicit_gemm_f16_i32_cin64_cout16_sg1",
     1},
    {64,
     32,
     "sparse_relation_conv_forward_implicit_gemm_f32_i32_cin64_cout32_sg1",
     "sparse_relation_conv_forward_implicit_gemm_f16_i32_cin64_cout32_sg1",
     1},
    {64,
     64,
     "sparse_relation_conv_forward_implicit_gemm_f32_i32_cin64_cout64_sg1",
     "sparse_relation_conv_forward_implicit_gemm_f16_i32_cin64_cout64_sg1",
     1},
};

const KernelNames* find_kernel(SparseConvShape shape) {
    for (const auto& kernel : kKernels) {
        if (shape.in_channels == kernel.in_channels &&
            shape.out_channels == kernel.out_channels) {
            return &kernel;
        }
    }
    return nullptr;
}

bool is_float16(const mx::array& array) { return array.dtype() == mx::float16; }

int stride_at(const mx::array& array, int dim) {
    return static_cast<int>(array.strides(dim));
}
#endif

} // namespace

bool supports(SparseConvShape shape) {
    auto supported_channels = [](int channels) {
        return channels == 16 || channels == 32 || channels == 64;
    };
    return supported_channels(shape.in_channels) &&
           supported_channels(shape.out_channels) && shape.n_kernels == 27;
}

bool is_preferred(SparseConvShape shape, const mx::Stream& stream) {
    (void)shape;
    (void)stream;
    // Correct full-grid dispatch is slower than the classic forward kernels for
    // all measured 16/32/64 channel pairs at 100K rows. Keep the implementation
    // available for experiments, but do not route production workloads here.
    return false;
}

void encode(
    SparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    mx::array& out
) {
#ifdef _METAL_
    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);
    const auto* selected = find_kernel(shape);
    if (selected == nullptr) {
        throw std::runtime_error(
            "Unsupported TensorOps forward convolution shape."
        );
    }
    auto kernel = device.get_kernel(
        is_float16(inputs[0]) ? selected->fp16 : selected->fp32, library
    );

    encoder.set_compute_pipeline_state(kernel);
    for (int index = 0; index < int(inputs.size()); ++index) {
        encoder.set_input_array(inputs[index], index);
    }
    encoder.set_output_array(out, 7);
    encoder.set_bytes(static_cast<int>(inputs[2].shape(0)), 8);
    encoder.set_bytes(shape.out_capacity, 9);
    encoder.set_bytes(shape.n_kernels, 10);
    encoder.set_bytes(shape.in_channels, 11);
    encoder.set_bytes(shape.out_channels, 12);
    encoder.set_bytes(stride_at(inputs[0], 0), 13);
    encoder.set_bytes(stride_at(inputs[0], 1), 14);
    encoder.set_bytes(stride_at(inputs[1], 0), 15);
    encoder.set_bytes(stride_at(inputs[1], 1), 16);
    encoder.set_bytes(stride_at(inputs[1], 2), 17);
    encoder.set_bytes(inputs[1].ndim() == 5 ? stride_at(inputs[1], 3) : 0, 18);
    encoder.set_bytes(inputs[1].ndim() == 5 ? stride_at(inputs[1], 4) : 0, 19);
    encoder.set_bytes(shape.weight_layout, 20);
    encoder.set_bytes(shape.kernel_x, 21);
    encoder.set_bytes(shape.kernel_y, 22);
    encoder.set_bytes(shape.kernel_z, 23);
    auto row_tiles = static_cast<std::size_t>(
        (shape.out_capacity + kChannels - 1) / kChannels
    );
    auto co_groups = static_cast<std::size_t>(
        (shape.out_channels / kChannels + selected->co_blocks_per_threadgroup -
         1) /
        selected->co_blocks_per_threadgroup
    );
    encoder.dispatch_threadgroups(
        MTL::Size(row_tiles * co_groups, 1, 1),
        MTL::Size(
            static_cast<std::size_t>(selected->co_blocks_per_threadgroup) * 32,
            1,
            1
        )
    );
#else
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)out;
    throw std::runtime_error("Metal support is not available.");
#endif
}

} // namespace mlx_lattice::backend::metal::tensor_ops::conv::forward
