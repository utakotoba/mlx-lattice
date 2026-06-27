#include "features/pooling/metal/runtime.h"

#include <algorithm>
#include <stdexcept>

#include "foundation/array_utils.h"
#include "platform/metal/runtime_utils.h"

#ifdef _METAL_
#endif

namespace mlx_lattice::backend::metal::pool {
namespace {

int reduce_id(PoolReduceOp op) {
    switch (op) {
    case PoolReduceOp::Sum:
        return 0;
    case PoolReduceOp::Max:
        return 1;
    case PoolReduceOp::Avg:
        return 2;
    }
}

int stride_at(const mx::array& array, int dim) {
    return static_cast<int>(array.strides(dim));
}

#ifdef _METAL_
template <typename Encoder>
void bind_forward_shape(
    Encoder& encoder,
    const std::vector<mx::array>& inputs,
    PoolReduceOp reduce,
    SparsePoolShape shape
) {
    set_bytes_range(
        encoder,
        7,
        reduce_id(reduce),
        shape.out_capacity,
        shape.channels,
        stride_at(inputs[0], 0),
        stride_at(inputs[0], 1)
    );
}

template <typename Encoder>
void bind_autodiff_shape(
    Encoder& encoder,
    const std::vector<mx::array>& inputs,
    PoolReduceOp reduce,
    SparsePoolShape shape,
    int first_index
) {
    set_bytes_range(
        encoder,
        first_index,
        reduce_id(reduce),
        shape.in_capacity,
        shape.out_capacity,
        shape.n_kernels,
        shape.channels,
        stride_at(inputs[0], 0),
        stride_at(inputs[0], 1),
        stride_at(inputs[1], 0),
        stride_at(inputs[1], 1),
        stride_at(inputs[2], 0),
        stride_at(inputs[2], 1)
    );
}
#endif

} // namespace

void eval(
    PoolReduceOp reduce,
    SparsePoolShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
#ifdef _METAL_
    auto& out = outputs[0];
    allocate(out);
    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);
    auto kernel = device.get_kernel("sparse_pool_relation_f32_i32", library);
    encoder.set_compute_pipeline_state(kernel);
    for (int index = 0; index < 6; ++index) {
        encoder.set_input_array(inputs[index], index);
    }
    encoder.set_output_array(out, 6);
    bind_forward_shape(encoder, inputs, reduce, shape);
    dispatch_1d(
        encoder,
        kernel,
        static_cast<size_t>(shape.out_capacity) *
            static_cast<size_t>(shape.channels)
    );
#else
    (void)reduce;
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

void eval_grad(
    PoolReduceOp reduce,
    SparsePoolShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
#ifdef _METAL_
    auto& out = outputs[0];
    allocate(out);
    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);

    auto kernel_name = "sparse_pool_relation_sum_avg_input_grad_f32_i32";
    if (shape.input_exclusive) {
        kernel_name = "sparse_pool_relation_exclusive_input_grad_f32_i32";
    } else if (reduce == PoolReduceOp::Max) {
        kernel_name = "sparse_pool_relation_max_input_grad_f32_i32";
    }
    auto kernel = device.get_kernel(kernel_name, library);
    encoder.set_compute_pipeline_state(kernel);
    for (int index = 0; index < int(inputs.size()); ++index) {
        encoder.set_input_array(inputs[index], index);
    }
    encoder.set_output_array(out, 10);
    bind_autodiff_shape(encoder, inputs, reduce, shape, 11);
    dispatch_1d(
        encoder,
        kernel,
        static_cast<size_t>(shape.in_capacity) *
            static_cast<size_t>(shape.channels)
    );
#else
    (void)reduce;
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

void eval_jvp(
    PoolReduceOp reduce,
    SparsePoolShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
#ifdef _METAL_
    auto& out = outputs[0];
    allocate(out);
    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);
    auto kernel =
        device.get_kernel("sparse_pool_relation_jvp_f32_i32", library);
    encoder.set_compute_pipeline_state(kernel);
    for (int index = 0; index < int(inputs.size()); ++index) {
        encoder.set_input_array(inputs[index], index);
    }
    encoder.set_output_array(out, 8);
    bind_autodiff_shape(encoder, inputs, reduce, shape, 9);
    dispatch_1d(
        encoder,
        kernel,
        static_cast<size_t>(shape.out_capacity) *
            static_cast<size_t>(shape.channels)
    );
#else
    (void)reduce;
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

} // namespace mlx_lattice::backend::metal::pool
