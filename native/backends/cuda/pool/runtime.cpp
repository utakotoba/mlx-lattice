#include "backends/cuda/pool/runtime.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "backends/array_utils.h"
#include "backends/cuda/pool/kernels.cuh"
#include "backends/cuda/runtime_api.h"
#include "backends/cuda/runtime_utils.h"

namespace mlx_lattice::backend::cuda::pool {
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

void require_f32(const mx::array& input, const char* name) {
    if (input.dtype() != mx::float32) {
        throw std::invalid_argument(
            std::string("CUDA sparse pooling requires float32 ") + name + "."
        );
    }
}

void require_i32(const mx::array& input, const char* name) {
    if (input.dtype() != mx::int32) {
        throw std::invalid_argument(
            std::string("CUDA sparse pooling requires int32 ") + name + "."
        );
    }
}

template <typename Kernel, typename... Args>
void add_1d(
    const mx::Stream& stream,
    Kernel kernel,
    std::size_t elements,
    Args&&... args
) {
    auto& encoder = mx::cu::get_command_encoder(stream);
    auto launch = launch_1d(elements);
    encoder.add_kernel_node(
        kernel, launch.grid, launch.block, std::forward<Args>(args)...
    );
}

template <typename Kernel, typename... Args>
void add_pool_blocks(
    const mx::Stream& stream,
    Kernel kernel,
    int out_capacity,
    int channels,
    Args&&... args
) {
    auto& encoder = mx::cu::get_command_encoder(stream);
    encoder.add_kernel_node(
        kernel,
        dim3(
            static_cast<unsigned int>(std::max(out_capacity, 1)),
            static_cast<unsigned int>(std::max(channels, 1)),
            1
        ),
        dim3(128, 1, 1),
        std::forward<Args>(args)...
    );
}

bool should_use_block_forward(SparsePoolShape shape) {
    if (shape.out_capacity <= 0 || shape.channels <= 0) {
        return false;
    }
    auto average_degree = shape.n_kernels > 8
                              ? shape.n_kernels
                              : shape.in_capacity / shape.out_capacity;
    return average_degree >= 8 && shape.channels <= 256;
}

template <typename Array> auto ptr(const Array& array) {
    return mx::gpu_ptr<void>(array);
}

} // namespace

void eval(
    PoolReduceOp reduce,
    SparsePoolShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    require_f32(inputs[0], "features");
    for (int index = 1; index < 6; ++index) {
        require_i32(inputs[index], "relation array");
    }

    auto& out = outputs[0];
    allocate(out);
    auto& encoder = mx::cu::get_command_encoder(stream);
    for (const auto& input : inputs) {
        encoder.set_input_array(input);
    }
    encoder.set_output_array(out);

    if (should_use_block_forward(shape)) {
        auto kernel = sparse_pool_relation_block_sum_f32_i32;
        if (reduce == PoolReduceOp::Max) {
            kernel = sparse_pool_relation_block_max_f32_i32;
        } else if (reduce == PoolReduceOp::Avg) {
            kernel = sparse_pool_relation_block_avg_f32_i32;
        }
        add_pool_blocks(
            stream,
            kernel,
            shape.out_capacity,
            shape.channels,
            static_cast<const float*>(ptr(inputs[0])),
            static_cast<const int*>(ptr(inputs[1])),
            static_cast<const int*>(ptr(inputs[4])),
            static_cast<const int*>(ptr(inputs[5])),
            static_cast<float*>(ptr(out)),
            shape.out_capacity,
            shape.channels,
            stride_at(inputs[0], 0),
            stride_at(inputs[0], 1)
        );
        return;
    }

    add_1d(
        stream,
        sparse_pool_relation_f32_i32,
        static_cast<std::size_t>(shape.out_capacity) *
            static_cast<std::size_t>(shape.channels),
        static_cast<const float*>(ptr(inputs[0])),
        static_cast<const int*>(ptr(inputs[1])),
        static_cast<const int*>(ptr(inputs[2])),
        static_cast<const int*>(ptr(inputs[3])),
        static_cast<const int*>(ptr(inputs[4])),
        static_cast<const int*>(ptr(inputs[5])),
        static_cast<float*>(ptr(out)),
        reduce_id(reduce),
        shape.out_capacity,
        shape.channels,
        stride_at(inputs[0], 0),
        stride_at(inputs[0], 1)
    );
}

void eval_grad(
    PoolReduceOp reduce,
    SparsePoolShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    require_f32(inputs[0], "cotangent");
    require_f32(inputs[1], "features");
    require_f32(inputs[2], "pooled features");
    for (int index = 3; index < int(inputs.size()); ++index) {
        require_i32(inputs[index], "relation array");
    }

    auto& out = outputs[0];
    allocate(out);
    auto& encoder = mx::cu::get_command_encoder(stream);
    for (const auto& input : inputs) {
        encoder.set_input_array(input);
    }
    encoder.set_output_array(out);

    auto kernel = sparse_pool_relation_sum_avg_input_grad_f32_i32;
    if (shape.input_exclusive) {
        kernel = sparse_pool_relation_exclusive_input_grad_f32_i32;
    } else if (reduce == PoolReduceOp::Max) {
        kernel = sparse_pool_relation_max_input_grad_f32_i32;
    }
    add_1d(
        stream,
        kernel,
        static_cast<std::size_t>(shape.in_capacity) *
            static_cast<std::size_t>(shape.channels),
        static_cast<const float*>(ptr(inputs[0])),
        static_cast<const float*>(ptr(inputs[1])),
        static_cast<const float*>(ptr(inputs[2])),
        static_cast<const int*>(ptr(inputs[3])),
        static_cast<const int*>(ptr(inputs[4])),
        static_cast<const int*>(ptr(inputs[5])),
        static_cast<const int*>(ptr(inputs[6])),
        static_cast<const int*>(ptr(inputs[7])),
        static_cast<const int*>(ptr(inputs[8])),
        static_cast<const int*>(ptr(inputs[9])),
        static_cast<float*>(ptr(out)),
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

void eval_jvp(
    PoolReduceOp reduce,
    SparsePoolShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    require_f32(inputs[0], "tangent");
    require_f32(inputs[1], "features");
    require_f32(inputs[2], "pooled features");
    for (int index = 3; index < int(inputs.size()); ++index) {
        require_i32(inputs[index], "relation array");
    }

    auto& out = outputs[0];
    allocate(out);
    auto& encoder = mx::cu::get_command_encoder(stream);
    for (const auto& input : inputs) {
        encoder.set_input_array(input);
    }
    encoder.set_output_array(out);

    add_1d(
        stream,
        sparse_pool_relation_jvp_f32_i32,
        static_cast<std::size_t>(shape.out_capacity) *
            static_cast<std::size_t>(shape.channels),
        static_cast<const float*>(ptr(inputs[0])),
        static_cast<const float*>(ptr(inputs[1])),
        static_cast<const float*>(ptr(inputs[2])),
        static_cast<const int*>(ptr(inputs[3])),
        static_cast<const int*>(ptr(inputs[4])),
        static_cast<const int*>(ptr(inputs[5])),
        static_cast<const int*>(ptr(inputs[6])),
        static_cast<const int*>(ptr(inputs[7])),
        static_cast<float*>(ptr(out)),
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

} // namespace mlx_lattice::backend::cuda::pool
