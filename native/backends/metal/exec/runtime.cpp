#include "backends/metal/exec/runtime.h"

#include <algorithm>

#include "backends/array_utils.h"
#include "backends/metal/runtime_utils.h"
#include "mlx/device.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace mlx_lattice::exec::metal {

namespace {

int map_op_id(SparseMapOp op) {
    switch (op) {
    case SparseMapOp::Forward:
        return 0;
    case SparseMapOp::Transposed:
        return 1;
    case SparseMapOp::Generative:
        return 2;
    }
}

int pool_op_id(PoolReduceOp op) {
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

bool metal_runtime_available() {
#if MLX_LATTICE_HAS_METAL
    return mx::is_available(mx::Device::gpu);
#else
    return false;
#endif
}

bool is_pointwise_forward(
    SparseMapOp op,
    SparseConvShape shape,
    Triple stride,
    Triple padding
) {
    return op == SparseMapOp::Forward && shape.n_kernels == 1 &&
           stride == Triple{1, 1, 1} && padding == Triple{0, 0, 0};
}

bool is_identity_forward_coords(Triple stride, Triple padding) {
    return stride == Triple{1, 1, 1} && padding == Triple{0, 0, 0};
}

bool is_nonoverlapping_transposed(
    SparseMapOp op,
    SparseConvShape shape,
    Triple stride,
    Triple padding
) {
    return op == SparseMapOp::Transposed && padding == Triple{0, 0, 0} &&
           shape.kernel_x == stride[0] && shape.kernel_y == stride[1] &&
           shape.kernel_z == stride[2];
}

bool is_nonoverlapping_forward_pool(
    SparsePoolShape shape,
    Triple stride,
    Triple padding
) {
    return padding == Triple{0, 0, 0} && stride[0] > 0 && stride[1] > 0 &&
           stride[2] > 0 &&
           shape.n_kernels == stride[0] * stride[1] * stride[2];
}

#ifdef _METAL_
mx::array make_int32_temp(int elements) {
    auto count = std::max(elements, 1);
    return mx::array(
        mx::allocator::malloc(static_cast<size_t>(count) * sizeof(int32_t)),
        mx::Shape{count},
        mx::int32
    );
}

int next_power_of_two(int value) {
    auto out = 1;
    while (out < value) {
        out <<= 1;
    }
    return out;
}

template <typename Encoder, typename Kernel>
void dispatch_1d(Encoder& encoder, Kernel* kernel, size_t elements) {
    auto threads = std::max<size_t>(elements, 1);
    auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}
#endif

} // namespace

bool can_run_sparse_conv(
    const mx::array& coords, // NOLINT(bugprone-easily-swappable-parameters)
    const mx::array& active_rows,
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& offsets
) {
#if MLX_LATTICE_HAS_METAL
    return metal_runtime_available() && coords.dtype() == mx::int32 &&
           active_rows.dtype() == mx::int32 && feats.dtype() == mx::float32 &&
           weights.dtype() == mx::float32 && offsets.dtype() == mx::int32;
#else
    (void)coords;
    (void)active_rows;
    (void)feats;
    (void)weights;
    (void)offsets;
    return false;
#endif
}

bool can_run_sparse_pool(
    const mx::array& coords, // NOLINT(bugprone-easily-swappable-parameters)
    const mx::array& active_rows,
    const mx::array& feats,
    const mx::array& offsets
) {
#if MLX_LATTICE_HAS_METAL
    return metal_runtime_available() && coords.dtype() == mx::int32 &&
           active_rows.dtype() == mx::int32 && feats.dtype() == mx::float32 &&
           offsets.dtype() == mx::int32;
#else
    (void)coords;
    (void)active_rows;
    (void)feats;
    (void)offsets;
    return false;
#endif
}

void eval_sparse_conv(
    SparseMapOp op,
    SparseConvShape shape,
    Triple stride, // NOLINT(bugprone-easily-swappable-parameters)
    Triple padding,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
#ifdef _METAL_
    backend::allocate_all(outputs);

    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);
    if (is_pointwise_forward(op, shape, stride, padding)) {
        auto kernel =
            device.get_kernel("sparse_conv_pointwise_f32_i32", library);
        encoder.set_compute_pipeline_state(kernel);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_input_array(inputs[1], 1);
        encoder.set_input_array(inputs[2], 2);
        encoder.set_input_array(inputs[3], 3);
        encoder.set_output_array(outputs[SparseOutCoords], 4);
        encoder.set_output_array(outputs[SparseOutFeats], 5);
        encoder.set_output_array(outputs[SparseCounts], 6);
        encoder.set_bytes(shape.in_capacity, 7);
        encoder.set_bytes(shape.out_capacity, 8);
        encoder.set_bytes(shape.in_channels, 9);
        encoder.set_bytes(shape.out_channels, 10);
        encoder.set_bytes(stride_at(inputs[2], 0), 11);
        encoder.set_bytes(stride_at(inputs[2], 1), 12);
        encoder.set_bytes(stride_at(inputs[3], 0), 13);
        encoder.set_bytes(stride_at(inputs[3], 1), 14);
        encoder.set_bytes(stride_at(inputs[3], 2), 15);
        encoder.set_bytes(
            inputs[3].ndim() == 5 ? stride_at(inputs[3], 3) : 0, 16
        );
        encoder.set_bytes(
            inputs[3].ndim() == 5 ? stride_at(inputs[3], 4) : 0, 17
        );
        encoder.set_bytes(shape.weight_layout, 18);
        encoder.set_bytes(shape.kernel_x, 19);
        encoder.set_bytes(shape.kernel_y, 20);
        encoder.set_bytes(shape.kernel_z, 21);
        auto coord_elements = static_cast<size_t>(shape.out_capacity) * 4;
        auto feat_elements = static_cast<size_t>(shape.out_capacity) *
                             static_cast<size_t>(shape.out_channels);
        dispatch_1d(encoder, kernel, std::max(coord_elements, feat_elements));
        return;
    }

    if (op == SparseMapOp::Generative ||
        is_nonoverlapping_transposed(op, shape, stride, padding)) {
        auto kernel =
            device.get_kernel("sparse_conv_generative_f32_i32", library);
        encoder.set_compute_pipeline_state(kernel);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_input_array(inputs[1], 1);
        encoder.set_input_array(inputs[2], 2);
        encoder.set_input_array(inputs[3], 3);
        encoder.set_input_array(inputs[4], 4);
        encoder.set_output_array(outputs[SparseOutCoords], 5);
        encoder.set_output_array(outputs[SparseOutFeats], 6);
        encoder.set_output_array(outputs[SparseCounts], 7);
        encoder.set_bytes(shape.in_capacity, 8);
        encoder.set_bytes(shape.out_capacity, 9);
        encoder.set_bytes(shape.n_kernels, 10);
        encoder.set_bytes(shape.in_channels, 11);
        encoder.set_bytes(shape.out_channels, 12);
        encoder.set_bytes(stride[0], 13);
        encoder.set_bytes(stride[1], 14);
        encoder.set_bytes(stride[2], 15);
        encoder.set_bytes(stride_at(inputs[2], 0), 16);
        encoder.set_bytes(stride_at(inputs[2], 1), 17);
        encoder.set_bytes(stride_at(inputs[3], 0), 18);
        encoder.set_bytes(stride_at(inputs[3], 1), 19);
        encoder.set_bytes(stride_at(inputs[3], 2), 20);
        encoder.set_bytes(
            inputs[3].ndim() == 5 ? stride_at(inputs[3], 3) : 0, 21
        );
        encoder.set_bytes(
            inputs[3].ndim() == 5 ? stride_at(inputs[3], 4) : 0, 22
        );
        encoder.set_bytes(shape.weight_layout, 23);
        encoder.set_bytes(shape.kernel_x, 24);
        encoder.set_bytes(shape.kernel_y, 25);
        encoder.set_bytes(shape.kernel_z, 26);
        auto coord_elements = static_cast<size_t>(shape.out_capacity) * 4;
        auto feat_elements = static_cast<size_t>(shape.out_capacity) *
                             static_cast<size_t>(shape.out_channels);
        dispatch_1d(encoder, kernel, std::max(coord_elements, feat_elements));
        return;
    }

    if (op == SparseMapOp::Forward) {
        auto coord_elements = static_cast<size_t>(shape.out_capacity) * 4;
        auto feat_elements = static_cast<size_t>(shape.out_capacity) *
                             static_cast<size_t>(shape.out_channels);
        auto max_edges = shape.out_capacity * shape.n_kernels;

        if (is_identity_forward_coords(stride, padding)) {
            auto table_capacity = next_power_of_two(
                std::max(
                    shape.in_capacity * (shape.in_capacity >= 65536 ? 4 : 2), 2
                )
            );
            auto table_keys = make_int32_temp(table_capacity);
            auto table_rows = make_int32_temp(table_capacity);
            encoder.add_temporaries({table_keys, table_rows});

            constexpr int empty_key = 0x7fffffff;
            auto hash_clear =
                device.get_kernel("sparse_conv_hash_clear_i32", library);
            encoder.set_compute_pipeline_state(hash_clear);
            encoder.set_output_array(table_keys, 0);
            encoder.set_output_array(table_rows, 1);
            encoder.set_output_array(outputs[SparseCounts], 2);
            encoder.set_bytes(table_capacity, 3);
            encoder.set_bytes(empty_key, 4);
            dispatch_1d(
                encoder, hash_clear, static_cast<size_t>(table_capacity)
            );

            auto hash_insert =
                device.get_kernel("sparse_conv_hash_insert_i32", library);
            encoder.set_compute_pipeline_state(hash_insert);
            encoder.set_input_array(inputs[0], 0);
            encoder.set_input_array(inputs[1], 1);
            encoder.set_output_array(table_keys, 2);
            encoder.set_output_array(table_rows, 3);
            encoder.set_bytes(shape.in_capacity, 4);
            encoder.set_bytes(table_capacity, 5);
            encoder.set_bytes(empty_key, 6);
            dispatch_1d(
                encoder, hash_insert, static_cast<size_t>(shape.in_capacity)
            );

            if (shape.out_channels % 4 == 0 && shape.in_capacity >= 8192) {
                auto plan_in_rows = make_int32_temp(max_edges);
                encoder.add_temporary(plan_in_rows);

                auto plan = device.get_kernel(
                    "sparse_conv_identity_hash_plan_i32", library
                );
                encoder.set_compute_pipeline_state(plan);
                encoder.set_input_array(inputs[0], 0);
                encoder.set_input_array(inputs[1], 1);
                encoder.set_input_array(inputs[4], 2);
                encoder.set_input_array(table_keys, 3);
                encoder.set_input_array(table_rows, 4);
                encoder.set_output_array(outputs[SparseOutCoords], 5);
                encoder.set_output_array(outputs[SparseCounts], 6);
                encoder.set_output_array(plan_in_rows, 7);
                encoder.set_bytes(shape.in_capacity, 8);
                encoder.set_bytes(shape.n_kernels, 9);
                encoder.set_bytes(table_capacity, 10);
                encoder.set_bytes(empty_key, 11);
                dispatch_1d(
                    encoder,
                    plan,
                    std::max(coord_elements, static_cast<size_t>(max_edges))
                );

                auto gather = device.get_kernel(
                    "sparse_conv_identity_plan_gather_vec4_f32_i32", library
                );
                encoder.set_compute_pipeline_state(gather);
                encoder.set_input_array(inputs[2], 0);
                encoder.set_input_array(inputs[3], 1);
                encoder.set_input_array(plan_in_rows, 2);
                encoder.set_input_array(outputs[SparseCounts], 3);
                encoder.set_output_array(outputs[SparseOutFeats], 4);
                encoder.set_bytes(shape.out_capacity, 5);
                encoder.set_bytes(shape.n_kernels, 6);
                encoder.set_bytes(shape.in_channels, 7);
                encoder.set_bytes(shape.out_channels, 8);
                encoder.set_bytes(stride_at(inputs[2], 0), 9);
                encoder.set_bytes(stride_at(inputs[2], 1), 10);
                encoder.set_bytes(stride_at(inputs[3], 0), 11);
                encoder.set_bytes(stride_at(inputs[3], 1), 12);
                encoder.set_bytes(stride_at(inputs[3], 2), 13);
                encoder.set_bytes(
                    inputs[3].ndim() == 5 ? stride_at(inputs[3], 3) : 0, 14
                );
                encoder.set_bytes(
                    inputs[3].ndim() == 5 ? stride_at(inputs[3], 4) : 0, 15
                );
                encoder.set_bytes(shape.weight_layout, 16);
                encoder.set_bytes(shape.kernel_x, 17);
                encoder.set_bytes(shape.kernel_y, 18);
                encoder.set_bytes(shape.kernel_z, 19);
                dispatch_1d(
                    encoder,
                    gather,
                    static_cast<size_t>(shape.out_capacity) *
                        static_cast<size_t>(shape.out_channels / 4)
                );
                return;
            }

            auto use_vec4 = shape.out_channels % 4 == 0;
            auto conv = device.get_kernel(
                use_vec4 ? "sparse_conv_identity_hash_vec4_f32_i32"
                         : "sparse_conv_identity_hash_f32_i32",
                library
            );
            encoder.set_compute_pipeline_state(conv);
            encoder.set_input_array(inputs[0], 0);
            encoder.set_input_array(inputs[1], 1);
            encoder.set_input_array(inputs[2], 2);
            encoder.set_input_array(inputs[3], 3);
            encoder.set_input_array(inputs[4], 4);
            encoder.set_input_array(table_keys, 5);
            encoder.set_input_array(table_rows, 6);
            encoder.set_output_array(outputs[SparseOutCoords], 7);
            encoder.set_output_array(outputs[SparseOutFeats], 8);
            encoder.set_output_array(outputs[SparseCounts], 9);
            encoder.set_bytes(shape.in_capacity, 10);
            encoder.set_bytes(shape.out_capacity, 11);
            encoder.set_bytes(shape.n_kernels, 12);
            encoder.set_bytes(shape.in_channels, 13);
            encoder.set_bytes(shape.out_channels, 14);
            encoder.set_bytes(table_capacity, 15);
            encoder.set_bytes(empty_key, 16);
            encoder.set_bytes(stride_at(inputs[2], 0), 17);
            encoder.set_bytes(stride_at(inputs[2], 1), 18);
            encoder.set_bytes(stride_at(inputs[3], 0), 19);
            encoder.set_bytes(stride_at(inputs[3], 1), 20);
            encoder.set_bytes(stride_at(inputs[3], 2), 21);
            encoder.set_bytes(
                inputs[3].ndim() == 5 ? stride_at(inputs[3], 3) : 0, 22
            );
            encoder.set_bytes(
                inputs[3].ndim() == 5 ? stride_at(inputs[3], 4) : 0, 23
            );
            encoder.set_bytes(shape.weight_layout, 24);
            encoder.set_bytes(shape.kernel_x, 25);
            encoder.set_bytes(shape.kernel_y, 26);
            encoder.set_bytes(shape.kernel_z, 27);
            dispatch_1d(
                encoder,
                conv,
                std::max(
                    coord_elements,
                    use_vec4 ? static_cast<size_t>(shape.out_capacity) *
                                   static_cast<size_t>(shape.out_channels / 4)
                             : feat_elements
                )
            );
            return;
        }

        auto clear = device.get_kernel("sparse_conv_clear_f32_i32", library);
        encoder.set_compute_pipeline_state(clear);
        encoder.set_output_array(outputs[SparseOutCoords], 0);
        encoder.set_output_array(outputs[SparseOutFeats], 1);
        encoder.set_output_array(outputs[SparseCounts], 2);
        auto coord_total = static_cast<int>(coord_elements);
        auto feat_total = static_cast<int>(feat_elements);
        encoder.set_bytes(coord_total, 3);
        encoder.set_bytes(feat_total, 4);
        dispatch_1d(encoder, clear, std::max(coord_elements, feat_elements));

        auto plan_in_rows = make_int32_temp(max_edges);
        auto plan_kernel_ids = make_int32_temp(max_edges);
        auto plan_row_offsets = make_int32_temp(shape.out_capacity + 1);
        encoder.add_temporaries(
            {plan_in_rows, plan_kernel_ids, plan_row_offsets}
        );

        auto plan = device.get_kernel("sparse_conv_forward_plan_i32", library);
        encoder.set_compute_pipeline_state(plan);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_input_array(inputs[1], 1);
        encoder.set_input_array(inputs[4], 2);
        encoder.set_output_array(outputs[SparseOutCoords], 3);
        encoder.set_output_array(outputs[SparseCounts], 4);
        encoder.set_output_array(plan_in_rows, 5);
        encoder.set_output_array(plan_kernel_ids, 6);
        encoder.set_output_array(plan_row_offsets, 7);
        encoder.set_bytes(shape.in_capacity, 8);
        encoder.set_bytes(shape.out_capacity, 9);
        encoder.set_bytes(shape.n_kernels, 10);
        encoder.set_bytes(stride[0], 11);
        encoder.set_bytes(stride[1], 12);
        encoder.set_bytes(stride[2], 13);
        encoder.set_bytes(padding[0], 14);
        encoder.set_bytes(padding[1], 15);
        encoder.set_bytes(padding[2], 16);
        auto identity = is_identity_forward_coords(stride, padding) ? 1 : 0;
        encoder.set_bytes(identity, 17);
        encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));

        auto gather = device.get_kernel(
            "sparse_conv_forward_plan_gather_f32_i32", library
        );
        encoder.set_compute_pipeline_state(gather);
        encoder.set_input_array(inputs[2], 0);
        encoder.set_input_array(inputs[3], 1);
        encoder.set_input_array(plan_in_rows, 2);
        encoder.set_input_array(plan_kernel_ids, 3);
        encoder.set_input_array(plan_row_offsets, 4);
        encoder.set_input_array(outputs[SparseCounts], 5);
        encoder.set_output_array(outputs[SparseOutFeats], 6);
        encoder.set_bytes(shape.out_capacity, 7);
        encoder.set_bytes(shape.in_channels, 8);
        encoder.set_bytes(shape.out_channels, 9);
        encoder.set_bytes(stride_at(inputs[2], 0), 10);
        encoder.set_bytes(stride_at(inputs[2], 1), 11);
        encoder.set_bytes(stride_at(inputs[3], 0), 12);
        encoder.set_bytes(stride_at(inputs[3], 1), 13);
        encoder.set_bytes(stride_at(inputs[3], 2), 14);
        encoder.set_bytes(
            inputs[3].ndim() == 5 ? stride_at(inputs[3], 3) : 0, 15
        );
        encoder.set_bytes(
            inputs[3].ndim() == 5 ? stride_at(inputs[3], 4) : 0, 16
        );
        encoder.set_bytes(shape.weight_layout, 17);
        encoder.set_bytes(shape.kernel_x, 18);
        encoder.set_bytes(shape.kernel_y, 19);
        encoder.set_bytes(shape.kernel_z, 20);
        dispatch_1d(encoder, gather, feat_elements);
        return;
    }

    auto kernel = device.get_kernel("sparse_conv_f32_i32_serial", library);

    encoder.set_compute_pipeline_state(kernel);
    for (int i = 0; i < int(inputs.size()); ++i) {
        encoder.set_input_array(inputs[i], i);
    }
    encoder.set_output_array(outputs[SparseOutCoords], 5);
    encoder.set_output_array(outputs[SparseOutFeats], 6);
    encoder.set_output_array(outputs[SparseCounts], 7);
    encoder.set_bytes(map_op_id(op), 8);
    encoder.set_bytes(shape.in_capacity, 9);
    encoder.set_bytes(shape.out_capacity, 10);
    encoder.set_bytes(shape.n_kernels, 11);
    encoder.set_bytes(shape.in_channels, 12);
    encoder.set_bytes(shape.out_channels, 13);
    encoder.set_bytes(stride[0], 14);
    encoder.set_bytes(stride[1], 15);
    encoder.set_bytes(stride[2], 16);
    encoder.set_bytes(padding[0], 17);
    encoder.set_bytes(padding[1], 18);
    encoder.set_bytes(padding[2], 19);
    encoder.set_bytes(stride_at(inputs[2], 0), 20);
    encoder.set_bytes(stride_at(inputs[2], 1), 21);
    encoder.set_bytes(stride_at(inputs[3], 0), 22);
    encoder.set_bytes(stride_at(inputs[3], 1), 23);
    encoder.set_bytes(stride_at(inputs[3], 2), 24);
    encoder.set_bytes(inputs[3].ndim() == 5 ? stride_at(inputs[3], 3) : 0, 25);
    encoder.set_bytes(inputs[3].ndim() == 5 ? stride_at(inputs[3], 4) : 0, 26);
    encoder.set_bytes(shape.weight_layout, 27);
    encoder.set_bytes(shape.kernel_x, 28);
    encoder.set_bytes(shape.kernel_y, 29);
    encoder.set_bytes(shape.kernel_z, 30);
    encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
#else
    (void)op;
    (void)shape;
    (void)stride;
    (void)padding;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

void eval_sparse_conv_input_grad(
    SparseMapOp op,
    SparseConvShape shape,
    Triple stride, // NOLINT(bugprone-easily-swappable-parameters)
    Triple padding,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
#ifdef _METAL_
    auto& out = outputs[0];
    backend::allocate(out);

    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);
    if (is_pointwise_forward(op, shape, stride, padding)) {
        auto kernel = device.get_kernel(
            "sparse_conv_pointwise_input_grad_f32_i32", library
        );
        encoder.set_compute_pipeline_state(kernel);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_input_array(inputs[2], 1);
        encoder.set_input_array(inputs[3], 2);
        encoder.set_output_array(out, 3);
        encoder.set_bytes(shape.in_capacity, 4);
        encoder.set_bytes(shape.in_channels, 5);
        encoder.set_bytes(shape.out_channels, 6);
        encoder.set_bytes(stride_at(inputs[0], 0), 7);
        encoder.set_bytes(stride_at(inputs[0], 1), 8);
        encoder.set_bytes(stride_at(inputs[3], 0), 9);
        encoder.set_bytes(stride_at(inputs[3], 1), 10);
        encoder.set_bytes(stride_at(inputs[3], 2), 11);
        encoder.set_bytes(
            inputs[3].ndim() == 5 ? stride_at(inputs[3], 3) : 0, 12
        );
        encoder.set_bytes(
            inputs[3].ndim() == 5 ? stride_at(inputs[3], 4) : 0, 13
        );
        encoder.set_bytes(shape.weight_layout, 14);
        encoder.set_bytes(shape.kernel_x, 15);
        encoder.set_bytes(shape.kernel_y, 16);
        encoder.set_bytes(shape.kernel_z, 17);
        dispatch_1d(
            encoder,
            kernel,
            static_cast<size_t>(shape.in_capacity) *
                static_cast<size_t>(shape.in_channels)
        );
        return;
    }

    auto kernel =
        device.get_kernel("sparse_conv_input_grad_f32_i32_serial", library);

    encoder.set_compute_pipeline_state(kernel);
    for (int i = 0; i < int(inputs.size()); ++i) {
        encoder.set_input_array(inputs[i], i);
    }
    encoder.set_output_array(out, 5);
    encoder.set_bytes(map_op_id(op), 6);
    encoder.set_bytes(shape.in_capacity, 7);
    encoder.set_bytes(shape.out_capacity, 8);
    encoder.set_bytes(shape.n_kernels, 9);
    encoder.set_bytes(shape.in_channels, 10);
    encoder.set_bytes(shape.out_channels, 11);
    encoder.set_bytes(stride[0], 12);
    encoder.set_bytes(stride[1], 13);
    encoder.set_bytes(stride[2], 14);
    encoder.set_bytes(padding[0], 15);
    encoder.set_bytes(padding[1], 16);
    encoder.set_bytes(padding[2], 17);
    encoder.set_bytes(stride_at(inputs[0], 0), 18);
    encoder.set_bytes(stride_at(inputs[0], 1), 19);
    encoder.set_bytes(stride_at(inputs[3], 0), 20);
    encoder.set_bytes(stride_at(inputs[3], 1), 21);
    encoder.set_bytes(stride_at(inputs[3], 2), 22);
    encoder.set_bytes(inputs[3].ndim() == 5 ? stride_at(inputs[3], 3) : 0, 23);
    encoder.set_bytes(inputs[3].ndim() == 5 ? stride_at(inputs[3], 4) : 0, 24);
    encoder.set_bytes(shape.weight_layout, 25);
    encoder.set_bytes(shape.kernel_x, 26);
    encoder.set_bytes(shape.kernel_y, 27);
    encoder.set_bytes(shape.kernel_z, 28);
    encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
#else
    (void)op;
    (void)shape;
    (void)stride;
    (void)padding;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

void eval_sparse_conv_weight_grad(
    SparseMapOp op,
    SparseConvShape shape,
    Triple stride, // NOLINT(bugprone-easily-swappable-parameters)
    Triple padding,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
#ifdef _METAL_
    auto& out = outputs[0];
    backend::allocate(out);

    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);
    if (is_pointwise_forward(op, shape, stride, padding)) {
        auto kernel = device.get_kernel(
            "sparse_conv_pointwise_weight_grad_f32_i32", library
        );
        encoder.set_compute_pipeline_state(kernel);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_input_array(inputs[1], 1);
        encoder.set_input_array(inputs[3], 2);
        encoder.set_output_array(out, 3);
        encoder.set_bytes(shape.in_capacity, 4);
        encoder.set_bytes(shape.in_channels, 5);
        encoder.set_bytes(shape.out_channels, 6);
        encoder.set_bytes(stride_at(inputs[0], 0), 7);
        encoder.set_bytes(stride_at(inputs[0], 1), 8);
        encoder.set_bytes(stride_at(inputs[1], 0), 9);
        encoder.set_bytes(stride_at(inputs[1], 1), 10);
        encoder.set_bytes(shape.weight_layout, 11);
        encoder.set_bytes(shape.kernel_x, 12);
        encoder.set_bytes(shape.kernel_y, 13);
        encoder.set_bytes(shape.kernel_z, 14);
        dispatch_1d(
            encoder,
            kernel,
            static_cast<size_t>(shape.in_channels) *
                static_cast<size_t>(shape.out_channels)
        );
        return;
    }

    auto kernel =
        device.get_kernel("sparse_conv_weight_grad_f32_i32_serial", library);

    encoder.set_compute_pipeline_state(kernel);
    for (int i = 0; i < int(inputs.size()); ++i) {
        encoder.set_input_array(inputs[i], i);
    }
    encoder.set_output_array(out, 5);
    encoder.set_bytes(map_op_id(op), 6);
    encoder.set_bytes(shape.in_capacity, 7);
    encoder.set_bytes(shape.out_capacity, 8);
    encoder.set_bytes(shape.n_kernels, 9);
    encoder.set_bytes(shape.in_channels, 10);
    encoder.set_bytes(shape.out_channels, 11);
    encoder.set_bytes(stride[0], 12);
    encoder.set_bytes(stride[1], 13);
    encoder.set_bytes(stride[2], 14);
    encoder.set_bytes(padding[0], 15);
    encoder.set_bytes(padding[1], 16);
    encoder.set_bytes(padding[2], 17);
    encoder.set_bytes(stride_at(inputs[0], 0), 18);
    encoder.set_bytes(stride_at(inputs[0], 1), 19);
    encoder.set_bytes(stride_at(inputs[1], 0), 20);
    encoder.set_bytes(stride_at(inputs[1], 1), 21);
    encoder.set_bytes(shape.weight_layout, 22);
    encoder.set_bytes(shape.kernel_x, 23);
    encoder.set_bytes(shape.kernel_y, 24);
    encoder.set_bytes(shape.kernel_z, 25);
    encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
#else
    (void)op;
    (void)shape;
    (void)stride;
    (void)padding;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

void eval_sparse_conv_features(
    SparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
#ifdef _METAL_
    auto& out = outputs[0];
    backend::allocate(out);

    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);

    auto clear = device.get_kernel("sparse_relation_conv_clear_f32", library);
    encoder.set_compute_pipeline_state(clear);
    encoder.set_output_array(out, 0);
    auto out_total = static_cast<int>(out.size());
    encoder.set_bytes(out_total, 1);
    dispatch_1d(encoder, clear, static_cast<size_t>(out_total));

    auto kernel = device.get_kernel("sparse_relation_conv_f32_i32", library);
    encoder.set_compute_pipeline_state(kernel);
    for (int i = 0; i < int(inputs.size()); ++i) {
        encoder.set_input_array(inputs[i], i);
    }
    encoder.set_output_array(out, 6);
    auto edge_capacity = static_cast<int>(inputs[2].shape(0));
    encoder.set_bytes(edge_capacity, 7);
    encoder.set_bytes(shape.out_capacity, 8);
    encoder.set_bytes(shape.in_channels, 9);
    encoder.set_bytes(shape.out_channels, 10);
    encoder.set_bytes(stride_at(inputs[0], 0), 11);
    encoder.set_bytes(stride_at(inputs[0], 1), 12);
    encoder.set_bytes(stride_at(inputs[1], 0), 13);
    encoder.set_bytes(stride_at(inputs[1], 1), 14);
    encoder.set_bytes(stride_at(inputs[1], 2), 15);
    encoder.set_bytes(inputs[1].ndim() == 5 ? stride_at(inputs[1], 3) : 0, 16);
    encoder.set_bytes(inputs[1].ndim() == 5 ? stride_at(inputs[1], 4) : 0, 17);
    encoder.set_bytes(shape.weight_layout, 18);
    encoder.set_bytes(shape.kernel_x, 19);
    encoder.set_bytes(shape.kernel_y, 20);
    encoder.set_bytes(shape.kernel_z, 21);
    dispatch_1d(
        encoder,
        kernel,
        static_cast<size_t>(edge_capacity) *
            static_cast<size_t>(shape.out_channels)
    );
#else
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

void eval_sparse_conv_features_input_grad(
    SparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
#ifdef _METAL_
    auto& out = outputs[0];
    backend::allocate(out);

    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);

    auto clear = device.get_kernel("sparse_relation_conv_clear_f32", library);
    encoder.set_compute_pipeline_state(clear);
    encoder.set_output_array(out, 0);
    auto out_total = static_cast<int>(out.size());
    encoder.set_bytes(out_total, 1);
    dispatch_1d(encoder, clear, static_cast<size_t>(out_total));

    auto kernel =
        device.get_kernel("sparse_relation_conv_input_grad_f32_i32", library);
    encoder.set_compute_pipeline_state(kernel);
    for (int i = 0; i < int(inputs.size()); ++i) {
        encoder.set_input_array(inputs[i], i);
    }
    encoder.set_output_array(out, 6);
    auto edge_capacity = static_cast<int>(inputs[2].shape(0));
    encoder.set_bytes(edge_capacity, 7);
    encoder.set_bytes(shape.out_capacity, 8);
    encoder.set_bytes(shape.in_channels, 9);
    encoder.set_bytes(shape.out_channels, 10);
    encoder.set_bytes(stride_at(inputs[0], 0), 11);
    encoder.set_bytes(stride_at(inputs[0], 1), 12);
    encoder.set_bytes(stride_at(inputs[1], 0), 13);
    encoder.set_bytes(stride_at(inputs[1], 1), 14);
    encoder.set_bytes(stride_at(inputs[1], 2), 15);
    encoder.set_bytes(inputs[1].ndim() == 5 ? stride_at(inputs[1], 3) : 0, 16);
    encoder.set_bytes(inputs[1].ndim() == 5 ? stride_at(inputs[1], 4) : 0, 17);
    encoder.set_bytes(shape.weight_layout, 18);
    encoder.set_bytes(shape.kernel_x, 19);
    encoder.set_bytes(shape.kernel_y, 20);
    encoder.set_bytes(shape.kernel_z, 21);
    dispatch_1d(
        encoder,
        kernel,
        static_cast<size_t>(edge_capacity) *
            static_cast<size_t>(shape.in_channels)
    );
#else
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

void eval_sparse_conv_features_weight_grad(
    SparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
#ifdef _METAL_
    auto& out = outputs[0];
    backend::allocate(out);

    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);

    auto clear = device.get_kernel("sparse_relation_conv_clear_f32", library);
    encoder.set_compute_pipeline_state(clear);
    encoder.set_output_array(out, 0);
    auto out_total = static_cast<int>(out.size());
    encoder.set_bytes(out_total, 1);
    dispatch_1d(encoder, clear, static_cast<size_t>(out_total));

    auto kernel =
        device.get_kernel("sparse_relation_conv_weight_grad_f32_i32", library);
    encoder.set_compute_pipeline_state(kernel);
    for (int i = 0; i < int(inputs.size()); ++i) {
        encoder.set_input_array(inputs[i], i);
    }
    encoder.set_output_array(out, 6);
    auto edge_capacity = static_cast<int>(inputs[2].shape(0));
    encoder.set_bytes(edge_capacity, 7);
    encoder.set_bytes(shape.out_capacity, 8);
    encoder.set_bytes(shape.in_channels, 9);
    encoder.set_bytes(shape.out_channels, 10);
    encoder.set_bytes(stride_at(inputs[0], 0), 11);
    encoder.set_bytes(stride_at(inputs[0], 1), 12);
    encoder.set_bytes(stride_at(inputs[1], 0), 13);
    encoder.set_bytes(stride_at(inputs[1], 1), 14);
    encoder.set_bytes(shape.weight_layout, 15);
    encoder.set_bytes(shape.kernel_x, 16);
    encoder.set_bytes(shape.kernel_y, 17);
    encoder.set_bytes(shape.kernel_z, 18);
    dispatch_1d(
        encoder,
        kernel,
        static_cast<size_t>(edge_capacity) *
            static_cast<size_t>(shape.in_channels) *
            static_cast<size_t>(shape.out_channels)
    );
#else
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

void eval_sparse_pool(
    PoolReduceOp reduce,
    SparsePoolShape shape,
    Triple stride, // NOLINT(bugprone-easily-swappable-parameters)
    Triple padding,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
#ifdef _METAL_
    backend::allocate_all(outputs);

    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);
    auto coord_elements = static_cast<size_t>(shape.out_capacity) * 4;
    auto feat_elements = static_cast<size_t>(shape.out_capacity) *
                         static_cast<size_t>(shape.channels);
    auto clear = device.get_kernel("sparse_pool_clear_f32_i32", library);
    encoder.set_compute_pipeline_state(clear);
    encoder.set_output_array(outputs[SparseOutCoords], 0);
    encoder.set_output_array(outputs[SparseOutFeats], 1);
    encoder.set_output_array(outputs[SparseCounts], 2);
    auto coord_total = static_cast<int>(coord_elements);
    auto feat_total = static_cast<int>(feat_elements);
    auto reduce_id = pool_op_id(reduce);
    encoder.set_bytes(coord_total, 3);
    encoder.set_bytes(feat_total, 4);
    encoder.set_bytes(reduce_id, 5);
    dispatch_1d(encoder, clear, std::max(coord_elements, feat_elements));

    auto use_downsample_pool =
        is_nonoverlapping_forward_pool(shape, stride, padding);
    if (use_downsample_pool) {
        auto table_capacity =
            next_power_of_two(std::max(shape.in_capacity * 2, 1));
        auto empty_key = int(0x7fffffff);
        auto table_keys = make_int32_temp(table_capacity);
        auto table_rows = make_int32_temp(table_capacity);
        auto coords_kernel = device.get_kernel(
            "sparse_pool_downsample_coords_hash_i32", library
        );
        encoder.set_compute_pipeline_state(coords_kernel);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_input_array(inputs[1], 1);
        encoder.set_output_array(table_keys, 2);
        encoder.set_output_array(table_rows, 3);
        encoder.set_output_array(outputs[SparseOutCoords], 4);
        encoder.set_output_array(outputs[SparseCounts], 5);
        encoder.set_bytes(shape.in_capacity, 6);
        encoder.set_bytes(table_capacity, 7);
        encoder.set_bytes(empty_key, 8);
        encoder.set_bytes(stride[0], 9);
        encoder.set_bytes(stride[1], 10);
        encoder.set_bytes(stride[2], 11);
        encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
    } else {
        auto coords_kernel = device.get_kernel(
            is_identity_forward_coords(stride, padding)
                ? "sparse_forward_identity_coords_i32"
                : "sparse_conv_forward_coords_i32",
            library
        );
        encoder.set_compute_pipeline_state(coords_kernel);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_input_array(inputs[1], 1);
        encoder.set_input_array(inputs[3], 2);
        encoder.set_output_array(outputs[SparseOutCoords], 3);
        encoder.set_output_array(outputs[SparseCounts], 4);
        encoder.set_bytes(shape.in_capacity, 5);
        encoder.set_bytes(shape.n_kernels, 6);
        if (!is_identity_forward_coords(stride, padding)) {
            encoder.set_bytes(stride[0], 7);
            encoder.set_bytes(stride[1], 8);
            encoder.set_bytes(stride[2], 9);
            encoder.set_bytes(padding[0], 10);
            encoder.set_bytes(padding[1], 11);
            encoder.set_bytes(padding[2], 12);
        }
        dispatch_1d(
            encoder,
            coords_kernel,
            is_identity_forward_coords(stride, padding)
                ? static_cast<size_t>(shape.in_capacity) * 4
                : static_cast<size_t>(shape.in_capacity)
        );
    }

    auto use_downsample_row_gather =
        use_downsample_pool && reduce != PoolReduceOp::Sum;
    auto gather = device.get_kernel(
        use_downsample_pool
            ? (use_downsample_row_gather
                   ? "sparse_pool_downsample_gather_rows_f32_i32"
                   : "sparse_pool_downsample_gather_f32_i32")
            : "sparse_pool_forward_gather_f32_i32",
        library
    );
    encoder.set_compute_pipeline_state(gather);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[1], 1);
    encoder.set_input_array(inputs[2], 2);
    if (use_downsample_pool) {
        encoder.set_input_array(outputs[SparseOutCoords], 3);
        encoder.set_input_array(outputs[SparseCounts], 4);
        encoder.set_output_array(outputs[SparseOutFeats], 5);
        encoder.set_bytes(reduce_id, 6);
        encoder.set_bytes(shape.in_capacity, 7);
        encoder.set_bytes(shape.out_capacity, 8);
        encoder.set_bytes(shape.channels, 9);
        encoder.set_bytes(stride[0], 10);
        encoder.set_bytes(stride[1], 11);
        encoder.set_bytes(stride[2], 12);
        encoder.set_bytes(stride_at(inputs[2], 0), 13);
        encoder.set_bytes(stride_at(inputs[2], 1), 14);
    } else {
        encoder.set_input_array(inputs[3], 3);
        encoder.set_input_array(outputs[SparseOutCoords], 4);
        encoder.set_input_array(outputs[SparseCounts], 5);
        encoder.set_output_array(outputs[SparseOutFeats], 6);
        encoder.set_bytes(reduce_id, 7);
        encoder.set_bytes(shape.in_capacity, 8);
        encoder.set_bytes(shape.out_capacity, 9);
        encoder.set_bytes(shape.n_kernels, 10);
        encoder.set_bytes(shape.channels, 11);
        encoder.set_bytes(stride[0], 12);
        encoder.set_bytes(stride[1], 13);
        encoder.set_bytes(stride[2], 14);
        encoder.set_bytes(padding[0], 15);
        encoder.set_bytes(padding[1], 16);
        encoder.set_bytes(padding[2], 17);
        encoder.set_bytes(stride_at(inputs[2], 0), 18);
        encoder.set_bytes(stride_at(inputs[2], 1), 19);
    }
    dispatch_1d(
        encoder,
        gather,
        use_downsample_row_gather ? static_cast<size_t>(shape.out_capacity)
                                  : feat_elements
    );
#else
    (void)reduce;
    (void)shape;
    (void)stride;
    (void)padding;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

void eval_sparse_pool_grad(
    PoolReduceOp reduce,
    SparsePoolShape shape,
    Triple stride, // NOLINT(bugprone-easily-swappable-parameters)
    Triple padding,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
#ifdef _METAL_
    auto& out = outputs[0];
    backend::allocate(out);

    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);
    auto kernel = device.get_kernel("sparse_pool_grad_f32_i32_serial", library);

    encoder.set_compute_pipeline_state(kernel);
    for (int i = 0; i < int(inputs.size()); ++i) {
        encoder.set_input_array(inputs[i], i);
    }
    encoder.set_output_array(out, 6);
    encoder.set_bytes(pool_op_id(reduce), 7);
    encoder.set_bytes(shape.in_capacity, 8);
    encoder.set_bytes(shape.out_capacity, 9);
    encoder.set_bytes(shape.n_kernels, 10);
    encoder.set_bytes(shape.channels, 11);
    encoder.set_bytes(stride[0], 12);
    encoder.set_bytes(stride[1], 13);
    encoder.set_bytes(stride[2], 14);
    encoder.set_bytes(padding[0], 15);
    encoder.set_bytes(padding[1], 16);
    encoder.set_bytes(padding[2], 17);
    encoder.set_bytes(stride_at(inputs[0], 0), 18);
    encoder.set_bytes(stride_at(inputs[0], 1), 19);
    encoder.set_bytes(stride_at(inputs[1], 0), 20);
    encoder.set_bytes(stride_at(inputs[1], 1), 21);
    encoder.set_bytes(stride_at(inputs[2], 0), 22);
    encoder.set_bytes(stride_at(inputs[2], 1), 23);
    encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
#else
    (void)reduce;
    (void)shape;
    (void)stride;
    (void)padding;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

void eval_sparse_pool_jvp(
    PoolReduceOp reduce,
    SparsePoolShape shape,
    Triple stride, // NOLINT(bugprone-easily-swappable-parameters)
    Triple padding,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
#ifdef _METAL_
    auto& out = outputs[0];
    backend::allocate(out);

    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);
    auto kernel = device.get_kernel("sparse_pool_jvp_f32_i32_serial", library);

    encoder.set_compute_pipeline_state(kernel);
    for (int i = 0; i < int(inputs.size()); ++i) {
        encoder.set_input_array(inputs[i], i);
    }
    encoder.set_output_array(out, 6);
    encoder.set_bytes(pool_op_id(reduce), 7);
    encoder.set_bytes(shape.in_capacity, 8);
    encoder.set_bytes(shape.out_capacity, 9);
    encoder.set_bytes(shape.n_kernels, 10);
    encoder.set_bytes(shape.channels, 11);
    encoder.set_bytes(stride[0], 12);
    encoder.set_bytes(stride[1], 13);
    encoder.set_bytes(stride[2], 14);
    encoder.set_bytes(padding[0], 15);
    encoder.set_bytes(padding[1], 16);
    encoder.set_bytes(padding[2], 17);
    encoder.set_bytes(stride_at(inputs[0], 0), 18);
    encoder.set_bytes(stride_at(inputs[0], 1), 19);
    encoder.set_bytes(stride_at(inputs[1], 0), 20);
    encoder.set_bytes(stride_at(inputs[1], 1), 21);
    encoder.set_bytes(stride_at(inputs[2], 0), 22);
    encoder.set_bytes(stride_at(inputs[2], 1), 23);
    encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
#else
    (void)reduce;
    (void)shape;
    (void)stride;
    (void)padding;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

} // namespace mlx_lattice::exec::metal
