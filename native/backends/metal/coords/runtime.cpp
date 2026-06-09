#include "backends/metal/coords/runtime.h"

#include <algorithm>
#include <initializer_list>
#include <stdexcept>
#include <string>

#include "backends/array_utils.h"
#include "backends/metal/runtime_utils.h"
#include "mlx/stream.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace mlx_lattice::coords::metal {

namespace {

const char* set_kernel_name(CoordSetOp op) {
    switch (op) {
    case CoordSetOp::Downsample:
        return "downsample_coords_i32";
    case CoordSetOp::Union:
        return "union_coords_i32";
    case CoordSetOp::Intersection:
        return "intersection_coords_i32";
    }
}

const char* relation_kernel_name(CoordRelationOp op) {
    switch (op) {
    case CoordRelationOp::Forward:
        return "build_forward_kernel_relation_i32";
    case CoordRelationOp::Transposed:
        return "build_transposed_kernel_relation_i32";
    }
}

int neighbor_relation_op_id(NeighborRelationOp op) {
    switch (op) {
    case NeighborRelationOp::Knn:
        return 0;
    case NeighborRelationOp::Radius:
        return 1;
    }
}

// MARK: - guards

void require_i32_inputs(
    const std::vector<mx::array>& inputs,
    std::initializer_list<const char*> names
) {
    int index = 0;
    for (auto name : names) {
        if (inputs[index++].dtype() != mx::int32) {
            throw std::invalid_argument(
                std::string("Metal coordinate kernels require int32 ") + name +
                "."
            );
        }
    }
}

} // namespace

// MARK: - set ops

void eval_set_coords(
    CoordSetOp op,
    Triple stride,
    CoordSetShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    require_i32_inputs(inputs, {"coords"});
    if (op != CoordSetOp::Downsample) {
        require_i32_inputs(inputs, {"coords", "rhs coords"});
    }

#ifdef _METAL_
    auto& out_coords = outputs[0];
    auto& count = outputs[1];
    backend::allocate(out_coords);
    backend::allocate(count);

    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);
    auto kernel = device.get_kernel(set_kernel_name(op), library);

    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(inputs[0], 0);
    if (op != CoordSetOp::Downsample) {
        encoder.set_input_array(inputs[1], 1);
        encoder.set_output_array(out_coords, 2);
        encoder.set_output_array(count, 3);
        encoder.set_bytes(shape.lhs_rows, 4);
        encoder.set_bytes(shape.rhs_rows, 5);
    } else {
        encoder.set_output_array(out_coords, 1);
        encoder.set_output_array(count, 2);
        encoder.set_bytes(shape.lhs_rows, 3);
        encoder.set_bytes(stride[0], 4);
        encoder.set_bytes(stride[1], 5);
        encoder.set_bytes(stride[2], 6);
    }
    encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
#else
    (void)op;
    (void)stride;
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

void eval_lookup_coords(
    CoordLookupShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    require_i32_inputs(inputs, {"coords", "queries"});

#ifdef _METAL_
    auto& out = outputs[0];
    backend::allocate(out);

    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);
    auto kernel = device.get_kernel("lookup_coords_i32", library);
    auto group = std::min(
        static_cast<size_t>(std::max(shape.query_rows, 1)),
        kernel->maxTotalThreadsPerThreadgroup()
    );

    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[1], 1);
    encoder.set_output_array(out, 2);
    encoder.set_bytes(shape.rows, 3);
    encoder.set_bytes(shape.query_rows, 4);
    encoder.dispatch_threads(
        MTL::Size(static_cast<size_t>(std::max(shape.query_rows, 1)), 1, 1),
        MTL::Size(group, 1, 1)
    );
#else
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

// MARK: - relations

void eval_generic_kernel_relation(
    CoordRelationOp op,
    int rows, // NOLINT(bugprone-easily-swappable-parameters)
    int kernel_count,
    Triple stride, // NOLINT(bugprone-easily-swappable-parameters)
    Triple padding,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    require_i32_inputs(inputs, {"coords", "kernel offsets", "active rows"});

#ifdef _METAL_
    backend::allocate_all(outputs);

    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);
    auto kernel = device.get_kernel(relation_kernel_name(op), library);

    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[1], 1);
    encoder.set_input_array(inputs[2], 2);
    for (int i = 0; i < int(outputs.size()); ++i) {
        encoder.set_output_array(outputs[i], i + 3);
    }
    encoder.set_bytes(rows, 8);
    encoder.set_bytes(kernel_count, 9);
    encoder.set_bytes(stride[0], 10);
    encoder.set_bytes(stride[1], 11);
    encoder.set_bytes(stride[2], 12);
    encoder.set_bytes(padding[0], 13);
    encoder.set_bytes(padding[1], 14);
    encoder.set_bytes(padding[2], 15);
    encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
#else
    (void)op;
    (void)rows;
    (void)kernel_count;
    (void)stride;
    (void)padding;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

void eval_generative_kernel_relation(
    int rows, // NOLINT(bugprone-easily-swappable-parameters)
    int kernel_count,
    Triple stride,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    require_i32_inputs(inputs, {"coords", "kernel offsets", "active rows"});

#ifdef _METAL_
    backend::allocate_all(outputs);

    auto thread_count = std::max(rows * kernel_count, 1);

    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);
    auto kernel =
        device.get_kernel("build_generative_kernel_relation_i32", library);
    auto group = std::min(
        static_cast<size_t>(thread_count),
        kernel->maxTotalThreadsPerThreadgroup()
    );

    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[1], 1);
    encoder.set_input_array(inputs[2], 2);
    for (int i = 0; i < int(outputs.size()); ++i) {
        encoder.set_output_array(outputs[i], i + 3);
    }
    encoder.set_bytes(rows, 8);
    encoder.set_bytes(kernel_count, 9);
    encoder.set_bytes(stride[0], 10);
    encoder.set_bytes(stride[1], 11);
    encoder.set_bytes(stride[2], 12);
    encoder.dispatch_threads(
        MTL::Size(static_cast<size_t>(thread_count), 1, 1),
        MTL::Size(group, 1, 1)
    );
#else
    (void)rows;
    (void)kernel_count;
    (void)stride;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

void eval_neighbor_relation(
    NeighborRelationOp op,
    NeighborRelationShape shape,
    float radius_squared,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    require_i32_inputs(
        inputs,
        {"source coords",
         "query coords",
         "source active rows",
         "query active rows"}
    );

#ifdef _METAL_
    backend::allocate_all(outputs);

    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);
    auto kernel = device.get_kernel("build_neighbor_relation_i32", library);

    encoder.set_compute_pipeline_state(kernel);
    for (int i = 0; i < int(inputs.size()); ++i) {
        encoder.set_input_array(inputs[i], i);
    }
    for (int i = 0; i < int(outputs.size()); ++i) {
        encoder.set_output_array(outputs[i], i + 4);
    }
    encoder.set_bytes(neighbor_relation_op_id(op), 9);
    encoder.set_bytes(shape.source_rows, 10);
    encoder.set_bytes(shape.query_rows, 11);
    encoder.set_bytes(shape.max_neighbors, 12);
    encoder.set_bytes(radius_squared, 13);
    encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
#else
    (void)op;
    (void)shape;
    (void)radius_squared;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

} // namespace mlx_lattice::coords::metal
