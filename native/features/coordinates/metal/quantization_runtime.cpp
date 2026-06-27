#include "features/coordinates/metal/runtime_detail.h"

namespace mlx_lattice::coords::metal {
void eval_sparse_quantize(
    QuantizationSpec spec,
    int rows,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    require_f32_input(inputs[0], "points");
    require_i32_input(inputs[1], "batch indices");
    require_i32_input(inputs[2], "active rows");

#ifdef _METAL_
    backend::allocate_all(outputs);

    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);
    auto table_capacity = coord_hash_capacity(rows);
    auto table = make_int32_temp(table_capacity);
    auto quantized_coords = make_int32_temp(rows * 4);
    auto selected = make_int32_temp(rows);
    auto compact_buffers = make_stable_compact_buffers(rows);
    encoder.add_temporaries({table, quantized_coords, selected});
    clear_coord_hash(device, library, encoder, table, table_capacity);

    auto clear = device.get_kernel("clear_sparse_quantization_i32", library);
    encoder.set_compute_pipeline_state(clear);
    encoder.set_output_array(outputs[0], 0);
    encoder.set_output_array(outputs[1], 1);
    encoder.set_output_array(outputs[2], 2);
    encoder.set_output_array(outputs[3], 3);
    encoder.set_bytes(rows, 4);
    dispatch_1d(encoder, clear, static_cast<size_t>(rows) * 4);

    auto quantize = device.get_kernel("quantize_points_i32", library);
    encoder.set_compute_pipeline_state(quantize);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[1], 1);
    encoder.set_input_array(inputs[2], 2);
    encoder.set_output_array(quantized_coords, 3);
    encoder.set_bytes(rows, 4);
    encoder.set_bytes(spec.voxel_size[0], 5);
    encoder.set_bytes(spec.voxel_size[1], 6);
    encoder.set_bytes(spec.voxel_size[2], 7);
    encoder.set_bytes(spec.origin[0], 8);
    encoder.set_bytes(spec.origin[1], 9);
    encoder.set_bytes(spec.origin[2], 10);
    dispatch_1d(encoder, quantize, static_cast<size_t>(rows));

    auto build = device.get_kernel("build_quantized_point_hash_i32", library);
    encoder.set_compute_pipeline_state(build);
    encoder.set_input_array(quantized_coords, 0);
    encoder.set_input_array(inputs[2], 1);
    encoder.set_output_array(table, 2);
    encoder.set_bytes(rows, 3);
    encoder.set_bytes(table_capacity, 4);
    dispatch_1d(encoder, build, static_cast<size_t>(rows));

    auto plan = device.get_kernel("plan_quantized_points_i32", library);
    encoder.set_compute_pipeline_state(plan);
    encoder.set_input_array(quantized_coords, 0);
    encoder.set_input_array(inputs[2], 1);
    encoder.set_input_array(table, 2);
    encoder.set_output_array(selected, 3);
    encoder.set_bytes(rows, 4);
    encoder.set_bytes(table_capacity, 5);
    dispatch_1d(encoder, plan, static_cast<size_t>(rows));

    encode_stable_compact_offsets(
        device, library, encoder, selected, outputs[1], compact_buffers, rows
    );

    auto fill = device.get_kernel("fill_quantized_points_i32", library);
    encoder.set_compute_pipeline_state(fill);
    encoder.set_input_array(quantized_coords, 0);
    encoder.set_input_array(inputs[2], 1);
    encoder.set_input_array(selected, 2);
    encoder.set_input_array(compact_buffers.local_offsets, 3);
    encoder.set_input_array(compact_buffers.block_offsets, 4);
    encoder.set_output_array(outputs[0], 5);
    encoder.set_bytes(rows, 6);
    dispatch_1d(encoder, fill, static_cast<size_t>(rows));

    auto map = device.get_kernel("map_quantized_points_i32", library);
    encoder.set_compute_pipeline_state(map);
    encoder.set_input_array(quantized_coords, 0);
    encoder.set_input_array(inputs[2], 1);
    encoder.set_input_array(table, 2);
    encoder.set_input_array(compact_buffers.local_offsets, 3);
    encoder.set_input_array(compact_buffers.block_offsets, 4);
    encoder.set_output_array(outputs[2], 5);
    encoder.set_output_array(outputs[3], 6);
    encoder.set_bytes(rows, 7);
    encoder.set_bytes(table_capacity, 8);
    dispatch_1d(encoder, map, static_cast<size_t>(rows));
#else
    (void)spec;
    (void)rows;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

void eval_voxelize_features(
    VoxelReduceOp reduce,
    VoxelFeatureShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    require_f32_input(inputs[0], "features");
    require_i32_input(inputs[1], "inverse rows");
    require_i32_input(inputs[2], "voxel counts");
    require_i32_input(inputs[3], "active rows");

#ifdef _METAL_
    backend::allocate_all(outputs);

    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);
    auto elements = shape.voxel_rows * shape.channels;
    auto clear = device.get_kernel("clear_voxelized_features_f32", library);
    encoder.set_compute_pipeline_state(clear);
    encoder.set_output_array(outputs[0], 0);
    encoder.set_bytes(elements, 1);
    dispatch_1d(encoder, clear, static_cast<size_t>(elements));

    auto scatter =
        device.get_kernel("scatter_voxelized_features_f32_i32", library);
    encoder.set_compute_pipeline_state(scatter);
    bind_input_arrays(encoder, inputs);
    encoder.set_output_array(outputs[0], 4);
    encoder.set_bytes(voxel_reduce_op_id(reduce), 5);
    encoder.set_bytes(shape.point_rows, 6);
    encoder.set_bytes(shape.voxel_rows, 7);
    encoder.set_bytes(shape.channels, 8);
    dispatch_1d(
        encoder,
        scatter,
        static_cast<size_t>(shape.point_rows) *
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

void eval_voxelize_feature_grad(
    VoxelReduceOp reduce,
    VoxelFeatureShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    require_f32_input(inputs[0], "cotangent");
    require_i32_input(inputs[1], "inverse rows");
    require_i32_input(inputs[2], "voxel counts");
    require_i32_input(inputs[3], "active rows");

#ifdef _METAL_
    backend::allocate_all(outputs);

    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);
    auto kernel = device.get_kernel("voxelize_feature_grad_f32_i32", library);

    encoder.set_compute_pipeline_state(kernel);
    bind_input_arrays(encoder, inputs);
    encoder.set_output_array(outputs[0], 4);
    encoder.set_bytes(voxel_reduce_op_id(reduce), 5);
    encoder.set_bytes(shape.point_rows, 6);
    encoder.set_bytes(shape.voxel_rows, 7);
    encoder.set_bytes(shape.channels, 8);
    dispatch_1d(
        encoder,
        kernel,
        static_cast<size_t>(shape.point_rows) *
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

// MARK: - relations

} // namespace mlx_lattice::coords::metal
