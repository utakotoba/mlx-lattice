#include "features/coordinates/metal/runtime_detail.h"

namespace mlx_lattice::coords::metal {
void eval_occupancy_downsample(
    CoordActiveRowsShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    require_i32_inputs(inputs, {"coords", "active rows"});

#ifdef _METAL_
    backend::allocate_all(outputs);

    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);
    auto table_capacity = coord_hash_capacity(shape.rows);
    auto table = make_int32_temp(table_capacity);
    auto selected = make_int32_temp(shape.rows);
    encoder.add_temporaries({table, selected});
    clear_coord_hash(device, library, encoder, table, table_capacity);

    auto clear = device.get_kernel("clear_occupancy_downsample_i32", library);
    encoder.set_compute_pipeline_state(clear);
    encoder.set_output_array(outputs[2], 0);
    encoder.set_bytes(shape.rows, 1);
    dispatch_1d(encoder, clear, static_cast<size_t>(shape.rows));

    auto build =
        device.get_kernel("build_occupancy_downsample_hash_i32", library);
    encoder.set_compute_pipeline_state(build);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[1], 1);
    encoder.set_output_array(table, 2);
    encoder.set_bytes(shape.rows, 3);
    encoder.set_bytes(table_capacity, 4);
    dispatch_1d(encoder, build, static_cast<size_t>(shape.rows));

    auto plan = device.get_kernel("plan_occupancy_downsample_i32", library);
    encoder.set_compute_pipeline_state(plan);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[1], 1);
    encoder.set_input_array(table, 2);
    encoder.set_output_array(selected, 3);
    encoder.set_bytes(shape.rows, 4);
    encoder.set_bytes(table_capacity, 5);
    dispatch_1d(encoder, plan, static_cast<size_t>(shape.rows));

    auto buffers = make_stable_compact_buffers(shape.rows);
    encode_stable_compact_offsets(
        device, library, encoder, selected, outputs[1], buffers, shape.rows
    );

    auto scatter =
        device.get_kernel("scatter_occupancy_downsample_i32", library);
    encoder.set_compute_pipeline_state(scatter);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(selected, 1);
    encoder.set_input_array(buffers.local_offsets, 2);
    encoder.set_input_array(buffers.block_offsets, 3);
    encoder.set_output_array(outputs[0], 4);
    encoder.set_bytes(shape.rows, 5);
    dispatch_1d(encoder, scatter, static_cast<size_t>(shape.rows));

    auto fill = device.get_kernel("fill_occupancy_downsample_i32", library);
    encoder.set_compute_pipeline_state(fill);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[1], 1);
    encoder.set_input_array(table, 2);
    encoder.set_input_array(buffers.local_offsets, 3);
    encoder.set_input_array(buffers.block_offsets, 4);
    encoder.set_output_array(outputs[2], 5);
    encoder.set_bytes(shape.rows, 6);
    encoder.set_bytes(table_capacity, 7);
    dispatch_1d(encoder, fill, static_cast<size_t>(shape.rows));
#else
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

void eval_occupancy_expand(
    CoordActiveRowsShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    require_i32_inputs(inputs, {"coords", "active rows", "occupancy"});

#ifdef _METAL_
    backend::allocate_all(outputs);

    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);
    auto total_rows = shape.rows * 8;
    auto selected = make_int32_temp(total_rows);
    encoder.add_temporary(selected);

    auto plan = device.get_kernel("plan_occupancy_expand_i32", library);
    encoder.set_compute_pipeline_state(plan);
    encoder.set_input_array(inputs[1], 0);
    encoder.set_input_array(inputs[2], 1);
    encoder.set_output_array(selected, 2);
    encoder.set_bytes(shape.rows, 3);
    dispatch_1d(encoder, plan, static_cast<size_t>(total_rows));

    auto buffers = make_stable_compact_buffers(total_rows);
    encode_stable_compact_offsets(
        device, library, encoder, selected, outputs[1], buffers, total_rows
    );

    auto scatter = device.get_kernel("scatter_occupancy_expand_i32", library);
    encoder.set_compute_pipeline_state(scatter);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(selected, 1);
    encoder.set_input_array(buffers.local_offsets, 2);
    encoder.set_input_array(buffers.block_offsets, 3);
    encoder.set_output_array(outputs[0], 4);
    encoder.set_output_array(outputs[2], 5);
    encoder.set_output_array(outputs[3], 6);
    encoder.set_bytes(shape.rows, 7);
    dispatch_1d(encoder, scatter, static_cast<size_t>(total_rows));
#else
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

void eval_child_coords_from_indices(
    CoordRowsShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    require_i32_inputs(inputs, {"parent coords", "child indices"});

#ifdef _METAL_
    auto& out = outputs[0];
    backend::allocate(out);

    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);
    auto kernel = device.get_kernel("child_coords_from_indices_i32", library);
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[1], 1);
    encoder.set_output_array(out, 2);
    encoder.set_bytes(shape.rows, 3);
    dispatch_1d(encoder, kernel, static_cast<size_t>(shape.rows));
#else
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

// MARK: - quantization

} // namespace mlx_lattice::coords::metal
