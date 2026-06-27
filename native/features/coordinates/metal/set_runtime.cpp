#include "features/coordinates/metal/runtime_detail.h"

namespace mlx_lattice::coords::metal {
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

    if (op == CoordSetOp::Downsample) {
        auto table_capacity = coord_hash_capacity(shape.lhs_rows);
        auto table = make_int32_temp(table_capacity);
        auto selected = make_int32_temp(shape.lhs_rows);
        encoder.add_temporaries({table, selected});
        clear_coord_hash(device, library, encoder, table, table_capacity);
        auto build =
            device.get_kernel("build_downsample_coord_hash_i32", library);
        encoder.set_compute_pipeline_state(build);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_output_array(table, 1);
        encoder.set_bytes(shape.lhs_rows, 2);
        encoder.set_bytes(table_capacity, 3);
        bind_triple_bytes(encoder, stride, 4);
        dispatch_1d(encoder, build, static_cast<size_t>(shape.lhs_rows));

        auto plan = device.get_kernel("plan_downsample_coord_set_i32", library);
        encoder.set_compute_pipeline_state(plan);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_input_array(table, 1);
        encoder.set_output_array(selected, 2);
        encoder.set_bytes(shape.lhs_rows, 3);
        encoder.set_bytes(table_capacity, 4);
        bind_triple_bytes(encoder, stride, 5);
        dispatch_1d(encoder, plan, static_cast<size_t>(shape.lhs_rows));

        auto compact = device.get_kernel(
            shape.lhs_rows >= kParallelCompactThreshold
                ? "scatter_downsample_coord_set_i32"
                : "compact_downsample_coord_set_i32",
            library
        );
        if (shape.lhs_rows >= kParallelCompactThreshold) {
            auto buffers = make_stable_compact_buffers(shape.lhs_rows);
            encode_stable_compact_offsets(
                device,
                library,
                encoder,
                selected,
                count,
                buffers,
                shape.lhs_rows
            );
            encoder.set_compute_pipeline_state(compact);
            encoder.set_input_array(inputs[0], 0);
            encoder.set_input_array(selected, 1);
            encoder.set_input_array(buffers.local_offsets, 2);
            encoder.set_input_array(buffers.block_offsets, 3);
            encoder.set_output_array(out_coords, 4);
            encoder.set_bytes(shape.lhs_rows, 5);
            bind_triple_bytes(encoder, stride, 6);
            dispatch_1d(encoder, compact, static_cast<size_t>(shape.lhs_rows));
            return;
        }
        encoder.set_compute_pipeline_state(compact);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_input_array(selected, 1);
        encoder.set_output_array(out_coords, 2);
        encoder.set_output_array(count, 3);
        encoder.set_bytes(shape.lhs_rows, 4);
        bind_triple_bytes(encoder, stride, 5);
        dispatch_single(encoder);
    } else if (op == CoordSetOp::Union) {
        auto total_rows = shape.lhs_rows + shape.rhs_rows;
        auto lhs_table_capacity = coord_hash_capacity(shape.lhs_rows);
        auto rhs_table_capacity = coord_hash_capacity(shape.rhs_rows);
        auto lhs_table = make_int32_temp(lhs_table_capacity);
        auto rhs_table = make_int32_temp(rhs_table_capacity);
        auto selected = make_int32_temp(total_rows);
        encoder.add_temporaries({lhs_table, rhs_table, selected});
        clear_coord_hash(
            device, library, encoder, lhs_table, lhs_table_capacity
        );
        clear_coord_hash(
            device, library, encoder, rhs_table, rhs_table_capacity
        );
        insert_coord_hash(
            device,
            library,
            encoder,
            inputs[0],
            lhs_table,
            CoordHashShape{shape.lhs_rows, lhs_table_capacity}
        );
        insert_coord_hash(
            device,
            library,
            encoder,
            inputs[1],
            rhs_table,
            CoordHashShape{shape.rhs_rows, rhs_table_capacity}
        );

        auto plan = device.get_kernel("plan_union_coord_set_i32", library);
        encoder.set_compute_pipeline_state(plan);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_input_array(inputs[1], 1);
        encoder.set_input_array(lhs_table, 2);
        encoder.set_input_array(rhs_table, 3);
        encoder.set_output_array(selected, 4);
        encoder.set_bytes(shape.lhs_rows, 5);
        encoder.set_bytes(shape.rhs_rows, 6);
        encoder.set_bytes(lhs_table_capacity, 7);
        encoder.set_bytes(rhs_table_capacity, 8);
        dispatch_1d(encoder, plan, static_cast<size_t>(total_rows));

        auto compact = device.get_kernel(
            total_rows >= kParallelCompactThreshold
                ? "scatter_union_coord_set_i32"
                : "compact_union_coord_set_i32",
            library
        );
        if (total_rows >= kParallelCompactThreshold) {
            auto buffers = make_stable_compact_buffers(total_rows);
            encode_stable_compact_offsets(
                device, library, encoder, selected, count, buffers, total_rows
            );
            encoder.set_compute_pipeline_state(compact);
            encoder.set_input_array(inputs[0], 0);
            encoder.set_input_array(inputs[1], 1);
            encoder.set_input_array(selected, 2);
            encoder.set_input_array(buffers.local_offsets, 3);
            encoder.set_input_array(buffers.block_offsets, 4);
            encoder.set_output_array(out_coords, 5);
            encoder.set_bytes(shape.lhs_rows, 6);
            encoder.set_bytes(shape.rhs_rows, 7);
            dispatch_1d(encoder, compact, static_cast<size_t>(total_rows));
            return;
        }
        encoder.set_compute_pipeline_state(compact);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_input_array(inputs[1], 1);
        encoder.set_input_array(selected, 2);
        encoder.set_output_array(out_coords, 3);
        encoder.set_output_array(count, 4);
        encoder.set_bytes(shape.lhs_rows, 5);
        encoder.set_bytes(shape.rhs_rows, 6);
        dispatch_single(encoder);
    } else {
        auto rhs_table_capacity = coord_hash_capacity(shape.rhs_rows);
        auto lhs_table_capacity = coord_hash_capacity(shape.lhs_rows);
        auto rhs_table = make_int32_temp(rhs_table_capacity);
        auto lhs_table = make_int32_temp(lhs_table_capacity);
        auto selected = make_int32_temp(shape.lhs_rows);
        encoder.add_temporaries({rhs_table, lhs_table, selected});
        clear_coord_hash(
            device, library, encoder, rhs_table, rhs_table_capacity
        );
        clear_coord_hash(
            device, library, encoder, lhs_table, lhs_table_capacity
        );
        insert_coord_hash(
            device,
            library,
            encoder,
            inputs[1],
            rhs_table,
            CoordHashShape{shape.rhs_rows, rhs_table_capacity}
        );
        insert_coord_hash(
            device,
            library,
            encoder,
            inputs[0],
            lhs_table,
            CoordHashShape{shape.lhs_rows, lhs_table_capacity}
        );
        auto plan =
            device.get_kernel("plan_intersection_coord_set_i32", library);
        encoder.set_compute_pipeline_state(plan);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_input_array(inputs[1], 1);
        encoder.set_input_array(rhs_table, 2);
        encoder.set_input_array(lhs_table, 3);
        encoder.set_output_array(selected, 4);
        encoder.set_bytes(shape.lhs_rows, 5);
        encoder.set_bytes(rhs_table_capacity, 6);
        encoder.set_bytes(lhs_table_capacity, 7);
        dispatch_1d(encoder, plan, static_cast<size_t>(shape.lhs_rows));

        auto compact = device.get_kernel(
            shape.lhs_rows >= kParallelCompactThreshold
                ? "scatter_intersection_coord_set_i32"
                : "compact_intersection_coord_set_i32",
            library
        );
        if (shape.lhs_rows >= kParallelCompactThreshold) {
            auto buffers = make_stable_compact_buffers(shape.lhs_rows);
            encode_stable_compact_offsets(
                device,
                library,
                encoder,
                selected,
                count,
                buffers,
                shape.lhs_rows
            );
            encoder.set_compute_pipeline_state(compact);
            encoder.set_input_array(inputs[0], 0);
            encoder.set_input_array(selected, 1);
            encoder.set_input_array(buffers.local_offsets, 2);
            encoder.set_input_array(buffers.block_offsets, 3);
            encoder.set_output_array(out_coords, 4);
            encoder.set_bytes(shape.lhs_rows, 5);
            dispatch_1d(encoder, compact, static_cast<size_t>(shape.lhs_rows));
            return;
        }
        encoder.set_compute_pipeline_state(compact);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_input_array(selected, 1);
        encoder.set_output_array(out_coords, 2);
        encoder.set_output_array(count, 3);
        encoder.set_bytes(shape.lhs_rows, 4);
        dispatch_single(encoder);
    }
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
    auto table_capacity = coord_hash_capacity(shape.rows);
    auto table = make_int32_temp(table_capacity);
    encoder.add_temporary(table);
    clear_coord_hash(device, library, encoder, table, table_capacity);
    insert_coord_hash(
        device,
        library,
        encoder,
        inputs[0],
        table,
        CoordHashShape{shape.rows, table_capacity}
    );

    auto kernel = device.get_kernel("lookup_coords_hash_i32", library);
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[1], 1);
    encoder.set_input_array(table, 2);
    encoder.set_output_array(out, 3);
    encoder.set_bytes(shape.query_rows, 4);
    encoder.set_bytes(table_capacity, 5);
    dispatch_1d(encoder, kernel, static_cast<size_t>(shape.query_rows));
#else
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

} // namespace mlx_lattice::coords::metal
