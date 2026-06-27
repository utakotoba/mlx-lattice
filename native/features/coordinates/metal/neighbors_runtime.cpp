#include "features/coordinates/metal/runtime_detail.h"

namespace mlx_lattice::coords::metal {
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
    auto clear = device.get_kernel("build_neighbor_relation_i32", library);
    constexpr int kKnnHashThreshold = 512;
    auto use_knn_hash = op == NeighborRelationOp::Knn &&
                        shape.max_neighbors <= 16 &&
                        shape.source_rows > kKnnHashThreshold;
    auto use_radius_hash = op == NeighborRelationOp::Radius;

    encoder.set_compute_pipeline_state(clear);
    encoder.set_input_array(inputs[3], 0);
    encoder.set_output_array(outputs[NeighborQueryRows], 1);
    encoder.set_output_array(outputs[NeighborSourceRows], 2);
    encoder.set_output_array(outputs[NeighborIds], 3);
    encoder.set_output_array(outputs[NeighborDistances], 4);
    encoder.set_output_array(outputs[NeighborRowOffsets], 5);
    encoder.set_output_array(outputs[NeighborCounts], 6);
    encoder.set_bytes(shape.query_rows, 7);
    encoder.set_bytes(shape.max_neighbors, 8);
    dispatch_1d(
        encoder,
        clear,
        use_knn_hash || use_radius_hash
            ? size_t{1}
            : static_cast<size_t>(shape.query_rows) *
                  static_cast<size_t>(shape.max_neighbors)
    );

    if (use_knn_hash) {
        auto table_capacity = coord_hash_capacity(shape.source_rows);
        auto table = make_int32_temp(table_capacity);
        encoder.add_temporary(table);
        clear_coord_hash(device, library, encoder, table, table_capacity);

        auto insert =
            device.get_kernel("coord_hash_insert_active_rows_i32", library);
        encoder.set_compute_pipeline_state(insert);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_input_array(inputs[2], 1);
        encoder.set_output_array(table, 2);
        encoder.set_bytes(shape.source_rows, 3);
        encoder.set_bytes(table_capacity, 4);
        dispatch_1d(encoder, insert, static_cast<size_t>(shape.source_rows));

        auto search_radius = std::max(
            2,
            static_cast<int>(
                std::ceil(std::cbrt(static_cast<double>(shape.max_neighbors)))
            ) * 2
        );
        auto count = device.get_kernel("count_knn_relation_hash_i32", library);
        encoder.set_compute_pipeline_state(count);
        bind_input_arrays(encoder, inputs);
        encoder.set_input_array(table, 4);
        encoder.set_output_array(outputs[NeighborRowOffsets], 5);
        encoder.set_bytes(shape.source_rows, 6);
        encoder.set_bytes(shape.query_rows, 7);
        encoder.set_bytes(shape.max_neighbors, 8);
        encoder.set_bytes(search_radius, 9);
        encoder.set_bytes(table_capacity, 10);
        dispatch_1d(encoder, count, static_cast<size_t>(shape.query_rows));

        encode_neighbor_row_offsets(
            device,
            library,
            encoder,
            {outputs[NeighborRowOffsets], outputs[NeighborCounts]},
            shape.query_rows
        );

        auto fill =
            device.get_kernel("fill_knn_relation_compact_hash_i32", library);
        encoder.set_compute_pipeline_state(fill);
        bind_input_arrays(encoder, inputs);
        encoder.set_input_array(table, 4);
        encoder.set_input_array(outputs[NeighborRowOffsets], 5);
        encoder.set_output_array(outputs[NeighborQueryRows], 6);
        encoder.set_output_array(outputs[NeighborSourceRows], 7);
        encoder.set_output_array(outputs[NeighborIds], 8);
        encoder.set_output_array(outputs[NeighborDistances], 9);
        encoder.set_bytes(shape.source_rows, 10);
        encoder.set_bytes(shape.query_rows, 11);
        encoder.set_bytes(shape.max_neighbors, 12);
        encoder.set_bytes(search_radius, 13);
        encoder.set_bytes(table_capacity, 14);
        dispatch_1d(encoder, fill, static_cast<size_t>(shape.query_rows));
        return;
    }

    if (op == NeighborRelationOp::Knn && shape.max_neighbors <= 16 &&
        !use_knn_hash) {
        auto fill = device.get_kernel("fill_knn_relation_topk_i32", library);
        encoder.set_compute_pipeline_state(fill);
        bind_input_arrays(encoder, inputs);
        encoder.set_output_array(outputs[NeighborQueryRows], 4);
        encoder.set_output_array(outputs[NeighborSourceRows], 5);
        encoder.set_output_array(outputs[NeighborIds], 6);
        encoder.set_output_array(outputs[NeighborDistances], 7);
        encoder.set_bytes(shape.source_rows, 8);
        encoder.set_bytes(shape.query_rows, 9);
        encoder.set_bytes(shape.max_neighbors, 10);
        encoder.dispatch_threadgroups(
            MTL::Size(static_cast<size_t>(shape.query_rows), 1, 1),
            MTL::Size(128, 1, 1)
        );
    } else if (use_radius_hash) {
        auto table_capacity = coord_hash_capacity(shape.source_rows);
        auto table = make_int32_temp(table_capacity);
        encoder.add_temporary(table);
        clear_coord_hash(device, library, encoder, table, table_capacity);

        auto insert =
            device.get_kernel("coord_hash_insert_active_rows_i32", library);
        encoder.set_compute_pipeline_state(insert);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_input_array(inputs[2], 1);
        encoder.set_output_array(table, 2);
        encoder.set_bytes(shape.source_rows, 3);
        encoder.set_bytes(table_capacity, 4);
        dispatch_1d(encoder, insert, static_cast<size_t>(shape.source_rows));

        auto ceil_radius =
            static_cast<int>(std::ceil(std::sqrt(radius_squared)));
        auto count =
            device.get_kernel("count_radius_relation_hash_i32", library);
        encoder.set_compute_pipeline_state(count);
        bind_input_arrays(encoder, inputs);
        encoder.set_input_array(table, 4);
        encoder.set_output_array(outputs[NeighborRowOffsets], 5);
        encoder.set_bytes(shape.source_rows, 6);
        encoder.set_bytes(shape.query_rows, 7);
        encoder.set_bytes(shape.max_neighbors, 8);
        encoder.set_bytes(radius_squared, 9);
        encoder.set_bytes(ceil_radius, 10);
        encoder.set_bytes(table_capacity, 11);
        dispatch_1d(encoder, count, static_cast<size_t>(shape.query_rows));

        encode_neighbor_row_offsets(
            device,
            library,
            encoder,
            {outputs[NeighborRowOffsets], outputs[NeighborCounts]},
            shape.query_rows
        );

        auto fill =
            device.get_kernel("fill_radius_relation_compact_hash_i32", library);
        encoder.set_compute_pipeline_state(fill);
        bind_input_arrays(encoder, inputs);
        encoder.set_input_array(table, 4);
        encoder.set_input_array(outputs[NeighborRowOffsets], 5);
        encoder.set_output_array(outputs[NeighborQueryRows], 6);
        encoder.set_output_array(outputs[NeighborSourceRows], 7);
        encoder.set_output_array(outputs[NeighborIds], 8);
        encoder.set_output_array(outputs[NeighborDistances], 9);
        encoder.set_bytes(shape.source_rows, 10);
        encoder.set_bytes(shape.query_rows, 11);
        encoder.set_bytes(shape.max_neighbors, 12);
        encoder.set_bytes(radius_squared, 13);
        encoder.set_bytes(ceil_radius, 14);
        encoder.set_bytes(table_capacity, 15);
        dispatch_1d(encoder, fill, static_cast<size_t>(shape.query_rows));
        return;
    } else {
        auto fill = device.get_kernel("fill_neighbor_relation_i32", library);
        encoder.set_compute_pipeline_state(fill);
        bind_input_arrays(encoder, inputs);
        encoder.set_output_array(outputs[NeighborQueryRows], 4);
        encoder.set_output_array(outputs[NeighborSourceRows], 5);
        encoder.set_output_array(outputs[NeighborIds], 6);
        encoder.set_output_array(outputs[NeighborDistances], 7);
        encoder.set_bytes(neighbor_relation_op_id(op), 8);
        encoder.set_bytes(shape.source_rows, 9);
        encoder.set_bytes(shape.query_rows, 10);
        encoder.set_bytes(shape.max_neighbors, 11);
        encoder.set_bytes(radius_squared, 12);
        dispatch_1d(encoder, fill, static_cast<size_t>(shape.query_rows));
    }

    auto compact = device.get_kernel("compact_neighbor_relation_i32", library);
    encoder.set_compute_pipeline_state(compact);
    encoder.set_output_array(outputs[NeighborQueryRows], 0);
    encoder.set_output_array(outputs[NeighborSourceRows], 1);
    encoder.set_output_array(outputs[NeighborIds], 2);
    encoder.set_output_array(outputs[NeighborDistances], 3);
    encoder.set_output_array(outputs[NeighborRowOffsets], 4);
    encoder.set_output_array(outputs[NeighborCounts], 5);
    encoder.set_bytes(shape.query_rows, 6);
    encoder.set_bytes(shape.max_neighbors, 7);
    dispatch_single(encoder);
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
