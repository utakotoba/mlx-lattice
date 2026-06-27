#include "features/coordinates/metal/runtime_detail.h"

namespace mlx_lattice::coords::metal {
void eval_generic_kernel_relation(
    CoordRelationOp op,
    int rows, // NOLINT(bugprone-easily-swappable-parameters)
    int kernel_count,
    Triple stride, // NOLINT(bugprone-easily-swappable-parameters)
    Triple padding,
    bool direct,
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

    if (op == CoordRelationOp::Forward &&
        outputs.size() > RelationBaseOutputCount) {
        auto table_capacity =
            static_cast<int>(outputs[RelationBaseOutputCount].shape(0));
        clear_coord_hash(
            device,
            library,
            encoder,
            outputs[RelationBaseOutputCount],
            table_capacity
        );

        auto insert =
            device.get_kernel("coord_hash_insert_active_rows_i32", library);
        encoder.set_compute_pipeline_state(insert);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_input_array(inputs[2], 1);
        encoder.set_output_array(outputs[RelationBaseOutputCount], 2);
        encoder.set_bytes(rows, 3);
        encoder.set_bytes(table_capacity, 4);
        dispatch_1d(encoder, insert, static_cast<size_t>(rows));

        if (!is_identity_forward_relation(stride, padding)) {
            auto out_table = make_int32_temp(table_capacity);
            auto selected = make_int32_temp(rows);
            encoder.add_temporaries({out_table, selected});
            clear_coord_hash(
                device, library, encoder, out_table, table_capacity
            );

            auto build_outputs = device.get_kernel(
                "build_strided_relation_output_hash_i32", library
            );
            encoder.set_compute_pipeline_state(build_outputs);
            encoder.set_input_array(inputs[0], 0);
            encoder.set_input_array(inputs[2], 1);
            encoder.set_output_array(out_table, 2);
            encoder.set_bytes(rows, 3);
            encoder.set_bytes(table_capacity, 4);
            bind_triple_bytes(encoder, stride, 5);
            dispatch_1d(encoder, build_outputs, static_cast<size_t>(rows));

            auto plan_outputs = device.get_kernel(
                "plan_strided_relation_output_coords_i32", library
            );
            encoder.set_compute_pipeline_state(plan_outputs);
            encoder.set_input_array(inputs[0], 0);
            encoder.set_input_array(inputs[2], 1);
            encoder.set_input_array(out_table, 2);
            encoder.set_output_array(selected, 3);
            encoder.set_bytes(rows, 4);
            encoder.set_bytes(table_capacity, 5);
            bind_triple_bytes(encoder, stride, 6);
            dispatch_1d(encoder, plan_outputs, static_cast<size_t>(rows));

            if (rows >= kParallelCompactThreshold) {
                auto buffers = make_stable_compact_buffers(rows);
                encode_relation_compact_offsets(
                    device,
                    library,
                    encoder,
                    selected,
                    outputs[RelationCounts],
                    buffers,
                    rows
                );
                auto compact_outputs = device.get_kernel(
                    "scatter_downsample_coord_set_i32", library
                );
                encoder.set_compute_pipeline_state(compact_outputs);
                encoder.set_input_array(inputs[0], 0);
                encoder.set_input_array(selected, 1);
                encoder.set_input_array(buffers.local_offsets, 2);
                encoder.set_input_array(buffers.block_offsets, 3);
                encoder.set_output_array(outputs[RelationOutCoords], 4);
                encoder.set_bytes(rows, 5);
                bind_triple_bytes(encoder, stride, 6);
                dispatch_1d(
                    encoder, compact_outputs, static_cast<size_t>(rows)
                );
            } else {
                auto compact_outputs = device.get_kernel(
                    "compact_strided_relation_output_coords_i32", library
                );
                encoder.set_compute_pipeline_state(compact_outputs);
                encoder.set_input_array(inputs[0], 0);
                encoder.set_input_array(selected, 1);
                encoder.set_output_array(outputs[RelationOutCoords], 2);
                encoder.set_output_array(outputs[RelationCounts], 3);
                encoder.set_bytes(rows, 4);
                bind_triple_bytes(encoder, stride, 5);
                dispatch_single(encoder);
            }

            auto slot_in_rows = make_int32_temp(rows * kernel_count);
            auto slot_out_rows = make_int32_temp(rows * kernel_count);
            auto slot_kernel_ids = make_int32_temp(rows * kernel_count);
            encoder.add_temporaries(
                {slot_in_rows, slot_out_rows, slot_kernel_ids}
            );

            auto slots = device.get_kernel(
                "build_strided_forward_relation_slots_i32", library
            );
            encoder.set_compute_pipeline_state(slots);
            encoder.set_input_array(inputs[0], 0);
            encoder.set_input_array(inputs[1], 1);
            encoder.set_input_array(outputs[RelationOutCoords], 2);
            encoder.set_output_array(outputs[RelationCounts], 3);
            encoder.set_input_array(outputs[RelationBaseOutputCount], 4);
            encoder.set_output_array(slot_in_rows, 5);
            encoder.set_output_array(slot_out_rows, 6);
            encoder.set_output_array(slot_kernel_ids, 7);
            encoder.set_output_array(outputs[RelationRowOffsets], 8);
            encoder.set_bytes(rows, 9);
            encoder.set_bytes(kernel_count, 10);
            encoder.set_bytes(table_capacity, 11);
            bind_triple_bytes(encoder, stride, 12);
            bind_triple_bytes(encoder, padding, 15);
            dispatch_1d(
                encoder,
                slots,
                std::max(
                    static_cast<size_t>(rows + 1),
                    static_cast<size_t>(rows) *
                        static_cast<size_t>(kernel_count)
                )
            );
            compact_forward_relation_slots(
                device,
                library,
                encoder,
                ForwardRelationSlotArrays{
                    .in_rows = slot_in_rows,
                    .out_rows = slot_out_rows,
                    .kernel_ids = slot_kernel_ids,
                },
                outputs,
                RelationSlotShape{.rows = rows, .kernel_count = kernel_count}
            );
            return;
        }

        auto row_degrees = make_int32_temp(rows);
        auto buffers = make_stable_compact_buffers(rows);
        encoder.add_temporaries(
            {row_degrees, buffers.local_offsets, buffers.block_offsets}
        );

        auto count = device.get_kernel(
            "count_identity_forward_relation_degrees_i32", library
        );
        encoder.set_compute_pipeline_state(count);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_input_array(inputs[1], 1);
        encoder.set_input_array(inputs[2], 2);
        encoder.set_input_array(outputs[RelationBaseOutputCount], 3);
        encoder.set_output_array(row_degrees, 4);
        encoder.set_output_array(outputs[RelationOutCoords], 5);
        encoder.set_output_array(outputs[RelationCounts], 6);
        encoder.set_bytes(rows, 7);
        encoder.set_bytes(kernel_count, 8);
        encoder.set_bytes(table_capacity, 9);
        dispatch_1d(encoder, count, static_cast<size_t>(rows));

        auto scan =
            device.get_kernel("scan_relation_row_degrees_blocks_i32", library);
        encoder.set_compute_pipeline_state(scan);
        encoder.set_input_array(row_degrees, 0);
        encoder.set_output_array(buffers.local_offsets, 1);
        encoder.set_output_array(buffers.block_offsets, 2);
        encoder.set_bytes(rows, 3);
        encoder.dispatch_threadgroups(
            MTL::Size(static_cast<size_t>(buffers.blocks), 1, 1),
            MTL::Size(kStableCompactBlockSize, 1, 1)
        );

        auto prefix =
            device.get_kernel("prefix_coord_set_selected_blocks_i32", library);
        encoder.set_compute_pipeline_state(prefix);
        encoder.set_output_array(buffers.block_offsets, 0);
        encoder.set_output_array(outputs[RelationCounts], 1);
        encoder.set_bytes(buffers.blocks, 2);
        dispatch_single(encoder);

        auto finalize = device.get_kernel(
            "finalize_forward_relation_row_offsets_i32", library
        );
        encoder.set_compute_pipeline_state(finalize);
        encoder.set_input_array(buffers.local_offsets, 0);
        encoder.set_input_array(buffers.block_offsets, 1);
        encoder.set_output_array(outputs[RelationRowOffsets], 2);
        encoder.set_input_array(outputs[RelationCounts], 3);
        encoder.set_bytes(rows, 4);
        dispatch_1d(encoder, finalize, static_cast<size_t>(rows) + size_t{1});

        auto fill = device.get_kernel(
            "fill_identity_forward_relation_compact_i32", library
        );
        encoder.set_compute_pipeline_state(fill);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_input_array(inputs[1], 1);
        encoder.set_input_array(outputs[RelationBaseOutputCount], 2);
        encoder.set_input_array(outputs[RelationRowOffsets], 3);
        encoder.set_input_array(outputs[RelationCounts], 4);
        encoder.set_output_array(outputs[RelationInRows], 5);
        encoder.set_output_array(outputs[RelationOutRows], 6);
        encoder.set_output_array(outputs[RelationKernelIds], 7);
        encoder.set_bytes(rows, 8);
        encoder.set_bytes(kernel_count, 9);
        encoder.set_bytes(table_capacity, 10);
        dispatch_1d(encoder, fill, static_cast<size_t>(rows));
        return;
    }

    if (op == CoordRelationOp::Transposed && direct) {
        auto kernel =
            device.get_kernel("build_transposed_direct_relation_i32", library);
        encoder.set_compute_pipeline_state(kernel);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_input_array(inputs[1], 1);
        encoder.set_input_array(inputs[2], 2);
        bind_output_arrays(encoder, outputs, 3, RelationBaseOutputCount);
        encoder.set_bytes(rows, 9);
        encoder.set_bytes(kernel_count, 10);
        bind_triple_bytes(encoder, stride, 11);
        bind_triple_bytes(encoder, padding, 14);
        dispatch_1d(encoder, kernel, static_cast<size_t>(rows) * kernel_count);
        return;
    }

    if (op == CoordRelationOp::Forward) {
        throw std::runtime_error(
            "Metal forward relations require hash scratch storage."
        );
    }
    auto kernel =
        device.get_kernel("build_transposed_kernel_relation_i32", library);

    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[1], 1);
    encoder.set_input_array(inputs[2], 2);
    bind_output_arrays(encoder, outputs, 3, RelationBaseOutputCount);
    encoder.set_bytes(rows, 9);
    encoder.set_bytes(kernel_count, 10);
    bind_triple_bytes(encoder, stride, 11);
    bind_triple_bytes(encoder, padding, 14);
    dispatch_single(encoder);
#else
    (void)op;
    (void)rows;
    (void)kernel_count;
    (void)stride;
    (void)padding;
    (void)direct;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

void eval_target_kernel_relation(
    int rows, // NOLINT(bugprone-easily-swappable-parameters)
    int target_rows,
    int kernel_count,
    Triple stride, // NOLINT(bugprone-easily-swappable-parameters)
    Triple padding,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    require_i32_inputs(
        inputs,
        {"coords",
         "kernel offsets",
         "active rows",
         "target coords",
         "target active rows"}
    );

#ifdef _METAL_
    backend::allocate_all(outputs);

    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);
    auto table_capacity =
        static_cast<int>(outputs[RelationBaseOutputCount].shape(0));
    clear_coord_hash(
        device,
        library,
        encoder,
        outputs[RelationBaseOutputCount],
        table_capacity
    );

    auto insert =
        device.get_kernel("coord_hash_insert_active_rows_i32", library);
    encoder.set_compute_pipeline_state(insert);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[2], 1);
    encoder.set_output_array(outputs[RelationBaseOutputCount], 2);
    encoder.set_bytes(rows, 3);
    encoder.set_bytes(table_capacity, 4);
    dispatch_1d(encoder, insert, static_cast<size_t>(rows));

    auto slot_in_rows = make_int32_temp(target_rows * kernel_count);
    auto slot_out_rows = make_int32_temp(target_rows * kernel_count);
    auto slot_kernel_ids = make_int32_temp(target_rows * kernel_count);
    encoder.add_temporaries({slot_in_rows, slot_out_rows, slot_kernel_ids});

    auto slots =
        device.get_kernel("build_target_forward_relation_slots_i32", library);
    encoder.set_compute_pipeline_state(slots);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[1], 1);
    encoder.set_input_array(inputs[3], 2);
    encoder.set_input_array(inputs[4], 3);
    encoder.set_input_array(outputs[RelationBaseOutputCount], 4);
    encoder.set_output_array(slot_in_rows, 5);
    encoder.set_output_array(slot_out_rows, 6);
    encoder.set_output_array(slot_kernel_ids, 7);
    encoder.set_output_array(outputs[RelationRowOffsets], 8);
    encoder.set_output_array(outputs[RelationCounts], 9);
    encoder.set_bytes(target_rows, 10);
    encoder.set_bytes(kernel_count, 11);
    encoder.set_bytes(table_capacity, 12);
    bind_triple_bytes(encoder, stride, 13);
    bind_triple_bytes(encoder, padding, 16);
    dispatch_1d(
        encoder,
        slots,
        std::max(
            {static_cast<size_t>(target_rows + 1),
             static_cast<size_t>(target_rows) *
                 static_cast<size_t>(kernel_count)}
        )
    );
    compact_forward_relation_slots(
        device,
        library,
        encoder,
        ForwardRelationSlotArrays{
            .in_rows = slot_in_rows,
            .out_rows = slot_out_rows,
            .kernel_ids = slot_kernel_ids,
        },
        outputs,
        RelationSlotShape{.rows = target_rows, .kernel_count = kernel_count}
    );
#else
    (void)rows;
    (void)target_rows;
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
    bind_output_arrays(encoder, outputs, 3, RelationBaseOutputCount);
    encoder.set_bytes(rows, 9);
    encoder.set_bytes(kernel_count, 10);
    bind_triple_bytes(encoder, stride, 11);
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

void eval_relation_grouped_view(
    RelationGroupedViewShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    require_i32_inputs(inputs, {"group ids", "counts"});

#ifdef _METAL_
    backend::allocate_all(outputs);
    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);
    encode_relation_grouped_view(
        device, library, encoder, inputs, outputs, shape
    );
#else
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

void eval_relation_direct_view(
    RelationGroupedViewShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    require_i32_inputs(inputs, {"group ids", "counts"});

#ifdef _METAL_
    backend::allocate_all(outputs);
    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);
    encode_relation_direct_view(
        device, library, encoder, inputs, outputs, shape
    );
#else
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

void eval_relation_implicit_gemm_view(
    RelationImplicitGemmViewShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    require_i32_inputs(
        inputs,
        {"source coords",
         "source active rows",
         "output coords",
         "output active rows",
         "kernel offsets"}
    );

#ifdef _METAL_
    backend::allocate_all(outputs);

    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);

    auto table_capacity = coord_hash_capacity(shape.source_rows);
    auto table = make_int32_temp(table_capacity);
    encoder.add_temporary(table);
    clear_coord_hash(device, library, encoder, table, table_capacity);
    insert_coord_hash(
        device,
        library,
        encoder,
        inputs[0],
        table,
        CoordHashShape{shape.source_rows, table_capacity}
    );

    auto total_slots = shape.output_rows * shape.kernel_count;
    auto clear =
        device.get_kernel("clear_relation_implicit_gemm_view_i32", library);
    encoder.set_compute_pipeline_state(clear);
    encoder.set_output_array(outputs[RelationImplicitGemmOutInMap], 0);
    encoder.set_output_array(outputs[RelationImplicitGemmRowMasks], 1);
    encoder.set_bytes(total_slots, 2);
    auto total_mask_words = shape.output_rows * shape.mask_words;
    encoder.set_bytes(total_mask_words, 3);
    dispatch_1d(
        encoder,
        clear,
        static_cast<size_t>(std::max(total_slots, total_mask_words))
    );

    auto build =
        device.get_kernel("build_relation_implicit_gemm_view_i32", library);
    encoder.set_compute_pipeline_state(build);
    bind_input_arrays(encoder, inputs, 0);
    encoder.set_input_array(table, 5);
    encoder.set_output_array(outputs[RelationImplicitGemmOutInMap], 6);
    encoder.set_output_array(outputs[RelationImplicitGemmRowMasks], 7);
    encoder.set_bytes(shape.source_rows, 8);
    encoder.set_bytes(shape.output_rows, 9);
    encoder.set_bytes(shape.kernel_count, 10);
    encoder.set_bytes(table_capacity, 11);
    encoder.set_bytes(shape.mask_words, 12);
    bind_triple_bytes(encoder, shape.stride, 13);
    bind_triple_bytes(encoder, shape.padding, 16);
    dispatch_1d(encoder, build, static_cast<size_t>(total_slots));
#else
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

} // namespace mlx_lattice::coords::metal
