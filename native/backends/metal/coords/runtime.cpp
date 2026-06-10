#include "backends/metal/coords/runtime.h"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <string>

#include "backends/array_utils.h"
#include "backends/metal/runtime_utils.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace mlx_lattice::coords::metal {

namespace {

int neighbor_relation_op_id(NeighborRelationOp op) {
    switch (op) {
    case NeighborRelationOp::Knn:
        return 0;
    case NeighborRelationOp::Radius:
        return 1;
    }
}

int voxel_reduce_op_id(VoxelReduceOp op) {
    switch (op) {
    case VoxelReduceOp::Sum:
        return 0;
    case VoxelReduceOp::Mean:
        return 1;
    }
}

bool is_identity_forward_relation(Triple stride, Triple padding) {
    return stride == Triple{1, 1, 1} && padding == Triple{0, 0, 0};
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

int coord_hash_capacity(int rows) {
    return next_power_of_two(std::max(rows * 2, 1));
}

struct CoordHashShape {
    int rows;
    int capacity;
};

struct ForwardRelationSlotArrays {
    const mx::array& in_rows;
    const mx::array& out_rows;
    const mx::array& kernel_ids;
};

struct RelationSlotShape {
    int rows;
    int kernel_count;
};

struct NeighborRowOutputs {
    mx::array& offsets;
    mx::array& counts;
};

struct StableCompactBuffers {
    mx::array local_offsets;
    mx::array block_offsets;
    int blocks;
};

constexpr int kStableCompactBlockSize = 256;
constexpr int kParallelCompactThreshold = 4096;

template <typename Encoder, typename Kernel>
void dispatch_1d(Encoder& encoder, Kernel* kernel, size_t elements) {
    auto threads = std::max<size_t>(elements, 1);
    auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}

StableCompactBuffers make_stable_compact_buffers(int rows) {
    auto blocks = std::max(
        (rows + kStableCompactBlockSize - 1) / kStableCompactBlockSize, 1
    );
    return {
        make_int32_temp(rows),
        make_int32_temp(blocks),
        blocks,
    };
}

template <typename Device, typename Library, typename Encoder>
void encode_stable_compact_offsets(
    Device& device,
    Library& library,
    Encoder& encoder,
    const mx::array& selected,
    mx::array& count,
    StableCompactBuffers& buffers,
    int rows
) {
    encoder.add_temporaries({buffers.local_offsets, buffers.block_offsets});
    auto scan =
        device.get_kernel("scan_coord_set_selected_blocks_i32", library);
    encoder.set_compute_pipeline_state(scan);
    encoder.set_input_array(selected, 0);
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
    encoder.set_output_array(count, 1);
    encoder.set_bytes(buffers.blocks, 2);
    encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
}

template <typename Device, typename Library, typename Encoder>
void encode_relation_compact_offsets(
    Device& device,
    Library& library,
    Encoder& encoder,
    const mx::array& selected,
    mx::array& counts,
    StableCompactBuffers& buffers,
    int rows
) {
    encoder.add_temporaries({buffers.local_offsets, buffers.block_offsets});
    auto scan =
        device.get_kernel("scan_coord_set_selected_blocks_i32", library);
    encoder.set_compute_pipeline_state(scan);
    encoder.set_input_array(selected, 0);
    encoder.set_output_array(buffers.local_offsets, 1);
    encoder.set_output_array(buffers.block_offsets, 2);
    encoder.set_bytes(rows, 3);
    encoder.dispatch_threadgroups(
        MTL::Size(static_cast<size_t>(buffers.blocks), 1, 1),
        MTL::Size(kStableCompactBlockSize, 1, 1)
    );

    auto prefix =
        device.get_kernel("prefix_relation_selected_blocks_i32", library);
    encoder.set_compute_pipeline_state(prefix);
    encoder.set_output_array(buffers.block_offsets, 0);
    encoder.set_output_array(counts, 1);
    encoder.set_bytes(buffers.blocks, 2);
    encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
}

template <typename Device, typename Library, typename Encoder>
void encode_neighbor_row_offsets(
    Device& device,
    Library& library,
    Encoder& encoder,
    NeighborRowOutputs outputs,
    int query_rows
) {
    if (query_rows < kParallelCompactThreshold) {
        auto prefix =
            device.get_kernel("prefix_neighbor_row_offsets_i32", library);
        encoder.set_compute_pipeline_state(prefix);
        encoder.set_output_array(outputs.offsets, 0);
        encoder.set_output_array(outputs.counts, 1);
        encoder.set_bytes(query_rows, 2);
        encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
        return;
    }

    auto buffers = make_stable_compact_buffers(query_rows);
    encoder.add_temporaries({buffers.local_offsets, buffers.block_offsets});
    auto scan =
        device.get_kernel("scan_relation_row_degrees_blocks_i32", library);
    encoder.set_compute_pipeline_state(scan);
    encoder.set_input_array(outputs.offsets, 0);
    encoder.set_output_array(buffers.local_offsets, 1);
    encoder.set_output_array(buffers.block_offsets, 2);
    encoder.set_bytes(query_rows, 3);
    encoder.dispatch_threadgroups(
        MTL::Size(static_cast<size_t>(buffers.blocks), 1, 1),
        MTL::Size(kStableCompactBlockSize, 1, 1)
    );

    auto prefix =
        device.get_kernel("prefix_coord_set_selected_blocks_i32", library);
    encoder.set_compute_pipeline_state(prefix);
    encoder.set_output_array(buffers.block_offsets, 0);
    encoder.set_output_array(outputs.counts, 1);
    encoder.set_bytes(buffers.blocks, 2);
    encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));

    auto finalize =
        device.get_kernel("finalize_forward_relation_row_offsets_i32", library);
    encoder.set_compute_pipeline_state(finalize);
    encoder.set_input_array(buffers.local_offsets, 0);
    encoder.set_input_array(buffers.block_offsets, 1);
    encoder.set_output_array(outputs.offsets, 2);
    encoder.set_input_array(outputs.counts, 3);
    encoder.set_bytes(query_rows, 4);
    dispatch_1d(encoder, finalize, static_cast<size_t>(query_rows) + size_t{1});
}

template <typename Device, typename Library, typename Encoder>
void clear_coord_hash(
    Device& device,
    Library& library,
    Encoder& encoder,
    mx::array& table,
    int capacity
) {
    auto clear = device.get_kernel("coord_hash_clear_i32", library);
    encoder.set_compute_pipeline_state(clear);
    encoder.set_output_array(table, 0);
    encoder.set_bytes(capacity, 1);
    dispatch_1d(encoder, clear, static_cast<size_t>(capacity));
}

template <typename Device, typename Library, typename Encoder>
void insert_coord_hash(
    Device& device,
    Library& library,
    Encoder& encoder,
    const mx::array& coords,
    mx::array& table,
    CoordHashShape shape
) {
    auto insert = device.get_kernel("coord_hash_insert_rows_i32", library);
    encoder.set_compute_pipeline_state(insert);
    encoder.set_input_array(coords, 0);
    encoder.set_output_array(table, 1);
    encoder.set_bytes(shape.rows, 2);
    encoder.set_bytes(shape.capacity, 3);
    dispatch_1d(encoder, insert, static_cast<size_t>(shape.rows));
}

template <typename Device, typename Library, typename Encoder>
void encode_relation_grouped_view(
    Device& device,
    Library& library,
    Encoder& encoder,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs,
    RelationGroupedViewShape shape
) {
    auto cursors = make_int32_temp(shape.group_count + 1);
    encoder.add_temporaries({cursors});

    auto clear = device.get_kernel("clear_relation_grouped_view_i32", library);
    encoder.set_compute_pipeline_state(clear);
    encoder.set_output_array(outputs[RelationViewRowOffsets], 0);
    encoder.set_bytes(shape.group_count, 1);
    dispatch_1d(
        encoder, clear, static_cast<size_t>(shape.group_count) + size_t{1}
    );

    auto count = device.get_kernel("count_relation_grouped_view_i32", library);
    encoder.set_compute_pipeline_state(count);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[1], 1);
    encoder.set_output_array(outputs[RelationViewRowOffsets], 2);
    encoder.set_bytes(shape.edge_capacity, 3);
    encoder.set_bytes(shape.group_count, 4);
    dispatch_1d(encoder, count, static_cast<size_t>(shape.edge_capacity));

    if (shape.group_count >= kParallelCompactThreshold) {
        auto buffers = make_stable_compact_buffers(shape.group_count);
        auto total_count = make_int32_temp(1);
        encoder.add_temporaries(
            {buffers.local_offsets, buffers.block_offsets, total_count}
        );

        auto scan =
            device.get_kernel("scan_relation_grouped_view_blocks_i32", library);
        encoder.set_compute_pipeline_state(scan);
        encoder.set_input_array(outputs[RelationViewRowOffsets], 0);
        encoder.set_output_array(buffers.local_offsets, 1);
        encoder.set_output_array(buffers.block_offsets, 2);
        encoder.set_bytes(shape.group_count, 3);
        encoder.dispatch_threadgroups(
            MTL::Size(static_cast<size_t>(buffers.blocks), 1, 1),
            MTL::Size(kStableCompactBlockSize, 1, 1)
        );

        auto prefix =
            device.get_kernel("prefix_coord_set_selected_blocks_i32", library);
        encoder.set_compute_pipeline_state(prefix);
        encoder.set_output_array(buffers.block_offsets, 0);
        encoder.set_output_array(total_count, 1);
        encoder.set_bytes(buffers.blocks, 2);
        encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));

        auto finalize =
            device.get_kernel("finalize_relation_grouped_view_i32", library);
        encoder.set_compute_pipeline_state(finalize);
        encoder.set_input_array(buffers.local_offsets, 0);
        encoder.set_input_array(buffers.block_offsets, 1);
        encoder.set_input_array(total_count, 2);
        encoder.set_output_array(outputs[RelationViewRowOffsets], 3);
        encoder.set_output_array(cursors, 4);
        encoder.set_bytes(shape.group_count, 5);
        dispatch_1d(
            encoder,
            finalize,
            static_cast<size_t>(shape.group_count) + size_t{1}
        );
    } else {
        auto prefix =
            device.get_kernel("prefix_relation_grouped_view_i32", library);
        encoder.set_compute_pipeline_state(prefix);
        encoder.set_output_array(outputs[RelationViewRowOffsets], 0);
        encoder.set_output_array(cursors, 1);
        encoder.set_bytes(shape.group_count, 2);
        encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
    }

    auto fill = device.get_kernel("fill_relation_grouped_view_i32", library);
    encoder.set_compute_pipeline_state(fill);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[1], 1);
    encoder.set_input_array(cursors, 2);
    encoder.set_output_array(outputs[RelationViewEdgeIds], 3);
    encoder.set_bytes(shape.edge_capacity, 4);
    encoder.set_bytes(shape.group_count, 5);
    dispatch_1d(encoder, fill, static_cast<size_t>(shape.edge_capacity));
}

template <typename Device, typename Library, typename Encoder>
void encode_relation_direct_view(
    Device& device,
    Library& library,
    Encoder& encoder,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs,
    RelationGroupedViewShape shape
) {
    auto clear = device.get_kernel("clear_relation_direct_view_i32", library);
    encoder.set_compute_pipeline_state(clear);
    encoder.set_output_array(outputs[0], 0);
    encoder.set_bytes(shape.group_count, 1);
    dispatch_1d(encoder, clear, static_cast<size_t>(shape.group_count));

    auto fill = device.get_kernel("fill_relation_direct_view_i32", library);
    encoder.set_compute_pipeline_state(fill);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[1], 1);
    encoder.set_output_array(outputs[0], 2);
    encoder.set_bytes(shape.edge_capacity, 3);
    encoder.set_bytes(shape.group_count, 4);
    dispatch_1d(encoder, fill, static_cast<size_t>(shape.edge_capacity));
}

template <typename Device, typename Library, typename Encoder>
void compact_forward_relation_slots(
    Device& device,
    Library& library,
    Encoder& encoder,
    ForwardRelationSlotArrays slots,
    std::vector<mx::array>& outputs,
    RelationSlotShape shape
) {
    if (shape.rows >= kParallelCompactThreshold) {
        auto row_degrees = make_int32_temp(shape.rows);
        auto buffers = make_stable_compact_buffers(shape.rows);
        encoder.add_temporaries(
            {row_degrees, buffers.local_offsets, buffers.block_offsets}
        );

        auto count = device.get_kernel(
            "count_forward_relation_slot_degrees_i32", library
        );
        encoder.set_compute_pipeline_state(count);
        encoder.set_input_array(slots.in_rows, 0);
        encoder.set_input_array(slots.kernel_ids, 1);
        encoder.set_input_array(outputs[RelationCounts], 2);
        encoder.set_output_array(row_degrees, 3);
        encoder.set_bytes(shape.rows, 4);
        encoder.set_bytes(shape.kernel_count, 5);
        dispatch_1d(encoder, count, static_cast<size_t>(shape.rows));

        auto scan =
            device.get_kernel("scan_relation_row_degrees_blocks_i32", library);
        encoder.set_compute_pipeline_state(scan);
        encoder.set_input_array(row_degrees, 0);
        encoder.set_output_array(buffers.local_offsets, 1);
        encoder.set_output_array(buffers.block_offsets, 2);
        encoder.set_bytes(shape.rows, 3);
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
        encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));

        auto finalize = device.get_kernel(
            "finalize_forward_relation_row_offsets_i32", library
        );
        encoder.set_compute_pipeline_state(finalize);
        encoder.set_input_array(buffers.local_offsets, 0);
        encoder.set_input_array(buffers.block_offsets, 1);
        encoder.set_output_array(outputs[RelationRowOffsets], 2);
        encoder.set_input_array(outputs[RelationCounts], 3);
        encoder.set_bytes(shape.rows, 4);
        dispatch_1d(
            encoder, finalize, static_cast<size_t>(shape.rows) + size_t{1}
        );
    } else {
        auto count =
            device.get_kernel("count_forward_relation_slot_rows_i32", library);
        encoder.set_compute_pipeline_state(count);
        encoder.set_input_array(slots.in_rows, 0);
        encoder.set_input_array(slots.kernel_ids, 1);
        encoder.set_output_array(outputs[RelationRowOffsets], 2);
        encoder.set_input_array(outputs[RelationCounts], 3);
        encoder.set_bytes(shape.rows, 4);
        encoder.set_bytes(shape.kernel_count, 5);
        dispatch_1d(encoder, count, static_cast<size_t>(shape.rows));

        auto prefix =
            device.get_kernel("prefix_forward_relation_slot_rows_i32", library);
        encoder.set_compute_pipeline_state(prefix);
        encoder.set_output_array(outputs[RelationRowOffsets], 0);
        encoder.set_output_array(outputs[RelationCounts], 1);
        encoder.set_bytes(shape.rows, 2);
        encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
    }

    auto compact =
        device.get_kernel("compact_forward_relation_slots_i32", library);
    encoder.set_compute_pipeline_state(compact);
    encoder.set_input_array(slots.in_rows, 0);
    encoder.set_input_array(slots.out_rows, 1);
    encoder.set_input_array(slots.kernel_ids, 2);
    encoder.set_input_array(outputs[RelationRowOffsets], 3);
    encoder.set_input_array(outputs[RelationCounts], 4);
    encoder.set_output_array(outputs[RelationInRows], 5);
    encoder.set_output_array(outputs[RelationOutRows], 6);
    encoder.set_output_array(outputs[RelationKernelIds], 7);
    encoder.set_bytes(shape.rows, 8);
    encoder.set_bytes(shape.kernel_count, 9);
    dispatch_1d(encoder, compact, static_cast<size_t>(shape.rows));
}
#endif

// MARK: - guards

void require_f32_input(const mx::array& input, const char* name) {
    if (input.dtype() != mx::float32) {
        throw std::invalid_argument(
            std::string("Metal coordinate kernels require float32 ") + name +
            "."
        );
    }
}

void require_i32_input(const mx::array& input, const char* name) {
    if (input.dtype() != mx::int32) {
        throw std::invalid_argument(
            std::string("Metal coordinate kernels require int32 ") + name + "."
        );
    }
}

void require_i32_inputs(
    const std::vector<mx::array>& inputs,
    std::initializer_list<const char*> names
) {
    int index = 0;
    for (auto name : names) {
        require_i32_input(inputs[index++], name);
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
        encoder.set_bytes(stride[0], 4);
        encoder.set_bytes(stride[1], 5);
        encoder.set_bytes(stride[2], 6);
        dispatch_1d(encoder, build, static_cast<size_t>(shape.lhs_rows));

        auto plan = device.get_kernel("plan_downsample_coord_set_i32", library);
        encoder.set_compute_pipeline_state(plan);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_input_array(table, 1);
        encoder.set_output_array(selected, 2);
        encoder.set_bytes(shape.lhs_rows, 3);
        encoder.set_bytes(table_capacity, 4);
        encoder.set_bytes(stride[0], 5);
        encoder.set_bytes(stride[1], 6);
        encoder.set_bytes(stride[2], 7);
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
            encoder.set_bytes(stride[0], 6);
            encoder.set_bytes(stride[1], 7);
            encoder.set_bytes(stride[2], 8);
            dispatch_1d(encoder, compact, static_cast<size_t>(shape.lhs_rows));
            return;
        }
        encoder.set_compute_pipeline_state(compact);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_input_array(selected, 1);
        encoder.set_output_array(out_coords, 2);
        encoder.set_output_array(count, 3);
        encoder.set_bytes(shape.lhs_rows, 4);
        encoder.set_bytes(stride[0], 5);
        encoder.set_bytes(stride[1], 6);
        encoder.set_bytes(stride[2], 7);
        encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
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
        encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
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
        encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
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

// MARK: - quantization

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
    auto selected = make_int32_temp(rows);
    auto representative_voxels = make_int32_temp(rows);
    encoder.add_temporaries({table, selected, representative_voxels});
    clear_coord_hash(device, library, encoder, table, table_capacity);

    auto clear = device.get_kernel("clear_sparse_quantization_i32", library);
    encoder.set_compute_pipeline_state(clear);
    encoder.set_output_array(outputs[0], 0);
    encoder.set_output_array(outputs[1], 1);
    encoder.set_output_array(outputs[2], 2);
    encoder.set_output_array(outputs[3], 3);
    encoder.set_bytes(rows, 4);
    dispatch_1d(encoder, clear, static_cast<size_t>(rows) * 4);

    auto build = device.get_kernel("build_quantized_point_hash_i32", library);
    encoder.set_compute_pipeline_state(build);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[1], 1);
    encoder.set_input_array(inputs[2], 2);
    encoder.set_output_array(table, 3);
    encoder.set_bytes(rows, 4);
    encoder.set_bytes(table_capacity, 5);
    encoder.set_bytes(spec.voxel_size[0], 6);
    encoder.set_bytes(spec.voxel_size[1], 7);
    encoder.set_bytes(spec.voxel_size[2], 8);
    encoder.set_bytes(spec.origin[0], 9);
    encoder.set_bytes(spec.origin[1], 10);
    encoder.set_bytes(spec.origin[2], 11);
    dispatch_1d(encoder, build, static_cast<size_t>(rows));

    auto plan = device.get_kernel("plan_quantized_points_i32", library);
    encoder.set_compute_pipeline_state(plan);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[1], 1);
    encoder.set_input_array(inputs[2], 2);
    encoder.set_input_array(table, 3);
    encoder.set_output_array(selected, 4);
    encoder.set_bytes(rows, 5);
    encoder.set_bytes(table_capacity, 6);
    encoder.set_bytes(spec.voxel_size[0], 7);
    encoder.set_bytes(spec.voxel_size[1], 8);
    encoder.set_bytes(spec.voxel_size[2], 9);
    encoder.set_bytes(spec.origin[0], 10);
    encoder.set_bytes(spec.origin[1], 11);
    encoder.set_bytes(spec.origin[2], 12);
    dispatch_1d(encoder, plan, static_cast<size_t>(rows));

    auto prefix = device.get_kernel("prefix_quantized_points_i32", library);
    encoder.set_compute_pipeline_state(prefix);
    encoder.set_input_array(inputs[2], 0);
    encoder.set_input_array(selected, 1);
    encoder.set_output_array(outputs[1], 2);
    encoder.set_output_array(representative_voxels, 3);
    encoder.set_bytes(rows, 4);
    encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));

    auto fill = device.get_kernel("fill_quantized_points_i32", library);
    encoder.set_compute_pipeline_state(fill);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[1], 1);
    encoder.set_input_array(inputs[2], 2);
    encoder.set_input_array(selected, 3);
    encoder.set_input_array(representative_voxels, 4);
    encoder.set_output_array(outputs[0], 5);
    encoder.set_bytes(rows, 6);
    encoder.set_bytes(spec.voxel_size[0], 7);
    encoder.set_bytes(spec.voxel_size[1], 8);
    encoder.set_bytes(spec.voxel_size[2], 9);
    encoder.set_bytes(spec.origin[0], 10);
    encoder.set_bytes(spec.origin[1], 11);
    encoder.set_bytes(spec.origin[2], 12);
    dispatch_1d(encoder, fill, static_cast<size_t>(rows));

    auto map = device.get_kernel("map_quantized_points_i32", library);
    encoder.set_compute_pipeline_state(map);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[1], 1);
    encoder.set_input_array(inputs[2], 2);
    encoder.set_input_array(table, 3);
    encoder.set_input_array(representative_voxels, 4);
    encoder.set_output_array(outputs[2], 5);
    encoder.set_output_array(outputs[3], 6);
    encoder.set_bytes(rows, 7);
    encoder.set_bytes(table_capacity, 8);
    encoder.set_bytes(spec.voxel_size[0], 9);
    encoder.set_bytes(spec.voxel_size[1], 10);
    encoder.set_bytes(spec.voxel_size[2], 11);
    encoder.set_bytes(spec.origin[0], 12);
    encoder.set_bytes(spec.origin[1], 13);
    encoder.set_bytes(spec.origin[2], 14);
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
    for (int i = 0; i < int(inputs.size()); ++i) {
        encoder.set_input_array(inputs[i], i);
    }
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
    for (int i = 0; i < int(inputs.size()); ++i) {
        encoder.set_input_array(inputs[i], i);
    }
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
            encoder.set_bytes(stride[0], 5);
            encoder.set_bytes(stride[1], 6);
            encoder.set_bytes(stride[2], 7);
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
            encoder.set_bytes(stride[0], 6);
            encoder.set_bytes(stride[1], 7);
            encoder.set_bytes(stride[2], 8);
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
                encoder.set_bytes(stride[0], 6);
                encoder.set_bytes(stride[1], 7);
                encoder.set_bytes(stride[2], 8);
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
                encoder.set_bytes(stride[0], 5);
                encoder.set_bytes(stride[1], 6);
                encoder.set_bytes(stride[2], 7);
                encoder.dispatch_threads(
                    MTL::Size(1, 1, 1), MTL::Size(1, 1, 1)
                );
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
            encoder.set_bytes(stride[0], 12);
            encoder.set_bytes(stride[1], 13);
            encoder.set_bytes(stride[2], 14);
            encoder.set_bytes(padding[0], 15);
            encoder.set_bytes(padding[1], 16);
            encoder.set_bytes(padding[2], 17);
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

        auto slot_in_rows = make_int32_temp(rows * kernel_count);
        auto slot_out_rows = make_int32_temp(rows * kernel_count);
        auto slot_kernel_ids = make_int32_temp(rows * kernel_count);
        encoder.add_temporaries({slot_in_rows, slot_out_rows, slot_kernel_ids});

        auto slots = device.get_kernel(
            "build_identity_forward_relation_slots_i32", library
        );
        encoder.set_compute_pipeline_state(slots);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_input_array(inputs[1], 1);
        encoder.set_input_array(inputs[2], 2);
        encoder.set_input_array(outputs[RelationBaseOutputCount], 3);
        encoder.set_output_array(slot_in_rows, 4);
        encoder.set_output_array(slot_out_rows, 5);
        encoder.set_output_array(slot_kernel_ids, 6);
        encoder.set_output_array(outputs[RelationRowOffsets], 7);
        encoder.set_output_array(outputs[RelationOutCoords], 8);
        encoder.set_output_array(outputs[RelationCounts], 9);
        encoder.set_bytes(rows, 10);
        encoder.set_bytes(kernel_count, 11);
        encoder.set_bytes(table_capacity, 12);
        dispatch_1d(
            encoder,
            slots,
            std::max(
                {static_cast<size_t>(rows + 1),
                 static_cast<size_t>(rows) * 4,
                 static_cast<size_t>(rows) * static_cast<size_t>(kernel_count)}
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

    if (op == CoordRelationOp::Transposed && direct) {
        auto kernel =
            device.get_kernel("build_transposed_direct_relation_i32", library);
        encoder.set_compute_pipeline_state(kernel);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_input_array(inputs[1], 1);
        encoder.set_input_array(inputs[2], 2);
        for (int i = 0; i < int(RelationBaseOutputCount); ++i) {
            encoder.set_output_array(outputs[i], i + 3);
        }
        encoder.set_bytes(rows, 9);
        encoder.set_bytes(kernel_count, 10);
        encoder.set_bytes(stride[0], 11);
        encoder.set_bytes(stride[1], 12);
        encoder.set_bytes(stride[2], 13);
        encoder.set_bytes(padding[0], 14);
        encoder.set_bytes(padding[1], 15);
        encoder.set_bytes(padding[2], 16);
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
    for (int i = 0; i < int(RelationBaseOutputCount); ++i) {
        encoder.set_output_array(outputs[i], i + 3);
    }
    encoder.set_bytes(rows, 9);
    encoder.set_bytes(kernel_count, 10);
    encoder.set_bytes(stride[0], 11);
    encoder.set_bytes(stride[1], 12);
    encoder.set_bytes(stride[2], 13);
    encoder.set_bytes(padding[0], 14);
    encoder.set_bytes(padding[1], 15);
    encoder.set_bytes(padding[2], 16);
    encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
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
    encoder.set_bytes(stride[0], 13);
    encoder.set_bytes(stride[1], 14);
    encoder.set_bytes(stride[2], 15);
    encoder.set_bytes(padding[0], 16);
    encoder.set_bytes(padding[1], 17);
    encoder.set_bytes(padding[2], 18);
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
    for (int i = 0; i < int(RelationBaseOutputCount); ++i) {
        encoder.set_output_array(outputs[i], i + 3);
    }
    encoder.set_bytes(rows, 9);
    encoder.set_bytes(kernel_count, 10);
    encoder.set_bytes(stride[0], 11);
    encoder.set_bytes(stride[1], 12);
    encoder.set_bytes(stride[2], 13);
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
        for (int i = 0; i < int(inputs.size()); ++i) {
            encoder.set_input_array(inputs[i], i);
        }
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
        for (int i = 0; i < int(inputs.size()); ++i) {
            encoder.set_input_array(inputs[i], i);
        }
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
        for (int i = 0; i < int(inputs.size()); ++i) {
            encoder.set_input_array(inputs[i], i);
        }
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
        for (int i = 0; i < int(inputs.size()); ++i) {
            encoder.set_input_array(inputs[i], i);
        }
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
        for (int i = 0; i < int(inputs.size()); ++i) {
            encoder.set_input_array(inputs[i], i);
        }
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
        for (int i = 0; i < int(inputs.size()); ++i) {
            encoder.set_input_array(inputs[i], i);
        }
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
