#include "features/coordinates/metal/runtime.h"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <string>

#include "foundation/array_utils.h"
#include "platform/metal/runtime_utils.h"

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

int point_voxel_interpolation_op_id(PointVoxelInterpolationOp op) {
    switch (op) {
    case PointVoxelInterpolationOp::Nearest:
        return 0;
    case PointVoxelInterpolationOp::Linear:
        return 1;
    }
}

int sparse_join_op_id(SparseJoinOp op) {
    switch (op) {
    case SparseJoinOp::Inner:
        return 0;
    case SparseJoinOp::Left:
        return 1;
    case SparseJoinOp::Right:
        return 2;
    case SparseJoinOp::Outer:
        return 3;
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

template <typename Encoder>
void bind_input_arrays(
    Encoder& encoder,
    const std::vector<mx::array>& inputs,
    int first = 0
) {
    for (int i = 0; i < int(inputs.size()); ++i) {
        encoder.set_input_array(inputs[i], first + i);
    }
}

template <typename Encoder>
void bind_output_arrays(
    Encoder& encoder,
    std::vector<mx::array>& outputs,
    int first,
    int count
) {
    for (int i = 0; i < count; ++i) {
        encoder.set_output_array(outputs[i], first + i);
    }
}

template <typename Encoder>
void bind_triple_bytes(Encoder& encoder, Triple value, int first) {
    encoder.set_bytes(value[0], first);
    encoder.set_bytes(value[1], first + 1);
    encoder.set_bytes(value[2], first + 2);
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
    dispatch_single(encoder);
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
    dispatch_single(encoder);
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
        dispatch_single(encoder);
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
    dispatch_single(encoder);

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
        dispatch_single(encoder);

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
        dispatch_single(encoder);
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
        dispatch_single(encoder);

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
        dispatch_single(encoder);
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

} // namespace mlx_lattice::coords::metal
