#include "backends/cuda/coords/runtime.h"

#include <stdexcept>
#include <utility>

#include "backends/array_utils.h"
#include "backends/cuda/coords/kernels.cuh"
#include "backends/cuda/runtime_utils.h"
#include "mlx/backend/cuda/utils.h"

namespace mlx_lattice::backend::cuda::coords {
namespace {

void require_i32(const mx::array& input, const char* name) {
    if (input.dtype() != mx::int32) {
        throw std::invalid_argument(
            std::string("CUDA coordinate kernels require int32 ") + name + "."
        );
    }
}

void require_f32(const mx::array& input, const char* name) {
    if (input.dtype() != mx::float32) {
        throw std::invalid_argument(
            std::string("CUDA coordinate kernels require float32 ") + name + "."
        );
    }
}

TripleArgs triple_args(Triple value) {
    return TripleArgs{.x = value[0], .y = value[1], .z = value[2]};
}

template <typename Kernel, typename... Args>
void add_1d(
    const mx::Stream& stream,
    Kernel kernel,
    std::size_t elements,
    Args&&... args
) {
    auto launch = backend::cuda::launch_1d(elements);
    mx::cu::get_command_encoder(stream).add_kernel_node(
        kernel, launch.grid, launch.block, std::forward<Args>(args)...
    );
}

template <typename T> T* device_ptr(mx::array& array) {
    return mx::gpu_ptr<T>(array);
}

template <typename T> const T* device_ptr(const mx::array& array) {
    return mx::gpu_ptr<T>(array);
}

void add_clear_i32(
    const mx::Stream& stream,
    mx::array& out,
    int elements,
    int value = 0
) {
    add_1d(
        stream,
        clear_i32_kernel,
        static_cast<std::size_t>(elements),
        device_ptr<int>(out),
        elements,
        value
    );
}

void bind_arrays(
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    auto& encoder = mx::cu::get_command_encoder(stream);
    for (const auto& input : inputs) {
        encoder.set_input_array(input);
    }
    for (const auto& output : outputs) {
        encoder.set_output_array(output);
    }
}

int relation_op_id(CoordRelationOp op) {
    return op == CoordRelationOp::Forward ? 0 : 1;
}

int neighbor_op_id(NeighborRelationOp op) {
    return op == NeighborRelationOp::Knn ? 0 : 1;
}

int voxel_reduce_id(VoxelReduceOp op) {
    return op == VoxelReduceOp::Sum ? 0 : 1;
}

} // namespace

void eval_set_coords(
    CoordSetOp op,
    Triple stride,
    CoordSetShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    require_i32(inputs[0], "coords");
    if (op != CoordSetOp::Downsample) {
        require_i32(inputs[1], "rhs coords");
    }
    backend::allocate_all(outputs);
    bind_arrays(stream, inputs, outputs);
    add_1d(
        stream,
        set_coords_i32,
        1,
        device_ptr<int>(inputs[0]),
        op == CoordSetOp::Downsample ? nullptr : device_ptr<int>(inputs[1]),
        device_ptr<int>(outputs[0]),
        device_ptr<int>(outputs[1]),
        static_cast<int>(op),
        triple_args(stride),
        shape.lhs_rows,
        shape.rhs_rows
    );
}

void eval_lookup_coords(
    CoordLookupShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    require_i32(inputs[0], "coords");
    require_i32(inputs[1], "queries");
    backend::allocate_all(outputs);
    bind_arrays(stream, inputs, outputs);
    add_1d(
        stream,
        lookup_coords_i32,
        static_cast<std::size_t>(shape.query_rows),
        device_ptr<int>(inputs[0]),
        device_ptr<int>(inputs[1]),
        device_ptr<int>(outputs[0]),
        shape.rows,
        shape.query_rows
    );
}

void eval_morton_codes(
    CoordRowsShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    require_i32(inputs[0], "coords");
    backend::allocate_all(outputs);
    bind_arrays(stream, inputs, outputs);
    add_1d(
        stream,
        morton_codes_i32,
        static_cast<std::size_t>(shape.rows),
        device_ptr<int>(inputs[0]),
        device_ptr<long long>(outputs[0]),
        shape.rows
    );
}

void eval_occupancy_downsample(
    CoordActiveRowsShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    require_i32(inputs[0], "coords");
    require_i32(inputs[1], "active rows");
    backend::allocate_all(outputs);
    bind_arrays(stream, inputs, outputs);
    add_1d(
        stream,
        occupancy_downsample_i32,
        1,
        device_ptr<int>(inputs[0]),
        device_ptr<int>(inputs[1]),
        device_ptr<int>(outputs[0]),
        device_ptr<int>(outputs[1]),
        device_ptr<int>(outputs[2]),
        shape.rows
    );
}

void eval_occupancy_expand(
    CoordActiveRowsShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    require_i32(inputs[0], "coords");
    require_i32(inputs[1], "active rows");
    require_i32(inputs[2], "occupancy");
    backend::allocate_all(outputs);
    bind_arrays(stream, inputs, outputs);
    add_1d(
        stream,
        occupancy_expand_i32,
        1,
        device_ptr<int>(inputs[0]),
        device_ptr<int>(inputs[1]),
        device_ptr<int>(inputs[2]),
        device_ptr<int>(outputs[0]),
        device_ptr<int>(outputs[1]),
        device_ptr<int>(outputs[2]),
        device_ptr<int>(outputs[3]),
        shape.rows
    );
}

void eval_child_coords_from_indices(
    CoordRowsShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    require_i32(inputs[0], "parent coords");
    require_i32(inputs[1], "child indices");
    backend::allocate_all(outputs);
    bind_arrays(stream, inputs, outputs);
    add_1d(
        stream,
        child_coords_from_indices_i32,
        static_cast<std::size_t>(shape.rows),
        device_ptr<int>(inputs[0]),
        device_ptr<int>(inputs[1]),
        device_ptr<int>(outputs[0]),
        shape.rows
    );
}

void eval_sparse_quantize(
    QuantizationSpec spec,
    int rows,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    require_f32(inputs[0], "points");
    require_i32(inputs[1], "batch indices");
    require_i32(inputs[2], "active rows");
    backend::allocate_all(outputs);
    bind_arrays(stream, inputs, outputs);
    auto args = QuantizeArgs{
        .voxel_x = spec.voxel_size[0],
        .voxel_y = spec.voxel_size[1],
        .voxel_z = spec.voxel_size[2],
        .origin_x = spec.origin[0],
        .origin_y = spec.origin[1],
        .origin_z = spec.origin[2],
    };
    add_1d(
        stream,
        sparse_quantize_i32,
        1,
        device_ptr<float>(inputs[0]),
        device_ptr<int>(inputs[1]),
        device_ptr<int>(inputs[2]),
        device_ptr<int>(outputs[0]),
        device_ptr<int>(outputs[1]),
        device_ptr<int>(outputs[2]),
        device_ptr<int>(outputs[3]),
        args,
        rows
    );
}

void eval_voxelize_features(
    VoxelReduceOp reduce,
    VoxelFeatureShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    require_f32(inputs[0], "features");
    require_i32(inputs[1], "inverse rows");
    require_i32(inputs[2], "voxel counts");
    require_i32(inputs[3], "active rows");
    backend::allocate_all(outputs);
    bind_arrays(stream, inputs, outputs);
    add_1d(
        stream,
        clear_f32,
        static_cast<std::size_t>(shape.voxel_rows) *
            static_cast<std::size_t>(shape.channels),
        device_ptr<float>(outputs[0]),
        shape.voxel_rows * shape.channels
    );
    add_1d(
        stream,
        voxelize_features_f32,
        static_cast<std::size_t>(shape.point_rows) *
            static_cast<std::size_t>(shape.channels),
        device_ptr<float>(inputs[0]),
        device_ptr<int>(inputs[1]),
        device_ptr<int>(inputs[2]),
        device_ptr<int>(inputs[3]),
        device_ptr<float>(outputs[0]),
        voxel_reduce_id(reduce),
        shape.point_rows,
        shape.voxel_rows,
        shape.channels
    );
}

void eval_voxelize_feature_grad(
    VoxelReduceOp reduce,
    VoxelFeatureShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    require_f32(inputs[0], "cotangent");
    require_i32(inputs[1], "inverse rows");
    require_i32(inputs[2], "voxel counts");
    require_i32(inputs[3], "active rows");
    backend::allocate_all(outputs);
    bind_arrays(stream, inputs, outputs);
    add_1d(
        stream,
        voxelize_feature_grad_f32,
        static_cast<std::size_t>(shape.point_rows) *
            static_cast<std::size_t>(shape.channels),
        device_ptr<float>(inputs[0]),
        device_ptr<int>(inputs[1]),
        device_ptr<int>(inputs[2]),
        device_ptr<int>(inputs[3]),
        device_ptr<float>(outputs[0]),
        voxel_reduce_id(reduce),
        shape.point_rows,
        shape.voxel_rows,
        shape.channels
    );
}

void eval_generic_kernel_relation(
    CoordRelationOp op,
    int rows,
    int kernel_count,
    Triple stride,
    Triple padding,
    bool direct,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    bind_arrays(stream, inputs, outputs);
    add_1d(
        stream,
        generic_kernel_relation_i32,
        1,
        device_ptr<int>(inputs[0]),
        device_ptr<int>(inputs[1]),
        device_ptr<int>(inputs[2]),
        device_ptr<int>(outputs[RelationInRows]),
        device_ptr<int>(outputs[RelationOutRows]),
        device_ptr<int>(outputs[RelationKernelIds]),
        device_ptr<int>(outputs[RelationRowOffsets]),
        device_ptr<int>(outputs[RelationOutCoords]),
        device_ptr<int>(outputs[RelationCounts]),
        relation_op_id(op),
        rows,
        kernel_count,
        triple_args(stride),
        triple_args(padding),
        direct
    );
}

void eval_target_kernel_relation(
    int rows,
    int target_rows,
    int kernel_count,
    Triple stride,
    Triple padding,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    bind_arrays(stream, inputs, outputs);
    auto active_targets =
        std::min(target_rows, int(outputs[RelationOutCoords].shape(0)));
    auto row_cursors = backend::cuda::make_int32_temp(active_targets + 1);
    auto row_offsets = outputs[RelationRowOffsets];
    mx::cu::get_command_encoder(stream).set_output_array(row_cursors);
    add_clear_i32(stream, row_offsets, active_targets + 1);
    add_1d(
        stream,
        count_target_kernel_relation_i32,
        static_cast<std::size_t>(target_rows) *
            static_cast<std::size_t>(kernel_count),
        device_ptr<int>(inputs[0]),
        device_ptr<int>(inputs[1]),
        device_ptr<int>(inputs[2]),
        device_ptr<int>(inputs[3]),
        device_ptr<int>(inputs[4]),
        device_ptr<int>(row_offsets),
        rows,
        target_rows,
        kernel_count,
        triple_args(stride),
        triple_args(padding)
    );
    add_1d(
        stream,
        prefix_relation_rows_active_i32,
        1,
        device_ptr<int>(row_offsets),
        device_ptr<int>(outputs[RelationCounts]),
        device_ptr<int>(inputs[4]),
        active_targets,
        static_cast<int>(outputs[RelationInRows].shape(0))
    );
    add_clear_i32(stream, row_cursors, active_targets + 1);
    add_1d(
        stream,
        fill_target_kernel_relation_i32,
        static_cast<std::size_t>(target_rows) *
            static_cast<std::size_t>(kernel_count),
        device_ptr<int>(inputs[0]),
        device_ptr<int>(inputs[1]),
        device_ptr<int>(inputs[2]),
        device_ptr<int>(inputs[3]),
        device_ptr<int>(inputs[4]),
        device_ptr<int>(row_offsets),
        device_ptr<int>(row_cursors),
        device_ptr<int>(outputs[RelationInRows]),
        device_ptr<int>(outputs[RelationOutRows]),
        device_ptr<int>(outputs[RelationKernelIds]),
        rows,
        target_rows,
        kernel_count,
        triple_args(stride),
        triple_args(padding)
    );
}

void eval_generative_kernel_relation(
    int rows,
    int kernel_count,
    Triple stride,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    bind_arrays(stream, inputs, outputs);
    add_1d(
        stream,
        generative_kernel_relation_i32,
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(kernel_count),
        device_ptr<int>(inputs[0]),
        device_ptr<int>(inputs[1]),
        device_ptr<int>(inputs[2]),
        device_ptr<int>(outputs[RelationInRows]),
        device_ptr<int>(outputs[RelationOutRows]),
        device_ptr<int>(outputs[RelationKernelIds]),
        device_ptr<int>(outputs[RelationRowOffsets]),
        device_ptr<int>(outputs[RelationOutCoords]),
        device_ptr<int>(outputs[RelationCounts]),
        rows,
        kernel_count,
        triple_args(stride)
    );
}

void eval_relation_grouped_view(
    RelationGroupedViewShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    bind_arrays(stream, inputs, outputs);
    auto row_cursors = backend::cuda::make_int32_temp(shape.group_count + 1);
    mx::cu::get_command_encoder(stream).set_output_array(row_cursors);
    add_1d(
        stream,
        clear_relation_grouped_view_i32,
        static_cast<std::size_t>(
            std::max(shape.edge_capacity, shape.group_count + 1)
        ),
        device_ptr<int>(outputs[RelationViewRowOffsets]),
        device_ptr<int>(outputs[RelationViewEdgeIds]),
        shape.edge_capacity,
        shape.group_count
    );
    add_1d(
        stream,
        count_relation_grouped_view_i32,
        static_cast<std::size_t>(shape.edge_capacity),
        device_ptr<int>(inputs[0]),
        device_ptr<int>(inputs[1]),
        device_ptr<int>(outputs[RelationViewRowOffsets]),
        shape.edge_capacity,
        shape.group_count
    );
    add_1d(
        stream,
        prefix_relation_rows_i32,
        1,
        device_ptr<int>(outputs[RelationViewRowOffsets]),
        device_ptr<int>(row_cursors),
        shape.group_count,
        shape.edge_capacity
    );
    add_clear_i32(stream, row_cursors, shape.group_count + 1);
    add_1d(
        stream,
        fill_relation_grouped_view_i32,
        static_cast<std::size_t>(shape.edge_capacity),
        device_ptr<int>(inputs[0]),
        device_ptr<int>(outputs[RelationViewRowOffsets]),
        device_ptr<int>(row_cursors),
        device_ptr<int>(outputs[RelationViewEdgeIds]),
        shape.edge_capacity,
        shape.group_count
    );
}

void eval_relation_direct_view(
    RelationGroupedViewShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    bind_arrays(stream, inputs, outputs);
    add_1d(
        stream,
        clear_relation_direct_view_i32,
        static_cast<std::size_t>(shape.group_count),
        device_ptr<int>(outputs[0]),
        shape.group_count
    );
    add_1d(
        stream,
        fill_relation_direct_view_i32,
        static_cast<std::size_t>(shape.edge_capacity),
        device_ptr<int>(inputs[0]),
        device_ptr<int>(inputs[1]),
        device_ptr<int>(outputs[0]),
        shape.edge_capacity,
        shape.group_count
    );
}

void eval_neighbor_relation(
    NeighborRelationOp op,
    NeighborRelationShape shape,
    float radius_squared,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    bind_arrays(stream, inputs, outputs);
    add_1d(
        stream,
        neighbor_relation_i32,
        1,
        device_ptr<int>(inputs[0]),
        device_ptr<int>(inputs[1]),
        device_ptr<int>(inputs[2]),
        device_ptr<int>(inputs[3]),
        device_ptr<int>(outputs[NeighborQueryRows]),
        device_ptr<int>(outputs[NeighborSourceRows]),
        device_ptr<int>(outputs[NeighborIds]),
        device_ptr<float>(outputs[NeighborDistances]),
        device_ptr<int>(outputs[NeighborRowOffsets]),
        device_ptr<int>(outputs[NeighborCounts]),
        neighbor_op_id(op),
        shape.source_rows,
        shape.query_rows,
        shape.max_neighbors,
        radius_squared
    );
}

} // namespace mlx_lattice::backend::cuda::coords
