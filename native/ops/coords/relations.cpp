#include "ops/coords/factories.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <typeinfo>
#include <vector>

#include "backends/cpu/coords/algorithms.h"
#include "backends/metal/coords/runtime.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "ops/coords/streams.h"

namespace mlx_lattice {

namespace {

struct RelationOutputSpec {
    std::vector<mx::Shape> shapes;
    std::vector<mx::Dtype> dtypes;
};

mx::array make_offsets_array(const std::vector<Triple>& offsets) {
    std::vector<int32_t> flat;
    flat.reserve(offsets.size() * 3);
    for (auto offset : offsets) {
        flat.insert(flat.end(), offset.begin(), offset.end());
    }
    return mx::array(
        flat.begin(), mx::Shape{int(offsets.size()), 3}, mx::int32
    );
}

NativeKernelRelation relation_from_outputs(
    const std::vector<mx::array>& outputs,
    const NativeKernelRelationViews& views
) {
    return {
        outputs[RelationInRows],
        outputs[RelationOutRows],
        outputs[RelationKernelIds],
        outputs[RelationRowOffsets],
        outputs[RelationOutCoords],
        outputs[RelationCounts],
        views.in_row_offsets,
        views.in_edge_ids,
        views.kernel_row_offsets,
        views.kernel_edge_ids,
    };
}

NativeKernelRelation target_relation_from_outputs(
    const std::vector<mx::array>& outputs,
    const mx::array& target_coords,
    const NativeKernelRelationViews& views
) {
    return {
        outputs[RelationInRows],
        outputs[RelationOutRows],
        outputs[RelationKernelIds],
        outputs[RelationRowOffsets],
        target_coords,
        outputs[RelationCounts],
        views.in_row_offsets,
        views.in_edge_ids,
        views.kernel_row_offsets,
        views.kernel_edge_ids,
    };
}

RelationOutputSpec relation_output_spec(
    int edges, // NOLINT(bugprone-easily-swappable-parameters)
    int out_rows,
    mx::Dtype coord_dtype,
    int scratch_rows = 0
) {
    std::vector<mx::Shape> shapes(RelationBaseOutputCount, mx::Shape{edges});
    std::vector<mx::Dtype> dtypes(RelationBaseOutputCount, mx::int32);
    shapes[RelationRowOffsets] = mx::Shape{out_rows + 1};
    shapes[RelationOutCoords] = mx::Shape{out_rows, 4};
    shapes[RelationCounts] = mx::Shape{2};
    dtypes[RelationOutCoords] = coord_dtype;
    if (scratch_rows > 0) {
        shapes.push_back(mx::Shape{scratch_rows});
        dtypes.push_back(mx::int32);
    }
    return {shapes, dtypes};
}

bool is_gpu_device(const mx::Device& device) {
    return device == mx::Device(mx::Device::gpu);
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

bool can_use_direct_transposed_relation(
    const std::vector<Triple>& offsets,
    Triple stride
) {
    if (offsets.empty()) {
        return false;
    }
    for (int axis = 0; axis < 3; ++axis) {
        auto min_value = offsets.front()[axis];
        auto max_value = offsets.front()[axis];
        for (auto offset : offsets) {
            min_value = std::min(min_value, offset[axis]);
            max_value = std::max(max_value, offset[axis]);
        }
        if (stride[axis] <= max_value - min_value) {
            return false;
        }
    }
    return true;
}

std::vector<mx::array> make_relation_outputs(
    const RelationOutputSpec& spec,
    const std::shared_ptr<mx::Primitive>& primitive,
    const mx::array& coords,
    const mx::array& offsets,
    const mx::array& active_rows,
    mx::Device device
) {
    auto inputs = std::vector<mx::array>{
        mx::contiguous(coords, false, device),
        mx::contiguous(offsets, false, device),
        mx::contiguous(active_rows, false, device),
    };
    return mx::array::make_arrays(spec.shapes, spec.dtypes, primitive, inputs);
}

class GenericKernelRelation final : public mx::Primitive {
  public:
    GenericKernelRelation(
        mx::Stream stream,
        CoordRelationOp op,
        int rows, // NOLINT(bugprone-easily-swappable-parameters)
        int kernel_count,
        Triple stride, // NOLINT(bugprone-easily-swappable-parameters)
        Triple padding,
        bool direct = false
    )
        : mx::Primitive(stream), op_(op), rows_(rows),
          kernel_count_(kernel_count), stride_(stride), padding_(padding),
          direct_(direct) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        coords::cpu::eval_generic_kernel_relation(
            op_, stride_, padding_, stream(), inputs, outputs
        );
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        coords::metal::eval_generic_kernel_relation(
            op_,
            rows_,
            kernel_count_,
            stride_,
            padding_,
            direct_,
            stream(),
            inputs,
            outputs
        );
    }

    const char* name() const override {
        return "lattice::GenericKernelRelation";
    }

    bool is_equivalent(const mx::Primitive& other) const override {
        if (typeid(other) != typeid(GenericKernelRelation)) {
            return false;
        }
        const auto& relation = static_cast<const GenericKernelRelation&>(other);
        return op_ == relation.op_ && rows_ == relation.rows_ &&
               kernel_count_ == relation.kernel_count_ &&
               stride_ == relation.stride_ && padding_ == relation.padding_ &&
               direct_ == relation.direct_;
    }

  private:
    CoordRelationOp op_;
    int rows_;
    int kernel_count_;
    Triple stride_;
    Triple padding_;
    bool direct_;
};

class GenerativeKernelRelation final : public mx::Primitive {
  public:
    GenerativeKernelRelation(
        mx::Stream stream,
        int rows, // NOLINT(bugprone-easily-swappable-parameters)
        int kernel_count,
        Triple stride
    )
        : mx::Primitive(stream), rows_(rows), kernel_count_(kernel_count),
          stride_(stride) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        coords::cpu::eval_generative_kernel_relation(
            stride_, stream(), inputs, outputs
        );
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        coords::metal::eval_generative_kernel_relation(
            rows_, kernel_count_, stride_, stream(), inputs, outputs
        );
    }

    const char* name() const override {
        return "lattice::GenerativeKernelRelation";
    }

    bool is_equivalent(const mx::Primitive& other) const override {
        if (typeid(other) != typeid(GenerativeKernelRelation)) {
            return false;
        }
        const auto& relation =
            static_cast<const GenerativeKernelRelation&>(other);
        return rows_ == relation.rows_ &&
               kernel_count_ == relation.kernel_count_ &&
               stride_ == relation.stride_;
    }

  private:
    int rows_;
    int kernel_count_;
    Triple stride_;
};

class TargetKernelRelation final : public mx::Primitive {
  public:
    TargetKernelRelation(
        mx::Stream stream,
        int rows, // NOLINT(bugprone-easily-swappable-parameters)
        int target_rows,
        int kernel_count,
        Triple stride, // NOLINT(bugprone-easily-swappable-parameters)
        Triple padding
    )
        : mx::Primitive(stream), rows_(rows), target_rows_(target_rows),
          kernel_count_(kernel_count), stride_(stride), padding_(padding) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        coords::cpu::eval_target_kernel_relation(
            stride_, padding_, stream(), inputs, outputs
        );
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        coords::metal::eval_target_kernel_relation(
            rows_,
            target_rows_,
            kernel_count_,
            stride_,
            padding_,
            stream(),
            inputs,
            outputs
        );
    }

    const char* name() const override {
        return "lattice::TargetKernelRelation";
    }

    bool is_equivalent(const mx::Primitive& other) const override {
        if (typeid(other) != typeid(TargetKernelRelation)) {
            return false;
        }
        const auto& relation = static_cast<const TargetKernelRelation&>(other);
        return rows_ == relation.rows_ &&
               target_rows_ == relation.target_rows_ &&
               kernel_count_ == relation.kernel_count_ &&
               stride_ == relation.stride_ && padding_ == relation.padding_;
    }

  private:
    int rows_;
    int target_rows_;
    int kernel_count_;
    Triple stride_;
    Triple padding_;
};

class RelationGroupedView final : public mx::Primitive {
  public:
    RelationGroupedView(mx::Stream stream, RelationGroupedViewShape shape)
        : mx::Primitive(stream), shape_(shape) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        coords::cpu::eval_relation_grouped_view(
            shape_, stream(), inputs, outputs
        );
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        coords::metal::eval_relation_grouped_view(
            shape_, stream(), inputs, outputs
        );
    }

    const char* name() const override { return "lattice::RelationGroupedView"; }

    bool is_equivalent(const mx::Primitive& other) const override {
        if (typeid(other) != typeid(RelationGroupedView)) {
            return false;
        }
        const auto& views = static_cast<const RelationGroupedView&>(other);
        return shape_.edge_capacity == views.shape_.edge_capacity &&
               shape_.group_count == views.shape_.group_count;
    }

  private:
    RelationGroupedViewShape shape_;
};

class RelationDirectView final : public mx::Primitive {
  public:
    RelationDirectView(mx::Stream stream, RelationGroupedViewShape shape)
        : mx::Primitive(stream), shape_(shape) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        coords::cpu::eval_relation_direct_view(
            shape_, stream(), inputs, outputs
        );
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        coords::metal::eval_relation_direct_view(
            shape_, stream(), inputs, outputs
        );
    }

    const char* name() const override { return "lattice::RelationDirectView"; }

    bool is_equivalent(const mx::Primitive& other) const override {
        if (typeid(other) != typeid(RelationDirectView)) {
            return false;
        }
        const auto& view = static_cast<const RelationDirectView&>(other);
        return shape_.edge_capacity == view.shape_.edge_capacity &&
               shape_.group_count == view.shape_.group_count;
    }

  private:
    RelationGroupedViewShape shape_;
};

} // namespace

NativeRelationGroupedView make_relation_grouped_view(
    const mx::array& group_ids,
    const mx::array& counts,
    int group_count
) {
    auto shape = RelationGroupedViewShape{
        group_ids.shape(0),
        group_count,
    };
    auto device = coord_device();
    auto outputs = mx::array::make_arrays(
        {mx::Shape{group_count + 1}, mx::Shape{shape.edge_capacity}},
        {mx::int32, mx::int32},
        std::make_shared<RelationGroupedView>(coord_stream(device), shape),
        {mx::contiguous(group_ids, false, device),
         mx::contiguous(counts, false, device)}
    );
    return {
        outputs[RelationViewRowOffsets],
        outputs[RelationViewEdgeIds],
    };
}

NativeKernelRelationViews make_kernel_relation_views(
    const mx::array& in_rows,
    const mx::array& kernel_ids,
    const mx::array& counts,
    int in_capacity,
    int kernel_count
) {
    auto input = make_relation_grouped_view(in_rows, counts, in_capacity);
    auto kernel = make_relation_grouped_view(kernel_ids, counts, kernel_count);
    return {
        input.row_offsets,
        input.edge_ids,
        kernel.row_offsets,
        kernel.edge_ids,
    };
}

NativeRelationDirectView make_relation_direct_view(
    const mx::array& group_ids,
    const mx::array& counts,
    int group_count
) {
    auto shape = RelationGroupedViewShape{
        group_ids.shape(0),
        group_count,
    };
    auto device = coord_device();
    auto outputs = mx::array::make_arrays(
        {mx::Shape{group_count}},
        {mx::int32},
        std::make_shared<RelationDirectView>(coord_stream(device), shape),
        {mx::contiguous(group_ids, false, device),
         mx::contiguous(counts, false, device)}
    );
    return {outputs[0]};
}

NativeKernelRelation make_kernel_relation(
    const mx::array& coords,
    const mx::array& active_rows,
    Triple kernel_size, // NOLINT(bugprone-easily-swappable-parameters)
    Triple stride,
    Triple padding, // NOLINT(bugprone-easily-swappable-parameters)
    Triple dilation
) {
    auto offsets = kernel_offsets(kernel_size, dilation);
    auto offset_values = make_offsets_array(offsets);
    auto rows = coords.shape(0);
    auto kernel_count = int(offsets.size());
    auto max_out_rows = rows;
    auto max_edges = max_out_rows * kernel_count;
    auto device = coord_device();
    auto scratch_rows = is_gpu_device(device) ? coord_hash_capacity(rows) : 0;
    auto outputs = make_relation_outputs(
        relation_output_spec(
            max_edges, max_out_rows, coords.dtype(), scratch_rows
        ),
        std::make_shared<GenericKernelRelation>(
            coord_stream(device),
            CoordRelationOp::Forward,
            rows,
            kernel_count,
            stride,
            padding
        ),
        coords,
        offset_values,
        active_rows,
        device
    );
    auto views = make_kernel_relation_views(
        outputs[RelationInRows],
        outputs[RelationKernelIds],
        outputs[RelationCounts],
        rows,
        kernel_count
    );
    return relation_from_outputs(outputs, views);
}

NativeKernelRelation make_generative_relation(
    const mx::array& coords,
    const mx::array& active_rows,
    Triple kernel_size, // NOLINT(bugprone-easily-swappable-parameters)
    Triple stride
) {
    auto offsets = kernel_offsets(kernel_size);
    auto offset_values = make_offsets_array(offsets);
    auto rows = coords.shape(0);
    auto kernel_count = int(offsets.size());
    auto pair_count = rows * kernel_count;
    auto device = coord_device();
    auto outputs = make_relation_outputs(
        relation_output_spec(pair_count, pair_count, coords.dtype()),
        std::make_shared<GenerativeKernelRelation>(
            coord_stream(device), rows, kernel_count, stride
        ),
        coords,
        offset_values,
        active_rows,
        device
    );
    auto views = make_kernel_relation_views(
        outputs[RelationInRows],
        outputs[RelationKernelIds],
        outputs[RelationCounts],
        rows,
        kernel_count
    );
    return relation_from_outputs(outputs, views);
}

NativeKernelRelation make_transposed_kernel_relation(
    const mx::array& coords,
    const mx::array& active_rows,
    Triple kernel_size, // NOLINT(bugprone-easily-swappable-parameters)
    Triple stride,
    Triple padding, // NOLINT(bugprone-easily-swappable-parameters)
    Triple dilation
) {
    auto offsets = kernel_offsets(kernel_size, dilation);
    auto offset_values = make_offsets_array(offsets);
    auto rows = coords.shape(0);
    auto kernel_count = int(offsets.size());
    auto max_edges = rows * kernel_count;
    auto device = coord_device();
    auto direct = can_use_direct_transposed_relation(offsets, stride);
    auto outputs = make_relation_outputs(
        relation_output_spec(max_edges, max_edges, coords.dtype()),
        std::make_shared<GenericKernelRelation>(
            coord_stream(device),
            CoordRelationOp::Transposed,
            rows,
            kernel_count,
            stride,
            padding,
            direct
        ),
        coords,
        offset_values,
        active_rows,
        device
    );
    auto views = make_kernel_relation_views(
        outputs[RelationInRows],
        outputs[RelationKernelIds],
        outputs[RelationCounts],
        rows,
        kernel_count
    );
    return relation_from_outputs(outputs, views);
}

NativeKernelRelation make_target_kernel_relation(
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& target_coords,
    const mx::array& target_active_rows,
    Triple kernel_size, // NOLINT(bugprone-easily-swappable-parameters)
    Triple stride,
    Triple padding, // NOLINT(bugprone-easily-swappable-parameters)
    Triple dilation
) {
    auto offsets = kernel_offsets(kernel_size, dilation);
    auto offset_values = make_offsets_array(offsets);
    auto rows = coords.shape(0);
    auto target_rows = target_coords.shape(0);
    auto kernel_count = int(offsets.size());
    auto max_edges = target_rows * kernel_count;
    auto device = coord_device();
    auto scratch_rows = is_gpu_device(device) ? coord_hash_capacity(rows) : 0;
    auto spec = relation_output_spec(
        max_edges, target_rows, target_coords.dtype(), scratch_rows
    );
    spec.shapes[RelationOutCoords] = mx::Shape{1, 4};
    auto inputs = std::vector<mx::array>{
        mx::contiguous(coords, false, device),
        mx::contiguous(offset_values, false, device),
        mx::contiguous(active_rows, false, device),
        mx::contiguous(target_coords, false, device),
        mx::contiguous(target_active_rows, false, device),
    };
    auto outputs = mx::array::make_arrays(
        spec.shapes,
        spec.dtypes,
        std::make_shared<TargetKernelRelation>(
            coord_stream(device),
            rows,
            target_rows,
            kernel_count,
            stride,
            padding
        ),
        inputs
    );
    auto views = make_kernel_relation_views(
        outputs[RelationInRows],
        outputs[RelationKernelIds],
        outputs[RelationCounts],
        rows,
        kernel_count
    );
    return target_relation_from_outputs(outputs, target_coords, views);
}

} // namespace mlx_lattice
