#include "ops/coords/dispatch.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "backends/cpu/coords/algorithms.h"
#include "backends/metal/coords/runtime.h"
#include "mlx/device.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/transforms.h"

namespace mlx_lattice {

namespace {

struct RelationOutputSpec {
    std::vector<mx::Shape> shapes;
    std::vector<mx::Dtype> dtypes;
};

// MARK: - arrays

mx::Device device_for(const mx::array& coords) {
    if (coords::metal::supports(coords)) {
        return mx::Device::gpu;
    }
    return mx::Device::cpu;
}

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

mx::array compact_coords(const std::vector<mx::array>& outputs) {
    auto cpu_value = mx::contiguous(outputs[1], false, mx::Device::cpu);
    cpu_value.eval();
    cpu_value.wait();
    auto rows = cpu_value.data<int32_t>()[0];
    return mx::slice(outputs[0], {0, 0}, {rows, 4});
}

NativeKernelRelation
relation_from_outputs(const std::vector<mx::array>& outputs) {
    return {
        outputs[RelationInRows],
        outputs[RelationOutRows],
        outputs[RelationKernelIds],
        outputs[RelationOutCoords],
        outputs[RelationCounts],
    };
}

RelationOutputSpec relation_output_spec(
    int edges,    // NOLINT(bugprone-easily-swappable-parameters)
    int out_rows, // NOLINT(bugprone-easily-swappable-parameters)
    mx::Dtype coord_dtype
) {
    std::vector<mx::Shape> shapes(RelationOutputCount, mx::Shape{edges});
    std::vector<mx::Dtype> dtypes(RelationOutputCount, mx::int32);
    shapes[RelationOutCoords] = mx::Shape{out_rows, 4};
    shapes[RelationCounts] = mx::Shape{2};
    dtypes[RelationOutCoords] = coord_dtype;
    return {shapes, dtypes};
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
    mx::eval(inputs);
    return mx::array::make_arrays(spec.shapes, spec.dtypes, primitive, inputs);
}

// MARK: - primitives

class SetCoords final : public mx::Primitive {
  public:
    SetCoords(
        mx::Stream stream,
        CoordSetOp op,
        Triple stride,
        CoordSetShape shape
    )
        : mx::Primitive(stream), op_(op), stride_(stride), shape_(shape) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        mx::eval(inputs);
        coords::cpu::eval_set_coords(op_, stride_, inputs, outputs);
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        mx::eval(inputs);
        coords::metal::eval_set_coords(
            op_, stride_, shape_, stream(), inputs, outputs
        );
    }

    const char* name() const override { return "lattice::SetCoords"; }

    bool is_equivalent(const mx::Primitive& other) const override {
        const auto& op = static_cast<const SetCoords&>(other);
        return op_ == op.op_ && stride_ == op.stride_ &&
               shape_.lhs_rows == op.shape_.lhs_rows &&
               shape_.rhs_rows == op.shape_.rhs_rows;
    }

  private:
    CoordSetOp op_;
    Triple stride_;
    CoordSetShape shape_;
};

class LookupCoords final : public mx::Primitive {
  public:
    LookupCoords(mx::Stream stream, CoordLookupShape shape)
        : mx::Primitive(stream), shape_(shape) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        mx::eval(inputs);
        coords::cpu::eval_lookup_coords(inputs, outputs);
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        mx::eval(inputs);
        coords::metal::eval_lookup_coords(shape_, stream(), inputs, outputs);
    }

    const char* name() const override { return "lattice::LookupCoords"; }

    bool is_equivalent(const mx::Primitive& other) const override {
        const auto& op = static_cast<const LookupCoords&>(other);
        return shape_.rows == op.shape_.rows &&
               shape_.query_rows == op.shape_.query_rows;
    }

  private:
    CoordLookupShape shape_;
};

class GenericKernelRelation final : public mx::Primitive {
  public:
    GenericKernelRelation(
        mx::Stream stream,
        CoordRelationOp op,
        int rows, // NOLINT(bugprone-easily-swappable-parameters)
        int kernel_count,
        Triple stride, // NOLINT(bugprone-easily-swappable-parameters)
        Triple padding
    )
        : mx::Primitive(stream), op_(op), rows_(rows),
          kernel_count_(kernel_count), stride_(stride), padding_(padding) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        mx::eval(inputs);
        coords::cpu::eval_generic_kernel_relation(
            op_, stride_, padding_, inputs, outputs
        );
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        mx::eval(inputs);
        coords::metal::eval_generic_kernel_relation(
            op_,
            rows_,
            kernel_count_,
            stride_,
            padding_,
            stream(),
            inputs,
            outputs
        );
    }

    const char* name() const override {
        return "lattice::GenericKernelRelation";
    }

    bool is_equivalent(const mx::Primitive& other) const override {
        const auto& relation = static_cast<const GenericKernelRelation&>(other);
        return op_ == relation.op_ && rows_ == relation.rows_ &&
               kernel_count_ == relation.kernel_count_ &&
               stride_ == relation.stride_ && padding_ == relation.padding_;
    }

  private:
    CoordRelationOp op_;
    int rows_;
    int kernel_count_;
    Triple stride_;
    Triple padding_;
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
        mx::eval(inputs);
        coords::cpu::eval_generative_kernel_relation(stride_, inputs, outputs);
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        mx::eval(inputs);
        coords::metal::eval_generative_kernel_relation(
            rows_, kernel_count_, stride_, stream(), inputs, outputs
        );
    }

    const char* name() const override {
        return "lattice::GenerativeKernelRelation";
    }

    bool is_equivalent(const mx::Primitive& other) const override {
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

// MARK: - factories

std::vector<mx::array> make_set_outputs(
    CoordSetOp op,
    const mx::array& lhs,
    const mx::array& rhs,
    Triple stride
) {
    auto lhs_rows = lhs.shape(0);
    auto rhs_rows = rhs.size() == 0 ? 0 : rhs.shape(0);
    auto max_rows = lhs_rows;
    if (op == CoordSetOp::Union) {
        max_rows += rhs_rows;
    }

    auto device = device_for(lhs);
    auto stream = mx::default_stream(device);
    std::vector<mx::array> inputs = {
        mx::contiguous(lhs, false, device),
    };
    if (op != CoordSetOp::Downsample) {
        inputs.push_back(mx::contiguous(rhs, false, device));
    }

    return mx::array::make_arrays(
        {mx::Shape{max_rows, 4}, mx::Shape{1}},
        {lhs.dtype(), mx::int32},
        std::make_shared<SetCoords>(
            stream, op, stride, CoordSetShape{lhs_rows, rhs_rows}
        ),
        inputs
    );
}

} // namespace

// MARK: - set ops

mx::array dispatch_downsample_coords(const mx::array& coords, Triple stride) {
    auto outputs = make_set_outputs(
        CoordSetOp::Downsample, coords, mx::array({}, mx::int32), stride
    );
    return compact_coords(outputs);
}

mx::array dispatch_union_coords(const mx::array& lhs, const mx::array& rhs) {
    auto outputs =
        make_set_outputs(CoordSetOp::Union, lhs, rhs, Triple{1, 1, 1});
    return compact_coords(outputs);
}

mx::array
dispatch_intersection_coords(const mx::array& lhs, const mx::array& rhs) {
    auto outputs =
        make_set_outputs(CoordSetOp::Intersection, lhs, rhs, Triple{1, 1, 1});
    return compact_coords(outputs);
}

mx::array
dispatch_lookup_coords(const mx::array& coords, const mx::array& queries) {
    auto device = device_for(coords);
    auto outputs = mx::array::make_arrays(
        {mx::Shape{queries.shape(0)}},
        {mx::int32},
        std::make_shared<LookupCoords>(
            mx::default_stream(device),
            CoordLookupShape{coords.shape(0), queries.shape(0)}
        ),
        {mx::contiguous(coords, false, device),
         mx::contiguous(queries, false, device)}
    );
    return outputs[0];
}

// MARK: - relations

NativeKernelRelation dispatch_build_kernel_relation(
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
    auto device = device_for(coords);
    auto outputs = make_relation_outputs(
        relation_output_spec(max_edges, max_out_rows, coords.dtype()),
        std::make_shared<GenericKernelRelation>(
            mx::default_stream(device),
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
    return relation_from_outputs(outputs);
}

NativeKernelRelation dispatch_build_generative_relation(
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
    auto device = device_for(coords);
    auto outputs = make_relation_outputs(
        relation_output_spec(pair_count, pair_count, coords.dtype()),
        std::make_shared<GenerativeKernelRelation>(
            mx::default_stream(device), rows, kernel_count, stride
        ),
        coords,
        offset_values,
        active_rows,
        device
    );
    return relation_from_outputs(outputs);
}

NativeKernelRelation dispatch_build_transposed_kernel_relation(
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
    auto device = device_for(coords);
    auto outputs = make_relation_outputs(
        relation_output_spec(max_edges, max_edges, coords.dtype()),
        std::make_shared<GenericKernelRelation>(
            mx::default_stream(device),
            CoordRelationOp::Transposed,
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
    return relation_from_outputs(outputs);
}

} // namespace mlx_lattice
