#include "ops/coords/dispatch.h"

#include <cstdint>
#include <vector>

#include "backends/cpu/coords/algorithms.h"
#include "backends/metal/coords/runtime.h"
#include "mlx/device.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx_lattice {

namespace {

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

int scalar_i32(const mx::array& value) {
    auto cpu_value = mx::contiguous(value, false, mx::Device::cpu);
    cpu_value.eval();
    cpu_value.wait();
    return cpu_value.data<int32_t>()[0];
}

mx::array compact_coords(const std::vector<mx::array>& outputs) {
    auto rows = scalar_i32(outputs[1]);
    return mx::slice(outputs[0], {0, 0}, {rows, 4});
}

mx::array compact_rows(const mx::array& rows, int count) {
    return mx::slice(rows, {0}, {count});
}

mx::array compact_offsets(const mx::array& offsets, int count) {
    return mx::slice(offsets, {0}, {count + 1});
}

NativeKernelMap compact_map(
    const std::vector<mx::array>& outputs,
    const mx::array& kernel_offsets
) {
    auto count_values = mx::contiguous(outputs[4], false, mx::Device::cpu);
    count_values.eval();
    count_values.wait();
    auto counts = count_values.data<int32_t>();
    auto edge_count = counts[0];
    auto out_count = counts[1];
    auto kernel_count = kernel_offsets.shape(0);
    return {
        compact_rows(outputs[0], edge_count),
        compact_rows(outputs[1], edge_count),
        compact_rows(outputs[2], edge_count),
        mx::slice(outputs[3], {0, 0}, {out_count, 4}),
        kernel_offsets,
        {compact_offsets(outputs[5], out_count),
         compact_rows(outputs[6], edge_count),
         compact_rows(outputs[7], edge_count)},
        {compact_offsets(outputs[8], kernel_count),
         compact_rows(outputs[9], edge_count),
         compact_rows(outputs[10], edge_count)},
        {compact_offsets(outputs[11], outputs[11].shape(0) - 1),
         compact_rows(outputs[12], edge_count),
         compact_rows(outputs[13], edge_count)},
    };
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
        coords::cpu::eval_set_coords(op_, stride_, inputs, outputs);
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
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
        coords::cpu::eval_lookup_coords(inputs, outputs);
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
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

class GenericKernelMap final : public mx::Primitive {
  public:
    GenericKernelMap(
        mx::Stream stream,
        CoordMapOp op,
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
        coords::cpu::eval_generic_kernel_map(
            op_, stride_, padding_, inputs, outputs
        );
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        coords::metal::eval_generic_kernel_map(
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

    const char* name() const override { return "lattice::GenericKernelMap"; }

    bool is_equivalent(const mx::Primitive& other) const override {
        const auto& map = static_cast<const GenericKernelMap&>(other);
        return op_ == map.op_ && rows_ == map.rows_ &&
               kernel_count_ == map.kernel_count_ && stride_ == map.stride_ &&
               padding_ == map.padding_;
    }

  private:
    CoordMapOp op_;
    int rows_;
    int kernel_count_;
    Triple stride_;
    Triple padding_;
};

class GenerativeKernelMap final : public mx::Primitive {
  public:
    GenerativeKernelMap(
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
        coords::cpu::eval_generative_kernel_map(stride_, inputs, outputs);
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        coords::metal::eval_generative_kernel_map(
            rows_, kernel_count_, stride_, stream(), inputs, outputs
        );
    }

    const char* name() const override { return "lattice::GenerativeKernelMap"; }

    bool is_equivalent(const mx::Primitive& other) const override {
        const auto& map = static_cast<const GenerativeKernelMap&>(other);
        return rows_ == map.rows_ && kernel_count_ == map.kernel_count_ &&
               stride_ == map.stride_;
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

// MARK: - maps

NativeKernelMap dispatch_build_kernel_map(
    const mx::array& coords,
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
    auto outputs = mx::array::make_arrays(
        {mx::Shape{max_edges},
         mx::Shape{max_edges},
         mx::Shape{max_edges},
         mx::Shape{max_out_rows, 4},
         mx::Shape{2},
         mx::Shape{max_out_rows + 1},
         mx::Shape{max_edges},
         mx::Shape{max_edges},
         mx::Shape{kernel_count + 1},
         mx::Shape{max_edges},
         mx::Shape{max_edges},
         mx::Shape{rows + 1},
         mx::Shape{max_edges},
         mx::Shape{max_edges}},
        {mx::int32,
         mx::int32,
         mx::int32,
         coords.dtype(),
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32},
        std::make_shared<GenericKernelMap>(
            mx::default_stream(device),
            CoordMapOp::Forward,
            rows,
            kernel_count,
            stride,
            padding
        ),
        {mx::contiguous(coords, false, device),
         mx::contiguous(offset_values, false, device)}
    );
    return compact_map(outputs, offset_values);
}

NativeKernelMap dispatch_build_generative_map(
    const mx::array& coords,
    Triple kernel_size, // NOLINT(bugprone-easily-swappable-parameters)
    Triple stride
) {
    auto offsets = kernel_offsets(kernel_size);
    auto offset_values = make_offsets_array(offsets);
    auto rows = coords.shape(0);
    auto kernel_count = int(offsets.size());
    auto pair_count = rows * kernel_count;
    auto device = device_for(coords);
    auto outputs = mx::array::make_arrays(
        {mx::Shape{pair_count},
         mx::Shape{pair_count},
         mx::Shape{pair_count},
         mx::Shape{pair_count, 4},
         mx::Shape{pair_count + 1},
         mx::Shape{pair_count},
         mx::Shape{pair_count},
         mx::Shape{kernel_count + 1},
         mx::Shape{pair_count},
         mx::Shape{pair_count},
         mx::Shape{rows + 1},
         mx::Shape{pair_count},
         mx::Shape{pair_count}},
        {mx::int32,
         mx::int32,
         mx::int32,
         coords.dtype(),
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32},
        std::make_shared<GenerativeKernelMap>(
            mx::default_stream(device), rows, kernel_count, stride
        ),
        {mx::contiguous(coords, false, device),
         mx::contiguous(offset_values, false, device)}
    );

    return {
        outputs[0],
        outputs[1],
        outputs[2],
        outputs[3],
        offset_values,
        {outputs[4], outputs[5], outputs[6]},
        {outputs[7], outputs[8], outputs[9]},
        {outputs[10], outputs[11], outputs[12]},
    };
}

NativeKernelMap dispatch_build_transposed_kernel_map(
    const mx::array& coords,
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
    auto outputs = mx::array::make_arrays(
        {mx::Shape{max_edges},
         mx::Shape{max_edges},
         mx::Shape{max_edges},
         mx::Shape{max_edges, 4},
         mx::Shape{2},
         mx::Shape{max_edges + 1},
         mx::Shape{max_edges},
         mx::Shape{max_edges},
         mx::Shape{kernel_count + 1},
         mx::Shape{max_edges},
         mx::Shape{max_edges},
         mx::Shape{rows + 1},
         mx::Shape{max_edges},
         mx::Shape{max_edges}},
        {mx::int32,
         mx::int32,
         mx::int32,
         coords.dtype(),
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32,
         mx::int32},
        std::make_shared<GenericKernelMap>(
            mx::default_stream(device),
            CoordMapOp::Transposed,
            rows,
            kernel_count,
            stride,
            padding
        ),
        {mx::contiguous(coords, false, device),
         mx::contiguous(offset_values, false, device)}
    );
    return compact_map(outputs, offset_values);
}

} // namespace mlx_lattice
