#include "ops/coords/factories.h"

#include <memory>
#include <typeinfo>
#include <vector>

#include "backends/cpu/coords/algorithms.h"
#include "backends/gpu.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "ops/coords/streams.h"

namespace mlx_lattice {

namespace {

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
        backend::cpu::coords::eval_set_coords(
            op_, stride_, stream(), inputs, outputs
        );
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        backend::gpu::coords::eval_set_coords(
            op_, stride_, shape_, stream(), inputs, outputs
        );
    }

    const char* name() const override { return "lattice::SetCoords"; }

    bool is_equivalent(const mx::Primitive& other) const override {
        if (typeid(other) != typeid(SetCoords)) {
            return false;
        }
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

    auto device = coord_device();
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
            coord_stream(device), op, stride, CoordSetShape{lhs_rows, rhs_rows}
        ),
        inputs
    );
}

NativeCoordSet coord_set_from_outputs(const std::vector<mx::array>& outputs) {
    return {outputs[0], outputs[1]};
}

} // namespace

NativeCoordSet make_downsample_coords(const mx::array& coords, Triple stride) {
    auto outputs = make_set_outputs(
        CoordSetOp::Downsample, coords, mx::array({}, mx::int32), stride
    );
    return coord_set_from_outputs(outputs);
}

NativeCoordSet make_union_coords(const mx::array& lhs, const mx::array& rhs) {
    auto outputs =
        make_set_outputs(CoordSetOp::Union, lhs, rhs, Triple{1, 1, 1});
    return coord_set_from_outputs(outputs);
}

NativeCoordSet
make_intersection_coords(const mx::array& lhs, const mx::array& rhs) {
    auto outputs =
        make_set_outputs(CoordSetOp::Intersection, lhs, rhs, Triple{1, 1, 1});
    return coord_set_from_outputs(outputs);
}

} // namespace mlx_lattice
