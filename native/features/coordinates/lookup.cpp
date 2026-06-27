#include "features/coordinates/factories.h"

#include <memory>
#include <typeinfo>
#include <vector>

#include "features/coordinates/cpu/algorithms.h"
#include "features/coordinates/metal/runtime.h"
#include "features/coordinates/streams.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx_lattice {

namespace {

class LookupCoords final : public mx::Primitive {
  public:
    LookupCoords(mx::Stream stream, CoordLookupShape shape)
        : mx::Primitive(stream), shape_(shape) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        coords::cpu::eval_lookup_coords(stream(), inputs, outputs);
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        coords::metal::eval_lookup_coords(shape_, stream(), inputs, outputs);
    }

    const char* name() const override { return "lattice::LookupCoords"; }

    bool is_equivalent(const mx::Primitive& other) const override {
        if (typeid(other) != typeid(LookupCoords)) {
            return false;
        }
        const auto& op = static_cast<const LookupCoords&>(other);
        return shape_.rows == op.shape_.rows &&
               shape_.query_rows == op.shape_.query_rows;
    }

  private:
    CoordLookupShape shape_;
};

class SparseAlignment final : public mx::Primitive {
  public:
    SparseAlignment(
        mx::Stream stream,
        SparseJoinOp join,
        SparseAlignmentShape shape
    )
        : mx::Primitive(stream), join_(join), shape_(shape) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        coords::cpu::eval_sparse_alignment(
            join_, shape_, stream(), inputs, outputs
        );
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        coords::metal::eval_sparse_alignment(
            join_, shape_, stream(), inputs, outputs
        );
    }

    const char* name() const override { return "lattice::SparseAlignment"; }

    bool is_equivalent(const mx::Primitive& other) const override {
        if (typeid(other) != typeid(SparseAlignment)) {
            return false;
        }
        const auto& op = static_cast<const SparseAlignment&>(other);
        return join_ == op.join_ && shape_ == op.shape_;
    }

  private:
    SparseJoinOp join_;
    SparseAlignmentShape shape_;
};

} // namespace

mx::array
make_lookup_coords(const mx::array& coords, const mx::array& queries) {
    auto device = coord_device();
    auto outputs = mx::array::make_arrays(
        {mx::Shape{queries.shape(0)}},
        {mx::int32},
        std::make_shared<LookupCoords>(
            coord_stream(device),
            CoordLookupShape{coords.shape(0), queries.shape(0)}
        ),
        {mx::contiguous(coords, false, device),
         mx::contiguous(queries, false, device)}
    );
    return outputs[0];
}

NativeSparseAlignment make_sparse_alignment(
    const mx::array& lhs_coords,
    const mx::array& lhs_active_rows,
    const mx::array& rhs_coords,
    const mx::array& rhs_active_rows,
    SparseJoinOp join
) {
    auto output_rows = lhs_coords.shape(0);
    if (join == SparseJoinOp::Right) {
        output_rows = rhs_coords.shape(0);
    } else if (join == SparseJoinOp::Outer) {
        output_rows += rhs_coords.shape(0);
    }
    auto device = coord_device();
    auto outputs = mx::array::make_arrays(
        {mx::Shape{output_rows, 4},
         mx::Shape{1},
         mx::Shape{output_rows},
         mx::Shape{output_rows}},
        {lhs_coords.dtype(), mx::int32, mx::int32, mx::int32},
        std::make_shared<SparseAlignment>(
            coord_stream(device),
            join,
            SparseAlignmentShape{
                lhs_coords.shape(0),
                rhs_coords.shape(0),
                output_rows,
            }
        ),
        {mx::contiguous(lhs_coords, false, device),
         mx::contiguous(lhs_active_rows, false, device),
         mx::contiguous(rhs_coords, false, device),
         mx::contiguous(rhs_active_rows, false, device)}
    );
    return {outputs[0], outputs[1], outputs[2], outputs[3]};
}

} // namespace mlx_lattice
