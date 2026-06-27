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

} // namespace mlx_lattice
