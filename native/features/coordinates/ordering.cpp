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

class MortonCodes final : public mx::Primitive {
  public:
    MortonCodes(mx::Stream stream, CoordRowsShape shape)
        : mx::Primitive(stream), shape_(shape) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        coords::cpu::eval_morton_codes(stream(), inputs, outputs);
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        coords::metal::eval_morton_codes(shape_, stream(), inputs, outputs);
    }

    const char* name() const override { return "lattice::MortonCodes"; }

    bool is_equivalent(const mx::Primitive& other) const override {
        if (typeid(other) != typeid(MortonCodes)) {
            return false;
        }
        const auto& op = static_cast<const MortonCodes&>(other);
        return shape_.rows == op.shape_.rows;
    }

  private:
    CoordRowsShape shape_;
};

} // namespace

mx::array make_morton_codes(const mx::array& coords) {
    auto device = coord_device();
    auto outputs = mx::array::make_arrays(
        {mx::Shape{coords.shape(0)}},
        {mx::int64},
        std::make_shared<MortonCodes>(
            coord_stream(device), CoordRowsShape{coords.shape(0)}
        ),
        {mx::contiguous(coords, false, device)}
    );
    return outputs[0];
}

} // namespace mlx_lattice
