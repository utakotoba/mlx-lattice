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

NativeSparseOccupancy
occupancy_from_outputs(const std::vector<mx::array>& outputs) {
    return {outputs[0], outputs[1], outputs[2]};
}

NativeOccupancyExpansion
expansion_from_outputs(const std::vector<mx::array>& outputs) {
    return {outputs[0], outputs[1], outputs[2], outputs[3]};
}

class OccupancyDownsample final : public mx::Primitive {
  public:
    OccupancyDownsample(mx::Stream stream, CoordActiveRowsShape shape)
        : mx::Primitive(stream), shape_(shape) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        backend::cpu::coords::eval_occupancy_downsample(
            stream(), inputs, outputs
        );
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        backend::gpu::coords::eval_occupancy_downsample(
            shape_, stream(), inputs, outputs
        );
    }

    const char* name() const override { return "lattice::OccupancyDownsample"; }

    bool is_equivalent(const mx::Primitive& other) const override {
        if (typeid(other) != typeid(OccupancyDownsample)) {
            return false;
        }
        const auto& op = static_cast<const OccupancyDownsample&>(other);
        return shape_.rows == op.shape_.rows;
    }

  private:
    CoordActiveRowsShape shape_;
};

class OccupancyExpand final : public mx::Primitive {
  public:
    OccupancyExpand(mx::Stream stream, CoordActiveRowsShape shape)
        : mx::Primitive(stream), shape_(shape) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        backend::cpu::coords::eval_occupancy_expand(stream(), inputs, outputs);
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        backend::gpu::coords::eval_occupancy_expand(
            shape_, stream(), inputs, outputs
        );
    }

    const char* name() const override { return "lattice::OccupancyExpand"; }

    bool is_equivalent(const mx::Primitive& other) const override {
        if (typeid(other) != typeid(OccupancyExpand)) {
            return false;
        }
        const auto& op = static_cast<const OccupancyExpand&>(other);
        return shape_.rows == op.shape_.rows;
    }

  private:
    CoordActiveRowsShape shape_;
};

class ChildCoordsFromIndices final : public mx::Primitive {
  public:
    ChildCoordsFromIndices(mx::Stream stream, CoordRowsShape shape)
        : mx::Primitive(stream), shape_(shape) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        backend::cpu::coords::eval_child_coords_from_indices(
            stream(), inputs, outputs
        );
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        backend::gpu::coords::eval_child_coords_from_indices(
            shape_, stream(), inputs, outputs
        );
    }

    const char* name() const override {
        return "lattice::ChildCoordsFromIndices";
    }

    bool is_equivalent(const mx::Primitive& other) const override {
        if (typeid(other) != typeid(ChildCoordsFromIndices)) {
            return false;
        }
        const auto& op = static_cast<const ChildCoordsFromIndices&>(other);
        return shape_.rows == op.shape_.rows;
    }

  private:
    CoordRowsShape shape_;
};

} // namespace

NativeSparseOccupancy make_occupancy_downsample(
    const mx::array& coords,
    const mx::array& active_rows
) {
    auto device = coord_device();
    auto outputs = mx::array::make_arrays(
        {mx::Shape{coords.shape(0), 4},
         mx::Shape{1},
         mx::Shape{coords.shape(0)}},
        {coords.dtype(), mx::int32, mx::int32},
        std::make_shared<OccupancyDownsample>(
            coord_stream(device), CoordActiveRowsShape{coords.shape(0)}
        ),
        {mx::contiguous(coords, false, device),
         mx::contiguous(active_rows, false, device)}
    );
    return occupancy_from_outputs(outputs);
}

NativeOccupancyExpansion make_occupancy_expand(
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& occupancy
) {
    auto device = coord_device();
    auto out_rows = coords.shape(0) * 8;
    auto outputs = mx::array::make_arrays(
        {mx::Shape{out_rows, 4},
         mx::Shape{1},
         mx::Shape{out_rows},
         mx::Shape{out_rows}},
        {coords.dtype(), mx::int32, mx::int32, mx::int32},
        std::make_shared<OccupancyExpand>(
            coord_stream(device), CoordActiveRowsShape{coords.shape(0)}
        ),
        {mx::contiguous(coords, false, device),
         mx::contiguous(active_rows, false, device),
         mx::contiguous(occupancy, false, device)}
    );
    return expansion_from_outputs(outputs);
}

mx::array make_child_coords_from_indices(
    const mx::array& parent_coords,
    const mx::array& child_indices
) {
    auto device = coord_device();
    auto outputs = mx::array::make_arrays(
        {mx::Shape{parent_coords.shape(0), 4}},
        {parent_coords.dtype()},
        std::make_shared<ChildCoordsFromIndices>(
            coord_stream(device), CoordRowsShape{parent_coords.shape(0)}
        ),
        {mx::contiguous(parent_coords, false, device),
         mx::contiguous(child_indices, false, device)}
    );
    return outputs[0];
}

} // namespace mlx_lattice
