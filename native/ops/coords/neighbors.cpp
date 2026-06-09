#include "ops/coords/factories.h"

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

struct NeighborOutputSpec {
    std::vector<mx::Shape> shapes;
    std::vector<mx::Dtype> dtypes;
};

struct NeighborRelationConfig {
    NeighborRelationOp op;
    int max_neighbors;
    float radius_squared;
};

NativeNeighborRelation
neighbor_from_outputs(const std::vector<mx::array>& outputs) {
    return {
        outputs[NeighborQueryRows],
        outputs[NeighborSourceRows],
        outputs[NeighborIds],
        outputs[NeighborDistances],
        outputs[NeighborCounts],
    };
}

NeighborOutputSpec neighbor_output_spec(int edge_capacity) {
    std::vector<mx::Shape> shapes(
        NeighborOutputCount, mx::Shape{edge_capacity}
    );
    std::vector<mx::Dtype> dtypes(NeighborOutputCount, mx::int32);
    dtypes[NeighborDistances] = mx::float32;
    shapes[NeighborCounts] = mx::Shape{2};
    return {shapes, dtypes};
}

std::vector<mx::array> make_neighbor_outputs(
    const NeighborOutputSpec& spec,
    const std::shared_ptr<mx::Primitive>& primitive,
    const mx::array& source_coords,
    const mx::array& source_active_rows,
    const mx::array& query_coords,
    const mx::array& query_active_rows,
    mx::Device device
) {
    auto inputs = std::vector<mx::array>{
        mx::contiguous(source_coords, false, device),
        mx::contiguous(query_coords, false, device),
        mx::contiguous(source_active_rows, false, device),
        mx::contiguous(query_active_rows, false, device),
    };
    return mx::array::make_arrays(spec.shapes, spec.dtypes, primitive, inputs);
}

class NeighborRelationPrimitive final : public mx::Primitive {
  public:
    NeighborRelationPrimitive(
        mx::Stream stream,
        NeighborRelationOp op,
        NeighborRelationShape shape,
        float radius_squared
    )
        : mx::Primitive(stream), op_(op), shape_(shape),
          radius_squared_(radius_squared) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        coords::cpu::eval_neighbor_relation(
            op_, shape_, radius_squared_, stream(), inputs, outputs
        );
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        coords::metal::eval_neighbor_relation(
            op_, shape_, radius_squared_, stream(), inputs, outputs
        );
    }

    const char* name() const override { return "lattice::NeighborRelation"; }

    bool is_equivalent(const mx::Primitive& other) const override {
        if (typeid(other) != typeid(NeighborRelationPrimitive)) {
            return false;
        }
        const auto& relation =
            static_cast<const NeighborRelationPrimitive&>(other);
        return op_ == relation.op_ &&
               shape_.source_rows == relation.shape_.source_rows &&
               shape_.query_rows == relation.shape_.query_rows &&
               shape_.max_neighbors == relation.shape_.max_neighbors &&
               radius_squared_ == relation.radius_squared_;
    }

  private:
    NeighborRelationOp op_;
    NeighborRelationShape shape_;
    float radius_squared_;
};

NativeNeighborRelation make_neighbor_relation(
    NeighborRelationConfig config,
    const mx::array& source_coords,
    const mx::array& source_active_rows,
    const mx::array& query_coords,
    const mx::array& query_active_rows
) {
    auto shape = NeighborRelationShape{
        source_coords.shape(0),
        query_coords.shape(0),
        config.max_neighbors,
    };
    auto edge_capacity = shape.query_rows * shape.max_neighbors;
    auto device = coord_device();
    auto outputs = make_neighbor_outputs(
        neighbor_output_spec(edge_capacity),
        std::make_shared<NeighborRelationPrimitive>(
            coord_stream(device), config.op, shape, config.radius_squared
        ),
        source_coords,
        source_active_rows,
        query_coords,
        query_active_rows,
        device
    );
    return neighbor_from_outputs(outputs);
}

} // namespace

NativeNeighborRelation make_knn_relation(
    const mx::array& source_coords,
    const mx::array& source_active_rows,
    const mx::array& query_coords,
    const mx::array& query_active_rows,
    int k
) {
    return make_neighbor_relation(
        NeighborRelationConfig{
            NeighborRelationOp::Knn,
            k,
            0.0F,
        },
        source_coords,
        source_active_rows,
        query_coords,
        query_active_rows
    );
}

NativeNeighborRelation make_radius_relation(
    const mx::array& source_coords,
    const mx::array& source_active_rows,
    const mx::array& query_coords,
    const mx::array& query_active_rows,
    double radius,
    int max_neighbors
) {
    return make_neighbor_relation(
        NeighborRelationConfig{
            NeighborRelationOp::Radius,
            max_neighbors,
            static_cast<float>(radius * radius),
        },
        source_coords,
        source_active_rows,
        query_coords,
        query_active_rows
    );
}

} // namespace mlx_lattice
