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

NativeSparseQuantization
quantization_from_outputs(const std::vector<mx::array>& outputs) {
    return {outputs[0], outputs[1], outputs[2], outputs[3]};
}

class SparseQuantize final : public mx::Primitive {
  public:
    SparseQuantize(mx::Stream stream, QuantizationSpec spec, int rows)
        : mx::Primitive(stream), spec_(spec), rows_(rows) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        coords::cpu::eval_sparse_quantize(spec_, stream(), inputs, outputs);
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        coords::metal::eval_sparse_quantize(
            spec_, rows_, stream(), inputs, outputs
        );
    }

    const char* name() const override { return "lattice::SparseQuantize"; }

    bool is_equivalent(const mx::Primitive& other) const override {
        if (typeid(other) != typeid(SparseQuantize)) {
            return false;
        }
        const auto& op = static_cast<const SparseQuantize&>(other);
        return spec_.voxel_size == op.spec_.voxel_size &&
               spec_.origin == op.spec_.origin && rows_ == op.rows_;
    }

  private:
    QuantizationSpec spec_;
    int rows_;
};

class VoxelizeFeatureGrad;

mx::array make_voxelize_feature_grad(
    VoxelReduceOp reduce,
    const mx::array& cotangent,
    const mx::array& inverse_rows,
    const mx::array& voxel_counts,
    const mx::array& active_rows,
    VoxelFeatureShape shape
);

class VoxelizeFeatures final : public mx::Primitive {
  public:
    VoxelizeFeatures(
        mx::Stream stream,
        VoxelReduceOp reduce,
        VoxelFeatureShape shape
    )
        : mx::Primitive(stream), reduce_(reduce), shape_(shape) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        coords::cpu::eval_voxelize_features(
            reduce_, shape_, stream(), inputs, outputs
        );
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        coords::metal::eval_voxelize_features(
            reduce_, shape_, stream(), inputs, outputs
        );
    }

    const char* name() const override { return "lattice::VoxelizeFeatures"; }

    std::vector<mx::array>
    jvp(const std::vector<mx::array>& primals,
        const std::vector<mx::array>& tangents,
        const std::vector<int>& argnums) override {
        for (int index = 0; index < int(argnums.size()); ++index) {
            if (argnums[index] == 0) {
                return {make_voxelize_features(
                    tangents[index], primals[1], primals[2], primals[3], reduce_
                )};
            }
        }
        return {mx::zeros(
            mx::Shape{shape_.voxel_rows, shape_.channels},
            primals[0].dtype(),
            stream()
        )};
    }

    std::vector<mx::array>
    vjp(const std::vector<mx::array>& primals,
        const std::vector<mx::array>& cotangents,
        const std::vector<int>& argnums,
        const std::vector<mx::array>& outputs) override {
        (void)outputs;
        std::vector<mx::array> grads;
        grads.reserve(argnums.size());
        for (const auto argnum : argnums) {
            if (argnum == 0) {
                grads.push_back(make_voxelize_feature_grad(
                    reduce_,
                    cotangents[0],
                    primals[1],
                    primals[2],
                    primals[3],
                    shape_
                ));
            } else {
                grads.push_back(mx::zeros_like(primals[argnum], stream()));
            }
        }
        return grads;
    }

    bool is_equivalent(const mx::Primitive& other) const override {
        if (typeid(other) != typeid(VoxelizeFeatures)) {
            return false;
        }
        const auto& op = static_cast<const VoxelizeFeatures&>(other);
        return reduce_ == op.reduce_ && shape_ == op.shape_;
    }

  private:
    VoxelReduceOp reduce_;
    VoxelFeatureShape shape_;
};

class VoxelizeFeatureGrad final : public mx::Primitive {
  public:
    VoxelizeFeatureGrad(
        mx::Stream stream,
        VoxelReduceOp reduce,
        VoxelFeatureShape shape
    )
        : mx::Primitive(stream), reduce_(reduce), shape_(shape) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        coords::cpu::eval_voxelize_feature_grad(
            reduce_, shape_, stream(), inputs, outputs
        );
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        coords::metal::eval_voxelize_feature_grad(
            reduce_, shape_, stream(), inputs, outputs
        );
    }

    const char* name() const override { return "lattice::VoxelizeFeatureGrad"; }

    bool is_equivalent(const mx::Primitive& other) const override {
        if (typeid(other) != typeid(VoxelizeFeatureGrad)) {
            return false;
        }
        const auto& op = static_cast<const VoxelizeFeatureGrad&>(other);
        return reduce_ == op.reduce_ && shape_ == op.shape_;
    }

  private:
    VoxelReduceOp reduce_;
    VoxelFeatureShape shape_;
};

} // namespace

NativeSparseQuantization make_sparse_quantize(
    const mx::array& points,
    const mx::array& batch_indices,
    const mx::array& active_rows,
    QuantizationSpec spec
) {
    auto device = coord_device();
    auto inputs = std::vector<mx::array>{
        mx::contiguous(points, false, device),
        mx::contiguous(batch_indices, false, device),
        mx::contiguous(active_rows, false, device),
    };
    auto outputs = mx::array::make_arrays(
        {mx::Shape{points.shape(0), 4},
         mx::Shape{1},
         mx::Shape{points.shape(0)},
         mx::Shape{points.shape(0)}},
        {mx::int32, mx::int32, mx::int32, mx::int32},
        std::make_shared<SparseQuantize>(
            coord_stream(device), spec, points.shape(0)
        ),
        inputs
    );
    return quantization_from_outputs(outputs);
}

mx::array make_voxelize_features(
    const mx::array& feats,
    const mx::array& inverse_rows,
    const mx::array& voxel_counts,
    const mx::array& active_rows,
    VoxelReduceOp reduce
) {
    auto shape = VoxelFeatureShape{
        feats.shape(0),
        voxel_counts.shape(0),
        feats.shape(1),
    };
    auto device = coord_device();
    auto inputs = std::vector<mx::array>{
        mx::contiguous(feats, false, device),
        mx::contiguous(inverse_rows, false, device),
        mx::contiguous(voxel_counts, false, device),
        mx::contiguous(active_rows, false, device),
    };
    return mx::array::make_arrays(
        {mx::Shape{shape.voxel_rows, shape.channels}},
        {feats.dtype()},
        std::make_shared<VoxelizeFeatures>(coord_stream(device), reduce, shape),
        inputs
    )[0];
}

namespace {

mx::array make_voxelize_feature_grad(
    VoxelReduceOp reduce,
    const mx::array& cotangent,
    const mx::array& inverse_rows,
    const mx::array& voxel_counts,
    const mx::array& active_rows,
    VoxelFeatureShape shape
) {
    auto device = coord_device();
    auto inputs = std::vector<mx::array>{
        mx::contiguous(cotangent, false, device),
        mx::contiguous(inverse_rows, false, device),
        mx::contiguous(voxel_counts, false, device),
        mx::contiguous(active_rows, false, device),
    };
    return mx::array::make_arrays(
        {mx::Shape{shape.point_rows, shape.channels}},
        {cotangent.dtype()},
        std::make_shared<VoxelizeFeatureGrad>(
            coord_stream(device), reduce, shape
        ),
        inputs
    )[0];
}

} // namespace

} // namespace mlx_lattice
