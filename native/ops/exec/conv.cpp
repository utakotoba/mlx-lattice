#include "ops/exec/factories.h"

#include <memory>
#include <typeinfo>
#include <vector>

#include "backends/cpu/exec/algorithms.h"
#include "backends/metal/exec/runtime.h"
#include "mlx/ops.h"
#include "ops/exec/primitive.h"
#include "ops/exec/streams.h"

namespace mlx_lattice {

namespace {

mx::array make_sparse_conv_input_grad(
    SparseMapOp op,
    const mx::array& cotangent,
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& weights,
    const mx::array& offsets,
    Triple stride,
    Triple padding,
    SparseConvShape shape
);

mx::array make_sparse_conv_weight_grad(
    SparseMapOp op,
    const mx::array& feats,
    const mx::array& cotangent,
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& offsets,
    const mx::Shape& weight_shape,
    Triple stride,
    Triple padding,
    SparseConvShape shape
);

mx::array make_sparse_conv_features_input_grad(
    const mx::array& cotangent,
    const mx::array& weights,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    const mx::array& counts,
    SparseConvShape shape
);

mx::array make_sparse_conv_features_weight_grad(
    const mx::array& feats,
    const mx::array& cotangent,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    const mx::array& counts,
    const mx::Shape& weight_shape,
    SparseConvShape shape
);

class SparseConv final : public SparsePrimitive {
  public:
    SparseConv(
        mx::Stream stream,
        SparseMapOp op,
        SparseConvShape shape,
        Triple stride, // NOLINT(bugprone-easily-swappable-parameters)
        Triple padding
    )
        : SparsePrimitive(stream), op_(op), shape_(shape), stride_(stride),
          padding_(padding) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        exec::cpu::eval_sparse_conv(
            op_, shape_, stride_, padding_, stream(), inputs, outputs
        );
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        exec::metal::eval_sparse_conv(
            op_, shape_, stride_, padding_, stream(), inputs, outputs
        );
    }

    const char* name() const override { return "lattice::SparseConv"; }

    std::vector<mx::array>
    jvp(const std::vector<mx::array>& primals,
        const std::vector<mx::array>& tangents,
        const std::vector<int>& argnums) override {
        auto out = mx::zeros(
            mx::Shape{shape_.out_capacity, shape_.out_channels},
            primals[2].dtype(),
            stream()
        );
        auto has_tangent = false;
        for (int index = 0; index < int(argnums.size()); ++index) {
            if (argnums[index] == 2) {
                auto component = make_sparse_conv(
                                     op_,
                                     primals[0],
                                     primals[1],
                                     tangents[index],
                                     primals[3],
                                     primals[4],
                                     stride_,
                                     padding_
                )
                                     .feats;
                out =
                    has_tangent ? mx::add(out, component, stream()) : component;
                has_tangent = true;
            } else if (argnums[index] == 3) {
                auto component = make_sparse_conv(
                                     op_,
                                     primals[0],
                                     primals[1],
                                     primals[2],
                                     tangents[index],
                                     primals[4],
                                     stride_,
                                     padding_
                )
                                     .feats;
                out =
                    has_tangent ? mx::add(out, component, stream()) : component;
                has_tangent = true;
            }
        }
        return {
            out,
            mx::zeros(
                mx::Shape{shape_.out_capacity, 4}, primals[0].dtype(), stream()
            ),
            mx::zeros(mx::Shape{2}, mx::int32, stream()),
        };
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
            if (argnum == 2) {
                grads.push_back(make_sparse_conv_input_grad(
                    op_,
                    cotangents[0],
                    primals[0],
                    primals[1],
                    primals[3],
                    primals[4],
                    stride_,
                    padding_,
                    shape_
                ));
            } else if (argnum == 3) {
                grads.push_back(make_sparse_conv_weight_grad(
                    op_,
                    primals[2],
                    cotangents[0],
                    primals[0],
                    primals[1],
                    primals[4],
                    primals[3].shape(),
                    stride_,
                    padding_,
                    shape_
                ));
            } else {
                grads.push_back(mx::zeros_like(primals[argnum], stream()));
            }
        }
        return grads;
    }

    bool is_equivalent(const mx::Primitive& other) const override {
        if (typeid(other) != typeid(SparseConv)) {
            return false;
        }
        const auto& op = static_cast<const SparseConv&>(other);
        return op_ == op.op_ && stride_ == op.stride_ &&
               padding_ == op.padding_ &&
               shape_.in_capacity == op.shape_.in_capacity &&
               shape_.out_capacity == op.shape_.out_capacity &&
               shape_.n_kernels == op.shape_.n_kernels &&
               shape_.in_channels == op.shape_.in_channels &&
               shape_.out_channels == op.shape_.out_channels &&
               shape_.weight_layout == op.shape_.weight_layout &&
               shape_.kernel_x == op.shape_.kernel_x &&
               shape_.kernel_y == op.shape_.kernel_y &&
               shape_.kernel_z == op.shape_.kernel_z;
    }

  private:
    SparseMapOp op_;
    SparseConvShape shape_;
    Triple stride_;
    Triple padding_;
};

class SparseConvInputGrad : public SparsePrimitive {
  public:
    SparseConvInputGrad(
        mx::Stream stream,
        SparseMapOp op,
        SparseConvShape shape,
        Triple stride, // NOLINT(bugprone-easily-swappable-parameters)
        Triple padding
    )
        : SparsePrimitive(stream), op_(op), shape_(shape), stride_(stride),
          padding_(padding) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        exec::cpu::eval_sparse_conv_input_grad(
            op_, shape_, stride_, padding_, stream(), inputs, outputs
        );
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        exec::metal::eval_sparse_conv_input_grad(
            op_, shape_, stride_, padding_, stream(), inputs, outputs
        );
    }

    const char* name() const override { return "lattice::SparseConvInputGrad"; }

    bool is_equivalent(const mx::Primitive& other) const override {
        if (typeid(other) != typeid(SparseConvInputGrad)) {
            return false;
        }
        const auto& op = static_cast<const SparseConvInputGrad&>(other);
        return op_ == op.op_ && stride_ == op.stride_ &&
               padding_ == op.padding_ &&
               shape_.in_capacity == op.shape_.in_capacity &&
               shape_.out_capacity == op.shape_.out_capacity &&
               shape_.n_kernels == op.shape_.n_kernels &&
               shape_.in_channels == op.shape_.in_channels &&
               shape_.out_channels == op.shape_.out_channels &&
               shape_.weight_layout == op.shape_.weight_layout &&
               shape_.kernel_x == op.shape_.kernel_x &&
               shape_.kernel_y == op.shape_.kernel_y &&
               shape_.kernel_z == op.shape_.kernel_z;
    }

  protected:
    SparseMapOp op_;
    SparseConvShape shape_;
    Triple stride_;
    Triple padding_;
};

class SparseConvWeightGrad final : public SparseConvInputGrad {
  public:
    using SparseConvInputGrad::SparseConvInputGrad;

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        exec::cpu::eval_sparse_conv_weight_grad(
            op_, shape_, stride_, padding_, stream(), inputs, outputs
        );
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        exec::metal::eval_sparse_conv_weight_grad(
            op_, shape_, stride_, padding_, stream(), inputs, outputs
        );
    }

    const char* name() const override {
        return "lattice::SparseConvWeightGrad";
    }
};

class SparseConvFeatures final : public SparsePrimitive {
  public:
    SparseConvFeatures(mx::Stream stream, SparseConvShape shape)
        : SparsePrimitive(stream), shape_(shape) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        exec::cpu::eval_sparse_conv_features(shape_, stream(), inputs, outputs);
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        exec::metal::eval_sparse_conv_features(
            shape_, stream(), inputs, outputs
        );
    }

    const char* name() const override { return "lattice::SparseConvFeatures"; }

    std::vector<mx::array>
    jvp(const std::vector<mx::array>& primals,
        const std::vector<mx::array>& tangents,
        const std::vector<int>& argnums) override {
        auto out = mx::zeros(
            mx::Shape{shape_.out_capacity, shape_.out_channels},
            primals[0].dtype(),
            stream()
        );
        auto has_tangent = false;
        for (int index = 0; index < int(argnums.size()); ++index) {
            if (argnums[index] == 0) {
                auto component = make_sparse_conv_features(
                    tangents[index],
                    primals[1],
                    primals[2],
                    primals[3],
                    primals[4],
                    primals[5],
                    shape_.out_capacity,
                    shape_.n_kernels
                );
                out =
                    has_tangent ? mx::add(out, component, stream()) : component;
                has_tangent = true;
            } else if (argnums[index] == 1) {
                auto component = make_sparse_conv_features(
                    primals[0],
                    tangents[index],
                    primals[2],
                    primals[3],
                    primals[4],
                    primals[5],
                    shape_.out_capacity,
                    shape_.n_kernels
                );
                out =
                    has_tangent ? mx::add(out, component, stream()) : component;
                has_tangent = true;
            } else {
                continue;
            }
        }
        return {out};
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
                grads.push_back(make_sparse_conv_features_input_grad(
                    cotangents[0],
                    primals[1],
                    primals[2],
                    primals[3],
                    primals[4],
                    primals[5],
                    shape_
                ));
            } else if (argnum == 1) {
                grads.push_back(make_sparse_conv_features_weight_grad(
                    primals[0],
                    cotangents[0],
                    primals[2],
                    primals[3],
                    primals[4],
                    primals[5],
                    primals[1].shape(),
                    shape_
                ));
            } else {
                grads.push_back(mx::zeros_like(primals[argnum], stream()));
            }
        }
        return grads;
    }

    bool is_equivalent(const mx::Primitive& other) const override {
        if (typeid(other) != typeid(SparseConvFeatures)) {
            return false;
        }
        const auto& op = static_cast<const SparseConvFeatures&>(other);
        return shape_.in_capacity == op.shape_.in_capacity &&
               shape_.out_capacity == op.shape_.out_capacity &&
               shape_.n_kernels == op.shape_.n_kernels &&
               shape_.in_channels == op.shape_.in_channels &&
               shape_.out_channels == op.shape_.out_channels &&
               shape_.weight_layout == op.shape_.weight_layout &&
               shape_.kernel_x == op.shape_.kernel_x &&
               shape_.kernel_y == op.shape_.kernel_y &&
               shape_.kernel_z == op.shape_.kernel_z;
    }

  private:
    SparseConvShape shape_;
};

class SparseConvFeaturesInputGrad : public SparsePrimitive {
  public:
    SparseConvFeaturesInputGrad(mx::Stream stream, SparseConvShape shape)
        : SparsePrimitive(stream), shape_(shape) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        exec::cpu::eval_sparse_conv_features_input_grad(
            shape_, stream(), inputs, outputs
        );
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        exec::metal::eval_sparse_conv_features_input_grad(
            shape_, stream(), inputs, outputs
        );
    }

    const char* name() const override {
        return "lattice::SparseConvFeaturesInputGrad";
    }

    bool is_equivalent(const mx::Primitive& other) const override {
        if (typeid(other) != typeid(SparseConvFeaturesInputGrad)) {
            return false;
        }
        const auto& op = static_cast<const SparseConvFeaturesInputGrad&>(other);
        return shape_.in_capacity == op.shape_.in_capacity &&
               shape_.out_capacity == op.shape_.out_capacity &&
               shape_.n_kernels == op.shape_.n_kernels &&
               shape_.in_channels == op.shape_.in_channels &&
               shape_.out_channels == op.shape_.out_channels &&
               shape_.weight_layout == op.shape_.weight_layout &&
               shape_.kernel_x == op.shape_.kernel_x &&
               shape_.kernel_y == op.shape_.kernel_y &&
               shape_.kernel_z == op.shape_.kernel_z;
    }

  protected:
    SparseConvShape shape_;
};

class SparseConvFeaturesWeightGrad final : public SparseConvFeaturesInputGrad {
  public:
    using SparseConvFeaturesInputGrad::SparseConvFeaturesInputGrad;

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        exec::cpu::eval_sparse_conv_features_weight_grad(
            shape_, stream(), inputs, outputs
        );
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        exec::metal::eval_sparse_conv_features_weight_grad(
            shape_, stream(), inputs, outputs
        );
    }

    const char* name() const override {
        return "lattice::SparseConvFeaturesWeightGrad";
    }

    bool is_equivalent(const mx::Primitive& other) const override {
        if (typeid(other) != typeid(SparseConvFeaturesWeightGrad)) {
            return false;
        }
        const auto& op =
            static_cast<const SparseConvFeaturesWeightGrad&>(other);
        return shape_.in_capacity == op.shape_.in_capacity &&
               shape_.out_capacity == op.shape_.out_capacity &&
               shape_.n_kernels == op.shape_.n_kernels &&
               shape_.in_channels == op.shape_.in_channels &&
               shape_.out_channels == op.shape_.out_channels &&
               shape_.weight_layout == op.shape_.weight_layout &&
               shape_.kernel_x == op.shape_.kernel_x &&
               shape_.kernel_y == op.shape_.kernel_y &&
               shape_.kernel_z == op.shape_.kernel_z;
    }
};

} // namespace

NativeSparseTensorOutput make_sparse_conv(
    SparseMapOp op,
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& offsets,
    Triple stride,
    Triple padding
) {
    auto out_capacity = op == SparseMapOp::Forward
                            ? coords.shape(0)
                            : coords.shape(0) * offsets.shape(0);
    auto mapped_weight = weights.ndim() == 3;
    auto shape = SparseConvShape{
        coords.shape(0),
        out_capacity,
        offsets.shape(0),
        feats.shape(1),
        mapped_weight ? weights.shape(2) : weights.shape(0),
        mapped_weight ? 0 : 1,
        mapped_weight ? offsets.shape(0) : weights.shape(1),
        mapped_weight ? 1 : weights.shape(2),
        mapped_weight ? 1 : weights.shape(3),
    };
    auto stream =
        sparse_conv_stream(coords, active_rows, feats, weights, offsets);
    auto primitive =
        std::make_shared<SparseConv>(stream, op, shape, stride, padding);
    auto inputs = std::vector<mx::array>{
        coords,
        active_rows,
        feats,
        weights,
        offsets,
    };
    auto outputs = mx::array::make_arrays(
        {mx::Shape{out_capacity, shape.out_channels},
         mx::Shape{out_capacity, 4},
         mx::Shape{2}},
        {feats.dtype(), coords.dtype(), mx::int32},
        primitive,
        inputs
    );
    return {
        outputs[SparseOutCoords],
        outputs[SparseOutFeats],
        outputs[SparseCounts],
    };
}

mx::array make_sparse_conv_features(
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    const mx::array& counts,
    int out_capacity,
    int n_kernels
) {
    auto mapped_weight = weights.ndim() == 3;
    auto shape = SparseConvShape{
        feats.shape(0),
        out_capacity,
        n_kernels,
        feats.shape(1),
        mapped_weight ? weights.shape(2) : weights.shape(0),
        mapped_weight ? 0 : 1,
        mapped_weight ? n_kernels : weights.shape(1),
        mapped_weight ? 1 : weights.shape(2),
        mapped_weight ? 1 : weights.shape(3),
    };
    auto stream = sparse_conv_features_stream(
        feats, weights, in_rows, out_rows, kernel_ids, counts
    );
    auto primitive = std::make_shared<SparseConvFeatures>(stream, shape);
    auto inputs = std::vector<mx::array>{
        feats,
        weights,
        in_rows,
        out_rows,
        kernel_ids,
        counts,
    };
    return mx::array::make_arrays(
        {mx::Shape{out_capacity, shape.out_channels}},
        {feats.dtype()},
        primitive,
        inputs
    )[0];
}

namespace {

mx::array make_sparse_conv_input_grad(
    SparseMapOp op,
    const mx::array& cotangent,
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& weights,
    const mx::array& offsets,
    Triple stride,
    Triple padding,
    SparseConvShape shape
) {
    auto stream =
        sparse_conv_stream(coords, active_rows, cotangent, weights, offsets);
    auto primitive = std::make_shared<SparseConvInputGrad>(
        stream, op, shape, stride, padding
    );
    auto inputs = std::vector<mx::array>{
        cotangent,
        coords,
        active_rows,
        weights,
        offsets,
    };
    return mx::array::make_arrays(
        {mx::Shape{shape.in_capacity, shape.in_channels}},
        {cotangent.dtype()},
        primitive,
        inputs
    )[0];
}

mx::array make_sparse_conv_weight_grad(
    SparseMapOp op,
    const mx::array& feats,
    const mx::array& cotangent,
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& offsets,
    const mx::Shape& weight_shape,
    Triple stride,
    Triple padding,
    SparseConvShape shape
) {
    auto stream =
        sparse_conv_stream(coords, active_rows, feats, cotangent, offsets);
    auto primitive = std::make_shared<SparseConvWeightGrad>(
        stream, op, shape, stride, padding
    );
    auto inputs = std::vector<mx::array>{
        feats,
        cotangent,
        coords,
        active_rows,
        offsets,
    };
    return mx::array::make_arrays(
        {weight_shape}, {cotangent.dtype()}, primitive, inputs
    )[0];
}

mx::array make_sparse_conv_features_input_grad(
    const mx::array& cotangent,
    const mx::array& weights,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    const mx::array& counts,
    SparseConvShape shape
) {
    auto stream = sparse_conv_features_stream(
        cotangent, weights, in_rows, out_rows, kernel_ids, counts
    );
    auto primitive =
        std::make_shared<SparseConvFeaturesInputGrad>(stream, shape);
    auto inputs = std::vector<mx::array>{
        cotangent,
        weights,
        in_rows,
        out_rows,
        kernel_ids,
        counts,
    };
    return mx::array::make_arrays(
        {mx::Shape{shape.in_capacity, shape.in_channels}},
        {cotangent.dtype()},
        primitive,
        inputs
    )[0];
}

mx::array make_sparse_conv_features_weight_grad(
    const mx::array& feats,
    const mx::array& cotangent,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    const mx::array& counts,
    const mx::Shape& weight_shape,
    SparseConvShape shape
) {
    auto stream = sparse_conv_features_stream(
        feats, cotangent, in_rows, out_rows, kernel_ids, counts
    );
    auto primitive =
        std::make_shared<SparseConvFeaturesWeightGrad>(stream, shape);
    auto inputs = std::vector<mx::array>{
        feats,
        cotangent,
        in_rows,
        out_rows,
        kernel_ids,
        counts,
    };
    return mx::array::make_arrays(
        {weight_shape}, {cotangent.dtype()}, primitive, inputs
    )[0];
}

} // namespace

} // namespace mlx_lattice
