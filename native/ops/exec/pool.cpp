#include "ops/exec/factories.h"

#include <memory>
#include <typeinfo>
#include <vector>

#include "backends/cpu/pool/algorithms.h"
#include "backends/metal/pool/runtime.h"
#include "mlx/ops.h"
#include "ops/coords/factories.h"
#include "ops/exec/primitive.h"
#include "ops/exec/streams.h"

namespace mlx_lattice {

namespace {

mx::array make_sparse_pool_features_grad(
    PoolReduceOp reduce,
    const mx::array& cotangent,
    const mx::array& feats,
    const mx::array& pooled,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    const mx::array& row_offsets,
    const mx::array& counts,
    const mx::array& in_row_offsets,
    const mx::array& in_edge_ids,
    SparsePoolShape shape
);

mx::array make_sparse_pool_features_jvp(
    PoolReduceOp reduce,
    const mx::array& tangent,
    const mx::array& feats,
    const mx::array& pooled,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    const mx::array& row_offsets,
    const mx::array& counts,
    SparsePoolShape shape
);

class SparsePoolFeatures final : public SparsePrimitive {
  public:
    SparsePoolFeatures(
        mx::Stream stream,
        PoolReduceOp reduce,
        SparsePoolShape shape
    )
        : SparsePrimitive(stream), reduce_(reduce), shape_(shape) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        backend::cpu::pool::eval(reduce_, shape_, stream(), inputs, outputs);
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        backend::metal::pool::eval(reduce_, shape_, stream(), inputs, outputs);
    }

    const char* name() const override { return "lattice::SparsePoolFeatures"; }

    std::vector<mx::array>
    jvp(const std::vector<mx::array>& primals,
        const std::vector<mx::array>& tangents,
        const std::vector<int>& argnums) override {
        for (int index = 0; index < int(argnums.size()); ++index) {
            if (argnums[index] == 0) {
                auto pooled = make_sparse_pool_features(
                    reduce_,
                    primals[0],
                    primals[1],
                    primals[2],
                    primals[3],
                    primals[4],
                    primals[5],
                    shape_.out_capacity,
                    shape_.n_kernels,
                    shape_.input_exclusive ? PoolInputLayout::Exclusive
                                           : PoolInputLayout::Overlap
                );
                return {make_sparse_pool_features_jvp(
                    reduce_,
                    tangents[index],
                    primals[0],
                    pooled,
                    primals[1],
                    primals[2],
                    primals[3],
                    primals[4],
                    primals[5],
                    shape_
                )};
            }
        }
        return {mx::zeros(
            mx::Shape{shape_.out_capacity, shape_.channels},
            primals[0].dtype(),
            stream()
        )};
    }

    std::vector<mx::array>
    vjp(const std::vector<mx::array>& primals,
        const std::vector<mx::array>& cotangents,
        const std::vector<int>& argnums,
        const std::vector<mx::array>& outputs) override {
        std::vector<mx::array> grads;
        grads.reserve(argnums.size());
        for (const auto argnum : argnums) {
            if (argnum == 0) {
                if (shape_.input_exclusive) {
                    auto view = make_relation_direct_view(
                        primals[1], primals[5], shape_.in_capacity
                    );
                    grads.push_back(make_sparse_pool_features_grad(
                        reduce_,
                        cotangents[0],
                        primals[0],
                        outputs[0],
                        primals[1],
                        primals[2],
                        primals[3],
                        primals[4],
                        primals[5],
                        primals[4],
                        view.edge_ids,
                        shape_
                    ));
                    continue;
                }
                auto view = make_relation_grouped_view(
                    primals[1], primals[5], shape_.in_capacity
                );
                grads.push_back(make_sparse_pool_features_grad(
                    reduce_,
                    cotangents[0],
                    primals[0],
                    outputs[0],
                    primals[1],
                    primals[2],
                    primals[3],
                    primals[4],
                    primals[5],
                    view.row_offsets,
                    view.edge_ids,
                    shape_
                ));
            } else {
                grads.push_back(mx::zeros_like(primals[argnum], stream()));
            }
        }
        return grads;
    }

    bool is_equivalent(const mx::Primitive& other) const override {
        if (typeid(other) != typeid(SparsePoolFeatures)) {
            return false;
        }
        const auto& op = static_cast<const SparsePoolFeatures&>(other);
        return reduce_ == op.reduce_ &&
               shape_.in_capacity == op.shape_.in_capacity &&
               shape_.out_capacity == op.shape_.out_capacity &&
               shape_.n_kernels == op.shape_.n_kernels &&
               shape_.channels == op.shape_.channels &&
               shape_.input_exclusive == op.shape_.input_exclusive;
    }

  private:
    PoolReduceOp reduce_;
    SparsePoolShape shape_;
};

class SparsePoolFeaturesGrad : public SparsePrimitive {
  public:
    SparsePoolFeaturesGrad(
        mx::Stream stream,
        PoolReduceOp reduce,
        SparsePoolShape shape
    )
        : SparsePrimitive(stream), reduce_(reduce), shape_(shape) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        backend::cpu::pool::eval_grad(
            reduce_, shape_, stream(), inputs, outputs
        );
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        backend::metal::pool::eval_grad(
            reduce_, shape_, stream(), inputs, outputs
        );
    }

    const char* name() const override {
        return "lattice::SparsePoolFeaturesGrad";
    }

    bool is_equivalent(const mx::Primitive& other) const override {
        if (typeid(other) != typeid(SparsePoolFeaturesGrad)) {
            return false;
        }
        const auto& op = static_cast<const SparsePoolFeaturesGrad&>(other);
        return reduce_ == op.reduce_ &&
               shape_.in_capacity == op.shape_.in_capacity &&
               shape_.out_capacity == op.shape_.out_capacity &&
               shape_.n_kernels == op.shape_.n_kernels &&
               shape_.channels == op.shape_.channels &&
               shape_.input_exclusive == op.shape_.input_exclusive;
    }

  protected:
    PoolReduceOp reduce_;
    SparsePoolShape shape_;
};

class SparsePoolFeaturesJvp final : public SparsePoolFeaturesGrad {
  public:
    using SparsePoolFeaturesGrad::SparsePoolFeaturesGrad;

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        backend::cpu::pool::eval_jvp(
            reduce_, shape_, stream(), inputs, outputs
        );
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        backend::metal::pool::eval_jvp(
            reduce_, shape_, stream(), inputs, outputs
        );
    }

    const char* name() const override {
        return "lattice::SparsePoolFeaturesJvp";
    }
};

mx::Stream pool_stream(
    const mx::array& feats,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    const mx::array& row_offsets,
    const mx::array& counts
) {
    return sparse_pool_features_stream(
        feats, in_rows, out_rows, kernel_ids, row_offsets, counts
    );
}

} // namespace

mx::array make_sparse_pool_features(
    PoolReduceOp reduce,
    const mx::array& feats,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    const mx::array& row_offsets,
    const mx::array& counts,
    int out_capacity,
    int n_kernels,
    PoolInputLayout input_layout
) {
    auto shape = SparsePoolShape{
        feats.shape(0),
        out_capacity,
        n_kernels,
        feats.shape(1),
        input_layout == PoolInputLayout::Exclusive,
    };
    auto stream =
        pool_stream(feats, in_rows, out_rows, kernel_ids, row_offsets, counts);
    auto primitive =
        std::make_shared<SparsePoolFeatures>(stream, reduce, shape);
    auto inputs = std::vector<mx::array>{
        feats,
        in_rows,
        out_rows,
        kernel_ids,
        row_offsets,
        counts,
    };
    return mx::array::make_arrays(
        {mx::Shape{out_capacity, feats.shape(1)}},
        {feats.dtype()},
        primitive,
        inputs
    )[0];
}

namespace {

mx::array make_sparse_pool_features_grad(
    PoolReduceOp reduce,
    const mx::array& cotangent,
    const mx::array& feats,
    const mx::array& pooled,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    const mx::array& row_offsets,
    const mx::array& counts,
    const mx::array& in_row_offsets,
    const mx::array& in_edge_ids,
    SparsePoolShape shape
) {
    auto stream = pool_stream(
        cotangent, in_rows, out_rows, kernel_ids, row_offsets, counts
    );
    auto primitive =
        std::make_shared<SparsePoolFeaturesGrad>(stream, reduce, shape);
    auto inputs = std::vector<mx::array>{
        cotangent,
        feats,
        pooled,
        in_rows,
        out_rows,
        kernel_ids,
        row_offsets,
        counts,
        in_row_offsets,
        in_edge_ids,
    };
    return mx::array::make_arrays(
        {mx::Shape{shape.in_capacity, shape.channels}},
        {cotangent.dtype()},
        primitive,
        inputs
    )[0];
}

mx::array make_sparse_pool_features_jvp(
    PoolReduceOp reduce,
    const mx::array& tangent,
    const mx::array& feats,
    const mx::array& pooled,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    const mx::array& row_offsets,
    const mx::array& counts,
    SparsePoolShape shape
) {
    auto stream = pool_stream(
        tangent, in_rows, out_rows, kernel_ids, row_offsets, counts
    );
    auto primitive =
        std::make_shared<SparsePoolFeaturesJvp>(stream, reduce, shape);
    auto inputs = std::vector<mx::array>{
        tangent,
        feats,
        pooled,
        in_rows,
        out_rows,
        kernel_ids,
        row_offsets,
        counts,
    };
    return mx::array::make_arrays(
        {mx::Shape{shape.out_capacity, shape.channels}},
        {tangent.dtype()},
        primitive,
        inputs
    )[0];
}

} // namespace

} // namespace mlx_lattice
