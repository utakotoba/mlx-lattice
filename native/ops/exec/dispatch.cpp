#include "ops/exec/dispatch.h"

#include <memory>
#include <vector>

#include "backends/cpu/exec/algorithms.h"
#include "backends/metal/exec/runtime.h"
#include "mlx/device.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/transforms.h"

namespace mlx_lattice {

mx::array dispatch_spmm_edges_input_grad(
    const mx::array& cotangent,
    const mx::array& weights,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    const mx::array& edge_count,
    SpmmEdgesShape shape
);

mx::array dispatch_spmm_edges_weight_grad(
    const mx::array& feats,
    const mx::array& cotangent,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    const mx::array& edge_count,
    SpmmEdgesShape shape
);

mx::array dispatch_pool_edges_grad(
    PoolReduceOp op,
    const mx::array& cotangent,
    const mx::array& feats,
    const mx::array& pooled,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& edge_count,
    PoolEdgesShape shape
);

mx::array dispatch_pool_max_edges_jvp(
    const mx::array& tangent,
    const mx::array& feats,
    const mx::array& pooled,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& edge_count,
    PoolEdgesShape shape
);

mx::array dispatch_sparse_conv_input_grad(
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

mx::array dispatch_sparse_conv_weight_grad(
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

mx::array dispatch_sparse_pool_grad(
    PoolReduceOp reduce,
    const mx::array& cotangent,
    const mx::array& feats,
    const mx::array& pooled,
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& offsets,
    Triple stride,
    Triple padding,
    SparsePoolShape shape
);

mx::array dispatch_sparse_pool_jvp(
    PoolReduceOp reduce,
    const mx::array& tangent,
    const mx::array& feats,
    const mx::array& pooled,
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& offsets,
    Triple stride,
    Triple padding,
    SparsePoolShape shape
);

namespace {

mx::Device device_for(
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    const mx::array& edge_count
) {
    if (exec::metal::supports(
            feats, weights, in_rows, out_rows, kernel_ids, edge_count
        )) {
        return mx::Device::gpu;
    }
    return mx::Device::cpu;
}

mx::Device device_for_pool(
    const mx::array& feats,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& edge_count
) {
    if (exec::metal::supports_pool(feats, in_rows, out_rows, edge_count)) {
        return mx::Device::gpu;
    }
    return mx::Device::cpu;
}

mx::Device device_for_spmm_input_grad(
    const mx::array& cotangent,
    const mx::array& weights,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    const mx::array& edge_count
) {
    if (exec::metal::supports_spmm_input_grad(
            cotangent, weights, in_rows, out_rows, kernel_ids, edge_count
        )) {
        return mx::Device::gpu;
    }
    return mx::Device::cpu;
}

mx::Device device_for_spmm_weight_grad(
    const mx::array& feats,
    const mx::array& cotangent,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    const mx::array& edge_count
) {
    if (exec::metal::supports_spmm_weight_grad(
            feats, cotangent, in_rows, out_rows, kernel_ids, edge_count
        )) {
        return mx::Device::gpu;
    }
    return mx::Device::cpu;
}

mx::Device device_for_pool_grad(
    const mx::array& cotangent,
    const mx::array& feats,
    const mx::array& pooled,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& edge_count
) {
    if (exec::metal::supports_pool_grad(
            cotangent, feats, pooled, in_rows, out_rows, edge_count
        )) {
        return mx::Device::gpu;
    }
    return mx::Device::cpu;
}

mx::Device device_for_sparse_conv(
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& offsets
) {
    if (exec::metal::supports_sparse_conv(
            coords, active_rows, feats, weights, offsets
        )) {
        return mx::Device::gpu;
    }
    return mx::Device::cpu;
}

mx::Device device_for_sparse_pool(
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& feats,
    const mx::array& offsets
) {
    if (exec::metal::supports_sparse_pool(
            coords, active_rows, feats, offsets
        )) {
        return mx::Device::gpu;
    }
    return mx::Device::cpu;
}

mx::Device device_for_sparse_conv_weight_grad(
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& feats,
    const mx::array& offsets
) {
    if (exec::metal::supports_sparse_pool(
            coords, active_rows, feats, offsets
        )) {
        return mx::Device::gpu;
    }
    return mx::Device::cpu;
}

class SpmmEdges final : public mx::Primitive {
  public:
    SpmmEdges(mx::Stream stream, SpmmEdgesShape shape)
        : mx::Primitive(stream), shape_(shape) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        mx::eval(inputs);
        exec::cpu::eval_spmm_edges(shape_, inputs, outputs);
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        mx::eval(inputs);
        exec::metal::eval_spmm_edges(shape_, stream(), inputs, outputs);
    }

    const char* name() const override { return "lattice::SpmmEdges"; }

    std::vector<mx::array>
    jvp(const std::vector<mx::array>& primals,
        const std::vector<mx::array>& tangents,
        const std::vector<int>& argnums) override {
        auto out = mx::zeros(
            mx::Shape{shape_.n_out_rows, shape_.out_channels},
            primals[0].dtype(),
            stream()
        );
        auto has_tangent = false;
        for (int index = 0; index < int(argnums.size()); ++index) {
            if (argnums[index] == 0) {
                auto component = dispatch_spmm_edges(
                    tangents[index],
                    primals[1],
                    primals[2],
                    primals[3],
                    primals[4],
                    primals[5],
                    shape_.n_out_rows
                );
                out =
                    has_tangent ? mx::add(out, component, stream()) : component;
                has_tangent = true;
            } else if (argnums[index] == 1) {
                auto component = dispatch_spmm_edges(
                    primals[0],
                    tangents[index],
                    primals[2],
                    primals[3],
                    primals[4],
                    primals[5],
                    shape_.n_out_rows
                );
                out =
                    has_tangent ? mx::add(out, component, stream()) : component;
                has_tangent = true;
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
                grads.push_back(dispatch_spmm_edges_input_grad(
                    cotangents[SparseOutFeats],
                    primals[1],
                    primals[2],
                    primals[3],
                    primals[4],
                    primals[5],
                    shape_
                ));
            } else if (argnum == 1) {
                grads.push_back(dispatch_spmm_edges_weight_grad(
                    primals[0],
                    cotangents[SparseOutFeats],
                    primals[2],
                    primals[3],
                    primals[4],
                    primals[5],
                    shape_
                ));
            } else {
                grads.push_back(mx::zeros_like(primals[argnum], stream()));
            }
        }
        return grads;
    }

    bool is_equivalent(const mx::Primitive& other) const override {
        const auto& op = static_cast<const SpmmEdges&>(other);
        return shape_.edge_count == op.shape_.edge_count &&
               shape_.in_channels == op.shape_.in_channels &&
               shape_.out_channels == op.shape_.out_channels &&
               shape_.n_in_rows == op.shape_.n_in_rows &&
               shape_.n_out_rows == op.shape_.n_out_rows &&
               shape_.n_kernels == op.shape_.n_kernels;
    }

  private:
    SpmmEdgesShape shape_;
};

class PoolEdges final : public mx::Primitive {
  public:
    PoolEdges(mx::Stream stream, PoolReduceOp op, PoolEdgesShape shape)
        : mx::Primitive(stream), op_(op), shape_(shape) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        mx::eval(inputs);
        exec::cpu::eval_pool_edges(op_, shape_, inputs, outputs);
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        mx::eval(inputs);
        exec::metal::eval_pool_edges(op_, shape_, stream(), inputs, outputs);
    }

    const char* name() const override { return "lattice::PoolEdges"; }

    std::vector<mx::array>
    jvp(const std::vector<mx::array>& primals,
        const std::vector<mx::array>& tangents,
        const std::vector<int>& argnums) override {
        for (int index = 0; index < int(argnums.size()); ++index) {
            if (argnums[index] != 0) {
                continue;
            }
            if (op_ == PoolReduceOp::Sum) {
                return {dispatch_pool_edges(
                    op_,
                    tangents[index],
                    primals[1],
                    primals[2],
                    primals[3],
                    shape_.n_out_rows
                )};
            }
            auto pooled = dispatch_pool_edges(
                op_,
                primals[0],
                primals[1],
                primals[2],
                primals[3],
                shape_.n_out_rows
            );
            return {dispatch_pool_max_edges_jvp(
                tangents[index],
                primals[0],
                pooled,
                primals[1],
                primals[2],
                primals[3],
                shape_
            )};
        }
        return {mx::zeros(
            mx::Shape{shape_.n_out_rows, shape_.channels},
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
                grads.push_back(dispatch_pool_edges_grad(
                    op_,
                    cotangents[SparseOutFeats],
                    primals[0],
                    outputs[0],
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
        const auto& op = static_cast<const PoolEdges&>(other);
        return op_ == op.op_ && shape_.edge_count == op.shape_.edge_count &&
               shape_.channels == op.shape_.channels &&
               shape_.n_in_rows == op.shape_.n_in_rows &&
               shape_.n_out_rows == op.shape_.n_out_rows;
    }

  private:
    PoolReduceOp op_;
    PoolEdgesShape shape_;
};

class SpmmEdgesInputGrad final : public mx::Primitive {
  public:
    SpmmEdgesInputGrad(mx::Stream stream, SpmmEdgesShape shape)
        : mx::Primitive(stream), shape_(shape) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        mx::eval(inputs);
        exec::cpu::eval_spmm_edges_input_grad(shape_, inputs, outputs);
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        mx::eval(inputs);
        exec::metal::eval_spmm_edges_input_grad(
            shape_, stream(), inputs, outputs
        );
    }

    const char* name() const override { return "lattice::SpmmEdgesInputGrad"; }

    bool is_equivalent(const mx::Primitive& other) const override {
        const auto& op = static_cast<const SpmmEdgesInputGrad&>(other);
        return shape_.edge_count == op.shape_.edge_count &&
               shape_.in_channels == op.shape_.in_channels &&
               shape_.out_channels == op.shape_.out_channels &&
               shape_.n_in_rows == op.shape_.n_in_rows &&
               shape_.n_out_rows == op.shape_.n_out_rows &&
               shape_.n_kernels == op.shape_.n_kernels;
    }

  private:
    SpmmEdgesShape shape_;
};

class SpmmEdgesWeightGrad final : public mx::Primitive {
  public:
    SpmmEdgesWeightGrad(mx::Stream stream, SpmmEdgesShape shape)
        : mx::Primitive(stream), shape_(shape) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        mx::eval(inputs);
        exec::cpu::eval_spmm_edges_weight_grad(shape_, inputs, outputs);
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        mx::eval(inputs);
        exec::metal::eval_spmm_edges_weight_grad(
            shape_, stream(), inputs, outputs
        );
    }

    const char* name() const override { return "lattice::SpmmEdgesWeightGrad"; }

    bool is_equivalent(const mx::Primitive& other) const override {
        const auto& op = static_cast<const SpmmEdgesWeightGrad&>(other);
        return shape_.edge_count == op.shape_.edge_count &&
               shape_.in_channels == op.shape_.in_channels &&
               shape_.out_channels == op.shape_.out_channels &&
               shape_.n_in_rows == op.shape_.n_in_rows &&
               shape_.n_out_rows == op.shape_.n_out_rows &&
               shape_.n_kernels == op.shape_.n_kernels;
    }

  private:
    SpmmEdgesShape shape_;
};

class PoolEdgesGrad final : public mx::Primitive {
  public:
    PoolEdgesGrad(mx::Stream stream, PoolReduceOp op, PoolEdgesShape shape)
        : mx::Primitive(stream), op_(op), shape_(shape) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        mx::eval(inputs);
        exec::cpu::eval_pool_edges_grad(op_, shape_, inputs, outputs);
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        mx::eval(inputs);
        exec::metal::eval_pool_edges_grad(
            op_, shape_, stream(), inputs, outputs
        );
    }

    const char* name() const override { return "lattice::PoolEdgesGrad"; }

    bool is_equivalent(const mx::Primitive& other) const override {
        const auto& op = static_cast<const PoolEdgesGrad&>(other);
        return op_ == op.op_ && shape_.edge_count == op.shape_.edge_count &&
               shape_.channels == op.shape_.channels &&
               shape_.n_in_rows == op.shape_.n_in_rows &&
               shape_.n_out_rows == op.shape_.n_out_rows;
    }

  private:
    PoolReduceOp op_;
    PoolEdgesShape shape_;
};

class PoolMaxEdgesJvp final : public mx::Primitive {
  public:
    PoolMaxEdgesJvp(mx::Stream stream, PoolEdgesShape shape)
        : mx::Primitive(stream), shape_(shape) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        mx::eval(inputs);
        exec::cpu::eval_pool_max_edges_jvp(shape_, inputs, outputs);
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        mx::eval(inputs);
        exec::metal::eval_pool_max_edges_jvp(shape_, stream(), inputs, outputs);
    }

    const char* name() const override { return "lattice::PoolMaxEdgesJvp"; }

    bool is_equivalent(const mx::Primitive& other) const override {
        const auto& op = static_cast<const PoolMaxEdgesJvp&>(other);
        return shape_.edge_count == op.shape_.edge_count &&
               shape_.channels == op.shape_.channels &&
               shape_.n_in_rows == op.shape_.n_in_rows &&
               shape_.n_out_rows == op.shape_.n_out_rows;
    }

  private:
    PoolEdgesShape shape_;
};

class SparseConv final : public mx::Primitive {
  public:
    SparseConv(
        mx::Stream stream,
        SparseMapOp op,
        SparseConvShape shape,
        Triple stride, // NOLINT(bugprone-easily-swappable-parameters)
        Triple padding
    )
        : mx::Primitive(stream), op_(op), shape_(shape), stride_(stride),
          padding_(padding) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        exec::cpu::eval_sparse_conv(
            op_, shape_, stride_, padding_, inputs, outputs
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
            mx::Shape{shape_.n_out_rows, shape_.out_channels},
            primals[2].dtype(),
            stream()
        );
        auto has_tangent = false;
        for (int index = 0; index < int(argnums.size()); ++index) {
            if (argnums[index] == 2) {
                auto component = dispatch_sparse_conv(
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
                auto component = dispatch_sparse_conv(
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
                mx::Shape{shape_.n_out_rows, 4}, primals[0].dtype(), stream()
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
                grads.push_back(dispatch_sparse_conv_input_grad(
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
                grads.push_back(dispatch_sparse_conv_weight_grad(
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
        const auto& op = static_cast<const SparseConv&>(other);
        return op_ == op.op_ && stride_ == op.stride_ &&
               padding_ == op.padding_ &&
               shape_.n_in_rows == op.shape_.n_in_rows &&
               shape_.n_out_rows == op.shape_.n_out_rows &&
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

class SparseConvInputGrad : public mx::Primitive {
  public:
    SparseConvInputGrad(
        mx::Stream stream,
        SparseMapOp op,
        SparseConvShape shape,
        Triple stride, // NOLINT(bugprone-easily-swappable-parameters)
        Triple padding
    )
        : mx::Primitive(stream), op_(op), shape_(shape), stride_(stride),
          padding_(padding) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        exec::cpu::eval_sparse_conv_input_grad(
            op_, shape_, stride_, padding_, inputs, outputs
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
        const auto& op = static_cast<const SparseConvInputGrad&>(other);
        return op_ == op.op_ && stride_ == op.stride_ &&
               padding_ == op.padding_ &&
               shape_.n_in_rows == op.shape_.n_in_rows &&
               shape_.n_out_rows == op.shape_.n_out_rows &&
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
            op_, shape_, stride_, padding_, inputs, outputs
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

class SparsePool final : public mx::Primitive {
  public:
    SparsePool(
        mx::Stream stream,
        PoolReduceOp reduce,
        SparsePoolShape shape,
        Triple stride, // NOLINT(bugprone-easily-swappable-parameters)
        Triple padding
    )
        : mx::Primitive(stream), reduce_(reduce), shape_(shape),
          stride_(stride), padding_(padding) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        exec::cpu::eval_sparse_pool(
            reduce_, shape_, stride_, padding_, inputs, outputs
        );
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        exec::metal::eval_sparse_pool(
            reduce_, shape_, stride_, padding_, stream(), inputs, outputs
        );
    }

    const char* name() const override { return "lattice::SparsePool"; }

    std::vector<mx::array>
    jvp(const std::vector<mx::array>& primals,
        const std::vector<mx::array>& tangents,
        const std::vector<int>& argnums) override {
        for (int index = 0; index < int(argnums.size()); ++index) {
            if (argnums[index] == 2) {
                auto outputs = dispatch_sparse_pool(
                    reduce_,
                    primals[0],
                    primals[1],
                    primals[2],
                    primals[3],
                    stride_,
                    padding_
                );
                auto tangent = dispatch_sparse_pool_jvp(
                    reduce_,
                    tangents[index],
                    primals[2],
                    outputs.feats,
                    primals[0],
                    primals[1],
                    primals[3],
                    stride_,
                    padding_,
                    shape_
                );
                return {
                    tangent,
                    mx::zeros(
                        mx::Shape{shape_.n_out_rows, 4},
                        primals[0].dtype(),
                        stream()
                    ),
                    mx::zeros(mx::Shape{2}, mx::int32, stream()),
                };
            }
        }
        return {
            mx::zeros(
                mx::Shape{shape_.n_out_rows, shape_.channels},
                primals[2].dtype(),
                stream()
            ),
            mx::zeros(
                mx::Shape{shape_.n_out_rows, 4}, primals[0].dtype(), stream()
            ),
            mx::zeros(mx::Shape{2}, mx::int32, stream()),
        };
    }

    std::vector<mx::array>
    vjp(const std::vector<mx::array>& primals,
        const std::vector<mx::array>& cotangents,
        const std::vector<int>& argnums,
        const std::vector<mx::array>& outputs) override {
        std::vector<mx::array> grads;
        grads.reserve(argnums.size());
        for (const auto argnum : argnums) {
            if (argnum == 2) {
                grads.push_back(dispatch_sparse_pool_grad(
                    reduce_,
                    cotangents[0],
                    primals[2],
                    outputs[SparseOutFeats],
                    primals[0],
                    primals[1],
                    primals[3],
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
        const auto& op = static_cast<const SparsePool&>(other);
        return reduce_ == op.reduce_ && stride_ == op.stride_ &&
               padding_ == op.padding_ &&
               shape_.n_in_rows == op.shape_.n_in_rows &&
               shape_.n_out_rows == op.shape_.n_out_rows &&
               shape_.n_kernels == op.shape_.n_kernels &&
               shape_.channels == op.shape_.channels;
    }

  private:
    PoolReduceOp reduce_;
    SparsePoolShape shape_;
    Triple stride_;
    Triple padding_;
};

class SparsePoolGrad : public mx::Primitive {
  public:
    SparsePoolGrad(
        mx::Stream stream,
        PoolReduceOp reduce,
        SparsePoolShape shape,
        Triple stride, // NOLINT(bugprone-easily-swappable-parameters)
        Triple padding
    )
        : mx::Primitive(stream), reduce_(reduce), shape_(shape),
          stride_(stride), padding_(padding) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        exec::cpu::eval_sparse_pool_grad(
            reduce_, shape_, stride_, padding_, inputs, outputs
        );
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        exec::metal::eval_sparse_pool_grad(
            reduce_, shape_, stride_, padding_, stream(), inputs, outputs
        );
    }

    const char* name() const override { return "lattice::SparsePoolGrad"; }

    bool is_equivalent(const mx::Primitive& other) const override {
        const auto& op = static_cast<const SparsePoolGrad&>(other);
        return reduce_ == op.reduce_ && stride_ == op.stride_ &&
               padding_ == op.padding_ &&
               shape_.n_in_rows == op.shape_.n_in_rows &&
               shape_.n_out_rows == op.shape_.n_out_rows &&
               shape_.n_kernels == op.shape_.n_kernels &&
               shape_.channels == op.shape_.channels;
    }

  protected:
    PoolReduceOp reduce_;
    SparsePoolShape shape_;
    Triple stride_;
    Triple padding_;
};

class SparsePoolJvp final : public SparsePoolGrad {
  public:
    using SparsePoolGrad::SparsePoolGrad;

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        exec::cpu::eval_sparse_pool_jvp(
            reduce_, shape_, stride_, padding_, inputs, outputs
        );
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        exec::metal::eval_sparse_pool_jvp(
            reduce_, shape_, stride_, padding_, stream(), inputs, outputs
        );
    }

    const char* name() const override { return "lattice::SparsePoolJvp"; }
};

} // namespace

mx::array dispatch_spmm_edges(
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    const mx::array& edge_count,
    int n_out_rows
) {
    auto shape = SpmmEdgesShape{
        in_rows.shape(0),
        feats.shape(1),
        weights.shape(2),
        feats.shape(0),
        n_out_rows,
        weights.shape(0),
    };
    auto device =
        device_for(feats, weights, in_rows, out_rows, kernel_ids, edge_count);
    auto stream = mx::default_stream(device);
    auto primitive = std::make_shared<SpmmEdges>(stream, shape);
    auto inputs = std::vector<mx::array>{
        mx::contiguous(feats, false, device),
        mx::contiguous(weights, false, device),
        mx::contiguous(in_rows, false, device),
        mx::contiguous(out_rows, false, device),
        mx::contiguous(kernel_ids, false, device),
        mx::contiguous(edge_count, false, device),
    };
    mx::eval(inputs);
    return mx::array::make_arrays(
        {mx::Shape{n_out_rows, weights.shape(2)}},
        {feats.dtype()},
        primitive,
        inputs
    )[0];
}

mx::array dispatch_pool_edges(
    PoolReduceOp op,
    const mx::array& feats,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& edge_count,
    int n_out_rows
) {
    auto shape = PoolEdgesShape{
        in_rows.shape(0),
        feats.shape(1),
        feats.shape(0),
        n_out_rows,
    };
    auto device = device_for_pool(feats, in_rows, out_rows, edge_count);
    auto stream = mx::default_stream(device);
    auto primitive = std::make_shared<PoolEdges>(stream, op, shape);
    auto inputs = std::vector<mx::array>{
        mx::contiguous(feats, false, device),
        mx::contiguous(in_rows, false, device),
        mx::contiguous(out_rows, false, device),
        mx::contiguous(edge_count, false, device),
    };
    mx::eval(inputs);
    return mx::array::make_arrays(
        {mx::Shape{n_out_rows, feats.shape(1)}},
        {feats.dtype()},
        primitive,
        inputs
    )[0];
}

mx::array dispatch_spmm_edges_input_grad(
    const mx::array& cotangent,
    const mx::array& weights,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    const mx::array& edge_count,
    SpmmEdgesShape shape
) {
    auto device = device_for_spmm_input_grad(
        cotangent, weights, in_rows, out_rows, kernel_ids, edge_count
    );
    auto stream = mx::default_stream(device);
    auto primitive = std::make_shared<SpmmEdgesInputGrad>(stream, shape);
    auto inputs = std::vector<mx::array>{
        mx::contiguous(cotangent, false, device),
        mx::contiguous(weights, false, device),
        mx::contiguous(in_rows, false, device),
        mx::contiguous(out_rows, false, device),
        mx::contiguous(kernel_ids, false, device),
        mx::contiguous(edge_count, false, device),
    };
    mx::eval(inputs);
    return mx::array::make_arrays(
        {mx::Shape{shape.n_in_rows, shape.in_channels}},
        {cotangent.dtype()},
        primitive,
        inputs
    )[0];
}

mx::array dispatch_spmm_edges_weight_grad(
    const mx::array& feats,
    const mx::array& cotangent,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    const mx::array& edge_count,
    SpmmEdgesShape shape
) {
    auto device = device_for_spmm_weight_grad(
        feats, cotangent, in_rows, out_rows, kernel_ids, edge_count
    );
    auto stream = mx::default_stream(device);
    auto primitive = std::make_shared<SpmmEdgesWeightGrad>(stream, shape);
    auto inputs = std::vector<mx::array>{
        mx::contiguous(feats, false, device),
        mx::contiguous(cotangent, false, device),
        mx::contiguous(in_rows, false, device),
        mx::contiguous(out_rows, false, device),
        mx::contiguous(kernel_ids, false, device),
        mx::contiguous(edge_count, false, device),
    };
    mx::eval(inputs);
    return mx::array::make_arrays(
        {mx::Shape{shape.n_kernels, shape.in_channels, shape.out_channels}},
        {cotangent.dtype()},
        primitive,
        inputs
    )[0];
}

mx::array dispatch_pool_edges_grad(
    PoolReduceOp op,
    const mx::array& cotangent,
    const mx::array& feats,
    const mx::array& pooled,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& edge_count,
    PoolEdgesShape shape
) {
    auto device = device_for_pool_grad(
        cotangent, feats, pooled, in_rows, out_rows, edge_count
    );
    auto stream = mx::default_stream(device);
    auto primitive = std::make_shared<PoolEdgesGrad>(stream, op, shape);
    auto inputs = std::vector<mx::array>{
        mx::contiguous(cotangent, false, device),
        mx::contiguous(feats, false, device),
        mx::contiguous(pooled, false, device),
        mx::contiguous(in_rows, false, device),
        mx::contiguous(out_rows, false, device),
        mx::contiguous(edge_count, false, device),
    };
    mx::eval(inputs);
    return mx::array::make_arrays(
        {mx::Shape{shape.n_in_rows, shape.channels}},
        {cotangent.dtype()},
        primitive,
        inputs
    )[0];
}

mx::array dispatch_pool_max_edges_jvp(
    const mx::array& tangent,
    const mx::array& feats,
    const mx::array& pooled,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& edge_count,
    PoolEdgesShape shape
) {
    auto device = device_for_pool_grad(
        tangent, feats, pooled, in_rows, out_rows, edge_count
    );
    auto stream = mx::default_stream(device);
    auto primitive = std::make_shared<PoolMaxEdgesJvp>(stream, shape);
    auto inputs = std::vector<mx::array>{
        mx::contiguous(tangent, false, device),
        mx::contiguous(feats, false, device),
        mx::contiguous(pooled, false, device),
        mx::contiguous(in_rows, false, device),
        mx::contiguous(out_rows, false, device),
        mx::contiguous(edge_count, false, device),
    };
    mx::eval(inputs);
    return mx::array::make_arrays(
        {mx::Shape{shape.n_out_rows, shape.channels}},
        {tangent.dtype()},
        primitive,
        inputs
    )[0];
}

NativeSparseTensorOutput dispatch_sparse_conv(
    SparseMapOp op,
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& offsets,
    Triple stride,
    Triple padding
) {
    auto n_out_rows = op == SparseMapOp::Forward
                          ? coords.shape(0)
                          : coords.shape(0) * offsets.shape(0);
    auto mapped_weight = weights.ndim() == 3;
    auto shape = SparseConvShape{
        coords.shape(0),
        n_out_rows,
        offsets.shape(0),
        feats.shape(1),
        mapped_weight ? weights.shape(2) : weights.shape(0),
        mapped_weight ? 0 : 1,
        mapped_weight ? offsets.shape(0) : weights.shape(1),
        mapped_weight ? 1 : weights.shape(2),
        mapped_weight ? 1 : weights.shape(3),
    };
    auto device =
        device_for_sparse_conv(coords, active_rows, feats, weights, offsets);
    auto stream = mx::default_stream(device);
    auto primitive =
        std::make_shared<SparseConv>(stream, op, shape, stride, padding);
    auto inputs = std::vector<mx::array>{
        coords,
        active_rows,
        feats,
        weights,
        offsets,
    };
    mx::eval(inputs);
    auto outputs = mx::array::make_arrays(
        {mx::Shape{n_out_rows, weights.shape(2)},
         mx::Shape{n_out_rows, 4},
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

NativeSparseTensorOutput dispatch_sparse_pool(
    PoolReduceOp reduce,
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& feats,
    const mx::array& offsets,
    Triple stride,
    Triple padding
) {
    auto n_out_rows = coords.shape(0);
    auto shape = SparsePoolShape{
        coords.shape(0),
        n_out_rows,
        offsets.shape(0),
        feats.shape(1),
    };
    auto device = device_for_sparse_pool(coords, active_rows, feats, offsets);
    auto stream = mx::default_stream(device);
    auto primitive =
        std::make_shared<SparsePool>(stream, reduce, shape, stride, padding);
    auto inputs = std::vector<mx::array>{
        coords,
        active_rows,
        feats,
        offsets,
    };
    mx::eval(inputs);
    auto outputs = mx::array::make_arrays(
        {mx::Shape{n_out_rows, feats.shape(1)},
         mx::Shape{n_out_rows, 4},
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

mx::array dispatch_sparse_conv_input_grad(
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
    auto device = device_for_sparse_conv(
        coords, active_rows, cotangent, weights, offsets
    );
    auto stream = mx::default_stream(device);
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
    mx::eval(inputs);
    return mx::array::make_arrays(
        {mx::Shape{shape.n_in_rows, shape.in_channels}},
        {cotangent.dtype()},
        primitive,
        inputs
    )[0];
}

mx::array dispatch_sparse_conv_weight_grad(
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
    auto device =
        device_for_sparse_conv_weight_grad(coords, active_rows, feats, offsets);
    auto stream = mx::default_stream(device);
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
    mx::eval(inputs);
    return mx::array::make_arrays(
        {weight_shape}, {cotangent.dtype()}, primitive, inputs
    )[0];
}

mx::array dispatch_sparse_pool_grad(
    PoolReduceOp reduce,
    const mx::array& cotangent,
    const mx::array& feats,
    const mx::array& pooled,
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& offsets,
    Triple stride,
    Triple padding,
    SparsePoolShape shape
) {
    (void)pooled;
    auto device = device_for_sparse_pool(coords, active_rows, feats, offsets);
    auto stream = mx::default_stream(device);
    auto primitive = std::make_shared<SparsePoolGrad>(
        stream, reduce, shape, stride, padding
    );
    auto inputs = std::vector<mx::array>{
        cotangent,
        feats,
        pooled,
        coords,
        active_rows,
        offsets,
    };
    mx::eval(inputs);
    return mx::array::make_arrays(
        {mx::Shape{shape.n_in_rows, shape.channels}},
        {cotangent.dtype()},
        primitive,
        inputs
    )[0];
}

mx::array dispatch_sparse_pool_jvp(
    PoolReduceOp reduce,
    const mx::array& tangent,
    const mx::array& feats,
    const mx::array& pooled,
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& offsets,
    Triple stride,
    Triple padding,
    SparsePoolShape shape
) {
    auto device = device_for_sparse_pool(coords, active_rows, feats, offsets);
    auto stream = mx::default_stream(device);
    auto primitive =
        std::make_shared<SparsePoolJvp>(stream, reduce, shape, stride, padding);
    auto inputs = std::vector<mx::array>{
        tangent,
        feats,
        pooled,
        coords,
        active_rows,
        offsets,
    };
    mx::eval(inputs);
    return mx::array::make_arrays(
        {mx::Shape{shape.n_out_rows, shape.channels}},
        {tangent.dtype()},
        primitive,
        inputs
    )[0];
}

} // namespace mlx_lattice
