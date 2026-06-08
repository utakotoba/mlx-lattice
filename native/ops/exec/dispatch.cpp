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

namespace {

mx::Device device_for(
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids
) {
    if (exec::metal::supports(feats, weights, in_rows, out_rows, kernel_ids)) {
        return mx::Device::gpu;
    }
    return mx::Device::cpu;
}

mx::Device device_for_pool(
    const mx::array& feats,
    const mx::array& in_rows,
    const mx::array& out_rows
) {
    if (exec::metal::supports_pool(feats, in_rows, out_rows)) {
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
        exec::cpu::eval_spmm_edges(shape_, inputs, outputs);
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        exec::metal::eval_spmm_edges(shape_, stream(), inputs, outputs);
    }

    const char* name() const override { return "lattice::SpmmEdges"; }

    bool is_equivalent(const mx::Primitive& other) const override {
        const auto& op = static_cast<const SpmmEdges&>(other);
        return shape_.edge_count == op.shape_.edge_count &&
               shape_.in_channels == op.shape_.in_channels &&
               shape_.out_channels == op.shape_.out_channels &&
               shape_.n_out_rows == op.shape_.n_out_rows;
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
        exec::cpu::eval_pool_edges(op_, shape_, inputs, outputs);
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        exec::metal::eval_pool_edges(op_, shape_, stream(), inputs, outputs);
    }

    const char* name() const override { return "lattice::PoolEdges"; }

    bool is_equivalent(const mx::Primitive& other) const override {
        const auto& op = static_cast<const PoolEdges&>(other);
        return op_ == op.op_ && shape_.edge_count == op.shape_.edge_count &&
               shape_.channels == op.shape_.channels &&
               shape_.n_out_rows == op.shape_.n_out_rows;
    }

  private:
    PoolReduceOp op_;
    PoolEdgesShape shape_;
};

} // namespace

mx::array dispatch_spmm_edges(
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    int n_out_rows
) {
    auto shape = SpmmEdgesShape{
        in_rows.shape(0),
        feats.shape(1),
        weights.shape(2),
        n_out_rows,
    };
    auto device = device_for(feats, weights, in_rows, out_rows, kernel_ids);
    auto stream = mx::default_stream(device);
    auto primitive = std::make_shared<SpmmEdges>(stream, shape);
    auto inputs = std::vector<mx::array>{
        mx::contiguous(feats, false, device),
        mx::contiguous(weights, false, device),
        mx::contiguous(in_rows, false, device),
        mx::contiguous(out_rows, false, device),
        mx::contiguous(kernel_ids, false, device),
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
    int n_out_rows
) {
    auto shape = PoolEdgesShape{
        in_rows.shape(0),
        feats.shape(1),
        n_out_rows,
    };
    auto device = device_for_pool(feats, in_rows, out_rows);
    auto stream = mx::default_stream(device);
    auto primitive = std::make_shared<PoolEdges>(stream, op, shape);
    auto inputs = std::vector<mx::array>{
        mx::contiguous(feats, false, device),
        mx::contiguous(in_rows, false, device),
        mx::contiguous(out_rows, false, device),
    };
    mx::eval(inputs);
    return mx::array::make_arrays(
        {mx::Shape{n_out_rows, feats.shape(1)}},
        {feats.dtype()},
        primitive,
        inputs
    )[0];
}

} // namespace mlx_lattice
