#include "features/pooling/metal/runtime.h"

#include <stdexcept>
#include <vector>

namespace {
[[noreturn]] void unavailable() {
    throw std::runtime_error("Metal support is not available.");
}
} // namespace

namespace mlx_lattice::backend::metal::pool {

void eval(
    PoolReduceOp reduce,
    SparsePoolShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    (void)reduce;
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    unavailable();
}

void eval_grad(
    PoolReduceOp reduce,
    SparsePoolShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    (void)reduce;
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    unavailable();
}

void eval_jvp(
    PoolReduceOp reduce,
    SparsePoolShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    (void)reduce;
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    unavailable();
}

} // namespace mlx_lattice::backend::metal::pool
