#include "features/convolution/metal/runtime.h"

#include <stdexcept>
#include <vector>

namespace {
[[noreturn]] void unavailable() {
    throw std::runtime_error("Metal support is not available.");
}
} // namespace

namespace mlx_lattice::backend::metal::conv {

void eval(
    SparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    unavailable();
}

void eval_sorted_implicit_gemm(
    SparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    unavailable();
}

void eval_sorted_direct_reference(
    SparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    unavailable();
}

void eval_input_grad(
    SparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    unavailable();
}

void eval_weight_grad(
    SparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    unavailable();
}

} // namespace mlx_lattice::backend::metal::conv
