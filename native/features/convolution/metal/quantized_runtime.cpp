#include "features/convolution/metal/runtime.h"

#include "features/convolution/metal/direct/quantized_runtime.h"
#include "features/convolution/metal/sorted_igemm/quantized_runtime.h"
#include "features/convolution/metal/tensor_ops/quantized_forward/runtime.h"
#include "foundation/array_utils.h"

namespace mlx_lattice::backend::metal::conv {

void eval_quantized(
    QuantizedSparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
#ifdef _METAL_
    auto& out = outputs[0];
    allocate(out);
    if (quantized::tensor_ops::is_preferred(shape, stream, inputs)) {
        quantized::tensor_ops::encode(shape, stream, inputs, out);
        return;
    }
    if (quantized::sorted_igemm::is_preferred(shape, stream, inputs)) {
        quantized::sorted_igemm::encode(shape, stream, inputs, out);
        return;
    }
    quantized::direct::encode(shape, stream, inputs, out);
#else
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

} // namespace mlx_lattice::backend::metal::conv
