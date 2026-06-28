#pragma once

#include <vector>

#include "features/convolution/contract.h"
#include "mlx/stream.h"

namespace mlx_lattice::backend::metal::tensor_ops::conv::quantized_igemm {

bool is_preferred(QuantizedSparseConvShape shape, const mx::Stream& stream);

void encode(
    QuantizedSparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    mx::array& out
);

} // namespace mlx_lattice::backend::metal::tensor_ops::conv::quantized_igemm
