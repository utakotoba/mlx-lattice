#include "features/convolution/metal/tensor_ops/sorted_igemm/runtime.h"

#include <stdexcept>

#include "foundation/array_utils.h"
#include "platform/metal/runtime_utils.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace mlx_lattice::backend::metal::tensor_ops::conv::sorted_igemm {
namespace {

int stride_at(const mx::array& array, int dim) {
    return static_cast<int>(array.strides(dim));
}

} // namespace

bool supports(SparseConvShape shape, const std::vector<mx::array>& inputs) {
    auto supported_channels =
        shape.in_channels == shape.out_channels &&
        (shape.in_channels == 32 || shape.in_channels == 64);
    return inputs[0].dtype() == mx::float16 &&
           inputs[1].dtype() == mx::float16 && inputs[2].dtype() == mx::int32 &&
           inputs[3].dtype() == mx::int32 && inputs[4].dtype() == mx::int32 &&
           inputs[5].dtype() == mx::int32 && shape.weight_layout == 0 &&
           supported_channels && shape.n_kernels == 27 &&
           inputs[1].ndim() == 3 && stride_at(inputs[0], 1) == 1 &&
           stride_at(inputs[1], 2) == 1 &&
           stride_at(inputs[1], 1) == shape.out_channels &&
           stride_at(inputs[1], 0) == shape.in_channels * shape.out_channels;
}

void encode(
    SparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    mx::array& out
) {
#ifdef _METAL_
    if (!supports(shape, inputs)) {
        throw std::invalid_argument(
            "sorted implicit GEMM conv supports only contiguous "
            "fp16 mapped weights with K=27 and Cin=Cout in {32, 64}."
        );
    }

    auto& device = mx::metal::device(stream.device);
    auto library =
        device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
    auto& encoder = mx::metal::get_command_encoder(stream);
    auto kernel = device.get_kernel(
        shape.in_channels == 32
            ? "row_stationary_tensor_coop_devicew_rowfill4x2_kvmap_f16acc_"
              "sorted_c32_m64"
            : "row_stationary_tensor_coop_devicew_full64_kvmap_f16acc_sorted_"
              "c64_m64",
        library
    );
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[1], 1);
    encoder.set_input_array(inputs[3], 2);
    encoder.set_input_array(inputs[4], 3);
    encoder.set_input_array(inputs[5], 4);
    encoder.set_output_array(out, 5);
    set_bytes_range(encoder, 6, shape.out_capacity, shape.store_sorted);
    auto groups = static_cast<size_t>((shape.out_capacity + 63) / 64);
    encoder.dispatch_threadgroups(
        MTL::Size(groups, 1, 1), MTL::Size(128, 1, 1)
    );
#else
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)out;
    throw std::runtime_error("Metal support is not available.");
#endif
}

} // namespace mlx_lattice::backend::metal::tensor_ops::conv::sorted_igemm
