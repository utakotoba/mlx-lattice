#include "ops/exec/streams.h"

#include <stdexcept>

#include "mlx/device.h"

namespace mlx_lattice {

namespace {

bool is_gpu_device(const mx::Device& device) {
    return device == mx::Device(mx::Device::gpu);
}

bool is_conv_feature_dtype(mx::Dtype dtype) {
    return dtype == mx::float32 || dtype == mx::float16;
}

mx::Device sparse_exec_device() {
    auto device = mx::default_device();
    return is_gpu_device(device) ? mx::Device::gpu : mx::Device::cpu;
}

mx::Stream sparse_exec_stream(const mx::Device& device) {
    return mx::default_stream(device);
}

} // namespace

mx::Stream sparse_conv_features_stream(
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    const mx::array& counts,
    const mx::array& row_offsets
) {
    auto device = sparse_exec_device();
    if (is_gpu_device(device) &&
        (!is_conv_feature_dtype(feats.dtype()) ||
         weights.dtype() != feats.dtype() || in_rows.dtype() != mx::int32 ||
         out_rows.dtype() != mx::int32 || kernel_ids.dtype() != mx::int32 ||
         counts.dtype() != mx::int32 || row_offsets.dtype() != mx::int32)) {
        throw std::invalid_argument(
            "Metal sparse convolution features require int32 relation arrays "
            "and matching float32 or float16 features/weights."
        );
    }
    return sparse_exec_stream(device);
}

mx::Stream sparse_conv_implicit_gemm_stream(
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& out_in_map
) {
    auto device = sparse_exec_device();
    if (is_gpu_device(device) &&
        (!is_conv_feature_dtype(feats.dtype()) ||
         weights.dtype() != feats.dtype() || out_in_map.dtype() != mx::int32)) {
        throw std::invalid_argument(
            "Metal implicit GEMM sparse convolution requires int32 relation "
            "map and matching float32 or float16 features/weights."
        );
    }
    return sparse_exec_stream(device);
}

mx::Stream sparse_conv_grad_stream(
    const mx::array& lhs,
    const mx::array& rhs,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    const mx::array& counts,
    const mx::array& row_offsets,
    const mx::array& grouped_row_offsets,
    const mx::array& grouped_edge_ids
) {
    auto device = sparse_exec_device();
    if (is_gpu_device(device) &&
        (!is_conv_feature_dtype(lhs.dtype()) || rhs.dtype() != lhs.dtype() ||
         in_rows.dtype() != mx::int32 || out_rows.dtype() != mx::int32 ||
         kernel_ids.dtype() != mx::int32 || counts.dtype() != mx::int32 ||
         row_offsets.dtype() != mx::int32 ||
         grouped_row_offsets.dtype() != mx::int32 ||
         grouped_edge_ids.dtype() != mx::int32)) {
        throw std::invalid_argument(
            "Metal sparse convolution gradients require int32 plan arrays and "
            "matching float32 or float16 features/weights."
        );
    }
    return sparse_exec_stream(device);
}

mx::Stream sparse_pool_features_stream(
    const mx::array& feats,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    const mx::array& row_offsets,
    const mx::array& counts
) {
    auto device = sparse_exec_device();
    if (is_gpu_device(device) &&
        (feats.dtype() != mx::float32 || in_rows.dtype() != mx::int32 ||
         out_rows.dtype() != mx::int32 || kernel_ids.dtype() != mx::int32 ||
         row_offsets.dtype() != mx::int32 || counts.dtype() != mx::int32)) {
        throw std::invalid_argument(
            "Metal sparse pooling requires int32 relation arrays and float32 "
            "features."
        );
    }
    return sparse_exec_stream(device);
}

} // namespace mlx_lattice
