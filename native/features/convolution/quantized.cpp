#include "features/convolution/api.h"
#include "features/convolution/factories.h"

#include <memory>
#include <stdexcept>
#include <typeinfo>
#include <vector>

#include "features/convolution/cpu/algorithms.h"
#include "features/convolution/metal/runtime.h"
#include "features/convolution/streams.h"
#include "foundation/primitive.h"
#include "mlx/device.h"
#include "mlx/ops.h"

namespace mlx_lattice {
namespace {

class SparseQuantizedConvFeatures final : public SparsePrimitive {
  public:
    SparseQuantizedConvFeatures(
        mx::Stream stream,
        QuantizedSparseConvShape shape
    )
        : SparsePrimitive(stream), shape_(shape) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        backend::cpu::conv::eval_quantized(shape_, stream(), inputs, outputs);
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        backend::metal::conv::eval_quantized(shape_, stream(), inputs, outputs);
    }

    const char* name() const override {
        return "lattice::SparseQuantizedConvFeatures";
    }

    bool is_equivalent(const mx::Primitive& other) const override {
        if (typeid(other) != typeid(SparseQuantizedConvFeatures)) {
            return false;
        }
        const auto& op = static_cast<const SparseQuantizedConvFeatures&>(other);
        return shape_ == op.shape_;
    }

  private:
    QuantizedSparseConvShape shape_;
};

void validate_quantized_arrays(
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& scales,
    const mx::array& biases,
    QuantizedSparseConvShape shape
) {
    if (feats.ndim() != 2 ||
        (feats.dtype() != mx::float16 && feats.dtype() != mx::float32)) {
        throw std::invalid_argument(
            "quantized sparse convolution features must be a float16 or "
            "float32 matrix."
        );
    }
    if (weights.ndim() != 3 || weights.dtype() != mx::uint32) {
        throw std::invalid_argument(
            "packed weights must be a three-dimensional uint32 array."
        );
    }
    if (scales.ndim() != 3 || biases.shape() != scales.shape() ||
        scales.dtype() != feats.dtype() || biases.dtype() != feats.dtype()) {
        throw std::invalid_argument(
            "quantized scales and biases must have shape (K, C_out, G) "
            "and match the feature dtype."
        );
    }
    if (shape.bits != 4 && shape.bits != 8) {
        throw std::invalid_argument(
            "quantized convolution bits must be 4 or 8."
        );
    }
    if (shape.group_size != 32 && shape.group_size != 64 &&
        shape.group_size != 128) {
        throw std::invalid_argument(
            "quantized convolution group_size must be 32, 64, or 128."
        );
    }
    if (shape.in_channels <= 0 || shape.out_channels <= 0 ||
        shape.storage_in_channels < shape.in_channels ||
        shape.storage_in_channels % shape.group_size != 0) {
        throw std::invalid_argument(
            "quantized convolution channel metadata is invalid."
        );
    }
    auto packed_words = shape.storage_in_channels * shape.bits / 32;
    auto groups = shape.storage_in_channels / shape.group_size;
    if (weights.shape() !=
            mx::Shape{shape.n_kernels, packed_words, shape.out_channels} ||
        scales.shape() !=
            mx::Shape{shape.n_kernels, groups, shape.out_channels}) {
        throw std::invalid_argument(
            "packed weight shape does not match quantized convolution "
            "metadata."
        );
    }
    if (feats.shape(0) != shape.in_capacity ||
        feats.shape(1) != shape.in_channels) {
        throw std::invalid_argument(
            "feature shape does not match quantized convolution metadata."
        );
    }
}

} // namespace

mx::array make_sparse_quantized_conv_features(
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& scales,
    const mx::array& biases,
    const SparseRelationEdges& edges,
    const SparseRelationContract& contract,
    const mx::array& row_offsets,
    QuantizedSparseConvShape shape,
    const std::vector<mx::array>& sorted_inputs
) {
    auto stream = sparse_quantized_conv_stream(
        feats,
        weights,
        scales,
        biases,
        edges.in_rows,
        edges.out_rows,
        edges.kernel_ids,
        contract.counts,
        row_offsets
    );
    auto primitive =
        std::make_shared<SparseQuantizedConvFeatures>(stream, shape);
    auto device = mx::default_device() == mx::Device::gpu ? mx::Device::gpu
                                                          : mx::Device::cpu;
    auto inputs = std::vector<mx::array>{
        feats,
        mx::contiguous(weights, false, device),
        mx::contiguous(scales, false, device),
        mx::contiguous(biases, false, device),
        edges.in_rows,
        edges.out_rows,
        edges.kernel_ids,
        contract.counts,
        row_offsets,
    };
    inputs.insert(inputs.end(), sorted_inputs.begin(), sorted_inputs.end());
    return mx::array::make_arrays(
        {mx::Shape{shape.out_capacity, shape.out_channels}},
        {feats.dtype()},
        primitive,
        inputs
    )[0];
}

mx::array sparse_quantized_conv_features(
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& scales,
    const mx::array& biases,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    const mx::array& counts,
    const mx::array& row_offsets,
    int out_capacity,
    int n_kernels,
    int in_channels,
    int out_channels,
    int storage_in_channels,
    int group_size,
    int bits
) {
    if (out_capacity < 0 || n_kernels <= 0) {
        throw std::invalid_argument(
            "out_capacity must be nonnegative and n_kernels must be positive."
        );
    }
    auto shape = QuantizedSparseConvShape{
        static_cast<int>(feats.shape(0)),
        out_capacity,
        n_kernels,
        in_channels,
        out_channels,
        storage_in_channels,
        group_size,
        bits,
        0,
    };
    validate_quantized_arrays(feats, weights, scales, biases, shape);
    if (in_rows.ndim() != 1 || out_rows.ndim() != 1 || kernel_ids.ndim() != 1 ||
        in_rows.dtype() != mx::int32 || out_rows.dtype() != mx::int32 ||
        kernel_ids.dtype() != mx::int32 ||
        in_rows.shape() != out_rows.shape() ||
        in_rows.shape() != kernel_ids.shape()) {
        throw std::invalid_argument(
            "quantized convolution relation edges must be equal-length "
            "one-dimensional int32 arrays."
        );
    }
    if (counts.shape() != mx::Shape{2} || counts.dtype() != mx::int32) {
        throw std::invalid_argument(
            "quantized convolution counts must have shape (2,) and int32 "
            "dtype."
        );
    }
    if (row_offsets.shape() != mx::Shape{out_capacity + 1} ||
        row_offsets.dtype() != mx::int32) {
        throw std::invalid_argument(
            "quantized convolution row_offsets must have length "
            "out_capacity + 1 and int32 dtype."
        );
    }
    return make_sparse_quantized_conv_features(
        feats,
        weights,
        scales,
        biases,
        SparseRelationEdges{in_rows, out_rows, kernel_ids},
        SparseRelationContract{counts, out_capacity, n_kernels},
        row_offsets,
        shape
    );
}

mx::array sparse_quantized_conv_features_sorted(
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& scales,
    const mx::array& biases,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    const mx::array& counts,
    const mx::array& row_offsets,
    const mx::array& sorted_kv_out_in_map,
    const mx::array& reorder_rows,
    const mx::array& tile_masks,
    int out_capacity,
    int n_kernels,
    int in_channels,
    int out_channels,
    int storage_in_channels,
    int group_size,
    int bits
) {
    if (sorted_kv_out_in_map.shape() != mx::Shape{n_kernels, out_capacity} ||
        sorted_kv_out_in_map.dtype() != mx::int32 ||
        reorder_rows.shape() != mx::Shape{out_capacity} ||
        reorder_rows.dtype() != mx::int32 ||
        tile_masks.shape() != mx::Shape{4 * ((out_capacity + 63) / 64)} ||
        tile_masks.dtype() != mx::int32) {
        throw std::invalid_argument(
            "sorted quantized convolution view shapes are invalid."
        );
    }
    auto shape = QuantizedSparseConvShape{
        static_cast<int>(feats.shape(0)),
        out_capacity,
        n_kernels,
        in_channels,
        out_channels,
        storage_in_channels,
        group_size,
        bits,
        1,
    };
    validate_quantized_arrays(feats, weights, scales, biases, shape);
    if (in_rows.ndim() != 1 || out_rows.shape() != in_rows.shape() ||
        kernel_ids.shape() != in_rows.shape() || in_rows.dtype() != mx::int32 ||
        out_rows.dtype() != mx::int32 || kernel_ids.dtype() != mx::int32 ||
        counts.shape() != mx::Shape{2} || counts.dtype() != mx::int32 ||
        row_offsets.shape() != mx::Shape{out_capacity + 1} ||
        row_offsets.dtype() != mx::int32) {
        throw std::invalid_argument(
            "sorted quantized convolution relation arrays are invalid."
        );
    }
    return make_sparse_quantized_conv_features(
        feats,
        weights,
        scales,
        biases,
        SparseRelationEdges{in_rows, out_rows, kernel_ids},
        SparseRelationContract{counts, out_capacity, n_kernels},
        row_offsets,
        shape,
        {sorted_kv_out_in_map, reorder_rows, tile_masks}
    );
}

} // namespace mlx_lattice
