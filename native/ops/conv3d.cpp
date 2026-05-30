#include "ops/conv3d.h"

#include <algorithm>
#include <stdexcept>

#include "backends/cpu/conv3d.h"
#include "backends/metal/conv3d.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx_lattice {

namespace {

// MARK: - validation

void validate_conv3d_feats(
    const mx::array& feats,
    const mx::array& weight,
    const mx::array& maps,
    const mx::array& kernels,
    int out_rows
) {
    if (out_rows < 0) {
        throw std::invalid_argument("out_rows must be non-negative.");
    }
    if (feats.ndim() != 2) {
        throw std::invalid_argument("feats must have shape (N, Cin).");
    }
    if (weight.ndim() != 3) {
        throw std::invalid_argument("weight must have shape (K, Cin, Cout).");
    }
    if (maps.ndim() != 2 || maps.shape(1) != 2) {
        throw std::invalid_argument("maps must have shape (M, 2).");
    }
    if (kernels.ndim() != 1 || kernels.shape(0) != maps.shape(0)) {
        throw std::invalid_argument("kernels must have shape (M,).");
    }
    if (feats.dtype() != mx::float32 || weight.dtype() != mx::float32) {
        throw std::invalid_argument("conv3d_feats supports float32 tensors.");
    }
    if (maps.dtype() != mx::int32 || kernels.dtype() != mx::int32) {
        throw std::invalid_argument("maps and kernels must be int32.");
    }
    if (weight.shape(1) != feats.shape(1)) {
        throw std::invalid_argument("weight input channels must match feats.");
    }
}

class Conv3dFeats : public mx::Primitive {
  public:
    Conv3dFeats(mx::Stream stream, int rows, int in_channels, int out_channels)
        : mx::Primitive(stream), rows_(rows), in_channels_(in_channels),
          out_channels_(out_channels) {}

    // MARK: - primitive

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        cpu::eval_conv3d_feats(
            inputs, outputs, stream(), rows_, in_channels_, out_channels_
        );
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        metal::eval_conv3d_feats(
            inputs, outputs, stream(), rows_, in_channels_, out_channels_
        );
    }

    std::vector<mx::array>
    jvp(const std::vector<mx::array>&,
        const std::vector<mx::array>&,
        const std::vector<int>&) override {
        throw std::runtime_error("Conv3dFeats has no jvp implementation.");
    }

    std::vector<mx::array>
    vjp(const std::vector<mx::array>&,
        const std::vector<mx::array>&,
        const std::vector<int>&,
        const std::vector<mx::array>&) override {
        throw std::runtime_error("Conv3dFeats has no vjp implementation.");
    }

    std::pair<std::vector<mx::array>, std::vector<int>>
    vmap(const std::vector<mx::array>&, const std::vector<int>&) override {
        throw std::runtime_error("Conv3dFeats has no vmap implementation.");
    }

    const char* name() const override { return "Conv3dFeats"; }

    bool is_equivalent(const mx::Primitive& other) const override {
        const auto& conv = static_cast<const Conv3dFeats&>(other);
        return rows_ == conv.rows_ && in_channels_ == conv.in_channels_ &&
               out_channels_ == conv.out_channels_;
    }

  private:
    int rows_;
    int in_channels_;
    int out_channels_;
};

} // namespace

// MARK: - api

mx::array conv3d_feats(
    const mx::array& feats,
    const mx::array& weight,
    const mx::array& maps,
    const mx::array& kernels,
    int out_rows,
    mx::StreamOrDevice stream
) {
    validate_conv3d_feats(feats, weight, maps, kernels, out_rows);

    auto s = to_stream(stream);
    auto feats_contiguous = mx::contiguous(feats, false, s);
    auto weight_contiguous = mx::contiguous(weight, false, s);
    auto maps_contiguous = mx::contiguous(maps, false, s);
    auto kernels_contiguous = mx::contiguous(kernels, false, s);

    return mx::array(
        mx::Shape{out_rows, weight.shape(2)},
        mx::float32,
        std::make_shared<Conv3dFeats>(
            s, out_rows, feats.shape(1), weight.shape(2)
        ),
        {feats_contiguous,
         weight_contiguous,
         maps_contiguous,
         kernels_contiguous}
    );
}

} // namespace mlx_lattice
