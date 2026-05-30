#include "ops/conv3d.h"

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

void validate_conv3d_residual_feats(
    const mx::array& base,
    const mx::array& feats,
    const mx::array& weight,
    const mx::array& maps,
    const mx::array& kernels,
    const mx::array& offsets
) {
    validate_conv3d_feats(feats, weight, maps, kernels, base.shape(0));
    if (base.ndim() != 2 || base.dtype() != mx::float32) {
        throw std::invalid_argument("base must be a float32 matrix.");
    }
    if (base.shape(1) != weight.shape(2)) {
        throw std::invalid_argument("base output channels must match weight.");
    }
    if (offsets.ndim() != 1 || offsets.shape(0) != base.shape(0) + 1) {
        throw std::invalid_argument("offsets must have shape (rows + 1,).");
    }
    if (offsets.dtype() != mx::int32) {
        throw std::invalid_argument("offsets must be int32.");
    }
}

std::vector<mx::array> conv3d_vjp(
    const std::vector<mx::array>& primals,
    const std::vector<mx::array>& cotangents,
    const std::vector<int>& argnums,
    int rows
);

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
    vjp(const std::vector<mx::array>& primals,
        const std::vector<mx::array>& cotangents,
        const std::vector<int>& argnums,
        const std::vector<mx::array>&) override {
        return conv3d_vjp(primals, cotangents, argnums, primals[0].shape(0));
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

class Conv3dSubmFeats : public mx::Primitive {
  public:
    Conv3dSubmFeats(
        mx::Stream stream,
        int rows,
        int in_channels,
        int out_channels,
        int center_kernel
    )
        : mx::Primitive(stream), rows_(rows), in_channels_(in_channels),
          out_channels_(out_channels), center_kernel_(center_kernel) {}

    // MARK: - primitive

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        cpu::eval_conv3d_subm_feats(
            inputs,
            outputs,
            stream(),
            rows_,
            in_channels_,
            out_channels_,
            center_kernel_
        );
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        metal::eval_conv3d_subm_feats(
            inputs,
            outputs,
            stream(),
            rows_,
            in_channels_,
            out_channels_,
            center_kernel_
        );
    }

    std::vector<mx::array>
    jvp(const std::vector<mx::array>&,
        const std::vector<mx::array>&,
        const std::vector<int>&) override {
        throw std::runtime_error("Conv3dSubmFeats has no jvp implementation.");
    }

    std::vector<mx::array>
    vjp(const std::vector<mx::array>& primals,
        const std::vector<mx::array>& cotangents,
        const std::vector<int>& argnums,
        const std::vector<mx::array>&) override {
        return conv3d_vjp(primals, cotangents, argnums, primals[0].shape(0));
    }

    std::pair<std::vector<mx::array>, std::vector<int>>
    vmap(const std::vector<mx::array>&, const std::vector<int>&) override {
        throw std::runtime_error("Conv3dSubmFeats has no vmap implementation.");
    }

    const char* name() const override { return "Conv3dSubmFeats"; }

    bool is_equivalent(const mx::Primitive& other) const override {
        const auto& conv = static_cast<const Conv3dSubmFeats&>(other);
        return rows_ == conv.rows_ && in_channels_ == conv.in_channels_ &&
               out_channels_ == conv.out_channels_ &&
               center_kernel_ == conv.center_kernel_;
    }

  private:
    int rows_;
    int in_channels_;
    int out_channels_;
    int center_kernel_;
};

class Conv3dResidualFeats : public mx::Primitive {
  public:
    Conv3dResidualFeats(
        mx::Stream stream,
        int rows,
        int in_channels,
        int out_channels
    )
        : mx::Primitive(stream), rows_(rows), in_channels_(in_channels),
          out_channels_(out_channels) {}

    // MARK: - primitive

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        cpu::eval_conv3d_residual_feats(
            inputs, outputs, stream(), rows_, in_channels_, out_channels_
        );
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        metal::eval_conv3d_residual_feats(
            inputs, outputs, stream(), rows_, in_channels_, out_channels_
        );
    }

    std::vector<mx::array>
    jvp(const std::vector<mx::array>&,
        const std::vector<mx::array>&,
        const std::vector<int>&) override {
        throw std::runtime_error(
            "Conv3dResidualFeats has no jvp implementation."
        );
    }

    std::vector<mx::array>
    vjp(const std::vector<mx::array>& primals,
        const std::vector<mx::array>& cotangents,
        const std::vector<int>& argnums,
        const std::vector<mx::array>&) override {
        std::vector<mx::array> out;
        out.reserve(argnums.size());
        for (auto argnum : argnums) {
            if (argnum == 0) {
                out.push_back(cotangents[0]);
            } else if (argnum == 1) {
                out.push_back(conv3d_feats_grad(
                    cotangents[0],
                    primals[2],
                    primals[3],
                    primals[4],
                    primals[1].shape(0)
                ));
            } else if (argnum == 2) {
                out.push_back(conv3d_weight_grad(
                    primals[1],
                    cotangents[0],
                    primals[3],
                    primals[4],
                    primals[2].shape(0)
                ));
            } else {
                out.push_back(mx::zeros_like(primals[argnum]));
            }
        }
        return out;
    }

    std::pair<std::vector<mx::array>, std::vector<int>>
    vmap(const std::vector<mx::array>&, const std::vector<int>&) override {
        throw std::runtime_error(
            "Conv3dResidualFeats has no vmap implementation."
        );
    }

    const char* name() const override { return "Conv3dResidualFeats"; }

    bool is_equivalent(const mx::Primitive& other) const override {
        const auto& conv = static_cast<const Conv3dResidualFeats&>(other);
        return rows_ == conv.rows_ && in_channels_ == conv.in_channels_ &&
               out_channels_ == conv.out_channels_;
    }

  private:
    int rows_;
    int in_channels_;
    int out_channels_;
};

class Conv3dFeatsGrad : public mx::Primitive {
  public:
    Conv3dFeatsGrad(
        mx::Stream stream,
        int rows,
        int in_channels,
        int out_channels
    )
        : mx::Primitive(stream), rows_(rows), in_channels_(in_channels),
          out_channels_(out_channels) {}

    // MARK: - primitive

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        cpu::eval_conv3d_feats_grad(
            inputs, outputs, stream(), rows_, in_channels_, out_channels_
        );
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        metal::eval_conv3d_feats_grad(
            inputs, outputs, stream(), rows_, in_channels_, out_channels_
        );
    }

    std::vector<mx::array>
    jvp(const std::vector<mx::array>&,
        const std::vector<mx::array>&,
        const std::vector<int>&) override {
        throw std::runtime_error("Conv3dFeatsGrad has no jvp implementation.");
    }

    std::vector<mx::array>
    vjp(const std::vector<mx::array>&,
        const std::vector<mx::array>&,
        const std::vector<int>&,
        const std::vector<mx::array>&) override {
        throw std::runtime_error("Conv3dFeatsGrad has no vjp implementation.");
    }

    std::pair<std::vector<mx::array>, std::vector<int>>
    vmap(const std::vector<mx::array>&, const std::vector<int>&) override {
        throw std::runtime_error("Conv3dFeatsGrad has no vmap implementation.");
    }

    const char* name() const override { return "Conv3dFeatsGrad"; }

    bool is_equivalent(const mx::Primitive& other) const override {
        const auto& grad = static_cast<const Conv3dFeatsGrad&>(other);
        return rows_ == grad.rows_ && in_channels_ == grad.in_channels_ &&
               out_channels_ == grad.out_channels_;
    }

  private:
    int rows_;
    int in_channels_;
    int out_channels_;
};

class Conv3dWeightGrad : public mx::Primitive {
  public:
    Conv3dWeightGrad(
        mx::Stream stream,
        int kernels,
        int in_channels,
        int out_channels
    )
        : mx::Primitive(stream), kernels_(kernels), in_channels_(in_channels),
          out_channels_(out_channels) {}

    // MARK: - primitive

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        cpu::eval_conv3d_weight_grad(
            inputs, outputs, stream(), kernels_, in_channels_, out_channels_
        );
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        metal::eval_conv3d_weight_grad(
            inputs, outputs, stream(), kernels_, in_channels_, out_channels_
        );
    }

    std::vector<mx::array>
    jvp(const std::vector<mx::array>&,
        const std::vector<mx::array>&,
        const std::vector<int>&) override {
        throw std::runtime_error("Conv3dWeightGrad has no jvp implementation.");
    }

    std::vector<mx::array>
    vjp(const std::vector<mx::array>&,
        const std::vector<mx::array>&,
        const std::vector<int>&,
        const std::vector<mx::array>&) override {
        throw std::runtime_error("Conv3dWeightGrad has no vjp implementation.");
    }

    std::pair<std::vector<mx::array>, std::vector<int>>
    vmap(const std::vector<mx::array>&, const std::vector<int>&) override {
        throw std::runtime_error(
            "Conv3dWeightGrad has no vmap implementation."
        );
    }

    const char* name() const override { return "Conv3dWeightGrad"; }

    bool is_equivalent(const mx::Primitive& other) const override {
        const auto& grad = static_cast<const Conv3dWeightGrad&>(other);
        return kernels_ == grad.kernels_ && in_channels_ == grad.in_channels_ &&
               out_channels_ == grad.out_channels_;
    }

  private:
    int kernels_;
    int in_channels_;
    int out_channels_;
};

std::vector<mx::array> conv3d_vjp(
    const std::vector<mx::array>& primals,
    const std::vector<mx::array>& cotangents,
    const std::vector<int>& argnums,
    int rows
) {
    std::vector<mx::array> out;
    out.reserve(argnums.size());
    for (auto argnum : argnums) {
        if (argnum == 0) {
            out.push_back(conv3d_feats_grad(
                cotangents[0], primals[1], primals[2], primals[3], rows
            ));
        } else if (argnum == 1) {
            out.push_back(conv3d_weight_grad(
                primals[0],
                cotangents[0],
                primals[2],
                primals[3],
                primals[1].shape(0)
            ));
        } else {
            out.push_back(mx::zeros_like(primals[argnum]));
        }
    }
    return out;
}

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

mx::array conv3d_subm_feats(
    const mx::array& feats,
    const mx::array& weight,
    const mx::array& maps,
    const mx::array& kernels,
    int center_kernel,
    mx::StreamOrDevice stream
) {
    validate_conv3d_feats(feats, weight, maps, kernels, feats.shape(0));
    if (center_kernel < 0 || center_kernel >= weight.shape(0)) {
        throw std::invalid_argument("center_kernel is out of bounds.");
    }

    auto s = to_stream(stream);
    auto feats_contiguous = mx::contiguous(feats, false, s);
    auto weight_contiguous = mx::contiguous(weight, false, s);
    auto maps_contiguous = mx::contiguous(maps, false, s);
    auto kernels_contiguous = mx::contiguous(kernels, false, s);

    return mx::array(
        mx::Shape{feats.shape(0), weight.shape(2)},
        mx::float32,
        std::make_shared<Conv3dSubmFeats>(
            s, feats.shape(0), feats.shape(1), weight.shape(2), center_kernel
        ),
        {feats_contiguous,
         weight_contiguous,
         maps_contiguous,
         kernels_contiguous}
    );
}

mx::array conv3d_residual_feats(
    const mx::array& base,
    const mx::array& feats,
    const mx::array& weight,
    const mx::array& maps,
    const mx::array& kernels,
    const mx::array& offsets,
    mx::StreamOrDevice stream
) {
    validate_conv3d_residual_feats(base, feats, weight, maps, kernels, offsets);

    auto s = to_stream(stream);
    auto base_contiguous = mx::contiguous(base, false, s);
    auto feats_contiguous = mx::contiguous(feats, false, s);
    auto weight_contiguous = mx::contiguous(weight, false, s);
    auto maps_contiguous = mx::contiguous(maps, false, s);
    auto kernels_contiguous = mx::contiguous(kernels, false, s);
    auto offsets_contiguous = mx::contiguous(offsets, false, s);

    return mx::array(
        base.shape(),
        mx::float32,
        std::make_shared<Conv3dResidualFeats>(
            s, base.shape(0), feats.shape(1), weight.shape(2)
        ),
        {base_contiguous,
         feats_contiguous,
         weight_contiguous,
         maps_contiguous,
         kernels_contiguous,
         offsets_contiguous}
    );
}

mx::array conv3d_feats_grad(
    const mx::array& grad,
    const mx::array& weight,
    const mx::array& maps,
    const mx::array& kernels,
    int rows,
    mx::StreamOrDevice stream
) {
    if (rows < 0) {
        throw std::invalid_argument("rows must be non-negative.");
    }
    if (grad.ndim() != 2 || weight.ndim() != 3) {
        throw std::invalid_argument("grad must be 2D and weight must be 3D.");
    }
    if (grad.dtype() != mx::float32 || weight.dtype() != mx::float32) {
        throw std::invalid_argument(
            "conv3d gradients support float32 tensors."
        );
    }
    if (grad.shape(1) != weight.shape(2)) {
        throw std::invalid_argument("grad output channels must match weight.");
    }
    if (maps.ndim() != 2 || maps.shape(1) != 2 || kernels.ndim() != 1 ||
        kernels.shape(0) != maps.shape(0)) {
        throw std::invalid_argument("invalid map shapes.");
    }
    if (maps.dtype() != mx::int32 || kernels.dtype() != mx::int32) {
        throw std::invalid_argument("maps and kernels must be int32.");
    }

    auto s = to_stream(stream);
    auto grad_contiguous = mx::contiguous(grad, false, s);
    auto weight_contiguous = mx::contiguous(weight, false, s);
    auto maps_contiguous = mx::contiguous(maps, false, s);
    auto kernels_contiguous = mx::contiguous(kernels, false, s);

    return mx::array(
        mx::Shape{rows, weight.shape(1)},
        mx::float32,
        std::make_shared<Conv3dFeatsGrad>(
            s, rows, weight.shape(1), weight.shape(2)
        ),
        {grad_contiguous,
         weight_contiguous,
         maps_contiguous,
         kernels_contiguous}
    );
}

mx::array conv3d_weight_grad(
    const mx::array& feats,
    const mx::array& grad,
    const mx::array& maps,
    const mx::array& kernels,
    int kernel_count,
    mx::StreamOrDevice stream
) {
    if (kernel_count < 0) {
        throw std::invalid_argument("kernel_count must be non-negative.");
    }
    if (feats.ndim() != 2 || grad.ndim() != 2) {
        throw std::invalid_argument("feats and grad must be matrices.");
    }
    if (feats.dtype() != mx::float32 || grad.dtype() != mx::float32) {
        throw std::invalid_argument(
            "conv3d gradients support float32 tensors."
        );
    }
    if (maps.ndim() != 2 || maps.shape(1) != 2 || kernels.ndim() != 1 ||
        kernels.shape(0) != maps.shape(0)) {
        throw std::invalid_argument("invalid map shapes.");
    }
    if (maps.dtype() != mx::int32 || kernels.dtype() != mx::int32) {
        throw std::invalid_argument("maps and kernels must be int32.");
    }

    auto s = to_stream(stream);
    auto feats_contiguous = mx::contiguous(feats, false, s);
    auto grad_contiguous = mx::contiguous(grad, false, s);
    auto maps_contiguous = mx::contiguous(maps, false, s);
    auto kernels_contiguous = mx::contiguous(kernels, false, s);

    return mx::array(
        mx::Shape{kernel_count, feats.shape(1), grad.shape(1)},
        mx::float32,
        std::make_shared<Conv3dWeightGrad>(
            s, kernel_count, feats.shape(1), grad.shape(1)
        ),
        {feats_contiguous, grad_contiguous, maps_contiguous, kernels_contiguous}
    );
}

} // namespace mlx_lattice
