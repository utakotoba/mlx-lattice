#include "ops/conv3d/primitives.h"

#include <stdexcept>

#include "backends/cpu/conv3d.h"
#include "ops/conv3d.h"
#include "ops/conv3d/dispatch.h"

namespace mlx_lattice {

namespace {

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

class Conv3dFeats : public mx::Primitive {
  public:
    Conv3dFeats(mx::Stream stream, int rows, int in_channels, int out_channels)
        : mx::Primitive(stream), rows_(rows), in_channels_(in_channels),
          out_channels_(out_channels) {}

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
        eval_gpu_conv3d_feats(
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
        eval_gpu_conv3d_subm_feats(
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
        eval_gpu_conv3d_residual_feats(
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

class Pool3dFeats : public mx::Primitive {
  public:
    Pool3dFeats(mx::Stream stream, int rows, int channels)
        : mx::Primitive(stream), rows_(rows), channels_(channels) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        cpu::eval_pool3d_feats(inputs, outputs, stream(), rows_, channels_);
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        eval_gpu_pool3d_feats(inputs, outputs, stream(), rows_, channels_);
    }

    std::vector<mx::array>
    jvp(const std::vector<mx::array>&,
        const std::vector<mx::array>&,
        const std::vector<int>&) override {
        throw std::runtime_error("Pool3dFeats has no jvp implementation.");
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
                out.push_back(pool3d_feats_grad(
                    cotangents[0], primals[1], primals[2], primals[0].shape(0)
                ));
            } else {
                out.push_back(mx::zeros_like(primals[argnum]));
            }
        }
        return out;
    }

    std::pair<std::vector<mx::array>, std::vector<int>>
    vmap(const std::vector<mx::array>&, const std::vector<int>&) override {
        throw std::runtime_error("Pool3dFeats has no vmap implementation.");
    }

    const char* name() const override { return "Pool3dFeats"; }

    bool is_equivalent(const mx::Primitive& other) const override {
        const auto& pool = static_cast<const Pool3dFeats&>(other);
        return rows_ == pool.rows_ && channels_ == pool.channels_;
    }

  private:
    int rows_;
    int channels_;
};

class Pool3dFeatsGrad : public mx::Primitive {
  public:
    Pool3dFeatsGrad(mx::Stream stream, int rows, int channels)
        : mx::Primitive(stream), rows_(rows), channels_(channels) {}

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        cpu::eval_pool3d_feats_grad(
            inputs, outputs, stream(), rows_, channels_
        );
    }

    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override {
        eval_gpu_pool3d_feats_grad(inputs, outputs, stream(), rows_, channels_);
    }

    std::vector<mx::array>
    jvp(const std::vector<mx::array>&,
        const std::vector<mx::array>&,
        const std::vector<int>&) override {
        throw std::runtime_error("Pool3dFeatsGrad has no jvp implementation.");
    }

    std::vector<mx::array>
    vjp(const std::vector<mx::array>&,
        const std::vector<mx::array>&,
        const std::vector<int>&,
        const std::vector<mx::array>&) override {
        throw std::runtime_error("Pool3dFeatsGrad has no vjp implementation.");
    }

    std::pair<std::vector<mx::array>, std::vector<int>>
    vmap(const std::vector<mx::array>&, const std::vector<int>&) override {
        throw std::runtime_error("Pool3dFeatsGrad has no vmap implementation.");
    }

    const char* name() const override { return "Pool3dFeatsGrad"; }

    bool is_equivalent(const mx::Primitive& other) const override {
        const auto& grad = static_cast<const Pool3dFeatsGrad&>(other);
        return rows_ == grad.rows_ && channels_ == grad.channels_;
    }

  private:
    int rows_;
    int channels_;
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
        eval_gpu_conv3d_feats_grad(
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
        eval_gpu_conv3d_weight_grad(
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

} // namespace

std::shared_ptr<mx::Primitive> make_conv3d_feats_primitive(
    mx::Stream stream,
    int rows,
    int in_channels,
    int out_channels
) {
    return std::make_shared<Conv3dFeats>(
        stream, rows, in_channels, out_channels
    );
}

std::shared_ptr<mx::Primitive> make_conv3d_subm_feats_primitive(
    mx::Stream stream,
    int rows,
    int in_channels,
    int out_channels,
    int center_kernel
) {
    return std::make_shared<Conv3dSubmFeats>(
        stream, rows, in_channels, out_channels, center_kernel
    );
}

std::shared_ptr<mx::Primitive> make_conv3d_residual_feats_primitive(
    mx::Stream stream,
    int rows,
    int in_channels,
    int out_channels
) {
    return std::make_shared<Conv3dResidualFeats>(
        stream, rows, in_channels, out_channels
    );
}

std::shared_ptr<mx::Primitive>
make_pool3d_feats_primitive(mx::Stream stream, int rows, int channels) {
    return std::make_shared<Pool3dFeats>(stream, rows, channels);
}

std::shared_ptr<mx::Primitive>
make_pool3d_feats_grad_primitive(mx::Stream stream, int rows, int channels) {
    return std::make_shared<Pool3dFeatsGrad>(stream, rows, channels);
}

std::shared_ptr<mx::Primitive> make_conv3d_feats_grad_primitive(
    mx::Stream stream,
    int rows,
    int in_channels,
    int out_channels
) {
    return std::make_shared<Conv3dFeatsGrad>(
        stream, rows, in_channels, out_channels
    );
}

std::shared_ptr<mx::Primitive> make_conv3d_weight_grad_primitive(
    mx::Stream stream,
    int kernels,
    int in_channels,
    int out_channels
) {
    return std::make_shared<Conv3dWeightGrad>(
        stream, kernels, in_channels, out_channels
    );
}

} // namespace mlx_lattice
