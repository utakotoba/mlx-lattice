#include "backends/cpu/exec/algorithms.h"

#include "backends/cpu/exec/planning.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "backends/array_utils.h"
#include "backends/cpu/schedule.h"

namespace mlx_lattice::exec::cpu {

namespace {

void fill_zero(mx::array& out) {
    auto data = out.data<float>();
    std::fill(data, data + out.size(), 0.0F);
}

void fill_pool_init(mx::array& out, PoolReduceOp reduce) {
    auto data = out.data<float>();
    auto value = reduce == PoolReduceOp::Max
                     ? -std::numeric_limits<float>::infinity()
                     : 0.0F;
    std::fill(data, data + out.size(), value);
}

std::ptrdiff_t weight_offset(
    const mx::array& weights,
    const SparseConvShape& shape,
    int kernel,
    int in_channel,
    int out_channel
) {
    if (shape.weight_layout == 0) {
        return static_cast<std::ptrdiff_t>(kernel) * weights.strides(0) +
               static_cast<std::ptrdiff_t>(in_channel) * weights.strides(1) +
               static_cast<std::ptrdiff_t>(out_channel) * weights.strides(2);
    }

    auto xy = shape.kernel_y * shape.kernel_z;
    auto kx = kernel / xy;
    auto rem = kernel % xy;
    auto ky = rem / shape.kernel_z;
    auto kz = rem % shape.kernel_z;
    return static_cast<std::ptrdiff_t>(out_channel) * weights.strides(0) +
           static_cast<std::ptrdiff_t>(kx) * weights.strides(1) +
           static_cast<std::ptrdiff_t>(ky) * weights.strides(2) +
           static_cast<std::ptrdiff_t>(kz) * weights.strides(3) +
           static_cast<std::ptrdiff_t>(in_channel) * weights.strides(4);
}

std::ptrdiff_t dense_weight_offset(
    const SparseConvShape& shape,
    int kernel,
    int in_channel,
    int out_channel
) {
    if (shape.weight_layout == 0) {
        return (static_cast<std::ptrdiff_t>(kernel) * shape.in_channels +
                in_channel) *
                   shape.out_channels +
               out_channel;
    }

    auto xy = shape.kernel_y * shape.kernel_z;
    auto kx = kernel / xy;
    auto rem = kernel % xy;
    auto ky = rem / shape.kernel_z;
    auto kz = rem % shape.kernel_z;
    return (((static_cast<std::ptrdiff_t>(out_channel) * shape.kernel_x + kx) *
                 shape.kernel_y +
             ky) *
                shape.kernel_z +
            kz) *
               shape.in_channels +
           in_channel;
}

struct MaxTiePolicy {
    std::vector<int32_t> counts;
    std::vector<int32_t> first_ranks;
};

int pool_edge_rank(const Edge& edge, const SparsePoolShape& shape) {
    return edge[0] * shape.n_kernels + edge[2];
}

std::size_t pool_channel_key(int out_row, int channel, int channels) {
    return static_cast<std::size_t>(out_row) *
               static_cast<std::size_t>(channels) +
           static_cast<std::size_t>(channel);
}

MaxTiePolicy build_max_tie_policy(
    const Plan& plan,
    const mx::array& feats,
    const mx::array& pooled,
    SparsePoolShape shape
) {
    auto size = static_cast<std::size_t>(shape.out_capacity) *
                static_cast<std::size_t>(shape.channels);
    MaxTiePolicy policy{
        std::vector<int32_t>(size, 0),
        std::vector<int32_t>(size, std::numeric_limits<int32_t>::max()),
    };

    const auto* feat_data = feats.data<float>();
    const auto* pooled_data = pooled.data<float>();
    const auto feat_s0 = feats.strides(0);
    const auto feat_s1 = feats.strides(1);
    const auto pooled_s0 = pooled.strides(0);
    const auto pooled_s1 = pooled.strides(1);

    for (auto edge : plan.edges) {
        auto in_row = edge[0];
        auto out_row = edge[1];
        auto rank = pool_edge_rank(edge, shape);
        for (int channel = 0; channel < shape.channels; ++channel) {
            auto feat_value = feat_data
                [static_cast<std::ptrdiff_t>(in_row) * feat_s0 +
                 static_cast<std::ptrdiff_t>(channel) * feat_s1];
            auto pooled_value = pooled_data
                [static_cast<std::ptrdiff_t>(out_row) * pooled_s0 +
                 static_cast<std::ptrdiff_t>(channel) * pooled_s1];
            if (feat_value != pooled_value) {
                continue;
            }
            auto key = pool_channel_key(out_row, channel, shape.channels);
            policy.counts[key] += 1;
            policy.first_ranks[key] = std::min(policy.first_ranks[key], rank);
        }
    }
    return policy;
}

} // namespace

void eval_sparse_conv_features(
    SparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    backend::schedule_cpu(
        stream,
        inputs,
        outputs,
        [shape](
            const std::vector<mx::array>& ready,
            std::vector<mx::array>& task_outputs
        ) {
            const auto& feats = ready[0];
            const auto& weights = ready[1];
            const auto& in_rows = ready[2];
            const auto& out_rows = ready[3];
            const auto& kernel_ids = ready[4];
            const auto& counts = ready[5];

            auto& out = task_outputs[0];
            fill_zero(out);
            auto* out_data = out.data<float>();
            const auto* feat_data = feats.data<float>();
            const auto* weight_data = weights.data<float>();
            const auto* in_data = in_rows.data<int32_t>();
            const auto* out_row_data = out_rows.data<int32_t>();
            const auto* kernel_data = kernel_ids.data<int32_t>();
            auto edge_count = std::min(
                counts.data<int32_t>()[0], static_cast<int>(in_rows.shape(0))
            );
            const auto feat_s0 = feats.strides(0);
            const auto feat_s1 = feats.strides(1);

            for (int edge = 0; edge < edge_count; ++edge) {
                auto in_row = in_data[edge];
                auto out_row = out_row_data[edge];
                auto kernel = kernel_data[edge];
                if (in_row < 0 || out_row < 0 || kernel < 0 ||
                    out_row >= shape.out_capacity) {
                    continue;
                }
                auto* out_row_ptr =
                    out_data +
                    static_cast<std::ptrdiff_t>(out_row) * shape.out_channels;
                for (int ci = 0; ci < shape.in_channels; ++ci) {
                    auto value = feat_data
                        [static_cast<std::ptrdiff_t>(in_row) * feat_s0 +
                         static_cast<std::ptrdiff_t>(ci) * feat_s1];
                    for (int co = 0; co < shape.out_channels; ++co) {
                        out_row_ptr[co] +=
                            value * weight_data[weight_offset(
                                        weights, shape, kernel, ci, co
                                    )];
                    }
                }
            }
        }
    );
}

void eval_sparse_conv_features_input_grad(
    SparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    backend::schedule_cpu(
        stream,
        inputs,
        outputs,
        [shape](
            const std::vector<mx::array>& ready,
            std::vector<mx::array>& task_outputs
        ) {
            const auto& cotangent = ready[0];
            const auto& weights = ready[1];
            const auto& in_rows = ready[2];
            const auto& out_rows = ready[3];
            const auto& kernel_ids = ready[4];
            const auto& counts = ready[5];

            auto& grad = task_outputs[0];
            fill_zero(grad);
            auto* grad_data = grad.data<float>();
            const auto* cotangent_data = cotangent.data<float>();
            const auto* weight_data = weights.data<float>();
            const auto* in_data = in_rows.data<int32_t>();
            const auto* out_row_data = out_rows.data<int32_t>();
            const auto* kernel_data = kernel_ids.data<int32_t>();
            auto edge_count = std::min(
                counts.data<int32_t>()[0], static_cast<int>(in_rows.shape(0))
            );
            const auto cotangent_s0 = cotangent.strides(0);
            const auto cotangent_s1 = cotangent.strides(1);

            for (int edge = 0; edge < edge_count; ++edge) {
                auto in_row = in_data[edge];
                auto out_row = out_row_data[edge];
                auto kernel = kernel_data[edge];
                if (in_row < 0 || out_row < 0 || kernel < 0 ||
                    out_row >= shape.out_capacity) {
                    continue;
                }
                auto* grad_row =
                    grad_data +
                    static_cast<std::ptrdiff_t>(in_row) * shape.in_channels;
                for (int ci = 0; ci < shape.in_channels; ++ci) {
                    for (int co = 0; co < shape.out_channels; ++co) {
                        grad_row[ci] +=
                            cotangent_data
                                [static_cast<std::ptrdiff_t>(out_row) *
                                     cotangent_s0 +
                                 static_cast<std::ptrdiff_t>(co) *
                                     cotangent_s1] *
                            weight_data[weight_offset(
                                weights, shape, kernel, ci, co
                            )];
                    }
                }
            }
        }
    );
}

void eval_sparse_conv_features_weight_grad(
    SparseConvShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    backend::schedule_cpu(
        stream,
        inputs,
        outputs,
        [shape](
            const std::vector<mx::array>& ready,
            std::vector<mx::array>& task_outputs
        ) {
            const auto& feats = ready[0];
            const auto& cotangent = ready[1];
            const auto& in_rows = ready[2];
            const auto& out_rows = ready[3];
            const auto& kernel_ids = ready[4];
            const auto& counts = ready[5];

            auto& grad = task_outputs[0];
            fill_zero(grad);
            auto* grad_data = grad.data<float>();
            const auto* feat_data = feats.data<float>();
            const auto* cotangent_data = cotangent.data<float>();
            const auto* in_data = in_rows.data<int32_t>();
            const auto* out_row_data = out_rows.data<int32_t>();
            const auto* kernel_data = kernel_ids.data<int32_t>();
            auto edge_count = std::min(
                counts.data<int32_t>()[0], static_cast<int>(in_rows.shape(0))
            );
            const auto feat_s0 = feats.strides(0);
            const auto feat_s1 = feats.strides(1);
            const auto cotangent_s0 = cotangent.strides(0);
            const auto cotangent_s1 = cotangent.strides(1);

            for (int edge = 0; edge < edge_count; ++edge) {
                auto in_row = in_data[edge];
                auto out_row = out_row_data[edge];
                auto kernel = kernel_data[edge];
                if (in_row < 0 || out_row < 0 || kernel < 0 ||
                    out_row >= shape.out_capacity) {
                    continue;
                }
                for (int ci = 0; ci < shape.in_channels; ++ci) {
                    auto feat_value = feat_data
                        [static_cast<std::ptrdiff_t>(in_row) * feat_s0 +
                         static_cast<std::ptrdiff_t>(ci) * feat_s1];
                    for (int co = 0; co < shape.out_channels; ++co) {
                        grad_data[dense_weight_offset(shape, kernel, ci, co)] +=
                            feat_value *
                            cotangent_data
                                [static_cast<std::ptrdiff_t>(out_row) *
                                     cotangent_s0 +
                                 static_cast<std::ptrdiff_t>(co) *
                                     cotangent_s1];
                    }
                }
            }
        }
    );
}

void eval_sparse_pool(
    PoolReduceOp reduce,
    SparsePoolShape shape,
    Triple stride,
    Triple padding,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    backend::schedule_cpu(
        stream,
        inputs,
        outputs,
        [reduce, shape, stride, padding](
            const std::vector<mx::array>& ready,
            std::vector<mx::array>& task_outputs
        ) {
            const auto& coords = ready[0];
            const auto& active_rows = ready[1];
            const auto& feats = ready[2];
            const auto& offsets = ready[3];

            auto plan = build_plan(
                SparseMapOp::Forward,
                coords,
                active_rows,
                offsets,
                stride,
                padding
            );
            write_coords(task_outputs[SparseOutCoords], plan.out_coords);
            write_counts(task_outputs[SparseCounts], plan);

            auto& out = task_outputs[SparseOutFeats];
            fill_pool_init(out, reduce);
            auto* out_data = out.data<float>();
            const auto* feat_data = feats.data<float>();
            const auto feat_s0 = feats.strides(0);
            const auto feat_s1 = feats.strides(1);
            for (auto edge : plan.edges) {
                auto in_row = edge[0];
                auto out_row = edge[1];
                auto* out_row_data =
                    out_data +
                    static_cast<std::ptrdiff_t>(out_row) * shape.channels;
                for (int channel = 0; channel < shape.channels; ++channel) {
                    auto value = feat_data
                        [static_cast<std::ptrdiff_t>(in_row) * feat_s0 +
                         static_cast<std::ptrdiff_t>(channel) * feat_s1];
                    if (reduce == PoolReduceOp::Max) {
                        out_row_data[channel] =
                            std::max(out_row_data[channel], value);
                    } else {
                        out_row_data[channel] += value;
                    }
                }
            }

            if (reduce == PoolReduceOp::Avg) {
                auto degrees = pool_degrees(plan, shape.out_capacity);
                for (int row = 0; row < int(plan.out_coords.size()); ++row) {
                    auto denom = std::max(degrees[row], int32_t{1});
                    auto* out_row =
                        out_data +
                        static_cast<std::ptrdiff_t>(row) * shape.channels;
                    for (int channel = 0; channel < shape.channels; ++channel) {
                        out_row[channel] /= float(denom);
                    }
                }
            }
        }
    );
}

void eval_sparse_pool_grad(
    PoolReduceOp reduce,
    SparsePoolShape shape,
    Triple stride,
    Triple padding,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    backend::schedule_cpu(
        stream,
        inputs,
        outputs,
        [reduce, shape, stride, padding](
            const std::vector<mx::array>& ready,
            std::vector<mx::array>& task_outputs
        ) {
            const auto& cotangent = ready[0];
            const auto& feats = ready[1];
            const auto& pooled = ready[2];
            const auto& coords = ready[3];
            const auto& active_rows = ready[4];
            const auto& offsets = ready[5];

            auto plan = build_plan(
                SparseMapOp::Forward,
                coords,
                active_rows,
                offsets,
                stride,
                padding
            );
            auto degrees = pool_degrees(plan, shape.out_capacity);
            auto max_ties =
                reduce == PoolReduceOp::Max
                    ? build_max_tie_policy(plan, feats, pooled, shape)
                    : MaxTiePolicy{};
            auto& grad = task_outputs[0];
            fill_zero(grad);
            auto* grad_data = grad.data<float>();
            const auto* cotangent_data = cotangent.data<float>();
            const auto* feat_data = feats.data<float>();
            const auto* pooled_data = pooled.data<float>();
            const auto cotangent_s0 = cotangent.strides(0);
            const auto cotangent_s1 = cotangent.strides(1);
            const auto feat_s0 = feats.strides(0);
            const auto feat_s1 = feats.strides(1);
            const auto pooled_s0 = pooled.strides(0);
            const auto pooled_s1 = pooled.strides(1);

            for (auto edge : plan.edges) {
                auto in_row = edge[0];
                auto out_row = edge[1];
                auto* grad_row =
                    grad_data +
                    static_cast<std::ptrdiff_t>(in_row) * shape.channels;
                auto denom = std::max(degrees[out_row], int32_t{1});

                for (int channel = 0; channel < shape.channels; ++channel) {
                    auto feat_value = feat_data
                        [static_cast<std::ptrdiff_t>(in_row) * feat_s0 +
                         static_cast<std::ptrdiff_t>(channel) * feat_s1];
                    auto pooled_value = pooled_data
                        [static_cast<std::ptrdiff_t>(out_row) * pooled_s0 +
                         static_cast<std::ptrdiff_t>(channel) * pooled_s1];
                    auto scale = 1.0F;
                    if (reduce == PoolReduceOp::Max) {
                        if (feat_value != pooled_value) {
                            continue;
                        }
                        auto key =
                            pool_channel_key(out_row, channel, shape.channels);
                        auto tie_count = max_ties.counts[key];
                        if (tie_count == 0) {
                            continue;
                        }
                        scale = 1.0F / float(tie_count);
                    } else if (reduce == PoolReduceOp::Avg) {
                        scale = 1.0F / float(denom);
                    }
                    auto cotangent_value = cotangent_data
                        [static_cast<std::ptrdiff_t>(out_row) * cotangent_s0 +
                         static_cast<std::ptrdiff_t>(channel) * cotangent_s1];
                    grad_row[channel] += cotangent_value * scale;
                }
            }
        }
    );
}

void eval_sparse_pool_jvp(
    PoolReduceOp reduce,
    SparsePoolShape shape,
    Triple stride,
    Triple padding,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    backend::schedule_cpu(
        stream,
        inputs,
        outputs,
        [reduce, shape, stride, padding](
            const std::vector<mx::array>& ready,
            std::vector<mx::array>& task_outputs
        ) {
            const auto& tangent = ready[0];
            const auto& feats = ready[1];
            const auto& pooled = ready[2];
            const auto& coords = ready[3];
            const auto& active_rows = ready[4];
            const auto& offsets = ready[5];

            auto plan = build_plan(
                SparseMapOp::Forward,
                coords,
                active_rows,
                offsets,
                stride,
                padding
            );
            auto degrees = pool_degrees(plan, shape.out_capacity);
            auto max_ties =
                reduce == PoolReduceOp::Max
                    ? build_max_tie_policy(plan, feats, pooled, shape)
                    : MaxTiePolicy{};
            auto& out = task_outputs[0];
            fill_zero(out);
            auto* out_data = out.data<float>();
            const auto* tangent_data = tangent.data<float>();
            const auto* feat_data = feats.data<float>();
            const auto* pooled_data = pooled.data<float>();
            const auto tangent_s0 = tangent.strides(0);
            const auto tangent_s1 = tangent.strides(1);
            const auto feat_s0 = feats.strides(0);
            const auto feat_s1 = feats.strides(1);
            const auto pooled_s0 = pooled.strides(0);
            const auto pooled_s1 = pooled.strides(1);

            for (auto edge : plan.edges) {
                auto in_row = edge[0];
                auto out_row = edge[1];
                auto* out_row_data =
                    out_data +
                    static_cast<std::ptrdiff_t>(out_row) * shape.channels;
                auto denom = std::max(degrees[out_row], int32_t{1});

                for (int channel = 0; channel < shape.channels; ++channel) {
                    auto feat_value = feat_data
                        [static_cast<std::ptrdiff_t>(in_row) * feat_s0 +
                         static_cast<std::ptrdiff_t>(channel) * feat_s1];
                    auto pooled_value = pooled_data
                        [static_cast<std::ptrdiff_t>(out_row) * pooled_s0 +
                         static_cast<std::ptrdiff_t>(channel) * pooled_s1];
                    auto scale = 1.0F;
                    if (reduce == PoolReduceOp::Max) {
                        if (feat_value != pooled_value) {
                            continue;
                        }
                        auto key =
                            pool_channel_key(out_row, channel, shape.channels);
                        if (pool_edge_rank(edge, shape) !=
                            max_ties.first_ranks[key]) {
                            continue;
                        }
                    } else if (reduce == PoolReduceOp::Avg) {
                        scale = 1.0F / float(denom);
                    }
                    auto tangent_value = tangent_data
                        [static_cast<std::ptrdiff_t>(in_row) * tangent_s0 +
                         static_cast<std::ptrdiff_t>(channel) * tangent_s1];
                    out_row_data[channel] += tangent_value * scale;
                }
            }
        }
    );
}

} // namespace mlx_lattice::exec::cpu
