#include "backends/cpu/pool/algorithms.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "backends/array_utils.h"
#include "backends/cpu/schedule.h"

namespace mlx_lattice::backend::cpu::pool {
namespace {

void fill_zero(mx::array& out) {
    auto* data = out.data<float>();
    std::fill(data, data + out.size(), 0.0F);
}

void fill_reduction_identity(mx::array& out, PoolReduceOp reduce) {
    auto* data = out.data<float>();
    auto value = reduce == PoolReduceOp::Max
                     ? -std::numeric_limits<float>::infinity()
                     : 0.0F;
    std::fill(data, data + out.size(), value);
}

std::size_t channel_key(int out_row, int channel, int channels) {
    return static_cast<std::size_t>(out_row) *
               static_cast<std::size_t>(channels) +
           static_cast<std::size_t>(channel);
}

int edge_rank(
    const int32_t* in_rows,
    const int32_t* kernel_ids,
    int edge,
    SparsePoolShape shape
) {
    return in_rows[edge] * shape.n_kernels + kernel_ids[edge];
}

struct RelationView {
    const int32_t* in_rows;
    const int32_t* out_rows;
    const int32_t* kernel_ids;
    const int32_t* row_offsets;
    const int32_t* counts;
    const int32_t* in_row_offsets;
    const int32_t* in_edge_ids;
};

RelationView relation_view(const std::vector<mx::array>& inputs) {
    return {
        inputs[1].data<int32_t>(),
        inputs[2].data<int32_t>(),
        inputs[3].data<int32_t>(),
        inputs[4].data<int32_t>(),
        inputs[5].data<int32_t>(),
        nullptr,
        nullptr,
    };
}

RelationView autodiff_relation_view(const std::vector<mx::array>& inputs) {
    return {
        inputs[3].data<int32_t>(),
        inputs[4].data<int32_t>(),
        inputs[5].data<int32_t>(),
        inputs[6].data<int32_t>(),
        inputs[7].data<int32_t>(),
        inputs[8].data<int32_t>(),
        inputs[9].data<int32_t>(),
    };
}

RelationView
autodiff_forward_relation_view(const std::vector<mx::array>& inputs) {
    return {
        inputs[3].data<int32_t>(),
        inputs[4].data<int32_t>(),
        inputs[5].data<int32_t>(),
        inputs[6].data<int32_t>(),
        inputs[7].data<int32_t>(),
        nullptr,
        nullptr,
    };
}

int32_t row_degree(RelationView relation, int out_row) {
    return relation.row_offsets[out_row + 1] - relation.row_offsets[out_row];
}

std::vector<int32_t> row_degrees(RelationView relation, int out_capacity) {
    std::vector<int32_t> out(static_cast<std::size_t>(out_capacity), 0);
    for (int row = 0; row < out_capacity; ++row) {
        out[static_cast<std::size_t>(row)] = row_degree(relation, row);
    }
    return out;
}

struct MaxTiePolicy {
    std::vector<int32_t> counts;
    std::vector<int32_t> first_ranks;
};

struct MaxTieQuery {
    float pooled_value;
    int out_row;
    int channel;
};

MaxTiePolicy build_max_tie_policy(
    RelationView relation,
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
    auto out_count = std::min(relation.counts[1], shape.out_capacity);

    for (int out_row = 0; out_row < out_count; ++out_row) {
        for (int edge = relation.row_offsets[out_row];
             edge < relation.row_offsets[out_row + 1];
             ++edge) {
            auto in_row = relation.in_rows[edge];
            for (int channel = 0; channel < shape.channels; ++channel) {
                auto feat_value = feat_data
                    [static_cast<std::ptrdiff_t>(in_row) * feats.strides(0) +
                     static_cast<std::ptrdiff_t>(channel) * feats.strides(1)];
                auto pooled_value = pooled_data
                    [static_cast<std::ptrdiff_t>(out_row) * pooled.strides(0) +
                     static_cast<std::ptrdiff_t>(channel) * pooled.strides(1)];
                if (feat_value != pooled_value) {
                    continue;
                }
                auto key = channel_key(out_row, channel, shape.channels);
                ++policy.counts[key];
                policy.first_ranks[key] = std::min(
                    policy.first_ranks[key],
                    edge_rank(
                        relation.in_rows, relation.kernel_ids, edge, shape
                    )
                );
            }
        }
    }
    return policy;
}

int max_tie_count(
    RelationView relation,
    const mx::array& feats,
    MaxTieQuery query
) {
    const auto* feat_data = feats.data<float>();
    auto count = 0;
    for (int edge = relation.row_offsets[query.out_row];
         edge < relation.row_offsets[query.out_row + 1];
         ++edge) {
        auto in_row = relation.in_rows[edge];
        auto feat_value = feat_data
            [static_cast<std::ptrdiff_t>(in_row) * feats.strides(0) +
             static_cast<std::ptrdiff_t>(query.channel) * feats.strides(1)];
        if (feat_value == query.pooled_value) {
            ++count;
        }
    }
    return count;
}

} // namespace

void eval(
    PoolReduceOp reduce,
    SparsePoolShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    allocate_all(outputs);
    schedule_cpu(
        stream,
        inputs,
        outputs,
        [reduce, shape](
            const std::vector<mx::array>& ready,
            std::vector<mx::array>& task_outputs
        ) {
            const auto& feats = ready[0];
            auto relation = relation_view(ready);
            auto& out = task_outputs[0];
            fill_reduction_identity(out, reduce);
            auto* out_data = out.data<float>();
            const auto* feat_data = feats.data<float>();
            auto out_count = std::min(relation.counts[1], shape.out_capacity);

            for (int out_row = 0; out_row < out_count; ++out_row) {
                auto* out_row_data =
                    out_data +
                    static_cast<std::ptrdiff_t>(out_row) * shape.channels;
                for (int edge = relation.row_offsets[out_row];
                     edge < relation.row_offsets[out_row + 1];
                     ++edge) {
                    auto in_row = relation.in_rows[edge];
                    for (int channel = 0; channel < shape.channels; ++channel) {
                        auto value = feat_data
                            [static_cast<std::ptrdiff_t>(in_row) *
                                 feats.strides(0) +
                             static_cast<std::ptrdiff_t>(channel) *
                                 feats.strides(1)];
                        if (reduce == PoolReduceOp::Max) {
                            out_row_data[channel] =
                                std::max(out_row_data[channel], value);
                        } else {
                            out_row_data[channel] += value;
                        }
                    }
                }
            }

            if (reduce != PoolReduceOp::Avg) {
                return;
            }
            auto degrees = row_degrees(relation, shape.out_capacity);
            for (int row = 0; row < out_count; ++row) {
                auto scale = 1.0F / float(std::max(degrees[row], int32_t{1}));
                auto* out_row_data =
                    out_data +
                    static_cast<std::ptrdiff_t>(row) * shape.channels;
                for (int channel = 0; channel < shape.channels; ++channel) {
                    out_row_data[channel] *= scale;
                }
            }
        }
    );
}

void eval_grad(
    PoolReduceOp reduce,
    SparsePoolShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    allocate_all(outputs);
    schedule_cpu(
        stream,
        inputs,
        outputs,
        [reduce, shape](
            const std::vector<mx::array>& ready,
            std::vector<mx::array>& task_outputs
        ) {
            const auto& cotangent = ready[0];
            const auto& feats = ready[1];
            const auto& pooled = ready[2];
            auto relation = autodiff_relation_view(ready);

            auto& grad = task_outputs[0];
            auto* grad_data = grad.data<float>();
            const auto* cotangent_data = cotangent.data<float>();
            const auto* feat_data = feats.data<float>();
            const auto* pooled_data = pooled.data<float>();
            auto out_count = std::min(relation.counts[1], shape.out_capacity);

            for (int in_row = 0; in_row < shape.in_capacity; ++in_row) {
                for (int channel = 0; channel < shape.channels; ++channel) {
                    auto value = 0.0F;
                    auto direct_edge = shape.input_exclusive
                                           ? relation.in_edge_ids[in_row]
                                           : -1;
                    auto start = shape.input_exclusive
                                     ? 0
                                     : relation.in_row_offsets[in_row];
                    auto stop = shape.input_exclusive
                                    ? int(direct_edge >= 0)
                                    : relation.in_row_offsets[in_row + 1];
                    for (int cursor = start; cursor < stop; ++cursor) {
                        auto edge = shape.input_exclusive
                                        ? direct_edge
                                        : relation.in_edge_ids[cursor];
                        if (edge < 0) {
                            continue;
                        }
                        auto out_row = relation.out_rows[edge];
                        if (out_row < 0 || out_row >= out_count) {
                            continue;
                        }
                        auto scale = 1.0F;
                        if (reduce == PoolReduceOp::Max) {
                            auto feat_value = feat_data
                                [static_cast<std::ptrdiff_t>(in_row) *
                                     feats.strides(0) +
                                 static_cast<std::ptrdiff_t>(channel) *
                                     feats.strides(1)];
                            auto pooled_value = pooled_data
                                [static_cast<std::ptrdiff_t>(out_row) *
                                     pooled.strides(0) +
                                 static_cast<std::ptrdiff_t>(channel) *
                                     pooled.strides(1)];
                            if (feat_value != pooled_value) {
                                continue;
                            }
                            auto count = max_tie_count(
                                relation,
                                feats,
                                MaxTieQuery{
                                    pooled_value,
                                    out_row,
                                    channel,
                                }
                            );
                            if (count == 0) {
                                continue;
                            }
                            scale = 1.0F / float(count);
                        } else if (reduce == PoolReduceOp::Avg) {
                            auto degree = row_degree(relation, out_row);
                            scale = 1.0F / float(std::max(degree, int32_t{1}));
                        }
                        value += cotangent_data
                                     [static_cast<std::ptrdiff_t>(out_row) *
                                          cotangent.strides(0) +
                                      static_cast<std::ptrdiff_t>(channel) *
                                          cotangent.strides(1)] *
                                 scale;
                    }
                    auto* grad_row =
                        grad_data +
                        static_cast<std::ptrdiff_t>(in_row) * shape.channels;
                    grad_row[channel] = value;
                }
            }
        }
    );
}

void eval_jvp(
    PoolReduceOp reduce,
    SparsePoolShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    allocate_all(outputs);
    schedule_cpu(
        stream,
        inputs,
        outputs,
        [reduce, shape](
            const std::vector<mx::array>& ready,
            std::vector<mx::array>& task_outputs
        ) {
            const auto& tangent = ready[0];
            const auto& feats = ready[1];
            const auto& pooled = ready[2];
            auto relation = autodiff_forward_relation_view(ready);
            auto degrees = row_degrees(relation, shape.out_capacity);
            auto ties =
                reduce == PoolReduceOp::Max
                    ? build_max_tie_policy(relation, feats, pooled, shape)
                    : MaxTiePolicy{};

            auto& out = task_outputs[0];
            fill_zero(out);
            auto* out_data = out.data<float>();
            const auto* tangent_data = tangent.data<float>();
            auto out_count = std::min(relation.counts[1], shape.out_capacity);

            for (int out_row = 0; out_row < out_count; ++out_row) {
                auto* out_row_data =
                    out_data +
                    static_cast<std::ptrdiff_t>(out_row) * shape.channels;
                for (int edge = relation.row_offsets[out_row];
                     edge < relation.row_offsets[out_row + 1];
                     ++edge) {
                    for (int channel = 0; channel < shape.channels; ++channel) {
                        auto scale = 1.0F;
                        if (reduce == PoolReduceOp::Max) {
                            auto key =
                                channel_key(out_row, channel, shape.channels);
                            if (edge_rank(
                                    relation.in_rows,
                                    relation.kernel_ids,
                                    edge,
                                    shape
                                ) != ties.first_ranks[key]) {
                                continue;
                            }
                        } else if (reduce == PoolReduceOp::Avg) {
                            scale =
                                1.0F /
                                float(std::max(degrees[out_row], int32_t{1}));
                        }
                        out_row_data[channel] +=
                            tangent_data
                                [static_cast<std::ptrdiff_t>(
                                     relation.in_rows[edge]
                                 ) * tangent.strides(0) +
                                 static_cast<std::ptrdiff_t>(channel) *
                                     tangent.strides(1)] *
                            scale;
                    }
                }
            }
        }
    );
}

} // namespace mlx_lattice::backend::cpu::pool
