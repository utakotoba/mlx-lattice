#include "backends/cpu/exec/algorithms.h"

#include "backends/cpu/exec/planning.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "mlx/transforms.h"

namespace mlx_lattice::exec::cpu {

namespace {

void wait_for_inputs(const std::vector<mx::array>& inputs) {
    auto ready = inputs;
    mx::eval(ready);
    for (auto& input : ready) {
        input.wait();
    }
}

void fill_zero(mx::array& out) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto data = out.data<float>();
    std::fill(data, data + out.size(), 0.0F);
}

void fill_pool_init(mx::array& out, PoolReduceOp reduce) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
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

} // namespace

void eval_spmm_edges(
    SpmmEdgesShape shape,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    const auto& feats = inputs[0];
    const auto& weights = inputs[1];
    const auto& in_rows = inputs[2];
    const auto& out_rows = inputs[3];
    const auto& kernel_ids = inputs[4];
    const auto& edge_count = inputs[5];

    auto& out = outputs[0];
    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto out_data = out.data<float>();
    std::fill(out_data, out_data + out.size(), 0.0F);

    const auto* feat_data = feats.data<float>();
    const auto* weight_data = weights.data<float>();
    const auto* in_data = in_rows.data<int32_t>();
    const auto* out_index_data = out_rows.data<int32_t>();
    const auto* kernel_data = kernel_ids.data<int32_t>();
    const auto active_edges = edge_count.data<int32_t>()[0];

    const auto in_channels = shape.in_channels;
    const auto out_channels = shape.out_channels;
    for (int edge = 0; edge < active_edges; ++edge) {
        const auto in_row = in_data[edge];
        const auto out_row = out_index_data[edge];
        const auto kernel = kernel_data[edge];
        if (in_row < 0 || in_row >= feats.shape(0) || out_row < 0 ||
            out_row >= shape.n_out_rows || kernel < 0 ||
            kernel >= weights.shape(0)) {
            throw std::out_of_range("spmm_edges row index is out of bounds.");
        }

        const auto* feat_row =
            feat_data + static_cast<std::ptrdiff_t>(in_row) * in_channels;
        const auto* weight_kernel =
            weight_data +
            static_cast<std::ptrdiff_t>(kernel) * in_channels * out_channels;
        auto* out_row_data =
            out_data + static_cast<std::ptrdiff_t>(out_row) * out_channels;

        for (int ci = 0; ci < in_channels; ++ci) {
            const auto value = feat_row[ci];
            const auto* weight_row =
                weight_kernel + static_cast<std::ptrdiff_t>(ci) * out_channels;
            for (int co = 0; co < out_channels; ++co) {
                out_row_data[co] += value * weight_row[co];
            }
        }
    }
}

void eval_spmm_edges_input_grad(
    SpmmEdgesShape shape,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    const auto& cotangent = inputs[0];
    const auto& weights = inputs[1];
    const auto& in_rows = inputs[2];
    const auto& out_rows = inputs[3];
    const auto& kernel_ids = inputs[4];
    const auto& edge_count = inputs[5];

    auto& grad = outputs[0];
    grad.set_data(mx::allocator::malloc(grad.nbytes()));
    auto grad_data = grad.data<float>();
    std::fill(grad_data, grad_data + grad.size(), 0.0F);

    const auto* cotangent_data = cotangent.data<float>();
    const auto* weight_data = weights.data<float>();
    const auto* in_data = in_rows.data<int32_t>();
    const auto* out_index_data = out_rows.data<int32_t>();
    const auto* kernel_data = kernel_ids.data<int32_t>();
    const auto active_edges = edge_count.data<int32_t>()[0];

    for (int edge = 0; edge < active_edges; ++edge) {
        const auto in_row = in_data[edge];
        const auto out_row = out_index_data[edge];
        const auto kernel = kernel_data[edge];
        if (in_row < 0 || in_row >= shape.n_in_rows || out_row < 0 ||
            out_row >= shape.n_out_rows || kernel < 0 ||
            kernel >= shape.n_kernels) {
            throw std::out_of_range(
                "spmm_edges input gradient row index is out of bounds."
            );
        }

        auto* grad_row =
            grad_data + static_cast<std::ptrdiff_t>(in_row) * shape.in_channels;
        const auto* cotangent_row =
            cotangent_data +
            static_cast<std::ptrdiff_t>(out_row) * shape.out_channels;
        const auto* weight_kernel =
            weight_data + static_cast<std::ptrdiff_t>(kernel) *
                              shape.in_channels * shape.out_channels;

        for (int ci = 0; ci < shape.in_channels; ++ci) {
            const auto* weight_row =
                weight_kernel +
                static_cast<std::ptrdiff_t>(ci) * shape.out_channels;
            for (int co = 0; co < shape.out_channels; ++co) {
                grad_row[ci] += cotangent_row[co] * weight_row[co];
            }
        }
    }
}

void eval_spmm_edges_weight_grad(
    SpmmEdgesShape shape,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    const auto& feats = inputs[0];
    const auto& cotangent = inputs[1];
    const auto& in_rows = inputs[2];
    const auto& out_rows = inputs[3];
    const auto& kernel_ids = inputs[4];
    const auto& edge_count = inputs[5];

    auto& grad = outputs[0];
    grad.set_data(mx::allocator::malloc(grad.nbytes()));
    auto grad_data = grad.data<float>();
    std::fill(grad_data, grad_data + grad.size(), 0.0F);

    const auto* feat_data = feats.data<float>();
    const auto* cotangent_data = cotangent.data<float>();
    const auto* in_data = in_rows.data<int32_t>();
    const auto* out_index_data = out_rows.data<int32_t>();
    const auto* kernel_data = kernel_ids.data<int32_t>();
    const auto active_edges = edge_count.data<int32_t>()[0];

    for (int edge = 0; edge < active_edges; ++edge) {
        const auto in_row = in_data[edge];
        const auto out_row = out_index_data[edge];
        const auto kernel = kernel_data[edge];
        if (in_row < 0 || in_row >= shape.n_in_rows || out_row < 0 ||
            out_row >= shape.n_out_rows || kernel < 0 ||
            kernel >= shape.n_kernels) {
            throw std::out_of_range(
                "spmm_edges weight gradient row index is out of bounds."
            );
        }

        const auto* feat_row =
            feat_data + static_cast<std::ptrdiff_t>(in_row) * shape.in_channels;
        const auto* cotangent_row =
            cotangent_data +
            static_cast<std::ptrdiff_t>(out_row) * shape.out_channels;
        auto* grad_kernel = grad_data + static_cast<std::ptrdiff_t>(kernel) *
                                            shape.in_channels *
                                            shape.out_channels;

        for (int ci = 0; ci < shape.in_channels; ++ci) {
            auto* grad_row = grad_kernel + static_cast<std::ptrdiff_t>(ci) *
                                               shape.out_channels;
            for (int co = 0; co < shape.out_channels; ++co) {
                grad_row[co] += feat_row[ci] * cotangent_row[co];
            }
        }
    }
}

void eval_pool_edges(
    PoolReduceOp op,
    PoolEdgesShape shape,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    const auto& feats = inputs[0];
    const auto& in_rows = inputs[1];
    const auto& out_rows = inputs[2];
    const auto& edge_count = inputs[3];

    auto& out = outputs[0];
    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto out_data = out.data<float>();
    if (op == PoolReduceOp::Sum) {
        std::fill(out_data, out_data + out.size(), 0.0F);
    } else {
        std::fill(
            out_data,
            out_data + out.size(),
            -std::numeric_limits<float>::infinity()
        );
    }

    const auto* feat_data = feats.data<float>();
    const auto* in_data = in_rows.data<int32_t>();
    const auto* out_index_data = out_rows.data<int32_t>();
    const auto active_edges = edge_count.data<int32_t>()[0];
    for (int edge = 0; edge < active_edges; ++edge) {
        const auto in_row = in_data[edge];
        const auto out_row = out_index_data[edge];
        if (in_row < 0 || in_row >= shape.n_in_rows || out_row < 0 ||
            out_row >= shape.n_out_rows) {
            throw std::out_of_range(
                "pool edge reduction row index is out of bounds."
            );
        }

        const auto* feat_row =
            feat_data + static_cast<std::ptrdiff_t>(in_row) * shape.channels;
        auto* out_row_data =
            out_data + static_cast<std::ptrdiff_t>(out_row) * shape.channels;
        for (int channel = 0; channel < shape.channels; ++channel) {
            if (op == PoolReduceOp::Sum) {
                out_row_data[channel] += feat_row[channel];
            } else {
                out_row_data[channel] =
                    std::max(out_row_data[channel], feat_row[channel]);
            }
        }
    }
}

void eval_pool_edges_grad(
    PoolReduceOp op,
    PoolEdgesShape shape,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    const auto& cotangent = inputs[0];
    const auto& feats = inputs[1];
    const auto& pooled = inputs[2];
    const auto& in_rows = inputs[3];
    const auto& out_rows = inputs[4];
    const auto& edge_count = inputs[5];

    auto& grad = outputs[0];
    grad.set_data(mx::allocator::malloc(grad.nbytes()));
    auto grad_data = grad.data<float>();
    std::fill(grad_data, grad_data + grad.size(), 0.0F);

    const auto* cotangent_data = cotangent.data<float>();
    const auto* feat_data = feats.data<float>();
    const auto* pooled_data = pooled.data<float>();
    const auto* in_data = in_rows.data<int32_t>();
    const auto* out_index_data = out_rows.data<int32_t>();
    const auto active_edges = edge_count.data<int32_t>()[0];

    for (int edge = 0; edge < active_edges; ++edge) {
        const auto in_row = in_data[edge];
        const auto out_row = out_index_data[edge];
        if (in_row < 0 || in_row >= shape.n_in_rows || out_row < 0 ||
            out_row >= shape.n_out_rows) {
            throw std::out_of_range(
                "pool edge gradient row index is out of bounds."
            );
        }

        const auto* cotangent_row =
            cotangent_data +
            static_cast<std::ptrdiff_t>(out_row) * shape.channels;
        const auto* feat_row =
            feat_data + static_cast<std::ptrdiff_t>(in_row) * shape.channels;
        const auto* pooled_row =
            pooled_data + static_cast<std::ptrdiff_t>(out_row) * shape.channels;
        auto* grad_row =
            grad_data + static_cast<std::ptrdiff_t>(in_row) * shape.channels;

        for (int channel = 0; channel < shape.channels; ++channel) {
            if (op == PoolReduceOp::Sum ||
                feat_row[channel] == pooled_row[channel]) {
                grad_row[channel] += cotangent_row[channel];
            }
        }
    }
}

void eval_pool_max_edges_jvp(
    PoolEdgesShape shape,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    const auto& tangent = inputs[0];
    const auto& feats = inputs[1];
    const auto& pooled = inputs[2];
    const auto& in_rows = inputs[3];
    const auto& out_rows = inputs[4];
    const auto& edge_count = inputs[5];

    auto& out = outputs[0];
    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto out_data = out.data<float>();
    std::fill(out_data, out_data + out.size(), 0.0F);

    const auto* tangent_data = tangent.data<float>();
    const auto* feat_data = feats.data<float>();
    const auto* pooled_data = pooled.data<float>();
    const auto* in_data = in_rows.data<int32_t>();
    const auto* out_index_data = out_rows.data<int32_t>();
    const auto active_edges = edge_count.data<int32_t>()[0];

    for (int edge = 0; edge < active_edges; ++edge) {
        const auto in_row = in_data[edge];
        const auto out_row = out_index_data[edge];
        if (in_row < 0 || in_row >= shape.n_in_rows || out_row < 0 ||
            out_row >= shape.n_out_rows) {
            throw std::out_of_range("pool max JVP row index is out of bounds.");
        }

        const auto* tangent_row =
            tangent_data + static_cast<std::ptrdiff_t>(in_row) * shape.channels;
        const auto* feat_row =
            feat_data + static_cast<std::ptrdiff_t>(in_row) * shape.channels;
        const auto* pooled_row =
            pooled_data + static_cast<std::ptrdiff_t>(out_row) * shape.channels;
        auto* out_row_data =
            out_data + static_cast<std::ptrdiff_t>(out_row) * shape.channels;

        for (int channel = 0; channel < shape.channels; ++channel) {
            if (feat_row[channel] == pooled_row[channel]) {
                out_row_data[channel] += tangent_row[channel];
            }
        }
    }
}

void eval_sparse_conv(
    SparseMapOp op,
    SparseConvShape shape,
    Triple stride,
    Triple padding,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    wait_for_inputs(inputs);
    const auto& coords = inputs[0];
    const auto& active_rows = inputs[1];
    const auto& feats = inputs[2];
    const auto& weights = inputs[3];
    const auto& offsets = inputs[4];

    auto plan = build_plan(op, coords, active_rows, offsets, stride, padding);
    write_coords(outputs[SparseOutCoords], plan.out_coords);
    write_counts(outputs[SparseCounts], plan);

    auto& out = outputs[SparseOutFeats];
    fill_zero(out);
    auto* out_data = out.data<float>();
    const auto* feat_data = feats.data<float>();
    const auto* weight_data = weights.data<float>();
    const auto feat_s0 = feats.strides(0);
    const auto feat_s1 = feats.strides(1);

    for (auto edge : plan.edges) {
        auto in_row = edge[0];
        auto out_row = edge[1];
        auto kernel = edge[2];
        auto* out_row_data = out_data + static_cast<std::ptrdiff_t>(out_row) *
                                            shape.out_channels;
        for (int ci = 0; ci < shape.in_channels; ++ci) {
            const auto value = feat_data
                [static_cast<std::ptrdiff_t>(in_row) * feat_s0 +
                 static_cast<std::ptrdiff_t>(ci) * feat_s1];
            for (int co = 0; co < shape.out_channels; ++co) {
                out_row_data[co] +=
                    value *
                    weight_data[weight_offset(weights, shape, kernel, ci, co)];
            }
        }
    }
}

void eval_sparse_conv_input_grad(
    SparseMapOp op,
    SparseConvShape shape,
    Triple stride,
    Triple padding,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    wait_for_inputs(inputs);
    const auto& cotangent = inputs[0];
    const auto& coords = inputs[1];
    const auto& active_rows = inputs[2];
    const auto& weights = inputs[3];
    const auto& offsets = inputs[4];

    auto plan = build_plan(op, coords, active_rows, offsets, stride, padding);
    auto& grad = outputs[0];
    fill_zero(grad);
    auto* grad_data = grad.data<float>();
    const auto* cotangent_data = cotangent.data<float>();
    const auto* weight_data = weights.data<float>();
    const auto cotangent_s0 = cotangent.strides(0);
    const auto cotangent_s1 = cotangent.strides(1);

    for (auto edge : plan.edges) {
        auto in_row = edge[0];
        auto out_row = edge[1];
        auto kernel = edge[2];
        auto* grad_row =
            grad_data + static_cast<std::ptrdiff_t>(in_row) * shape.in_channels;
        for (int ci = 0; ci < shape.in_channels; ++ci) {
            for (int co = 0; co < shape.out_channels; ++co) {
                auto cotangent_index =
                    static_cast<std::ptrdiff_t>(out_row) * cotangent_s0 +
                    static_cast<std::ptrdiff_t>(co) * cotangent_s1;
                grad_row[ci] +=
                    cotangent_data[cotangent_index] *
                    weight_data[weight_offset(weights, shape, kernel, ci, co)];
            }
        }
    }
}

void eval_sparse_conv_weight_grad(
    SparseMapOp op,
    SparseConvShape shape,
    Triple stride,
    Triple padding,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    wait_for_inputs(inputs);
    const auto& feats = inputs[0];
    const auto& cotangent = inputs[1];
    const auto& coords = inputs[2];
    const auto& active_rows = inputs[3];
    const auto& offsets = inputs[4];

    auto plan = build_plan(op, coords, active_rows, offsets, stride, padding);
    auto& grad = outputs[0];
    fill_zero(grad);
    auto* grad_data = grad.data<float>();
    const auto* feat_data = feats.data<float>();
    const auto* cotangent_data = cotangent.data<float>();
    const auto feat_s0 = feats.strides(0);
    const auto feat_s1 = feats.strides(1);
    const auto cotangent_s0 = cotangent.strides(0);
    const auto cotangent_s1 = cotangent.strides(1);

    for (auto edge : plan.edges) {
        auto in_row = edge[0];
        auto out_row = edge[1];
        auto kernel = edge[2];
        for (int ci = 0; ci < shape.in_channels; ++ci) {
            for (int co = 0; co < shape.out_channels; ++co) {
                auto feat_index =
                    static_cast<std::ptrdiff_t>(in_row) * feat_s0 +
                    static_cast<std::ptrdiff_t>(ci) * feat_s1;
                auto cotangent_index =
                    static_cast<std::ptrdiff_t>(out_row) * cotangent_s0 +
                    static_cast<std::ptrdiff_t>(co) * cotangent_s1;
                grad_data[dense_weight_offset(shape, kernel, ci, co)] +=
                    feat_data[feat_index] * cotangent_data[cotangent_index];
            }
        }
    }
}

void eval_sparse_pool(
    PoolReduceOp reduce,
    SparsePoolShape shape,
    Triple stride,
    Triple padding,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    wait_for_inputs(inputs);
    const auto& coords = inputs[0];
    const auto& active_rows = inputs[1];
    const auto& feats = inputs[2];
    const auto& offsets = inputs[3];

    auto plan = build_plan(
        SparseMapOp::Forward, coords, active_rows, offsets, stride, padding
    );
    write_coords(outputs[SparseOutCoords], plan.out_coords);
    write_counts(outputs[SparseCounts], plan);

    auto& out = outputs[SparseOutFeats];
    fill_pool_init(out, reduce);
    auto* out_data = out.data<float>();
    const auto* feat_data = feats.data<float>();
    const auto feat_s0 = feats.strides(0);
    const auto feat_s1 = feats.strides(1);
    for (auto edge : plan.edges) {
        auto in_row = edge[0];
        auto out_row = edge[1];
        auto* out_row_data =
            out_data + static_cast<std::ptrdiff_t>(out_row) * shape.channels;
        for (int channel = 0; channel < shape.channels; ++channel) {
            auto value = feat_data
                [static_cast<std::ptrdiff_t>(in_row) * feat_s0 +
                 static_cast<std::ptrdiff_t>(channel) * feat_s1];
            if (reduce == PoolReduceOp::Max) {
                out_row_data[channel] = std::max(out_row_data[channel], value);
            } else {
                out_row_data[channel] += value;
            }
        }
    }

    if (reduce == PoolReduceOp::Avg) {
        auto degrees = pool_degrees(plan, shape.n_out_rows);
        for (int row = 0; row < int(plan.out_coords.size()); ++row) {
            auto denom = std::max(degrees[row], int32_t{1});
            auto* out_row =
                out_data + static_cast<std::ptrdiff_t>(row) * shape.channels;
            for (int channel = 0; channel < shape.channels; ++channel) {
                out_row[channel] /= float(denom);
            }
        }
    }
}

void eval_sparse_pool_grad(
    PoolReduceOp reduce,
    SparsePoolShape shape,
    Triple stride,
    Triple padding,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    wait_for_inputs(inputs);
    const auto& cotangent = inputs[0];
    const auto& feats = inputs[1];
    const auto& pooled = inputs[2];
    const auto& coords = inputs[3];
    const auto& active_rows = inputs[4];
    const auto& offsets = inputs[5];

    auto plan = build_plan(
        SparseMapOp::Forward, coords, active_rows, offsets, stride, padding
    );
    auto degrees = pool_degrees(plan, shape.n_out_rows);
    auto& grad = outputs[0];
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
            grad_data + static_cast<std::ptrdiff_t>(in_row) * shape.channels;
        auto denom = std::max(degrees[out_row], int32_t{1});

        for (int channel = 0; channel < shape.channels; ++channel) {
            auto feat_value = feat_data
                [static_cast<std::ptrdiff_t>(in_row) * feat_s0 +
                 static_cast<std::ptrdiff_t>(channel) * feat_s1];
            auto pooled_value = pooled_data
                [static_cast<std::ptrdiff_t>(out_row) * pooled_s0 +
                 static_cast<std::ptrdiff_t>(channel) * pooled_s1];
            if (reduce == PoolReduceOp::Max && feat_value != pooled_value) {
                continue;
            }
            auto scale =
                reduce == PoolReduceOp::Avg ? 1.0F / float(denom) : 1.0F;
            auto cotangent_value = cotangent_data
                [static_cast<std::ptrdiff_t>(out_row) * cotangent_s0 +
                 static_cast<std::ptrdiff_t>(channel) * cotangent_s1];
            grad_row[channel] += cotangent_value * scale;
        }
    }
}

void eval_sparse_pool_jvp(
    PoolReduceOp reduce,
    SparsePoolShape shape,
    Triple stride,
    Triple padding,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    wait_for_inputs(inputs);
    const auto& tangent = inputs[0];
    const auto& feats = inputs[1];
    const auto& pooled = inputs[2];
    const auto& coords = inputs[3];
    const auto& active_rows = inputs[4];
    const auto& offsets = inputs[5];

    auto plan = build_plan(
        SparseMapOp::Forward, coords, active_rows, offsets, stride, padding
    );
    auto degrees = pool_degrees(plan, shape.n_out_rows);
    auto& out = outputs[0];
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
            out_data + static_cast<std::ptrdiff_t>(out_row) * shape.channels;
        auto denom = std::max(degrees[out_row], int32_t{1});

        for (int channel = 0; channel < shape.channels; ++channel) {
            auto feat_value = feat_data
                [static_cast<std::ptrdiff_t>(in_row) * feat_s0 +
                 static_cast<std::ptrdiff_t>(channel) * feat_s1];
            auto pooled_value = pooled_data
                [static_cast<std::ptrdiff_t>(out_row) * pooled_s0 +
                 static_cast<std::ptrdiff_t>(channel) * pooled_s1];
            if (reduce == PoolReduceOp::Max && feat_value != pooled_value) {
                continue;
            }
            auto scale =
                reduce == PoolReduceOp::Avg ? 1.0F / float(denom) : 1.0F;
            auto tangent_value = tangent_data
                [static_cast<std::ptrdiff_t>(in_row) * tangent_s0 +
                 static_cast<std::ptrdiff_t>(channel) * tangent_s1];
            out_row_data[channel] += tangent_value * scale;
        }
    }
}

} // namespace mlx_lattice::exec::cpu
