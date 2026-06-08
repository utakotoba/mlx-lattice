#include "backends/cpu/exec/algorithms.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mlx_lattice::exec::cpu {

namespace {

using Coord = std::array<int64_t, 4>;
using Edge = std::array<int32_t, 3>;

struct CoordHash {
    size_t operator()(const Coord& coord) const {
        size_t seed = 0;
        for (auto value : coord) {
            auto part = std::hash<int64_t>{}(value);
            seed ^= part + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

struct Plan {
    std::vector<Coord> out_coords;
    std::vector<Edge> edges;
};

int64_t floor_div(int64_t value, int64_t divisor) {
    auto quotient = value / divisor;
    auto remainder = value % divisor;
    if (remainder != 0 && ((remainder < 0) != (divisor < 0))) {
        --quotient;
    }
    return quotient;
}

std::vector<Coord> read_coords(const mx::array& coords, int active_rows) {
    std::vector<Coord> out;
    auto rows = std::min(active_rows, int(coords.shape(0)));
    out.reserve(rows);
    if (coords.dtype() == mx::int32) {
        auto data = coords.data<int32_t>();
        for (int row = 0; row < rows; ++row) {
            auto base = static_cast<std::ptrdiff_t>(row) * 4;
            out.push_back({
                data[base],
                data[base + 1],
                data[base + 2],
                data[base + 3],
            });
        }
        return out;
    }

    auto data = coords.data<int64_t>();
    for (int row = 0; row < rows; ++row) {
        auto base = static_cast<std::ptrdiff_t>(row) * 4;
        out.push_back({
            data[base],
            data[base + 1],
            data[base + 2],
            data[base + 3],
        });
    }
    return out;
}

std::vector<Triple> read_offsets(const mx::array& offsets) {
    std::vector<Triple> out;
    out.reserve(offsets.shape(0));
    auto data = offsets.data<int32_t>();
    for (int row = 0; row < offsets.shape(0); ++row) {
        auto base = static_cast<std::ptrdiff_t>(row) * 3;
        out.push_back({data[base], data[base + 1], data[base + 2]});
    }
    return out;
}

std::unordered_map<Coord, int32_t, CoordHash>
first_row_map(const std::vector<Coord>& coords) {
    std::unordered_map<Coord, int32_t, CoordHash> rows;
    rows.reserve(coords.size());
    for (int row = 0; row < int(coords.size()); ++row) {
        rows.emplace(coords[row], static_cast<int32_t>(row));
    }
    return rows;
}

std::vector<Coord>
downsample_values(const std::vector<Coord>& coords, Triple stride) {
    std::vector<Coord> out;
    out.reserve(coords.size());
    std::unordered_set<Coord, CoordHash> seen;
    seen.reserve(coords.size());
    for (auto coord : coords) {
        Coord quantized = {
            coord[0],
            floor_div(coord[1], stride[0]),
            floor_div(coord[2], stride[1]),
            floor_div(coord[3], stride[2]),
        };
        if (seen.insert(quantized).second) {
            out.push_back(quantized);
        }
    }
    return out;
}

Coord kernel_input_coord(
    Coord out_coord,
    Triple offset,
    Triple stride,
    Triple padding
) {
    return {
        out_coord[0],
        out_coord[1] * stride[0] + offset[0] - padding[0],
        out_coord[2] * stride[1] + offset[1] - padding[1],
        out_coord[3] * stride[2] + offset[2] - padding[2],
    };
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
Plan build_plan(
    SparseMapOp op,
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& offsets,
    Triple stride,
    Triple padding
) {
    auto values = read_coords(coords, active_rows.data<int32_t>()[0]);
    auto kernel_offsets = read_offsets(offsets);
    if (op == SparseMapOp::Generative) {
        Plan plan;
        plan.out_coords.reserve(values.size() * kernel_offsets.size());
        plan.edges.reserve(values.size() * kernel_offsets.size());
        for (int in_row = 0; in_row < int(values.size()); ++in_row) {
            auto coord = values[in_row];
            for (int kernel = 0; kernel < int(kernel_offsets.size());
                 ++kernel) {
                auto offset = kernel_offsets[kernel];
                auto out_row = int(plan.out_coords.size());
                plan.out_coords.push_back({
                    coord[0],
                    coord[1] * stride[0] + offset[0],
                    coord[2] * stride[1] + offset[1],
                    coord[3] * stride[2] + offset[2],
                });
                plan.edges.push_back({
                    static_cast<int32_t>(in_row),
                    static_cast<int32_t>(out_row),
                    static_cast<int32_t>(kernel),
                });
            }
        }
        return plan;
    }

    if (op == SparseMapOp::Transposed) {
        Plan plan;
        std::unordered_map<Coord, int32_t, CoordHash> out_rows;
        auto capacity = values.size() * kernel_offsets.size();
        plan.out_coords.reserve(capacity);
        plan.edges.reserve(capacity);
        out_rows.reserve(capacity);
        for (int in_row = 0; in_row < int(values.size()); ++in_row) {
            auto coord = values[in_row];
            for (int kernel = 0; kernel < int(kernel_offsets.size());
                 ++kernel) {
                auto offset = kernel_offsets[kernel];
                Coord candidate = {
                    coord[0],
                    coord[1] * stride[0] + offset[0] - padding[0],
                    coord[2] * stride[1] + offset[1] - padding[1],
                    coord[3] * stride[2] + offset[2] - padding[2],
                };
                auto [match, inserted] = out_rows.emplace(
                    candidate, static_cast<int32_t>(plan.out_coords.size())
                );
                if (inserted) {
                    plan.out_coords.push_back(candidate);
                }
                plan.edges.push_back({
                    static_cast<int32_t>(in_row),
                    match->second,
                    static_cast<int32_t>(kernel),
                });
            }
        }
        return plan;
    }

    Plan plan;
    auto rows = first_row_map(values);
    bool identity_out = stride == Triple{1, 1, 1} && padding == Triple{0, 0, 0};
    plan.out_coords = identity_out ? values : downsample_values(values, stride);
    plan.edges.reserve(plan.out_coords.size() * kernel_offsets.size());
    for (int kernel = 0; kernel < int(kernel_offsets.size()); ++kernel) {
        auto offset = kernel_offsets[kernel];
        for (int out_row = 0; out_row < int(plan.out_coords.size());
             ++out_row) {
            auto candidate = kernel_input_coord(
                plan.out_coords[out_row], offset, stride, padding
            );
            auto match = rows.find(candidate);
            if (match != rows.end()) {
                plan.edges.push_back({
                    match->second,
                    static_cast<int32_t>(out_row),
                    static_cast<int32_t>(kernel),
                });
            }
        }
    }
    return plan;
}
// NOLINTEND(bugprone-easily-swappable-parameters)

void write_coords(mx::array& out, const std::vector<Coord>& coords) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
    if (out.dtype() == mx::int32) {
        auto data = out.data<int32_t>();
        std::fill(data, data + out.size(), 0);
        for (int row = 0; row < int(coords.size()); ++row) {
            auto base = static_cast<std::ptrdiff_t>(row) * 4;
            for (int axis = 0; axis < 4; ++axis) {
                data[base + axis] = static_cast<int32_t>(coords[row][axis]);
            }
        }
        return;
    }

    auto data = out.data<int64_t>();
    std::fill(data, data + out.size(), 0);
    for (int row = 0; row < int(coords.size()); ++row) {
        auto base = static_cast<std::ptrdiff_t>(row) * 4;
        for (int axis = 0; axis < 4; ++axis) {
            data[base + axis] = coords[row][axis];
        }
    }
}

void write_counts(mx::array& out, const Plan& plan) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto data = out.data<int32_t>();
    std::fill(data, data + out.size(), 0);
    data[0] = int(plan.edges.size());
    data[1] = int(plan.out_coords.size());
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

std::vector<int32_t> pool_degrees(const Plan& plan, int n_out_rows) {
    std::vector<int32_t> degrees(n_out_rows, 0);
    for (auto edge : plan.edges) {
        ++degrees[edge[1]];
    }
    return degrees;
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

    for (auto edge : plan.edges) {
        auto in_row = edge[0];
        auto out_row = edge[1];
        auto kernel = edge[2];
        const auto* feat_row =
            feat_data + static_cast<std::ptrdiff_t>(in_row) * shape.in_channels;
        const auto* weight_kernel =
            weight_data + static_cast<std::ptrdiff_t>(kernel) *
                              shape.in_channels * shape.out_channels;
        auto* out_row_data = out_data + static_cast<std::ptrdiff_t>(out_row) *
                                            shape.out_channels;
        for (int ci = 0; ci < shape.in_channels; ++ci) {
            const auto value = feat_row[ci];
            const auto* weight_row =
                weight_kernel +
                static_cast<std::ptrdiff_t>(ci) * shape.out_channels;
            for (int co = 0; co < shape.out_channels; ++co) {
                out_row_data[co] += value * weight_row[co];
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

    for (auto edge : plan.edges) {
        auto in_row = edge[0];
        auto out_row = edge[1];
        auto kernel = edge[2];
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

void eval_sparse_conv_weight_grad(
    SparseMapOp op,
    SparseConvShape shape,
    Triple stride,
    Triple padding,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
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

    for (auto edge : plan.edges) {
        auto in_row = edge[0];
        auto out_row = edge[1];
        auto kernel = edge[2];
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

void eval_sparse_pool(
    PoolReduceOp reduce,
    SparsePoolShape shape,
    Triple stride,
    Triple padding,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
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
    for (auto edge : plan.edges) {
        auto in_row = edge[0];
        auto out_row = edge[1];
        const auto* feat_row =
            feat_data + static_cast<std::ptrdiff_t>(in_row) * shape.channels;
        auto* out_row_data =
            out_data + static_cast<std::ptrdiff_t>(out_row) * shape.channels;
        for (int channel = 0; channel < shape.channels; ++channel) {
            if (reduce == PoolReduceOp::Max) {
                out_row_data[channel] =
                    std::max(out_row_data[channel], feat_row[channel]);
            } else {
                out_row_data[channel] += feat_row[channel];
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

    for (auto edge : plan.edges) {
        auto in_row = edge[0];
        auto out_row = edge[1];
        auto* grad_row =
            grad_data + static_cast<std::ptrdiff_t>(in_row) * shape.channels;
        const auto* cotangent_row =
            cotangent_data +
            static_cast<std::ptrdiff_t>(out_row) * shape.channels;
        const auto* feat_row =
            feat_data + static_cast<std::ptrdiff_t>(in_row) * shape.channels;
        const auto* pooled_row =
            pooled_data + static_cast<std::ptrdiff_t>(out_row) * shape.channels;
        auto denom = std::max(degrees[out_row], int32_t{1});

        for (int channel = 0; channel < shape.channels; ++channel) {
            if (reduce == PoolReduceOp::Max &&
                feat_row[channel] != pooled_row[channel]) {
                continue;
            }
            auto scale =
                reduce == PoolReduceOp::Avg ? 1.0F / float(denom) : 1.0F;
            grad_row[channel] += cotangent_row[channel] * scale;
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

    for (auto edge : plan.edges) {
        auto in_row = edge[0];
        auto out_row = edge[1];
        const auto* tangent_row =
            tangent_data + static_cast<std::ptrdiff_t>(in_row) * shape.channels;
        const auto* feat_row =
            feat_data + static_cast<std::ptrdiff_t>(in_row) * shape.channels;
        const auto* pooled_row =
            pooled_data + static_cast<std::ptrdiff_t>(out_row) * shape.channels;
        auto* out_row_data =
            out_data + static_cast<std::ptrdiff_t>(out_row) * shape.channels;
        auto denom = std::max(degrees[out_row], int32_t{1});

        for (int channel = 0; channel < shape.channels; ++channel) {
            if (reduce == PoolReduceOp::Max &&
                feat_row[channel] != pooled_row[channel]) {
                continue;
            }
            auto scale =
                reduce == PoolReduceOp::Avg ? 1.0F / float(denom) : 1.0F;
            out_row_data[channel] += tangent_row[channel] * scale;
        }
    }
}

} // namespace mlx_lattice::exec::cpu
