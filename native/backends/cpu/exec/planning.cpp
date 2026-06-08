#include "backends/cpu/exec/planning.h"

#include <algorithm>
#include <cstddef>
#include <unordered_map>
#include <unordered_set>

namespace mlx_lattice::exec::cpu {
namespace {

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
            auto base = static_cast<std::ptrdiff_t>(row) * coords.strides(0);
            out.push_back({
                data[base],
                data[base + coords.strides(1)],
                data[base + 2 * coords.strides(1)],
                data[base + 3 * coords.strides(1)],
            });
        }
        return out;
    }

    auto data = coords.data<int64_t>();
    for (int row = 0; row < rows; ++row) {
        auto base = static_cast<std::ptrdiff_t>(row) * coords.strides(0);
        out.push_back({
            data[base],
            data[base + coords.strides(1)],
            data[base + 2 * coords.strides(1)],
            data[base + 3 * coords.strides(1)],
        });
    }
    return out;
}

std::vector<Triple> read_offsets(const mx::array& offsets) {
    std::vector<Triple> out;
    out.reserve(offsets.shape(0));
    auto data = offsets.data<int32_t>();
    for (int row = 0; row < offsets.shape(0); ++row) {
        auto base = static_cast<std::ptrdiff_t>(row) * offsets.strides(0);
        out.push_back({
            data[base],
            data[base + offsets.strides(1)],
            data[base + 2 * offsets.strides(1)],
        });
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

} // namespace

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

std::vector<int32_t> pool_degrees(const Plan& plan, int n_out_rows) {
    std::vector<int32_t> degrees(n_out_rows, 0);
    for (auto edge : plan.edges) {
        ++degrees[edge[1]];
    }
    return degrees;
}

} // namespace mlx_lattice::exec::cpu
