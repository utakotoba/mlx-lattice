#include "backends/cpu/coords.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mlx/device.h"
#include "mlx/ops.h"

namespace mlx_lattice::cpu {

namespace {

// MARK: - types

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

// MARK: - arrays

std::vector<Coord> read_coords(const mx::array& coords) {
    auto cpu_coords = mx::contiguous(coords, false, mx::Device::cpu);
    cpu_coords.eval();
    cpu_coords.wait();

    std::vector<Coord> out;
    out.reserve(cpu_coords.shape(0));
    if (cpu_coords.dtype() == mx::int32) {
        auto data = cpu_coords.data<int32_t>();
        for (int row = 0; row < cpu_coords.shape(0); ++row) {
            auto base = static_cast<ptrdiff_t>(row) * 4;
            out.push_back({
                data[base],
                data[base + 1],
                data[base + 2],
                data[base + 3],
            });
        }
        return out;
    }

    auto data = cpu_coords.data<int64_t>();
    for (int row = 0; row < cpu_coords.shape(0); ++row) {
        auto base = static_cast<ptrdiff_t>(row) * 4;
        out.push_back({
            data[base],
            data[base + 1],
            data[base + 2],
            data[base + 3],
        });
    }
    return out;
}

mx::array make_coords_array(const std::vector<Coord>& coords, mx::Dtype dtype) {
    if (dtype == mx::int32) {
        std::vector<int32_t> data;
        data.reserve(coords.size() * 4);
        for (auto coord : coords) {
            for (auto value : coord) {
                data.push_back(static_cast<int32_t>(value));
            }
        }
        return mx::array(data.begin(), mx::Shape{int(coords.size()), 4}, dtype);
    }

    std::vector<int64_t> data;
    data.reserve(coords.size() * 4);
    for (auto coord : coords) {
        data.insert(data.end(), coord.begin(), coord.end());
    }
    return mx::array(data.begin(), mx::Shape{int(coords.size()), 4}, dtype);
}

mx::array make_i32_array(const std::vector<int32_t>& data) {
    return mx::array(data.begin(), mx::Shape{int(data.size())}, mx::int32);
}

mx::array make_offset_array(const std::vector<Triple>& offsets) {
    std::vector<int32_t> data;
    data.reserve(offsets.size() * 3);
    for (auto offset : offsets) {
        data.insert(data.end(), offset.begin(), offset.end());
    }
    return mx::array(
        data.begin(), mx::Shape{int(offsets.size()), 3}, mx::int32
    );
}

// MARK: - coords

int64_t floor_div(int64_t value, int64_t divisor) {
    auto quotient = value / divisor;
    auto remainder = value % divisor;
    if (remainder != 0 && ((remainder < 0) != (divisor < 0))) {
        --quotient;
    }
    return quotient;
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

std::unordered_map<Coord, int32_t, CoordHash>
first_row_map(const std::vector<Coord>& coords) {
    std::unordered_map<Coord, int32_t, CoordHash> rows;
    rows.reserve(coords.size());
    for (int row = 0; row < int(coords.size()); ++row) {
        rows.emplace(coords[row], static_cast<int32_t>(row));
    }
    return rows;
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

NativeKernelMap make_map(
    const std::vector<Edge>& edges,
    const std::vector<Coord>& out_coords,
    const std::vector<Triple>& offsets,
    mx::Dtype coord_dtype
) {
    std::vector<int32_t> in_rows;
    std::vector<int32_t> out_rows;
    std::vector<int32_t> kernel_ids;
    in_rows.reserve(edges.size());
    out_rows.reserve(edges.size());
    kernel_ids.reserve(edges.size());
    for (auto edge : edges) {
        in_rows.push_back(edge[0]);
        out_rows.push_back(edge[1]);
        kernel_ids.push_back(edge[2]);
    }

    return {
        make_i32_array(in_rows),
        make_i32_array(out_rows),
        make_i32_array(kernel_ids),
        make_coords_array(out_coords, coord_dtype),
        make_offset_array(offsets),
    };
}

} // namespace

// MARK: - set ops

mx::array downsample_coords(const mx::array& coords, Triple stride) {
    return make_coords_array(
        downsample_values(read_coords(coords), stride), coords.dtype()
    );
}

mx::array union_coords(const mx::array& lhs, const mx::array& rhs) {
    auto lhs_values = read_coords(lhs);
    auto rhs_values = read_coords(rhs);
    std::vector<Coord> out;
    out.reserve(lhs_values.size() + rhs_values.size());
    std::unordered_set<Coord, CoordHash> seen;
    seen.reserve(lhs_values.size() + rhs_values.size());

    for (auto coord : lhs_values) {
        if (seen.insert(coord).second) {
            out.push_back(coord);
        }
    }
    for (auto coord : rhs_values) {
        if (seen.insert(coord).second) {
            out.push_back(coord);
        }
    }
    return make_coords_array(out, lhs.dtype());
}

mx::array intersection_coords(const mx::array& lhs, const mx::array& rhs) {
    auto lhs_values = read_coords(lhs);
    auto rhs_values = read_coords(rhs);
    std::unordered_set<Coord, CoordHash> rhs_seen;
    rhs_seen.reserve(rhs_values.size());
    for (auto coord : rhs_values) {
        rhs_seen.insert(coord);
    }

    std::vector<Coord> out;
    out.reserve(std::min(lhs_values.size(), rhs_values.size()));
    std::unordered_set<Coord, CoordHash> emitted;
    emitted.reserve(lhs_values.size());
    for (auto coord : lhs_values) {
        if (rhs_seen.find(coord) != rhs_seen.end() &&
            emitted.insert(coord).second) {
            out.push_back(coord);
        }
    }
    return make_coords_array(out, lhs.dtype());
}

mx::array lookup_coords(const mx::array& coords, const mx::array& queries) {
    auto rows = first_row_map(read_coords(coords));
    auto query_values = read_coords(queries);
    std::vector<int32_t> out;
    out.reserve(query_values.size());
    for (auto coord : query_values) {
        auto match = rows.find(coord);
        out.push_back(match == rows.end() ? -1 : match->second);
    }
    return make_i32_array(out);
}

// MARK: - maps

NativeKernelMap build_kernel_map(
    const mx::array& coords,
    Triple kernel_size, // NOLINT(bugprone-easily-swappable-parameters)
    Triple stride,
    Triple padding, // NOLINT(bugprone-easily-swappable-parameters)
    Triple dilation
) {
    auto offsets = mlx_lattice::kernel_offsets(kernel_size, dilation);
    auto values = read_coords(coords);
    auto rows = first_row_map(values);
    bool identity_out = stride == Triple{1, 1, 1} && padding == Triple{0, 0, 0};
    auto out_values = identity_out ? values : downsample_values(values, stride);

    std::vector<Edge> edges;
    edges.reserve(out_values.size() * offsets.size());
    for (int kernel = 0; kernel < int(offsets.size()); ++kernel) {
        auto offset = offsets[kernel];
        for (int out_row = 0; out_row < int(out_values.size()); ++out_row) {
            auto candidate = kernel_input_coord(
                out_values[out_row], offset, stride, padding
            );
            auto match = rows.find(candidate);
            if (match != rows.end()) {
                edges.push_back({
                    match->second,
                    static_cast<int32_t>(out_row),
                    static_cast<int32_t>(kernel),
                });
            }
        }
    }

    return make_map(edges, out_values, offsets, coords.dtype());
}

NativeKernelMap build_generative_map(
    const mx::array& coords,
    Triple kernel_size, // NOLINT(bugprone-easily-swappable-parameters)
    Triple stride
) {
    auto offsets = mlx_lattice::kernel_offsets(kernel_size);
    auto values = read_coords(coords);
    std::vector<Edge> edges;
    std::vector<Coord> out_values;
    edges.reserve(values.size() * offsets.size());
    out_values.reserve(values.size() * offsets.size());

    for (int in_row = 0; in_row < int(values.size()); ++in_row) {
        auto coord = values[in_row];
        for (int kernel = 0; kernel < int(offsets.size()); ++kernel) {
            auto offset = offsets[kernel];
            auto out_row = int(out_values.size());
            out_values.push_back({
                coord[0],
                coord[1] * stride[0] + offset[0],
                coord[2] * stride[1] + offset[1],
                coord[3] * stride[2] + offset[2],
            });
            edges.push_back({
                static_cast<int32_t>(in_row),
                static_cast<int32_t>(out_row),
                static_cast<int32_t>(kernel),
            });
        }
    }

    return make_map(edges, out_values, offsets, coords.dtype());
}

NativeKernelMap build_transposed_kernel_map(
    const mx::array& coords,
    Triple kernel_size, // NOLINT(bugprone-easily-swappable-parameters)
    Triple stride,
    Triple padding, // NOLINT(bugprone-easily-swappable-parameters)
    Triple dilation
) {
    auto offsets = mlx_lattice::kernel_offsets(kernel_size, dilation);
    auto values = read_coords(coords);
    std::vector<Edge> edges;
    std::vector<Coord> out_values;
    std::unordered_map<Coord, int32_t, CoordHash> out_rows;
    edges.reserve(values.size() * offsets.size());
    out_values.reserve(values.size() * offsets.size());
    out_rows.reserve(values.size() * offsets.size());

    for (int in_row = 0; in_row < int(values.size()); ++in_row) {
        auto coord = values[in_row];
        for (int kernel = 0; kernel < int(offsets.size()); ++kernel) {
            auto offset = offsets[kernel];
            Coord candidate = {
                coord[0],
                coord[1] * stride[0] + offset[0] - padding[0],
                coord[2] * stride[1] + offset[1] - padding[1],
                coord[3] * stride[2] + offset[2] - padding[2],
            };
            auto [match, inserted] = out_rows.emplace(
                candidate, static_cast<int32_t>(out_values.size())
            );
            if (inserted) {
                out_values.push_back(candidate);
            }
            edges.push_back({
                static_cast<int32_t>(in_row),
                match->second,
                static_cast<int32_t>(kernel),
            });
        }
    }

    return make_map(edges, out_values, offsets, coords.dtype());
}

} // namespace mlx_lattice::cpu
