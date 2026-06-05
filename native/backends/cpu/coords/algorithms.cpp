#include "backends/cpu/coords/algorithms.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mlx/device.h"
#include "mlx/ops.h"
#include "ops/coords.h"

namespace mlx_lattice::coords::cpu {

namespace {

// MARK: - types

using Coord = std::array<int64_t, 4>;
using Edge = std::array<int32_t, 3>;

struct OutputCsrData {
    std::vector<int32_t> offsets;
    std::vector<int32_t> in_rows;
    std::vector<int32_t> kernel_ids;
};

struct KernelBucketData {
    std::vector<int32_t> offsets;
    std::vector<int32_t> in_rows;
    std::vector<int32_t> out_rows;
};

struct InputCsrData {
    std::vector<int32_t> offsets;
    std::vector<int32_t> out_rows;
    std::vector<int32_t> kernel_ids;
};

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

std::vector<Triple> read_offsets(const mx::array& offsets) {
    auto cpu_offsets = mx::contiguous(offsets, false, mx::Device::cpu);
    cpu_offsets.eval();
    cpu_offsets.wait();

    std::vector<Triple> out;
    out.reserve(cpu_offsets.shape(0));
    auto data = cpu_offsets.data<int32_t>();
    for (int row = 0; row < cpu_offsets.shape(0); ++row) {
        auto base = static_cast<ptrdiff_t>(row) * 3;
        out.push_back({data[base], data[base + 1], data[base + 2]});
    }
    return out;
}

void write_i32(mx::array& out, const std::vector<int32_t>& values) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto data = out.data<int32_t>();
    std::fill(data, data + out.size(), 0);
    std::copy(values.begin(), values.end(), data);
}

void write_count(mx::array& out, int first, int second = 0) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto data = out.data<int32_t>();
    std::fill(data, data + out.size(), 0);
    data[0] = first;
    if (out.size() > 1) {
        data[1] = second;
    }
}

void write_coords(
    mx::array& out,
    const std::vector<Coord>& coords,
    mx::Dtype dtype
) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
    if (dtype == mx::int32) {
        auto data = out.data<int32_t>();
        std::fill(data, data + out.size(), 0);
        for (int row = 0; row < int(coords.size()); ++row) {
            auto base = static_cast<ptrdiff_t>(row) * 4;
            for (int axis = 0; axis < 4; ++axis) {
                data[base + axis] = static_cast<int32_t>(coords[row][axis]);
            }
        }
        return;
    }

    auto data = out.data<int64_t>();
    std::fill(data, data + out.size(), 0);
    for (int row = 0; row < int(coords.size()); ++row) {
        auto base = static_cast<ptrdiff_t>(row) * 4;
        for (int axis = 0; axis < 4; ++axis) {
            data[base + axis] = coords[row][axis];
        }
    }
}

// MARK: - views

template <typename Key>
std::vector<int32_t>
view_offsets(const std::vector<Edge>& edges, int rows, Key key) {
    std::vector<int32_t> offsets(static_cast<size_t>(rows) + 1, 0);
    for (auto edge : edges) {
        offsets[static_cast<size_t>(key(edge)) + 1] += 1;
    }
    for (int row = 0; row < rows; ++row) {
        offsets[static_cast<size_t>(row) + 1] += offsets[row];
    }
    return offsets;
}

OutputCsrData output_csr_view(const std::vector<Edge>& edges, int n_out_rows) {
    auto offsets =
        view_offsets(edges, n_out_rows, [](Edge edge) { return edge[1]; });
    auto cursor = offsets;
    std::vector<int32_t> in_rows(edges.size());
    std::vector<int32_t> kernel_ids(edges.size());
    for (auto edge : edges) {
        auto index = cursor[static_cast<size_t>(edge[1])]++;
        in_rows[static_cast<size_t>(index)] = edge[0];
        kernel_ids[static_cast<size_t>(index)] = edge[2];
    }
    return {offsets, in_rows, kernel_ids};
}

KernelBucketData
kernel_bucket_view(const std::vector<Edge>& edges, int n_kernels) {
    auto offsets =
        view_offsets(edges, n_kernels, [](Edge edge) { return edge[2]; });
    auto cursor = offsets;
    std::vector<int32_t> in_rows(edges.size());
    std::vector<int32_t> out_rows(edges.size());
    for (auto edge : edges) {
        auto index = cursor[static_cast<size_t>(edge[2])]++;
        in_rows[static_cast<size_t>(index)] = edge[0];
        out_rows[static_cast<size_t>(index)] = edge[1];
    }
    return {offsets, in_rows, out_rows};
}

InputCsrData input_csr_view(const std::vector<Edge>& edges, int n_in_rows) {
    auto offsets =
        view_offsets(edges, n_in_rows, [](Edge edge) { return edge[0]; });
    auto cursor = offsets;
    std::vector<int32_t> out_rows(edges.size());
    std::vector<int32_t> kernel_ids(edges.size());
    for (auto edge : edges) {
        auto index = cursor[static_cast<size_t>(edge[0])]++;
        out_rows[static_cast<size_t>(index)] = edge[1];
        kernel_ids[static_cast<size_t>(index)] = edge[2];
    }
    return {offsets, out_rows, kernel_ids};
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

void write_compact_map(
    std::vector<mx::array>& outputs,
    const std::vector<Edge>& edges,
    const std::vector<Coord>& out_coords,
    const std::vector<Triple>& offsets,
    int n_in_rows,
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

    auto output_csr = output_csr_view(edges, int(out_coords.size()));
    auto kernel_buckets = kernel_bucket_view(edges, int(offsets.size()));
    auto input_csr = input_csr_view(edges, n_in_rows);

    write_i32(outputs[0], in_rows);
    write_i32(outputs[1], out_rows);
    write_i32(outputs[2], kernel_ids);
    write_coords(outputs[3], out_coords, coord_dtype);
    write_count(outputs[4], int(edges.size()), int(out_coords.size()));
    write_i32(outputs[5], output_csr.offsets);
    write_i32(outputs[6], output_csr.in_rows);
    write_i32(outputs[7], output_csr.kernel_ids);
    write_i32(outputs[8], kernel_buckets.offsets);
    write_i32(outputs[9], kernel_buckets.in_rows);
    write_i32(outputs[10], kernel_buckets.out_rows);
    write_i32(outputs[11], input_csr.offsets);
    write_i32(outputs[12], input_csr.out_rows);
    write_i32(outputs[13], input_csr.kernel_ids);
}

// MARK: - set ops

std::vector<Coord> union_values(const mx::array& lhs, const mx::array& rhs) {
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
    return out;
}

std::vector<Coord>
intersection_values(const mx::array& lhs, const mx::array& rhs) {
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
    return out;
}

std::vector<int32_t>
lookup_values(const mx::array& coords, const mx::array& queries) {
    auto rows = first_row_map(read_coords(coords));
    auto query_values = read_coords(queries);
    std::vector<int32_t> out;
    out.reserve(query_values.size());
    for (auto coord : query_values) {
        auto match = rows.find(coord);
        out.push_back(match == rows.end() ? -1 : match->second);
    }
    return out;
}

// MARK: - maps

void write_kernel_map(
    std::vector<mx::array>& outputs,
    const mx::array& coords,
    const std::vector<Triple>& offsets,
    Triple stride,
    Triple padding
) {
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

    write_compact_map(
        outputs, edges, out_values, offsets, int(values.size()), coords.dtype()
    );
}

void write_generative_map(
    std::vector<mx::array>& outputs,
    const mx::array& coords,
    const std::vector<Triple>& offsets,
    Triple stride
) {
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

    auto output_csr = output_csr_view(edges, int(out_values.size()));
    auto kernel_buckets = kernel_bucket_view(edges, int(offsets.size()));
    auto input_csr = input_csr_view(edges, int(values.size()));

    write_i32(outputs[0], in_rows);
    write_i32(outputs[1], out_rows);
    write_i32(outputs[2], kernel_ids);
    write_coords(outputs[3], out_values, coords.dtype());
    write_i32(outputs[4], output_csr.offsets);
    write_i32(outputs[5], output_csr.in_rows);
    write_i32(outputs[6], output_csr.kernel_ids);
    write_i32(outputs[7], kernel_buckets.offsets);
    write_i32(outputs[8], kernel_buckets.in_rows);
    write_i32(outputs[9], kernel_buckets.out_rows);
    write_i32(outputs[10], input_csr.offsets);
    write_i32(outputs[11], input_csr.out_rows);
    write_i32(outputs[12], input_csr.kernel_ids);
}

void write_transposed_kernel_map(
    std::vector<mx::array>& outputs,
    const mx::array& coords,
    const std::vector<Triple>& offsets,
    Triple stride,
    Triple padding
) {
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

    write_compact_map(
        outputs, edges, out_values, offsets, int(values.size()), coords.dtype()
    );
}

} // namespace

// MARK: - primitive eval

void eval_set_coords(
    CoordSetOp op,
    Triple stride,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    std::vector<Coord> values;
    switch (op) {
    case CoordSetOp::Downsample:
        values = downsample_values(read_coords(inputs[0]), stride);
        break;
    case CoordSetOp::Union:
        values = union_values(inputs[0], inputs[1]);
        break;
    case CoordSetOp::Intersection:
        values = intersection_values(inputs[0], inputs[1]);
        break;
    }

    write_coords(outputs[0], values, inputs[0].dtype());
    write_count(outputs[1], int(values.size()));
}

void eval_lookup_coords(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    write_i32(outputs[0], lookup_values(inputs[0], inputs[1]));
}

void eval_generic_kernel_map(
    CoordMapOp op,
    Triple stride,
    Triple padding,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    auto offsets = read_offsets(inputs[1]);

    switch (op) {
    case CoordMapOp::Forward:
        write_kernel_map(outputs, inputs[0], offsets, stride, padding);
        break;
    case CoordMapOp::Transposed:
        write_transposed_kernel_map(
            outputs, inputs[0], offsets, stride, padding
        );
        break;
    }
}

void eval_generative_kernel_map(
    Triple stride,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    write_generative_map(outputs, inputs[0], read_offsets(inputs[1]), stride);
}

} // namespace mlx_lattice::coords::cpu
