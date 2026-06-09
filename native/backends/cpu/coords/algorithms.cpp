#include "backends/cpu/coords/algorithms.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "backends/array_utils.h"
#include "backends/cpu/schedule.h"

namespace mlx_lattice::coords::cpu {

namespace {

// MARK: - types

using Coord = std::array<int64_t, 4>;
using Edge = std::array<int32_t, 3>;

struct NeighborCandidate {
    int32_t source_row;
    float distance;
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

std::vector<Coord> read_coords(const mx::array& coords) {
    std::vector<Coord> out;
    out.reserve(coords.shape(0));
    if (coords.dtype() == mx::int32) {
        auto data = coords.data<int32_t>();
        for (int row = 0; row < coords.shape(0); ++row) {
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

    auto data = coords.data<int64_t>();
    for (int row = 0; row < coords.shape(0); ++row) {
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
    std::vector<Triple> out;
    out.reserve(offsets.shape(0));
    auto data = offsets.data<int32_t>();
    for (int row = 0; row < offsets.shape(0); ++row) {
        auto base = static_cast<ptrdiff_t>(row) * 3;
        out.push_back({data[base], data[base + 1], data[base + 2]});
    }
    return out;
}

int read_scalar_i32(const mx::array& value) { return value.data<int32_t>()[0]; }

void write_i32(mx::array& out, const std::vector<int32_t>& values) {
    auto data = out.data<int32_t>();
    std::fill(data, data + out.size(), 0);
    std::copy(values.begin(), values.end(), data);
}

void write_f32(mx::array& out, const std::vector<float>& values) {
    auto data = out.data<float>();
    std::fill(data, data + out.size(), 0.0F);
    std::copy(values.begin(), values.end(), data);
}

void write_count(mx::array& out, int first, int second = 0) {
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

void write_map_rows(
    std::vector<mx::array>& outputs,
    const std::vector<Edge>& edges
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

    write_i32(outputs[RelationInRows], in_rows);
    write_i32(outputs[RelationOutRows], out_rows);
    write_i32(outputs[RelationKernelIds], kernel_ids);
}

void write_map(
    std::vector<mx::array>& outputs,
    const std::vector<Edge>& edges,
    const std::vector<Coord>& out_coords,
    mx::Dtype coord_dtype,
    bool compact
) {
    write_map_rows(outputs, edges);
    write_coords(outputs[RelationOutCoords], out_coords, coord_dtype);
    if (compact) {
        write_count(
            outputs[RelationCounts], int(edges.size()), int(out_coords.size())
        );
    }
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

bool same_batch(Coord lhs, Coord rhs) { return lhs[0] == rhs[0]; }

float squared_spatial_distance(Coord lhs, Coord rhs) {
    auto dx = static_cast<float>(lhs[1] - rhs[1]);
    auto dy = static_cast<float>(lhs[2] - rhs[2]);
    auto dz = static_cast<float>(lhs[3] - rhs[3]);
    return dx * dx + dy * dy + dz * dz;
}

bool closer_neighbor(
    const NeighborCandidate& lhs,
    const NeighborCandidate& rhs
) {
    if (lhs.distance != rhs.distance) {
        return lhs.distance < rhs.distance;
    }
    return lhs.source_row < rhs.source_row;
}

void write_neighbor_relation(
    std::vector<mx::array>& outputs,
    NeighborRelationOp op,
    const mx::array& source_coords,
    int source_active_rows,
    const mx::array& query_coords,
    int query_active_rows,
    NeighborRelationShape shape,
    float radius_squared
) {
    auto source_values = read_coords(source_coords);
    source_values.resize(
        std::min(source_active_rows, int(source_values.size()))
    );
    auto query_values = read_coords(query_coords);
    query_values.resize(std::min(query_active_rows, int(query_values.size())));

    std::vector<int32_t> query_rows;
    std::vector<int32_t> source_rows;
    std::vector<int32_t> neighbor_ids;
    std::vector<float> distances;
    query_rows.reserve(query_values.size() * shape.max_neighbors);
    source_rows.reserve(query_values.size() * shape.max_neighbors);
    neighbor_ids.reserve(query_values.size() * shape.max_neighbors);
    distances.reserve(query_values.size() * shape.max_neighbors);

    std::vector<NeighborCandidate> candidates;
    candidates.reserve(source_values.size());
    for (int query_row = 0; query_row < int(query_values.size()); ++query_row) {
        candidates.clear();
        auto query = query_values[query_row];
        for (int source_row = 0; source_row < int(source_values.size());
             ++source_row) {
            auto source = source_values[source_row];
            if (!same_batch(query, source)) {
                continue;
            }
            auto distance = squared_spatial_distance(query, source);
            if (op == NeighborRelationOp::Radius && distance > radius_squared) {
                continue;
            }
            candidates.push_back({static_cast<int32_t>(source_row), distance});
        }
        std::sort(candidates.begin(), candidates.end(), closer_neighbor);
        auto limit = std::min(shape.max_neighbors, int(candidates.size()));
        for (int neighbor = 0; neighbor < limit; ++neighbor) {
            query_rows.push_back(static_cast<int32_t>(query_row));
            source_rows.push_back(candidates[neighbor].source_row);
            neighbor_ids.push_back(static_cast<int32_t>(neighbor));
            distances.push_back(candidates[neighbor].distance);
        }
    }

    write_i32(outputs[NeighborQueryRows], query_rows);
    write_i32(outputs[NeighborSourceRows], source_rows);
    write_i32(outputs[NeighborIds], neighbor_ids);
    write_f32(outputs[NeighborDistances], distances);
    write_count(
        outputs[NeighborCounts],
        int(query_rows.size()),
        int(query_values.size())
    );
}

// MARK: - relations

void write_kernel_relation(
    std::vector<mx::array>& outputs,
    const mx::array& coords,
    int active_rows,
    const std::vector<Triple>& offsets,
    Triple stride,
    Triple padding
) {
    auto values = read_coords(coords);
    values.resize(std::min(active_rows, int(values.size())));
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

    write_map(outputs, edges, out_values, coords.dtype(), true);
}

void write_generative_relation(
    std::vector<mx::array>& outputs,
    const mx::array& coords,
    int active_rows,
    const std::vector<Triple>& offsets,
    Triple stride
) {
    auto values = read_coords(coords);
    values.resize(std::min(active_rows, int(values.size())));
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

    write_map(outputs, edges, out_values, coords.dtype(), true);
}

void write_transposed_kernel_relation(
    std::vector<mx::array>& outputs,
    const mx::array& coords,
    int active_rows,
    const std::vector<Triple>& offsets,
    Triple stride,
    Triple padding
) {
    auto values = read_coords(coords);
    values.resize(std::min(active_rows, int(values.size())));
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

    write_map(outputs, edges, out_values, coords.dtype(), true);
}

} // namespace

// MARK: - primitive eval

void eval_set_coords(
    CoordSetOp op,
    Triple stride,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    backend::schedule_cpu(
        stream,
        inputs,
        outputs,
        [op, stride](
            const std::vector<mx::array>& task_inputs,
            std::vector<mx::array>& task_outputs
        ) {
            std::vector<Coord> values;
            switch (op) {
            case CoordSetOp::Downsample:
                values = downsample_values(read_coords(task_inputs[0]), stride);
                break;
            case CoordSetOp::Union:
                values = union_values(task_inputs[0], task_inputs[1]);
                break;
            case CoordSetOp::Intersection:
                values = intersection_values(task_inputs[0], task_inputs[1]);
                break;
            }

            write_coords(task_outputs[0], values, task_inputs[0].dtype());
            write_count(task_outputs[1], int(values.size()));
        }
    );
}

void eval_lookup_coords(
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    backend::schedule_cpu(
        stream,
        inputs,
        outputs,
        [](const std::vector<mx::array>& task_inputs,
           std::vector<mx::array>& task_outputs) {
            write_i32(
                task_outputs[0], lookup_values(task_inputs[0], task_inputs[1])
            );
        }
    );
}

void eval_generic_kernel_relation(
    CoordRelationOp op,
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
        [op, stride, padding](
            const std::vector<mx::array>& task_inputs,
            std::vector<mx::array>& task_outputs
        ) {
            auto offsets = read_offsets(task_inputs[1]);
            auto active_rows = read_scalar_i32(task_inputs[2]);

            switch (op) {
            case CoordRelationOp::Forward:
                write_kernel_relation(
                    task_outputs,
                    task_inputs[0],
                    active_rows,
                    offsets,
                    stride,
                    padding
                );
                break;
            case CoordRelationOp::Transposed:
                write_transposed_kernel_relation(
                    task_outputs,
                    task_inputs[0],
                    active_rows,
                    offsets,
                    stride,
                    padding
                );
                break;
            }
        }
    );
}

void eval_generative_kernel_relation(
    Triple stride,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    backend::schedule_cpu(
        stream,
        inputs,
        outputs,
        [stride](
            const std::vector<mx::array>& task_inputs,
            std::vector<mx::array>& task_outputs
        ) {
            auto active_rows = read_scalar_i32(task_inputs[2]);
            write_generative_relation(
                task_outputs,
                task_inputs[0],
                active_rows,
                read_offsets(task_inputs[1]),
                stride
            );
        }
    );
}

void eval_neighbor_relation(
    NeighborRelationOp op,
    NeighborRelationShape shape,
    float radius_squared,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    backend::allocate_all(outputs);
    backend::schedule_cpu(
        stream,
        inputs,
        outputs,
        [op, shape, radius_squared](
            const std::vector<mx::array>& task_inputs,
            std::vector<mx::array>& task_outputs
        ) {
            write_neighbor_relation(
                task_outputs,
                op,
                task_inputs[0],
                read_scalar_i32(task_inputs[2]),
                task_inputs[1],
                read_scalar_i32(task_inputs[3]),
                shape,
                radius_squared
            );
        }
    );
}

} // namespace mlx_lattice::coords::cpu
